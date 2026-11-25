"""
Step 5: live attendance with face recognition + person segmentation.
 
Per frame:
  WHO   : detect faces -> track each face across frames -> ArcFace embedding
          -> identity from the gallery, using the mean of the last N embeddings of that face.
  WHERE : YOLO-seg gives one mask per person; each body takes the color of the face inside it.
 
- Known person (in the gallery): logged ONCE in attendance.csv.            -> green
- Unknown person: temporary id (Unknown 1, Unknown 2, ...) kept in memory
  while the camera runs, so the same stranger is not counted twice.        -> red
- Still deciding -> yellow.   Body with no visible face -> gray.
 
It also writes status.json about once per second (people in view, counters, fps),
so dashboard.py can show the live state without touching the camera.
 
Keys:
  e      enroll a person: type the name in the TERMINAL, then look at the camera
  r, r   press r twice within 2 s to reset attendance (old file is archived)
  d      toggle similarity scores
  q/ESC  quit
Run from the face_recognition/ folder:  python attendance_camera.py
"""
import json
import logging
import os
import time
from collections import deque
 
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO
 
import config as C
from attendance import AttendanceLog
from extract_faces import check_quality
from gallery import Gallery
from live_camera import iou
from live_camera_seg import ALPHA, SEG_MODEL, body_colors, person_masks
 
log = logging.getLogger("attendance_cam")
 
# ---------- Settings ----------
CAMERA_INDEX = 0
FRAME_SIZE = (1280, 720)
MIRROR = True
WINDOW = "Attendance"
SMOOTH_N = 7              # identity from the mean of the last N embeddings of a face
MIN_FRAMES = 3            # show "..." until a face has this many embeddings
UNKNOWN_MIN_FRAMES = 10   # a face must stay unknown this long before it gets an Unknown id
IOU_MATCH = 0.3           # box overlap to treat a detection as the same face as before
MAX_MISSED = 10           # forget a tracked face after this many frames without a match
ENROLL_SAMPLES = 25       # embeddings collected per enrollment
ENROLL_MIN = 10           # fewer good embeddings than this -> enrollment cancelled
ENROLL_TIMEOUT = 20.0     # seconds
RESET_CONFIRM_S = 2.0     # the second "r" must come within this time
FLASH_S = 2.5             # how long on-screen messages stay
STATUS_FILE = "status.json"   # live state for dashboard.py
STATUS_EVERY_S = 1.0          # write it about once per second
 
GREEN, RED, YELLOW, GRAY, WHITE = (0, 200, 0), (0, 0, 230), (0, 200, 230), (160, 160, 160), (255, 255, 255)
 
 
def normalize(v):
    return v / np.linalg.norm(v)
 
 
def face_area(f):
    return (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
 
 
class Track:
    """One face followed across frames, with its recent embeddings."""
 
    def __init__(self, bbox, frame_idx):
        self.bbox = bbox
        self.last_seen = frame_idx
        self.embs = deque(maxlen=SMOOTH_N)
        self.unknown_frames = 0
        self.unknown_id = None
 
 
def match_track(tracks, bbox, used, frame_idx):
    """Greedy match by box overlap; start a new track if nothing overlaps enough."""
    best, best_iou = None, IOU_MATCH
    for t in tracks:
        if t in used:
            continue
        v = iou(t.bbox, bbox)
        if v >= best_iou:
            best, best_iou = t, v
    if best is None:
        best = Track(bbox, frame_idx)
        tracks.append(best)
    used.add(best)
    return best
 
 
class UnknownMemory:
    """Strangers seen while the camera runs: one running-mean embedding per stranger (memory only)."""
 
    def __init__(self, threshold):
        self.threshold = threshold
        self.clear()
 
    def clear(self):
        self.centroids, self.counts = [], []
 
    def __len__(self):
        return len(self.centroids)
 
    def assign(self, emb):
        """Id (1, 2, ...) of the matching stranger, or a new id if nobody seen so far is close enough."""
        if self.centroids:
            sims = np.array(self.centroids) @ emb
            i = int(np.argmax(sims))
            if sims[i] >= self.threshold:
                n = self.counts[i]
                self.centroids[i] = normalize(self.centroids[i] * n + emb)   # running mean
                self.counts[i] = n + 1
                return i + 1
        self.centroids.append(emb)
        self.counts.append(1)
        return len(self.centroids)
 
 
def update_tracks(app, frame, gallery, att, unknowns, tracks, frame_idx, events):
    """Recognize every face in the frame (biggest first).
    Returns a list of (bbox, label, color, similarity_or_None)."""
    results, used = [], set()
    faces = sorted((f for f in app.get(frame) if f.det_score >= C.MIN_DET_SCORE), key=face_area, reverse=True)
    for f in faces:
        bbox = f.bbox.astype(float)
        t = match_track(tracks, bbox, used, frame_idx)
        t.bbox, t.last_seen = bbox, frame_idx
 
        if bbox[2] - bbox[0] < C.MIN_FACE_PX:                 # same size rule as the training data
            results.append((bbox, "too far", GRAY, None))
            continue
 
        t.embs.append(f.normed_embedding)
        if len(t.embs) < MIN_FRAMES:
            results.append((bbox, "...", YELLOW, None))
            continue
 
        emb = normalize(np.mean(t.embs, axis=0))              # mean of the last N embeddings
        name, sim = gallery.identify(emb)
        if name:
            t.unknown_frames = 0
            if len(t.embs) == SMOOTH_N and att.mark(name):     # log only once the window is full (stable)
                events.append(f"Logged: {name}")
            results.append((bbox, name, GREEN, sim))
        else:
            t.unknown_frames += 1
            if t.unknown_id is None and t.unknown_frames >= UNKNOWN_MIN_FRAMES:
                t.unknown_id = unknowns.assign(emb)
            if t.unknown_id:
                results.append((bbox, f"Unknown {t.unknown_id}", RED, sim))
            else:
                results.append((bbox, "...", YELLOW, sim))
 
    tracks[:] = [t for t in tracks if frame_idx - t.last_seen <= MAX_MISSED]
    return results
 
 
def draw_box(view, bbox, text, color, width):
    """Box + label; if the view is mirrored, flip the x coordinates too."""
    x1, y1, x2, y2 = bbox
    if MIRROR:
        x1, x2 = width - x2, width - x1
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
    cv2.putText(view, text, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
 
def draw_panel(view, att, unknowns, gallery, fps, messages, n_people=0):
    """Counters at the top, present names on the left, messages in the middle, key help at the bottom."""
    h, w = view.shape[:2]
    cv2.rectangle(view, (0, 0), (w, 36), (40, 40, 40), -1)
    cv2.putText(view, f"In view: {n_people}   Known present: {len(att)}   Unknown seen: {len(unknowns)}   "
                      f"Gallery: {len(gallery)}   {fps:.0f} FPS",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)
    for i, row in enumerate(att.rows[-8:]):                   # last 8 people logged
        cv2.putText(view, f"{row['name']}  {row['time']}", (10, 66 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
    for i, msg in enumerate(messages):
        cv2.putText(view, msg, (w // 2 - 200, h // 2 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2)
    cv2.putText(view, "e = enroll   r,r = reset attendance   d = scores   q = quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
 
 
def enroll(app, cap, name):
    """Collect embeddings of exactly ONE good-quality face for a few seconds. Returns a list of embeddings."""
    embs, t0 = [], time.time()
    while len(embs) < ENROLL_SAMPLES and time.time() - t0 < ENROLL_TIMEOUT:
        ok, frame = cap.read()
        if not ok:
            break
        faces = [f for f in app.get(frame) if f.det_score >= C.MIN_DET_SCORE]
        color = YELLOW
        if len(faces) != 1:
            msg, color = ("No face" if not faces else "Only ONE face in view, please"), RED
        else:
            reason, _ = check_quality(frame, faces[0])        # same filters as the training data
            emb = faces[0].normed_embedding.astype(np.float32)
            if reason:
                msg = f"Low quality ({reason}): come closer, hold still"
            elif embs and max(float(e @ emb) for e in embs) > C.DEDUP_SIM:
                msg = "Move your head slowly: left, right, up, down"
            else:
                embs.append(emb)
                msg, color = "Good", GREEN
 
        view = cv2.flip(frame, 1) if MIRROR else frame.copy()
        for f in faces:
            draw_box(view, f.bbox, "", color, frame.shape[1])
        cv2.putText(view, f"Enrolling {name}: {len(embs)}/{ENROLL_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        cv2.putText(view, msg, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(view, "q = cancel", (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        cv2.imshow(WINDOW, view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return []
    return embs
 
 
def run_enrollment(app, cap, gallery):
    """Ask for a name in the terminal, collect embeddings, warn about look-alikes, save. Returns a message."""
    print("\n--- Enrollment: click on this terminal ---")
    name = input("Name (Latin letters, empty = cancel): ").strip()
    if not name:
        return "Enrollment cancelled"
    embs = enroll(app, cap, name)
    if len(embs) < ENROLL_MIN:
        return f"Enrollment cancelled: only {len(embs)} good frames"
    other, sim = gallery.best_match(normalize(np.mean(embs, axis=0)))
    if other and other != name and sim >= C.MATCH_THRESHOLD:
        ans = input(f"This face looks like '{other}' (similarity {sim:.2f}). Save as '{name}' anyway? [y/N] ")
        if ans.strip().lower() != "y":
            return "Enrollment cancelled (looks like an existing person)"
    total = gallery.add(name, embs)
    log.info("enrolled %s: %d new, %d total embeddings", name, len(embs), total)
    return f"Enrolled {name} ({total} embeddings)"
 
 
def write_status(running, in_view, att, unknowns, gallery, fps, labels):
    """Live state for the dashboard. Written to a temp file, then renamed, so a reader never sees half a file."""
    data = {
        "running": running,
        "updated": time.time(),
        "in_view": in_view,
        "known_present": len(att),
        "unknown_seen": len(unknowns),
        "gallery_size": len(gallery),
        "fps": round(fps, 1),
        "labels": labels,
    }
    tmp = STATUS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, STATUS_FILE)
 
 
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gallery = Gallery()
    att = AttendanceLog()
    unknowns = UnknownMemory(C.MATCH_THRESHOLD)
    if not len(gallery):
        log.warning("gallery is empty: press e to enroll someone")
 
    app = FaceAnalysis(
        name=C.MODEL_NAME,
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=C.DET_SIZE)
    seg_model = YOLO(SEG_MODEL)
 
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
 
    tracks, frame_idx, debug, fps = [], 0, False, 0.0
    t_prev, last_r, flashes, last_status = time.perf_counter(), 0.0, [], 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("Camera read failed")
                break
 
            # 1) WHO: faces, identity, attendance, unknown ids
            events = []
            results = update_tracks(app, frame, gallery, att, unknowns, tracks, frame_idx, events)
            now = time.time()
            flashes += [(e, now + FLASH_S) for e in events]
            flashes = [(m, t) for m, t in flashes if t > now]
 
            # 2) WHERE: one mask per person, colored like the face inside it
            masks = person_masks(seg_model, frame)
            colors = body_colors(results, masks)
 
            tp = time.perf_counter()
            inst = 1.0 / max(tp - t_prev, 1e-6)
            fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
            t_prev = tp
 
            # 3) draw: masks first (on the original frame), then mirror, then boxes and text
            overlay = frame.copy()
            for mask, color in zip(masks, colors):
                overlay[mask.astype(bool)] = color
            view = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)
            if MIRROR:
                view = cv2.flip(view, 1)
            for bbox, label, color, sim in results:
                text = f"{label} {sim:.2f}" if (debug and sim is not None) else label
                draw_box(view, bbox, text, color, frame.shape[1])
            draw_panel(view, att, unknowns, gallery, fps, [m for m, _ in flashes], n_people=len(masks))
            cv2.imshow(WINDOW, view)
 
            if now - last_status >= STATUS_EVERY_S:            # live state for dashboard.py
                labels = [label for _, label, _, _ in results if label not in ("...", "too far")]
                write_status(True, len(masks), att, unknowns, gallery, fps, labels)
                last_status = now
 
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("e"):
                msg = run_enrollment(app, cap, gallery)
                flashes.append((msg, time.time() + FLASH_S))
                tracks.clear()                                 # identities are recomputed with the new gallery
                t_prev = time.perf_counter()
            if key == ord("r"):
                if now - last_r <= RESET_CONFIRM_S:
                    att.reset()
                    unknowns.clear()
                    tracks.clear()
                    flashes.append(("Attendance reset (old file archived)", now + FLASH_S))
                    last_r = 0.0
                else:
                    last_r = now
                    flashes.append(("Press r again to reset attendance", now + RESET_CONFIRM_S))
            frame_idx += 1
    finally:
        write_status(False, 0, att, unknowns, gallery, 0.0, [])   # tell the dashboard the camera stopped
        cap.release()
        cv2.destroyAllWindows()
        log.info("present: %d %s | unknown seen: %d", len(att), sorted(att.present), len(unknowns))
 
 
if __name__ == "__main__":
    main()
 