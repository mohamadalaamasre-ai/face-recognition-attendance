"""
Step 3: live webcam recognition.

Pipeline per frame:
  detect faces (SCRFD) -> embedding (ArcFace) -> score (model A or B, read from models/meta.json)
  -> temporal smoothing per tracked face (median of the last N scores) -> label.

Keys:  q or ESC = quit,  d = toggle debug scores.
Run from the face_recognition/ folder:  python live_camera.py
"""
import json
import logging
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
from insightface.app import FaceAnalysis

import config as C

log = logging.getLogger("live")

# ---------- Live settings (tunable) ----------
NAME = "Mohamad Alaa"      # label shown for the recognized person
CAMERA_INDEX = 0
FRAME_SIZE = (1280, 720)   # requested camera resolution (the camera may return something else)
MIRROR = True              # show a mirrored preview (detection still runs on the original frame)
SMOOTH_N = 7               # decide from the median of the last N scores of the same face
MIN_FRAMES = 3             # show "..." until a face has at least this many scores
IOU_MATCH = 0.3            # minimum box overlap to treat a detection as the same face as before
MAX_MISSED = 10            # forget a tracked face after this many frames without a match
MODELS_DIR = Path("models")

GREEN, RED, YELLOW, GRAY = (0, 200, 0), (0, 0, 230), (0, 200, 230), (160, 160, 160)  # BGR


def load_scorer():
    """Return (scoring function, model name, threshold) according to models/meta.json."""
    meta = json.loads((MODELS_DIR / "meta.json").read_text())
    model = meta["model"]
    if model == "A":
        centroid = np.load(MODELS_DIR / "centroid.npy")

        def scorer(emb):
            return float(emb @ centroid)          # cosine similarity (embeddings are L2-normalized)
    elif model == "B":
        clf = joblib.load(MODELS_DIR / "classifier.joblib")

        def scorer(emb):
            return float(clf.decision_function(emb.reshape(1, -1))[0])
    else:
        raise ValueError(f"Unknown model in meta.json: {model!r}")
    return scorer, model, float(meta["threshold"])


def iou(a, b):
    """Intersection-over-union of two boxes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Track:
    """One face followed across frames, with its recent scores."""

    def __init__(self, bbox, frame_idx):
        self.bbox = bbox
        self.scores = deque(maxlen=SMOOTH_N)
        self.last_seen = frame_idx


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


def process_frame(app, frame, scorer, threshold, tracks, frame_idx):
    """Detect, score, smooth. Returns a list of (bbox, label, color, smoothed_score_or_None)."""
    results = []
    used = set()
    faces = [f for f in app.get(frame) if f.det_score >= C.MIN_DET_SCORE]
    faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))   # biggest first
    for f in faces:
        bbox = f.bbox.astype(float)
        track = match_track(tracks, bbox, used, frame_idx)
        track.bbox, track.last_seen = bbox, frame_idx

        if bbox[2] - bbox[0] < C.MIN_FACE_PX:                  # same size rule as in training data
            results.append((bbox, "too far", GRAY, None))
            continue

        track.scores.append(scorer(f.normed_embedding))
        if len(track.scores) < MIN_FRAMES:
            results.append((bbox, "...", YELLOW, None))
            continue

        smoothed = float(np.median(track.scores))              # robust to a single bad frame
        if smoothed >= threshold:
            results.append((bbox, NAME, GREEN, smoothed))
        else:
            results.append((bbox, "Unknown", RED, smoothed))

    tracks[:] = [t for t in tracks if frame_idx - t.last_seen <= MAX_MISSED]
    return results


def draw(view, results, debug, mirror_width=None):
    """Draw boxes and labels. If the view is mirrored, flip the x coordinates too."""
    for bbox, label, color, score in results:
        x1, y1, x2, y2 = bbox
        if mirror_width is not None:
            x1, x2 = mirror_width - x2, mirror_width - x1
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}" if (debug and score is not None) else label
        cv2.putText(view, text, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scorer, model, threshold = load_scorer()
    log.info("model=%s threshold=%.4f smooth_n=%d", model, threshold, SMOOTH_N)

    app = FaceAnalysis(
        name=C.MODEL_NAME,
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=C.DET_SIZE)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

    tracks, frame_idx, debug, fps, t_prev = [], 0, True, 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("Camera read failed")
                break

            results = process_frame(app, frame, scorer, threshold, tracks, frame_idx)

            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
            t_prev = now

            view = cv2.flip(frame, 1) if MIRROR else frame
            draw(view, results, debug, mirror_width=frame.shape[1] if MIRROR else None)
            cv2.putText(view, f"model {model}  thr {threshold:.2f}  {fps:.0f} FPS", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Face recognition", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
