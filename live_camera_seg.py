"""
Step 4b: live face recognition + person segmentation.
 
Face recognition says WHO (model A or B from models/meta.json).
YOLO-seg says WHERE (the pixels of each person's body).
Each body mask takes the color of the face found inside it:
  green = Alaa, red = Unknown, yellow = still deciding, gray = no face found in that body.
 
Keys: q or ESC = quit,  d = toggle debug scores.
Run from the face_recognition/ folder:  python live_camera_seg.py
"""
import logging
import time
 
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from ultralytics import YOLO
 
import config as C
from live_camera import CAMERA_INDEX, FRAME_SIZE, MIRROR, GRAY, draw, load_scorer, process_frame
 
log = logging.getLogger("live_seg")
 
# ---------- Segmentation settings ----------
SEG_MODEL = "yolo11n-seg.pt"                          # pretrained on COCO
SEG_IMGSZ = 640                                       # 5 ms on our GPU (check_gpu.py)
DEVICE = 0 if torch.cuda.is_available() else "cpu"    # fall back to CPU instead of crashing
PERSON_CLASS = 0                                      # COCO class 0 = person
MIN_PERSON_CONF = 0.5                                 # ignore weak person detections
FACE_IN_BODY = 0.5                                    # a face belongs to a body if >= 50% of its box is inside the mask
ALPHA = 0.45                                          # mask transparency
 
 
def person_masks(seg_model, frame):
    """Run YOLO-seg and return one binary mask (H, W) per person, in frame pixels."""
    result = seg_model.predict(frame, imgsz=SEG_IMGSZ, classes=[PERSON_CLASS],
                               conf=MIN_PERSON_CONF, device=DEVICE, verbose=False)[0]
    if result.masks is None:
        return []
    h, w = frame.shape[:2]
    masks = []
    for poly in result.masks.xy:                      # outline of one person
        if len(poly) < 3:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)  # 1 = body pixel, 0 = not body
        masks.append(mask)
    return masks
 
 
def face_coverage(bbox, mask):
    """Fraction (0..1) of the face box that lies inside the body mask."""
    h, w = mask.shape
    x1, y1 = max(int(bbox[0]), 0), max(int(bbox[1]), 0)
    x2, y2 = min(int(bbox[2]), w), min(int(bbox[3]), h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].mean())
 
 
def body_colors(face_results, masks):
    """Give each body mask the color of the face inside it; bodies without a face stay gray."""
    colors = [GRAY] * len(masks)
    taken = set()
    for bbox, _label, color, _score in face_results:  # biggest faces first (process_frame sorts them)
        best, best_cov = None, FACE_IN_BODY
        for i, mask in enumerate(masks):
            if i in taken:
                continue
            cov = face_coverage(bbox, mask)
            if cov >= best_cov:
                best, best_cov = i, cov
        if best is not None:
            colors[best] = color
            taken.add(best)
    return colors
 
 
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scorer, model, threshold = load_scorer()
    log.info("model=%s threshold=%.4f | YOLO-seg device=%s imgsz=%d", model, threshold, DEVICE, SEG_IMGSZ)
 
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
 
    tracks, frame_idx, debug, fps, t_prev = [], 0, True, 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("Camera read failed")
                break
 
            # 1) WHO: faces, scores, smoothing (same code as live_camera.py)
            face_results = process_frame(app, frame, scorer, threshold, tracks, frame_idx)
            # 2) WHERE: one mask per person
            masks = person_masks(seg_model, frame)
            # 3) link them: each body takes the color of its face
            colors = body_colors(face_results, masks)
 
            overlay = frame.copy()
            for mask, color in zip(masks, colors):
                overlay[mask.astype(bool)] = color
            view = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)
 
            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
            t_prev = now
 
            if MIRROR:
                view = cv2.flip(view, 1)
            draw(view, face_results, debug, mirror_width=frame.shape[1] if MIRROR else None)
            cv2.putText(view, f"model {model}  thr {threshold:.2f}  people {len(masks)}  {fps:.0f} FPS",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Face recognition + segmentation", view)
 
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
 