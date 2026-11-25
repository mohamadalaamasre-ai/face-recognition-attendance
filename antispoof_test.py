"""
Step 6a: anti-spoofing on the live camera, with a small labeled measurement.
 
Shows REAL / SPOOF and the score for every face.
To measure, keep exactly ONE face in view and press a key to say what you are showing:
  1 = a real face (bona fide)    2 = a PHOTO (printed or on a phone)    3 = a VIDEO / screen
  0 = stop recording             q / ESC = quit and print APCER / BPCER
Frames with more than one face are not recorded (we could not tell which face the label is for).
Log: results/antispoof_log.csv
Run from the face_recognition/ folder:  python antispoof_test.py
"""
import csv
import logging
import time
from pathlib import Path
 
import cv2
import numpy as np
from insightface.app import FaceAnalysis
 
import config as C
from antispoof import AntiSpoof
 
LABELS = {ord("1"): ("real", "real face"), ord("2"): ("spoof", "photo"), ord("3"): ("spoof", "screen")}
LOG_FILE = Path("results/antispoof_log.csv")
CAMERA_INDEX, FRAME_SIZE, MIRROR = 0, (1280, 720), True
GREEN, RED, WHITE, YELLOW = (0, 200, 0), (0, 0, 230), (255, 255, 255), (0, 200, 230)
 
 
def summary(rows, thr):
    """APCER / BPCER at the threshold, per attack type, and a small threshold sweep."""
    real = np.array([s for lab, _, s in rows if lab == "real"])
    spoof = np.array([s for lab, _, s in rows if lab == "spoof"])
    print(f"\nrecorded frames: real={len(real)}  spoof={len(spoof)}   (threshold {thr})")
    if len(spoof):
        print(f"APCER = {(spoof >= thr).mean():.3f}   ({int((spoof >= thr).sum())} of {len(spoof)} attack frames accepted as real)")
    if len(real):
        print(f"BPCER = {(real < thr).mean():.3f}   ({int((real < thr).sum())} of {len(real)} real frames rejected)")
    for kind in ("photo", "screen"):
        k = np.array([s for _, kd, s in rows if kd == kind])
        if len(k):
            print(f"  {kind:<7} frames={len(k):<5} accepted as real={(k >= thr).mean():.3f}   highest score={k.max():.2f}")
    if len(real):
        print(f"  real    frames={len(real):<5} lowest score={real.min():.2f}")
    if len(real) and len(spoof):
        print("\nthreshold   APCER   BPCER")
        for t in np.arange(-3.0, 3.01, 0.5):
            print(f"{t:>8.1f}  {(spoof >= t).mean():>6.3f}  {(real < t).mean():>6.3f}")
 
 
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spoof_model = AntiSpoof()
    app = FaceAnalysis(name=C.MODEL_NAME, allowed_modules=["detection"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=C.DET_SIZE)
 
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
 
    rows, label, skipped = [], None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = [f for f in app.get(frame) if f.det_score >= C.MIN_DET_SCORE]
            scores = spoof_model.scores(frame, [f.bbox for f in faces])
 
            if label:
                if len(faces) == 1 and not np.isnan(scores[0]):
                    rows.append((label[0], label[1], float(scores[0])))
                elif faces:
                    skipped += 1
 
            view = cv2.flip(frame, 1) if MIRROR else frame.copy()
            w = frame.shape[1]
            for f, s in zip(faces, scores):
                x1, y1, x2, y2 = f.bbox
                if MIRROR:
                    x1, x2 = w - x2, w - x1
                real = not np.isnan(s) and spoof_model.is_real(s)
                color = GREEN if real else RED
                cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(view, f"{'REAL' if real else 'SPOOF'} {s:.2f}", (int(x1), max(int(y1) - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            status = f"recording: {label[1]}" if label else "not recording (1 real, 2 photo, 3 screen, 0 stop)"
            cv2.putText(view, f"{status}   frames: {len(rows)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        YELLOW if label else WHITE, 2)
            cv2.imshow("Anti-spoofing test", view)
 
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in LABELS:
                label = LABELS[key]
            if key == ord("0"):
                label = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
 
    if rows:
        LOG_FILE.parent.mkdir(exist_ok=True)
        new = not LOG_FILE.exists()
        with LOG_FILE.open("a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["time", "label", "kind", "score"])
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            wr.writerows([stamp, lab, kd, f"{s:.4f}"] for lab, kd, s in rows)
        print(f"saved {len(rows)} frames to {LOG_FILE} (skipped {skipped} frames with more than one face)")
        summary(rows, spoof_model.threshold)
 
 
if __name__ == "__main__":
    main()
 