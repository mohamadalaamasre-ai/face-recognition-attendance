"""
Step 4a: person segmentation only (no recognition) on the live webcam.
Goal: confirm YOLO-seg works on its own before merging it into live_camera.py.
Keys: q or ESC = quit.
"""
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

SEG_MODEL = "yolo11n-seg.pt"     # pretrained on COCO
SEG_IMGSZ = 640                  # 5 ms on the GPU in check_gpu.py, so we keep the most accurate size
DEVICE = 0 if torch.cuda.is_available() else "cpu"   # fall back to CPU instead of crashing
PERSON_CLASS = 0                 # in COCO, class 0 is "person"
MIN_CONF = 0.5                   # ignore weak person detections
ALPHA = 0.45                     # mask transparency (0 = invisible, 1 = solid)
CAMERA_INDEX = 0
FRAME_SIZE = (1280, 720)


def main():
    model = YOLO(SEG_MODEL)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    print(f"YOLO-seg on device={DEVICE}, imgsz={SEG_IMGSZ}")

    fps, t_prev = 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break

            result = model.predict(frame, imgsz=SEG_IMGSZ, classes=[PERSON_CLASS],
                                   conf=MIN_CONF, device=DEVICE, verbose=False)[0]

            overlay = frame.copy()
            n_people = 0
            if result.masks is not None:
                for poly in result.masks.xy:               # one outline per person, in frame pixels
                    if len(poly) < 3:
                        continue
                    cv2.fillPoly(overlay, [poly.astype(np.int32)], (0, 200, 0))
                    n_people += 1
            view = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
            t_prev = now

            view = cv2.flip(view, 1)                        # mirror AFTER drawing, so the text stays readable
            cv2.putText(view, f"people {n_people}  {fps:.0f} FPS  {DEVICE}  imgsz {SEG_IMGSZ}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Person segmentation test", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
