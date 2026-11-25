import time

import numpy as np
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis

import config as C

has_cuda = torch.cuda.is_available()
print("torch:", torch.__version__, "| CUDA in torch:", has_cuda)

app = FaceAnalysis(
    name=C.MODEL_NAME,
    allowed_modules=["detection", "recognition"],
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=C.DET_SIZE)
for name, model in app.models.items():
    print(f"InsightFace {name}: {model.session.get_providers()[0]}")

device = 0 if has_cuda else "cpu"
yolo = YOLO("yolo11n-seg.pt")
frame = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
for size in (640, 480, 320):
    yolo.predict(frame, imgsz=size, device=device, verbose=False)
    t0 = time.perf_counter()
    for _ in range(10):
        yolo.predict(frame, imgsz=size, device=device, verbose=False)
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"YOLO-seg on {device} imgsz={size}: {ms:.0f} ms per frame")
