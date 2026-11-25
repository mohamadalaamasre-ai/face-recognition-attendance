"""
Passive face anti-spoofing (liveness): is this a real face, or a photo / a screen?
 
Model: MiniFAS from github.com/facenox/face-antispoof-onnx (Apache-2.0),
file models/best/98.20/best_model.onnx -> saved here as models/antispoof/best_model.onnx
Input 128x128 RGB, output 2 logits [real, spoof].
 
The pre-processing below follows that repository (src/inference/preprocess.py), so the model
gets the same kind of input it was trained on:
  square crop around the face box, 1.5x larger (it keeps context: phone edges, screen borders),
  reflect padding at the image border, letterbox to 128x128, RGB, values 0..1, channels first.
 
Score = real_logit - spoof_logit.  score >= threshold -> real.  (threshold 0.0 = probability 0.5)
"""
import logging
from pathlib import Path
 
import cv2
import numpy as np
import onnxruntime as ort
 
import config as C
 
log = logging.getLogger("antispoof")
 
 
def square_crop(img, bbox, expansion):
    """Square crop centred on the face box, expansion x its longest side; reflect-pad outside the image."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    size = int(max(w, h) * expansion)
    x = int(x1 + w / 2 - size / 2)
    y = int(y1 + h / 2 - size / 2)
    H, W = img.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x), max(0, y), min(W, x + size), min(H, y + size)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    top, bottom = max(0, -y), max(0, y + size - H)
    left, right = max(0, -x), max(0, x + size - W)
    return cv2.copyMakeBorder(img[cy1:cy2, cx1:cx2], top, bottom, left, right, cv2.BORDER_REFLECT_101)
 
 
def to_input(crop_rgb, size):
    """Letterbox to size x size (reflect padding), scale to 0..1, HWC -> CHW."""
    h, w = crop_rgb.shape[:2]
    r = size / max(h, w)
    nh, nw = int(h * r), int(w * r)
    interp = cv2.INTER_LANCZOS4 if r > 1.0 else cv2.INTER_AREA
    img = cv2.resize(crop_rgb, (nw, nh), interpolation=interp)
    dh, dw = size - nh, size - nw
    img = cv2.copyMakeBorder(img, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2, cv2.BORDER_REFLECT_101)
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0
 
 
class AntiSpoof:
    def __init__(self, model_path=C.ANTISPOOF_MODEL, threshold=C.ANTISPOOF_THRESHOLD,
                 expansion=C.ANTISPOOF_EXPANSION, size=128):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"anti-spoof model not found: {model_path}")
        self.sess = ort.InferenceSession(str(model_path),
                                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.threshold, self.expansion, self.size = threshold, expansion, size
        log.info("anti-spoof model on %s, threshold %.2f", self.sess.get_providers()[0], threshold)
 
    def scores(self, frame_bgr, bboxes):
        """One score per face box (real_logit - spoof_logit); NaN where the crop failed."""
        out = np.full(len(bboxes), np.nan, np.float32)
        batch, idx = [], []
        for i, b in enumerate(bboxes):
            crop = square_crop(frame_bgr, b, self.expansion)
            if crop is None or crop.size == 0:
                continue
            batch.append(to_input(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), self.size))   # model expects RGB
            idx.append(i)
        if batch:
            logits = self.sess.run(None, {self.input_name: np.stack(batch)})[0]
            out[idx] = logits[:, 0] - logits[:, 1]
        return out
 
    def is_real(self, score):
        return bool(score >= self.threshold)
 