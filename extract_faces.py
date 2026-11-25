"""
Step 1: extract faces and embeddings from raw photos and videos.

For every image / sampled video frame:
  detect faces -> filter low quality -> align to 112x112 -> ArcFace embedding
  -> drop near-duplicates -> save crop (for review) + embedding (.npz)

Usage:
  python extract_faces.py                 # process all categories
  python extract_faces.py --only group    # re-process one category only

WARNING: re-processing a category deletes its review folder,
including any manual corrections you made there.
"""
import argparse
import logging
import shutil
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align

import config as C

log = logging.getLogger("extract")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


def load_model():
    log.info("ONNX Runtime providers available: %s", ort.get_available_providers())
    app = FaceAnalysis(
        name=C.MODEL_NAME,
        allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=C.DET_SIZE)
    return app


def iter_frames(path):
    """Yield (frame_name, BGR image) from an image file or a sampled video."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        img = cv2.imread(str(path))
        if img is None:
            log.warning("Could not read image: %s", path)
            return
        yield path.stem, img
    elif ext in VIDEO_EXT:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            log.warning("Could not open video: %s", path)
            return
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % C.FRAME_STEP == 0:
                yield f"{path.stem}_f{idx:05d}", frame
            idx += 1
        cap.release()


def identity_of(category, path):
    """Person id used later to split train/test by identity."""
    if category in ("alaa", "alaa_test"):
        return "alaa"
    if category == "group":
        return "group"
    rel = path.relative_to(C.RAW_DIR / category)
    if len(rel.parts) > 1:
        return f"{category}_{rel.parts[0]}"
    log.warning("File not inside a person sub-folder: %s", path)
    return f"{category}_unsorted"


def face_area(face):
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


def check_quality(img, face):
    """Return (reject_reason or None, aligned 112x112 crop)."""
    if face.det_score < C.MIN_DET_SCORE:
        return "low_score", None
    if face.bbox[2] - face.bbox[0] < C.MIN_FACE_PX:
        return "too_small", None
    crop = face_align.norm_crop(img, landmark=face.kps, image_size=112)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < C.MIN_SHARPNESS:
        return "blurry", None
    return None, crop


def load_alaa_centroid():
    """Mean of Alaa's training embeddings, L2-normalized."""
    path = C.OUT_DIR / "alaa.npz"
    if not path.exists():
        return None
    centroid = np.load(path)["embeddings"].mean(axis=0)
    return centroid / np.linalg.norm(centroid)


def process_category(app, category, spec, alaa_centroid=None):
    src_dir = C.RAW_DIR / category
    if not src_dir.exists():
        log.warning("Skipping '%s': folder not found (%s)", category, src_dir)
        return

    review_dir = C.REVIEW_DIR / category
    files = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT | VIDEO_EXT)
    cap = C.MAX_PER_IDENTITY.get(category)
    per_identity = Counter()
    stats = Counter()
    embeddings, labels, identities, crop_paths = [], [], [], []

    log.info("Processing '%s': %d files", category, len(files))
    for path in files:
        identity = identity_of(category, path)
        kept_from_file = []  # embeddings kept from this file, for de-duplication

        for frame_name, img in iter_frames(path):
            if cap and per_identity[identity] >= cap:
                stats["identity_cap"] += 1
                break

            faces = app.get(img)
            if not faces:
                stats["no_face"] += 1
                continue
            if spec["mode"] == "single":
                faces = [max(faces, key=face_area)]

            for i, face in enumerate(faces):
                reason, crop = check_quality(img, face)
                if reason:
                    stats[reason] += 1
                    continue

                emb = face.normed_embedding.astype(np.float32)
                if kept_from_file and max(float(e @ emb) for e in kept_from_file) > C.DEDUP_SIM:
                    stats["duplicate"] += 1
                    continue
                kept_from_file.append(emb)

                if spec["mode"] == "group":
                    sim = float(alaa_centroid @ emb)
                    sub = "likely_alaa" if sim >= C.GROUP_SPLIT_SIM else "likely_other"
                    out = review_dir / sub / f"{sim:.2f}__{frame_name}__{i}.jpg"
                    label = -1  # decided later by which review folder the crop ends up in
                else:
                    out = review_dir / identity / f"{frame_name}__{i}.jpg"
                    label = spec["label"]

                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), crop)

                embeddings.append(emb)
                labels.append(label)
                identities.append(identity)
                crop_paths.append(str(out))
                per_identity[identity] += 1
                stats["kept"] += 1

    log.info("'%s' done: %s", category, dict(stats))
    if not embeddings:
        log.warning("'%s': no faces kept, nothing saved", category)
        return

    np.savez(
        C.OUT_DIR / f"{category}.npz",
        embeddings=np.stack(embeddings),
        labels=np.array(labels),
        identities=np.array(identities),
        crop_paths=np.array(crop_paths),
    )


def main():
    parser = argparse.ArgumentParser(description="Extract faces and embeddings")
    parser.add_argument("--only", nargs="*", help="process only these categories")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = load_model()
    wanted = args.only or list(C.CATEGORIES)

    for category, spec in C.CATEGORIES.items():  # dict order: alaa runs before group
        if category not in wanted:
            continue

        centroid = None
        if spec["mode"] == "group":
            centroid = load_alaa_centroid()
            if centroid is None:
                log.error("Process 'alaa' before 'group' (needs alaa.npz)")
                continue

        shutil.rmtree(C.REVIEW_DIR / category, ignore_errors=True)
        process_category(app, category, spec, centroid)


if __name__ == "__main__":
    main()