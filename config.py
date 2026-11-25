"""
Central configuration for the face recognition project.
All tunable values live here so no script has hardcoded numbers.
"""
from pathlib import Path

# ---------- Paths ----------
RAW_DIR = Path("data/raw")            # input: photos and videos
OUT_DIR = Path("data/processed")      # output: embeddings (.npz)
REVIEW_DIR = Path("data/review")      # output: face crops for manual review

# ---------- Model ----------
# buffalo_l = SCRFD face detector + ArcFa
# ce (ResNet50) recognition model.
# Note: InsightFace pretrained weights are for non-commercial research use.
MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)                 # detector input size

# ---------- Video sampling ----------
FRAME_STEP = 10                       # keep 1 frame out of every N

# ---------- Quality filters ----------
MIN_DET_SCORE = 0.6                   # detector confidence threshold
MIN_FACE_PX = 60                      # minimum face width in pixels
MIN_SHARPNESS = 40.0                  # Laplacian variance on the 112x112 crop
DEDUP_SIM = 0.95                      # drop faces this similar to one already kept from the same file

# ---------- Group photos ----------
# Cosine similarity to Alaa's mean embedding used only to PRE-SORT
# group-photo faces for manual review. Not the final decision threshold.
GROUP_SPLIT_SIM = 0.40

# ---------- Per-identity caps ----------
# Some LFW identities have hundreds of photos; cap them to avoid imbalance.
MAX_PER_IDENTITY = {"lfw": 10}

# ---------- Categories ----------
# mode "single": one person per file -> keep the largest face only
# mode "group":  several people per file -> keep every face, label by review
CATEGORIES = {
    "alaa":      {"mode": "single", "label": 1},
    "alaa_test": {"mode": "single", "label": 1},
    "others":    {"mode": "single", "label": 0},
    "lfw":       {"mode": "single", "label": 0},
    "group":     {"mode": "group",  "label": None},
}
# ---------- Attendance system ----------
GALLERY_DIR = "gallery"      # one .npz file per known person
MATCH_THRESHOLD = 0.40       # cosine similarity needed to accept a match (validated for Alaa vs LFW)

ATTENDANCE_CSV = "attendance.csv"               # one row per person (name, date, time)
ATTENDANCE_ARCHIVE_DIR = "attendance_archive"   # old files are moved here on reset, never deleted
# ---------- Anti-spoofing ----------
ANTISPOOF_MODEL = "models/antispoof/best_model.onnx"
ANTISPOOF_THRESHOLD = 0.0      # real_logit - spoof_logit >= 0 -> real (probability 0.5)
ANTISPOOF_EXPANSION = 1.5      # crop 1.5x larger than the face box, like the original repository