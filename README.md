Face Recognition → Live Attendance System

Live face recognition on a laptop webcam, built in stages during a computer vision course:

One person: recognize one specific person and label everyone else Unknown.
Person segmentation: color each person's body by identity (who + where).
Measurement: FAR / FRR / ROC, precision / recall, per-stage inference time and FPS.
Attendance system: several people, enrollment with one key press, CSV attendance logged once per person, archive on reset, and a small Streamlit dashboard.

The model work is built on top of pretrained models. Most of the effort went into evaluating honestly: splitting data without leakage, distrusting perfect scores, and testing for shortcut learning.

Demo video / GIF: [add here]

Pipeline
camera frame
  ├─ SCRFD (InsightFace)        face boxes + 5 landmarks
  │    └─ alignment 112×112
  │         └─ ArcFace           512-number embedding (L2-normalized)
  │              └─ decision     who is it? (gallery + threshold)  ──► attendance.csv
  └─ YOLO11n-seg                 one mask per person
       └─ link                   each body takes the color of the face inside it
Face recognition, not object detection. Training YOLO on a class "Alaa" would teach "human face" and label anyone as Alaa.
Open-set. Unknown is not a trained class; it is a rejection decision by threshold.
Data
Source	Content	Faces	Role
alaa	8 videos (5 laptop, 3 phone)	585 (591 extracted, 6 removed by manual review)	train + validation
alaa_test	separate session: other day, clothes, place	56	final test
friend1	one friend, same kind of camera	93	train only (negative)
LFW (deep-funneled)	13,233 images, 5,749 identities, max 10 per identity	10,266	strangers

Quality filters during extraction: detection score ≥ 0.6, face width ≥ 60 px, sharpness (Laplacian variance) ≥ 40, near-duplicates removed (cosine > 0.95), one frame in 10. Counts were reconciled at every step (e.g. 591 kept + 44 rejected = 635 frames). The data is not in this repository (it contains real people's faces).

Split (the part that matters most)
Split	Alaa	Strangers	Use
Train	423 (6 videos)	≈ 7,047 LFW + 93 friend1	learning
Validation	162 (2 videos)	1,639 LFW	choosing the model and threshold
Test	56 (alaa_test)	1,580 LFW	used once, nothing tuned after it
Alaa's faces are split by video (frames of one video are near-duplicates, a random split leaks).
LFW is split by identity (70 / 15 / 15), checked with asserts, fixed seed 42.
Models
	A (baseline)	B
Method	cosine similarity to the centroid of Alaa's embeddings	logistic regression on frozen embeddings, class_weight="balanced"
Training	none	linear probing (ArcFace stays frozen; this is not full fine-tuning)
Threshold	0.40	0.0
Evaluation

Accuracy is misleading here: with ~16 strangers per Alaa face, answering "Unknown" every time scores ~95%. We used:

FAR (False Accept Rate): share of strangers accepted. FRR (False Reject Rate): share of Alaa's faces rejected.
ROC / AUC / EER to compare models over all thresholds.

What happened:

A perfect score we did not trust. The first validation gave AUC = 1.0 with a gap of +0.43 between the weakest Alaa score and the strongest stranger. The hard videos turned out to be in training; after swapping them the gap dropped to +0.11 (A and B still AUC = 1.0, EER = 0 on 162 / 1,639).
Shortcut test. Strangers are professional photos, Alaa is a laptop webcam: did B learn the camera instead of the face? We trained B without friend1 and scored friend1's 93 faces: 0 / 93 accepted (highest score −2.51, threshold 0.0). Strongest LFW stranger: −3.18. Only 1 / 162 Alaa faces fell below 0 (lowest −0.48).
Baseline A at 0.40 on validation: FRR ≈ 1.2%, 0 false accepts out of 8,686 LFW faces and 93 friend1 faces, margin +0.18 over the strongest stranger (0.22).
Decision: B at 0.0, A at 0.40 kept as baseline. The numbers did not separate them; the shortcut test showed B did not fall into the trap, not that it is better.
A silent bug caught. The first test run used model A: a default value marked "fill in our decision" was never changed. Found by reading the saved models/meta.json, not the code.
Final test (run once)
	predicted Alaa	predicted Unknown
actual Alaa	56	0
actual stranger	0	1,580

Same result for A and B. Zero errors on a small sample is not zero error. With the rule of three (95% upper bound ≈ 3/n): FAR ≤ 0.19%, FRR ≤ 5.4%. The 56 test faces are consecutive frames of one video, so the real bound is worse. Precision = recall = 1.0 on this set (lower bound ≈ 0.946).

Speed (RTX 3050 Laptop GPU, 300 frames of a fixed video at 1280×720, after 10 warm-up frames)
Stage	Mean ms	p95 ms	Share
face detection	13.90	14.13	31%
face embedding	12.69	13.73	29%
classification	0.40	0.46	1%
person segmentation	8.24	9.32	19%
face ↔ body link	0.14	0.19	0%
drawing	9.13	10.23	21%
total	44.50	47.30	≈ 22 FPS (pipeline only, camera excluded)

Drawing costs more than YOLO, so it is the cheapest place to speed things up. A sudden drop to ~35 ms near frame 265 was traced to the second person in the video leaving the frame (fewer faces to embed and masks to draw).

Attendance system
Why not B anymore: B answers one question ("Alaa or not?") and needs retraining for every new person. The attendance system uses a gallery: one centroid per person; a face goes to the closest centroid if similarity ≥ 0.40. Adding a person needs no training.
Consistency check: the gallery reproduces model A's results: 56 / 56 alaa_test faces accepted, 0 of 10,266 LFW faces.
Enrollment (e): 25 embeddings in ≤ 20 s, accepted only if exactly one face is in view, quality filters pass (same as training data), and the embedding is not a near-duplicate (cosine ≤ 0.95, so different angles are collected). Warns if the face looks like someone already enrolled. Stores embeddings, not images.
Attendance: logged after a stable window (mean of the last 7 embeddings), written immediately, loaded at start (no duplicates after a restart), reset with r pressed twice, old file archived, never deleted.
Unknown people: get an id only after 10 unknown frames, and each stranger is counted once while the camera runs (approximate).
Separation between enrolled people: Alaa vs Moaz centroid similarity 0.159, threshold 0.40.
Dashboard: the camera writes status.json about once a second (atomic write: temp file then rename); dashboard.py (Streamlit) reads it and the CSV files and marks the camera as stopped if nothing was written for 5 s.
Anti-spoofing: tested and not adopted

We tested a pretrained passive liveness model (MiniFAS, from facenox/face-antispoof-onnx, Apache-2.0), with the same pre-processing as the original repository. Labeled frames from our webcam (one person, one phone): 235 real, 521 photo, 137 screen.

Threshold	APCER (attacks accepted)	BPCER (real faces rejected)
−1.0	68.4%	24.7%
0.0 (default)	52.3%	41.3%
1.0	38.8%	65.5%

Screen attacks were accepted 91% of the time. No threshold gives usable values for both errors, so the model was not integrated. Its published accuracy was measured on its own data (CelebA-Spoof) and did not carry over to our camera. A possible cause on our side is that our face boxes (SCRFD) are tighter than the original repository's detector. antispoof.py and antispoof_test.py are kept as the record of this experiment. The model file is not included; to rerun it:

bash
mkdir -p models/antispoof
wget -O models/antispoof/best_model.onnx https://github.com/facenox/face-antispoof-onnx/raw/main/models/best/98.20/best_model.onnx
Limitations
Small test set: one session, one video, correlated frames.
Test strangers come from LFW (professional photos).
No working anti-spoofing: a photo of an enrolled person on a phone is accepted.
No IoU / Dice measured for the masks (no hand-drawn ground truth yet).
Counting unknown people is approximate; the simple IoU tracker can swap colors for a moment when people cross.
Project structure
config.py                  settings (filters, paths, thresholds)
extract_faces.py           videos/photos -> filtered, aligned faces + embeddings (.npz) + crops for review
01_train_and_evaluate.ipynb   split, models A and B, ROC/EER, shortcut test, threshold choice, final test
live_camera.py             live recognition of one person (model from models/meta.json)
check_gpu.py               checks that InsightFace and YOLO both run on the GPU
seg_camera_test.py         YOLO-seg alone on the webcam
live_camera_seg.py         recognition + person segmentation
02_benchmark.ipynb         per-stage inference time and FPS
gallery.py                 face gallery: enroll, match, check
attendance.py              attendance CSV: once per person, reset with archive
attendance_camera.py       live attendance + segmentation + enrollment + counters
dashboard.py               Streamlit dashboard
antispoof.py, antispoof_test.py   anti-spoofing experiment (not used)
results/                   benchmark.json, benchmark.png, antispoof_log.csv
Setup

Tested on Ubuntu, Python 3.11, NVIDIA RTX 3050 (6 GB).

bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python check_gpu.py                 # InsightFace must report CUDAExecutionProvider
python attendance_camera.py         # e = enroll, r r = reset, d = scores, q = quit
streamlit run dashboard.py          # in a second terminal

Use python -m pip (not bare pip) so packages go into the same environment that runs the code. Do not install the plain onnxruntime package next to onnxruntime-gpu: it hides the GPU.

Privacy

Face embeddings are biometric data. The gallery, attendance files, datasets and face images are excluded from this repository (.gitignore). Enroll people only with their consent.

Credits and licenses
InsightFace buffalo_l (SCRFD + ArcFace): the pretrained weights are for non-commercial research use.
Ultralytics YOLO11 segmentation: AGPL-3.0.
LFW (deep-funneled version) for stranger faces.
MiniFAS anti-spoofing model by facenox: Apache-2.0 (experiment only).

Thanks to the course coach for giving us problems instead of solutions.