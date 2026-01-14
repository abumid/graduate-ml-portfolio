# Safety Monitor (Ensemble Detection) — Real-time Triggers via Webcam

This folder contains a **public-friendly demo** that showcases real-time webcam computer vision for **safety/compliance monitoring** using an **ensemble detection approach**:

- **Sleeping** (eyes closed)
- **Fatigue** (yawning)
- **Phone Use**
- **Smoking**

> **Portfolio note:** This is a demo + implementation excerpt. Some production dependencies, datasets, and internal models may not be included publicly due to confidentiality and NDA constraints.

---

## What this project demonstrates (skills)

- **Real-time video pipeline** (webcam capture → inference → UI overlay)
- **Computer Vision + ML integration**
  - Face landmarks via `cvzone.FaceMeshModule`
  - Object detection via `ultralytics` YOLO
- **Ensemble logic** (combining a custom detector + general detector)
- **Event/trigger logic** (timers, thresholds, smoothing buffers, multi-alert handling)
- **Practical engineering**
  - configurable thresholds (`CONFIG`)
  - reproducible run steps (`requirements.txt`, `start.sh`)
  - clean “portfolio-style” packaging of an internal concept

---

## Visual proof (screenshots)


### Sleeping detected (eyes closed)
<img src="./assets/screenshots/sleeping.png" width="900" />

### Smoking detected
<img src="./assets/screenshots/smoking.png" width="900" />

### Phone in use detected
<img src="./assets/screenshots/phone_use.png" width="900" />

### Fatigue detected (yawn-based)
<img src="./assets/screenshots/fatigue.png" width="900" />

### Multiple violations at once (ensemble UI)
<img src="./assets/screenshots/multi_violation.png" width="900" />

---

## Included files (what’s in this folder)

### Core
- **`main.py`**
  - Main application: webcam capture, ensemble inference, UI overlay, violations panel, timers.
- **`requirements.txt`**
  - Minimal dependencies for running the demo.
- **`start.sh`**
  - Convenience script for fast start (Linux/macOS).

### Documentation / Notes
- **`ENSEMBLE_DETECTION.md`**
  - Notes describing how the ensemble setup works and how detections are combined.
- **`OPTIMIZATION_SUMMARY.md`**
  - Notes on optimization decisions and changes (model loading, pipeline improvements, etc).

---

## How it works (high-level)

1. **Webcam frames** are captured in real time.
2. A **FaceMesh** detector extracts key face landmarks to estimate:
   - eye state (open/closed) → “Sleeping”
   - yawn signal → “Fatigue”
3. A **YOLO ensemble** runs object detection:
   - **custom model** for domain classes (e.g., smoking)
   - **general YOLO model** for broader detection (e.g., phone)
4. A **trigger layer** applies:
   - confidence thresholds
   - smoothing (`buffer_size`)
   - per-class “consecutive frames” thresholds
5. The UI overlays:
   - bounding boxes + labels
   - warnings banner
   - session panel with timers + totals per violation

---

## Configuration

All important knobs are in `CONFIG` inside `main.py`, including:

- `use_ensemble` (enable/disable ensemble mode)
- `yolo_conf`, `yolo_iou`
- `eye_closed_threshold`, `yawn_threshold`
- per-class thresholds (frames needed before warning)
- per-class confidence thresholds

---

## How to run

### Option A — Quick run (Linux/macOS)
1) Create a virtual environment:
- python -m venv .venv
- source .venv/bin/activate

2) Install dependencies:
- pip install -r requirements.txt

3) Run:
- python main.py

---

### Option B — Using the start script (Linux/macOS)
1) Make it executable:
- chmod +x start.sh

2) Run:
- ./start.sh

---

## What is NOT included (expected in a real production repo)

Because this is a portfolio-friendly demo, some items may be intentionally excluded:

- **Training datasets** (private / client / internal)
- **Production configuration** (endpoints, secrets, internal services)
- **Some model weights**
  - If weights are large, they should be managed with Git LFS or shared separately.

> If you want, I can help you create a clean structure:
> - `weights/` kept locally and ignored in GitHub (`.gitignore`)
> - README explains how to add weights (without exposing private artifacts)

---

## Notes / Reproducibility

- Works best with a webcam and stable lighting.
- Tested as a real-time UI overlay demo.
- If you run into OpenCV webcam issues, try changing:
  - `CONFIG['camera_id'] = 0/1`
  - frame size settings

---

## Author / Context

Built as part of industry-focused computer vision work at **AI Implementation Group** (portfolio excerpt).
