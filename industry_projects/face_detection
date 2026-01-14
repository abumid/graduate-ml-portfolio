# Webcam Face Verification + Eye-Openness (Liveness Cue) — Demo + Pipeline Excerpt

This folder contains a lightweight webcam demo and a production-style pipeline excerpt showcasing real-time computer vision skills:
- webcam processing with OpenCV
- face verification UI overlays (identity label + bounding box)
- eye-openness signal as a basic liveness cue (left/right eye state)
- simple state logic (e.g., pause/continue timer based on eye-state)

> **Portfolio note:** This is a public-friendly demo. Some production dependencies/models may not be included for confidentiality and portability.

---

## Included in this folder

### Scripts
- **`eye_detection_demo.py`**  
  Launcher to run eye-openness detection in different modes (camera/image/video).

- **`eye_detection_script.py`**  
  Main OpenCV application. Runs webcam processing, draws overlays (eye-state + timing/stats), and supports basic controls.

- **`test_eye_detection.py`**  
  Quick test script to validate the eye-openness module and basic outputs.

- **`face_analysis.py`**  
  Production-style **pipeline excerpt** that shows how face verification and eye-openness state logic can be combined in a UI (matching the screenshots).  
  ⚠️ This file may not run standalone because it can depend on internal modules/models that are not included publicly.

### Screenshots (visual proof)
- **`Screenshot from 2026-01-14 11-53-44.png`** — example of “pass” state (green box, eye-state open)
- **`Screenshot from 2026-01-14 11-53-59.png`** — example of “pause/fail” state (red box, eye-state closed)

---

## What this demonstrates (skills)
- Real-time webcam video pipeline (capture → processing → overlay UI)
- Eye-openness / blink-style liveness cue (left/right eye state)
- System logic beyond detection (timers, pause conditions, UX overlay)
- Practical engineering (CLI options, testing script, modular detector usage)

---

# How to run

## 1) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
