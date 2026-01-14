# Face Detection / Verification (Webcam) + Eye-Openness (Liveness Cue)

This folder contains a public-friendly demo that showcases real-time webcam computer vision skills:
- face bounding box + identity label overlay
- eye-openness (left/right eye state) as a basic liveness cue
- simple state logic (e.g., pause/reset behavior based on eye-state)

> **Portfolio note:** This is a demo + implementation excerpt. Some production dependencies/models may be missing for confidentiality and portability.

---

## Screenshots (visual proof)

### Pass state (green box, eye-state open)
<img src="Screenshot%20from%202026-01-14%2011-53-44.png" width="900"/>

### Pause/fail state (red box, eye-state closed)
<img src="Screenshot%20from%202026-01-14%2011-53-59.png" width="900"/>
---

## What this demonstrates (skills)
- Real-time webcam video pipeline (capture → processing → overlay UI)
- Eye-openness / blink-style liveness cue (left/right eye state)
- System logic beyond detection (timers, pause conditions, UX overlay)
- Practical engineering (separate demo runner, testing script, modular detector usage)

---

## Included files

### Runnable demo scripts
- **`eye_detection_script.py`**  
  Main OpenCV application. Runs webcam processing and draws overlays (eye state + timing/stats).  
  Use this as the primary entry point for the demo.

- **`eye_detection_demo.py`**  
  Launcher to run eye-openness detection in different modes (camera/image/video).  
  Useful for switching between inputs and quickly testing behavior.

- **`test_eye_detection.py`**  
  Quick test script to validate the eye-openness module and basic outputs.

### Implementation excerpt
- **`face_analysis.py`**  
  Production-style **pipeline excerpt** showing how face verification and eye-openness state logic can be combined in a UI (matching the screenshots).  
  ⚠️ This file may not run standalone because it can depend on internal modules/models not included publicly.
---

## How to run

### 1) Create and activate a virtual environment (recommended)

macOS/Linux:
- `python -m venv .venv`
- `source .venv/bin/activate`

Windows:
- `python -m venv .venv`
- `.venv\Scripts\activate`

### 2) Install dependencies

Minimum:
- `pip install opencv-python numpy`

If your eye-openness detector uses extra packages:
- `pip install mediapipe onnxruntime`

> Tip: Install the minimum first, run the script, and if you get an ImportError — install the missing package mentioned in the error.

### 3) Run the webcam demo

Recommended:
- `python eye_detection_script.py`

Alternative launcher:
- `python eye_detection_demo.py --mode camera`

Quick test:
- `python test_eye_detection.py`

