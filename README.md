# Detection + Tracking — Clean Module

> **Scope:** Person detection from video + temporary-ID tracking only.
> No face recognition, no identity mapping, no attention scoring.

---

## Folder Structure

```
detection_tracking_clean/
├── main.py            ← entry point (run this)
├── detector.py        ← YOLOv8 person detector
├── tracker.py         ← IOU-based tracker (no DeepSORT)
├── visualizer.py      ← draws bounding boxes + track IDs
├── yolov8n.pt         ← model weights (copy from original folder OR auto-download)
├── input_video.mp4    ← place your classroom video here
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create & activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the model weights

**Option A – copy from the original project:**
```bash
# from classroom_attention root
copy yolov8n.pt detection_tracking_clean\yolov8n.pt
```

**Option B – auto-download (first run only):**
The `ultralytics` package downloads `yolov8n.pt` automatically the first time if it is not present.

### 4. Add your input video

Place your classroom video inside this folder and name it:
```
input_video.mp4
```
Or pass a custom path with `--video` (see below).

---

## How to Run

```bash
# default: reads input_video.mp4, opens a live window
python main.py

# custom video path
python main.py --video path/to/your_video.mp4

# headless (no display window), saves output only
python main.py --no-display

# custom YOLO model weights
python main.py --model yolov8s.pt
```

Press **Q** in the display window to stop early.

---

## What the Output Shows

| Element | Description |
|---|---|
| **Bounding box** | Coloured rectangle around each detected person |
| **ID label** | `ID 1`, `ID 2`, … — temporary numeric ID per person |
| **Top-left overlay** | Frame number + count of persons visible in that frame |
| **output_annotated.mp4** | Saved annotated video in the same folder |

> IDs are **temporary** and reset when you re-run the script.
> The same physical person keeps the same ID across consecutive frames (within one run).
> IDs are **not** names, roll numbers, or any identity information.

---

## Tracking Method

This module uses a **custom IOU (Intersection-over-Union) tracker** — no DeepSORT or similar library required.

- Each frame's detections are matched to existing tracks by highest IoU overlap.
- Unmatched detections get a new ID.
- Tracks missing for more than 30 consecutive frames are dropped.

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Video I/O and drawing |
| `numpy` | Array maths for IoU |
| `ultralytics` | YOLOv8 person detection |
