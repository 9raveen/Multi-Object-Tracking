# Multi-Object Detection and Persistent ID Tracking in Sports Footage

## 📌 Overview

This project implements a complete computer vision pipeline to detect, track, and analyze multiple objects (players) in sports footage. The system assigns persistent IDs, classifies teams based on jersey color, and handles real-world challenges such as occlusion, camera motion, and crowd interference.

---

## 🚀 Features

- Multi-object detection using YOLOv8
- Persistent ID tracking using DeepSORT
- Team classification using HSV-based color analysis
- Scene change detection for tracker reset
- Trajectory visualization for movement tracking
- Crowd filtering to remove spectators
- Annotated output video generation

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone <your-repo-link>
cd multi-object-tracking
```

### 2. Create environment (recommended)

```bash
conda create -n tracking python=3.10
conda activate tracking
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

- Python 3.10+
- OpenCV
- NumPy
- Ultralytics (YOLOv8)
- deep-sort-realtime
- scikit-learn

---

## ▶️ How to Run

### Step 1: Place video

Put your video here:

```
data/processed/trimmed.mp4
```

### Step 2: Run pipeline

```bash
python main.py
```

### Step 3: Output

Annotated video will be saved at:

```
outputs/output_video.mp4
```

---

## 🧠 Pipeline Overview

1. Detect players using YOLOv8
2. Track players using DeepSORT
3. Assign unique IDs
4. Classify teams using HSV color space
5. Filter crowd using spatial + size heuristics
6. Draw bounding boxes, IDs, and trajectories
7. Save annotated video

---

## ⚠️ Assumptions

- Players wear distinct team colors
- Camera angle is similar to broadcast sports footage
- Players are mostly in the lower region of the frame
- Crowd appears in upper regions or as large bounding boxes

---

## ❗ Limitations

- Small/distant players may not be classified
- Heavy occlusion can cause temporary ID switches
- Color-based classification may fail under extreme lighting
- Not robust to non-standard camera angles

---

## 🤖 Model & Tracker Choices

### YOLOv8 (Detection)

- Fast and accurate real-time detector
- Pretrained on COCO dataset
- Strong generalization for person detection

### DeepSORT (Tracking)

- Combines motion + appearance features
- Maintains identity across frames
- Handles occlusion better than simple trackers

---

## 📊 Output

- Annotated video with:
  - Bounding boxes
  - Unique IDs
  - Team labels
  - Trajectory lines

---

## 🎥 Demo

(Attach your demo video link here)

---

## 🔗 Video Source

https://youtu.be/X0we8220k74?si=ArpkK71w5bcPfjbj

---

## 📸 Sample Results

(Add screenshots here)

---
