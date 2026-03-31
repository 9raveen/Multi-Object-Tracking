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


## 📸 Results & Improvements

### 🔴 1. Crowd Misclassification Fix

**Before (Without Filtering):**
<img width="1895" height="1038" alt="before_crowd" src="https://github.com/user-attachments/assets/84307d73-94f2-4e25-b96e-c279b770921c" />

<img width="1908" height="1033" alt="before_crowd2" src="https://github.com/user-attachments/assets/9d4ae560-914d-404e-b0d7-672713586e95" />

* Spectators incorrectly detected as players
* Wrong team assignments

**After (With Spatial + Size Filtering):**
<img width="1818" height="999" alt="after_crowd" src="https://github.com/user-attachments/assets/dcd58c3d-918b-439d-8ab4-e41a988fe17b" />

<img width="1817" height="1014" alt="after_crowd2" src="https://github.com/user-attachments/assets/049f248f-c7ac-470d-b4fe-06d620300e82" />

* Crowd removed successfully
* Only field players tracked

---

### 🔵 2. Team Classification Improvement

**Before (Basic RGB / No HSV):**
<img width="1826" height="830" alt="before_classification" src="https://github.com/user-attachments/assets/00bc1e26-40f4-4699-8ba1-aa2339fb221b" />
<img width="1576" height="750" alt="before_players_team" src="https://github.com/user-attachments/assets/a7f72c6a-1a6d-41a3-9447-706e64dcd988" />

* Incorrect team labels
* Many players classified as "Other"

**After (HSV + KMeans + Filtering):**
<img width="1791" height="986" alt="after_classification" src="https://github.com/user-attachments/assets/412b9be5-d585-4a31-bbb7-3947b01838b4" />

* Accurate team classification
* Stable labels across frames

---

### 🟢 3. Tracking & Trajectory Visualization

**Final Output:**

<img width="1791" height="986" alt="after_classification" src="https://github.com/user-attachments/assets/42fc5347-5c44-481c-a52c-74b7e35946fe" />

* Persistent IDs
* Smooth trajectory lines
* Stable tracking across frames

/

---

---

## 🎥 Demo

(Attach your demo video link here)

---

## 🔗 Video Source

https://youtu.be/X0we8220k74?si=ArpkK71w5bcPfjbj

---

