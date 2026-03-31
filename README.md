# Multi-Object Detection and Persistent ID Tracking in Sports Footage
### YOLOv8m + DeepSORT | Football | FA Cup 2024

---

## 📌 Overview

This project implements a complete computer vision pipeline to detect, track, and analyze multiple objects (players) in sports footage. The system assigns persistent IDs, classifies players into teams based on jersey color, and handles real-world challenges such as occlusion, camera motion, scene changes, and crowd interference.

Built as both an assessment submission and a portfolio-grade computer vision project.

---

## 🚀 Features

- Multi-object detection using **YOLOv8m**
- Persistent ID tracking using **DeepSORT**
- Automatic **team classification** using HSV color analysis + K-Means clustering
- **Scene change detection** for tracker reset on camera cuts
- **Trajectory visualization** — 30-frame movement trails per player
- **Crowd filtering** using spatial and size heuristics
- GPU acceleration via CUDA
- Annotated output video with bounding boxes, IDs, team labels, and trails

---

## 📁 Project Structure

```
multi-object-tracking/
├── data/
│   ├── processed/            # trimmed input video (trimmed.mp4)
│   └── raw/                  # original downloaded footage
├── outputs/                  # annotated output video saved here
├── src/
│   ├── detector.py           # YOLOv8 detection module
│   ├── tracker.py            # DeepSORT tracking module
│   ├── team_classifier.py    # HSV-based team classification
│   └── scene_detector.py     # camera cut detection + tracker reset
├── scripts/
│   └── download_video.py     # yt-dlp video download script
├── main.py                   # full pipeline entry point
├── test_detector.py          # single-frame detection test
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd multi-object-tracking
```

### 2. Create Environment (Recommended)

```bash
conda create -n tracking python=3.10
conda activate tracking
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

```
ultralytics>=8.0.0
deep-sort-realtime>=1.3.2
scikit-learn>=1.3.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
```

### GPU Acceleration (Recommended)

This pipeline uses CUDA for significantly faster processing. Verify your setup:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, install CUDA-enabled PyTorch:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Tested on: NVIDIA GeForce RTX 3050 Laptop GPU, CUDA 12.1

---

## ▶️ How to Run

### Step 1 — Place Video

Put your input video here:

```
data/processed/trimmed.mp4
```

Or use the download script to fetch and trim automatically:

```bash
python scripts/download_video.py
```

### Step 2 — Test Detection (Optional)

Verify detection is working on a single frame:

```bash
python test_detector.py
```

### Step 3 — Run Full Pipeline

```bash
python main.py
```

### Step 4 — View Output

Annotated video saved at:

```
outputs/output_video.mp4
```

---

## ⚙️ Configuration

Key parameters in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `conf_threshold` | 0.5 | YOLOv8 detection confidence cutoff |
| `max_age` | 60 | Frames to hold a lost track before dropping |
| `n_init` | 5 | Consecutive frames required to confirm new ID |
| `MAX_DETECTIONS` | 20 | Detection limit — frames above this are crowd shots |
| `SHOW_PREVIEW` | False | Toggle live preview window (slower if True) |
| `update_interval` | 10 | Team classification runs every N frames |

---

## 🧠 Pipeline Overview

```
Input Video (trimmed.mp4)
        ↓
1. Scene Change Detection  →  Reset tracker on camera cuts
        ↓
2. YOLOv8m Detection       →  Bounding boxes + confidence scores + crops
        ↓
3. Crowd Filtering         →  Size, aspect ratio, position heuristics
        ↓
4. DeepSORT Tracking       →  Persistent IDs via Kalman Filter + Re-ID
        ↓
5. Team Classification     →  HSV K-Means on jersey torso crops
        ↓
6. Annotation              →  Boxes + IDs + Team labels + Trajectory trails
        ↓
Output Video (output_video.mp4)
```

---

## 🤖 Model & Tracker Choices

### YOLOv8m (Detection)

- Medium-sized model — better accuracy than YOLOv8n with acceptable speed
- Pretrained on COCO dataset (class 0 = person)
- Runs on GPU via CUDA for 7–11 FPS on RTX 3050
- Confidence threshold set to 0.5 to reduce crowd false positives

### DeepSORT (Tracking)

- Combines Kalman Filter (motion prediction) with appearance embeddings (Re-ID)
- `max_age=60` holds lost IDs for up to 60 frames (2.4 seconds) before dropping
- `n_init=5` requires 5 consecutive detections to confirm a track — eliminates ghost boxes
- `max_cosine_distance=0.2` enforces strict appearance matching for Re-ID
- Chosen over ByteTrack for better Re-ID capability with similar-looking players

### Team Classifier (HSV + K-Means)

- Crops torso region (30%–70% of bounding box height) to avoid grass/boots
- Removes grass-colored pixels (HSV hue 35–85) before clustering
- K-Means with k=2 finds dominant jersey color cluster
- Rule-based HSV classification:
  - Man Utd → Red (H < 20 or H > 160, S > 50, V > 50)
  - Man City → Light Blue (85 < H < 135, S > 50)
  - Other → Referee, staff, unclassified

---

## ⚠️ Assumptions

- Players wear distinct team colors (red vs light blue)
- Camera angle is consistent with standard broadcast football footage
- On-field players occupy the lower 80% of the frame
- Crowd appears in upper regions or as large bounding boxes
- Video is trimmed to a relevant match segment before processing

---

## ❗ Limitations

- **ID switching on camera cuts** — broadcast football cuts every 3–8 seconds. Scene change detection resets the tracker but IDs restart after each cut. This is a known limitation of single-camera tracking systems.
- **Re-entry after cut** — players who leave and re-enter frame after a cut receive new IDs
- **Heavy occlusion** — players hugging/celebrating cause temporary ID merges
- **Extreme lighting** — floodlight glare can shift HSV values and affect team classification
- **Non-standard angles** — close-up crowd shots or unusual camera angles bypass crowd filters

---

## 📊 Output

Each frame in the output video contains:

- **Colored bounding box** per player (red = Man Utd, blue = Man City, green = Other)
- **ID label** with team name (e.g. `ID 17 | Man Utd`)
- **30-frame trajectory trail** showing recent movement path
- **Frame counter, player count, and processing FPS** overlay

---

## 📸 Results & Improvements

### 🔴 1. Crowd Misclassification Fix

**Before (Without Filtering):**

<img width="1895" height="1038" alt="before_crowd" src="https://github.com/user-attachments/assets/84307d73-94f2-4e25-b96e-c279b770921c" />
<img width="1908" height="1033" alt="before_crowd2" src="https://github.com/user-attachments/assets/9d4ae560-914d-404e-b0d7-672713586e95" />

- Spectators incorrectly detected as players
- ID explosion (500+ IDs assigned during crowd shots)
- Wrong team assignments on crowd members

**After (With Spatial + Size Filtering):**

<img width="1818" height="999" alt="after_crowd" src="https://github.com/user-attachments/assets/dcd58c3d-918b-439d-8ab4-e41a988fe17b" />
<img width="1817" height="1014" alt="after_crowd2" src="https://github.com/user-attachments/assets/049f248f-c7ac-470d-b4fe-06d620300e82" />

- Crowd removed successfully
- Only on-field players tracked

---

### 🔵 2. Team Classification Improvement

**Before (Basic RGB / No HSV):**

<img width="1826" height="830" alt="before_classification" src="https://github.com/user-attachments/assets/00bc1e26-40f4-4699-8ba1-aa2339fb221b" />
<img width="1576" height="750" alt="before_players_team" src="https://github.com/user-attachments/assets/a7f72c6a-1a6d-41a3-9447-706e64dcd988" />

- Incorrect team labels
- Many players classified as "Other"

**After (HSV + K-Means + Grass Filtering):**

<img width="1791" height="986" alt="after_classification" src="https://github.com/user-attachments/assets/412b9be5-d585-4a31-bbb7-3947b01838b4" />

- Accurate team classification
- Stable labels across frames
- Correct color-coded bounding boxes

---

### 🟢 3. Tracking & Trajectory Visualization

**Final Output:**

<img width="1791" height="986" alt="final_output" src="https://github.com/user-attachments/assets/42fc5347-5c44-481c-a52c-74b7e35946fe" />

- Persistent IDs per player
- Smooth 30-frame trajectory trails
- Team-colored boxes and labels

---

## 🎥 Demo

> 📹 [Watch Demo Video](<[https://drive.google.com/drive/folders/1jxCMPlKkhEzqBOiDN2wj_1Kppb8Yx6WP?usp=sharing]>) — 3–5 min walkthrough of pipeline, code, and output

---

## 🔗 Video Source

FA Cup 2024 — Manchester City vs Manchester United (Public Broadcast Footage)

Source: https://youtu.be/X0we8220k74?si=ArpkK71w5bcPfjbj

---

## 👤 Author

**Praveen V**
Aspiring ML Engineer | Computer Vision | Data Science

---
