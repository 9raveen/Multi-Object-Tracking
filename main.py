import cv2
import os
import time
from src.detector import YOLODetector
from src.tracker import Tracker
from src.scene_detector import SceneChangeDetector
from src.team_classifier import TeamClassifier  

# ── CONFIG ───────────────────────────────────────────────────────────────
INPUT_VIDEO = "data/processed/trimmed.mp4"
OUTPUT_VIDEO = "outputs/output_video.mp4"
SHOW_PREVIEW = False
MAX_DETECTIONS = 20

os.makedirs("outputs", exist_ok=True)

# ── INIT ─────────────────────────────────────────────────────────────────
detector        = YOLODetector(model_path="yolov8m.pt", conf_threshold=0.5)
tracker         = Tracker(max_age=60, n_init=5)
scene_detector  = SceneChangeDetector(threshold=35.0)
team_classifier = TeamClassifier(update_interval=10)  
track_history   = {}

# ── VIDEO I/O ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 60:
    fps = 30

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"\n Video: {width}x{height} @ {fps:.1f} FPS")
print(f"Total Frames: {total}")
print(f"Output: {OUTPUT_VIDEO}")
print("Processing...\n")

start_time = time.time()
frame_idx  = 0
scene_cuts = 0

# ── MAIN LOOP ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # ── 1. SCENE CHANGE ───────────────────────────────────────────
    if scene_detector.is_scene_change(frame):
        tracker         = Tracker(max_age=60, n_init=5)
        track_history   = {}
        team_classifier.reset()
        scene_cuts += 1
        print(f"Scene cut at frame {frame_idx} (total: {scene_cuts})")
        out.write(frame)
        continue

    # ── 2. DETECTION ──────────────────────────────────────────────
    detections, crops = detector.detect(frame)

    if len(detections) > MAX_DETECTIONS:
        detections = detections[:MAX_DETECTIONS]
        crops      = crops[:MAX_DETECTIONS]

    # ── 3. TRACKING ───────────────────────────────────────────────
    tracks = tracker.update(detections, frame)

    # ── 4. TEAM CLASSIFICATION ────────────────────────────────────
    team_map = team_classifier.classify_by_crops(tracks, detections, crops, frame_idx)

    # ── 5. CLEAN TRACK HISTORY ───────────────────────────────────
    active_ids = set([t[4] for t in tracks])
    track_history = {
        tid: pts for tid, pts in track_history.items()
        if tid in active_ids
    }

    # ── 6. DRAW ──────────────────────────────────────────────────
    annotated = frame.copy()
    frame_h = frame.shape[0]

    for track in tracks:
        x1, y1, x2, y2, track_id = track

        if x2 <= x1 or y2 <= y1:
            continue

        # ── 🔥 BALANCED PLAYER FILTER (FIXED) ─────────────────────
        box_width  = x2 - x1
        box_height = y2 - y1

        if (
            box_width > 200 or
            box_height > 250 or
            (y1 < int(frame_h * 0.2) and box_width > 60)
        ):
            continue

        # Optional: remove noise but keep far players
        if box_width < 15:
            continue

        # Skip unclassified (clean output)
        if track_id not in team_map:
            continue

        # ── DRAW ────────────────────────────────────────────────
        color = team_classifier.get_team_color(track_id)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        team  = team_map.get(track_id, "Detecting...")
        label = f"ID {track_id} | {team}"

        (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, y1 - lh - base - 4), (x1 + lw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # ── TRAJECTORY ───────────────────────────────────────────
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append((cx, cy))

        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        for i in range(1, len(track_history[track_id])):
            cv2.line(annotated,
                     track_history[track_id][i - 1],
                     track_history[track_id][i],
                     color, 2)

    # ── 7. OVERLAY ───────────────────────────────────────────────
    elapsed     = time.time() - start_time
    fps_display = frame_idx / elapsed if elapsed > 0 else 0

    cv2.putText(annotated,
                f"Frame: {frame_idx}/{total} | Players: {len(tracks)} | FPS: {fps_display:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    # ── 8. SAVE OUTPUT ───────────────────────────────────────────
    out.write(annotated)

    if SHOW_PREVIEW:
        cv2.imshow("Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx}/{total} frames")

# ── CLEANUP ───────────────────────────────────────────────────────────────
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nTotal scene cuts detected: {scene_cuts}")
print(f"Done! Output saved at: {OUTPUT_VIDEO}")