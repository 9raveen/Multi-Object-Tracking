import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Team classification using HSV color space (robust to lighting).
    """

    TEAM_COLORS = {
        "Man Utd": (0, 0, 220),      # red
        "Man City": (220, 180, 0),   # light blue
        "Other": (0, 220, 0),
    }

    def __init__(self, update_interval=10):
        self.update_interval = update_interval
        self.team_assignments = {}
        self.frame_counter = 0

    # ── STEP 1: GET TORSO PIXELS (HSV) ─────────────────────────────

    def _get_torso_pixels(self, crop):
        if crop is None or crop.size == 0:
            return None

        h = crop.shape[0]

        # Focus on middle torso
        torso = crop[int(h * 0.3):int(h * 0.7), :]

        if torso.size == 0:
            return None

        torso = cv2.resize(torso, (20, 20))

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)

        # Remove grass (green hue)
        pixels = pixels[(pixels[:, 0] < 35) | (pixels[:, 0] > 85)]

        if len(pixels) < 20:
            return None

        return pixels.astype(np.float32)

    # ── STEP 2: DOMINANT COLOR ────────────────────────────────────

    def _dominant_color(self, pixels):
        if pixels is None:
            return None

        kmeans = KMeans(n_clusters=2, n_init=3, random_state=42)
        kmeans.fit(pixels)

        labels = kmeans.labels_
        counts = np.bincount(labels)
        dominant_idx = np.argmax(counts)

        return kmeans.cluster_centers_[dominant_idx]

    # ── STEP 3: CLASSIFY USING HSV (FIXED) ────────────────────────

    def _classify_color(self, hsv):
        if hsv is None:
            return "Other"

        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        # Ignore low saturation
        if s < 40:
            return "Other"

        # 🔴 RED (Man Utd) — FIXED (expanded range)
        if (h < 20 or h > 160) and s > 50 and v > 50:
            return "Man Utd"

        # 🔵 BLUE (Man City)
        if 85 < h < 135 and s > 50:
            return "Man City"

        return "Other"

    # ── MAIN CLASSIFICATION ───────────────────────────────────────

    def classify_by_crops(self, tracks, detections, crops, frame_idx):
        self.frame_counter += 1

        if self.frame_counter % self.update_interval != 0:
            return self.team_assignments

        if len(detections) == 0 or len(crops) == 0:
            return self.team_assignments

        for track in tracks:
            x1, y1, x2, y2, track_id = track

            track_cx = (x1 + x2) / 2
            track_cy = (y1 + y2) / 2

            best_idx = None
            best_dist = float("inf")

            for i, det in enumerate(detections):
                dx1, dy1, dx2, dy2 = det[:4]

                det_cx = (dx1 + dx2) / 2
                det_cy = (dy1 + dy2) / 2

                dist = ((track_cx - det_cx) ** 2 +
                        (track_cy - det_cy) ** 2) ** 0.5

                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            box_width = x2 - x1
            threshold = max(50, 0.3 * box_width)

            if best_idx is not None and best_dist < threshold and best_idx < len(crops):
                crop = crops[best_idx]

                pixels = self._get_torso_pixels(crop)
                if pixels is None:
                    continue

                dominant = self._dominant_color(pixels)
                team = self._classify_color(dominant)

                # ✅ IMPORTANT: allow correction if previously "Other"
                if track_id not in self.team_assignments or self.team_assignments[track_id] == "Other":
                    self.team_assignments[track_id] = team

        return self.team_assignments

    # ── UTILITIES ─────────────────────────────────────────────────

    def get_team_color(self, track_id):
        team = self.team_assignments.get(track_id, "Other")
        return self.TEAM_COLORS.get(team, (255, 255, 255))

    def reset(self):
        self.team_assignments = {}
        self.frame_counter = 0