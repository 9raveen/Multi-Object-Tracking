import cv2
import numpy as np

class SceneChangeDetector:
    def __init__(self, threshold=35.0):
        self.prev_frame = None
        self.threshold = threshold  # mean pixel difference threshold

    def is_scene_change(self, frame):
        """
        Returns True if this frame is a camera cut.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        diff = cv2.absdiff(gray, self.prev_frame)
        mean_diff = np.mean(diff)
        self.prev_frame = gray

        return mean_diff > self.threshold