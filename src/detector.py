import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8m.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        ''' self.model.to("cuda")  # GPU acceleration '''
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Run detection on a single frame.
        Returns:
            detections: np.ndarray of shape (N, 5) -> [x1, y1, x2, y2, conf]
            crops:      list of N BGR numpy arrays (cropped player regions)
        """
        results = self.model(frame, verbose=False)
        detections = []
        crops = []

        h, w = frame.shape[:2]
        frame_area = h * w

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue

                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                box_w = x2 - x1
                box_h = y2 - y1

                # Filter too-small boxes (distant crowd members)
                if box_h < 0.05 * h:
                    continue

                # Filter oversized boxes (crowd shots, banners)
                if (box_w * box_h) > 0.15 * frame_area:
                    continue

                # Filter wide boxes (crowd groups detected as one person)
                if (box_w / box_h) > 1.2:
                    continue

                detections.append([x1, y1, x2, y2, conf])

                # Clamp to frame boundaries
                cx1, cy1 = max(0, int(x1)), max(0, int(y1))
                cx2, cy2 = min(w, int(x2)), min(h, int(y2))
                crops.append(frame[cy1:cy2, cx1:cx2])

        if detections:
            return np.array(detections), crops
        return np.empty((0, 5)), []