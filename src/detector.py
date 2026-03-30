from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4):
        """
        Initialize YOLOv8 model
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Run detection on a single frame
        Returns:
            detections: list of [x1, y1, x2, y2, confidence]
        """
        results = self.model(frame, stream=True)

        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # class 0 = person (COCO dataset)
                if cls != 0:
                    continue

                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append([x1, y1, x2, y2, conf])

        return detections