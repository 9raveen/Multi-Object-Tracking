import cv2
from src.detector import YOLODetector

# load video
cap = cv2.VideoCapture("data/processed/trimmed.mp4")

detector = YOLODetector()

ret, frame = cap.read()

if ret:
    detections = detector.detect(frame)

    print("Detections:", detections)

    # draw boxes
    for det in detections:
        x1, y1, x2, y2, conf = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()