import cv2
from src.detector import YOLODetector

cap = cv2.VideoCapture("data/processed/trimmed.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

detector = YOLODetector()

ret, frame = cap.read()

if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

detections, crops = detector.detect(frame)

print(f"Total detections: {len(detections)}")
print(f"Total crops: {len(crops)}")

for det in detections:
    x1, y1, x2, y2, conf = det
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.putText(frame, f"Detections: {len(detections)}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2)


cv2.imshow("Frame", frame)

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


'''
## What This Means for the Full Pipeline

Your project structure is now naturally set up for team classification:

detector.detect(frame)
    → detections (boxes)       → tracker.update()  → IDs
    → crops (jersey images)    → TeamClassifier     → team label per ID
'''