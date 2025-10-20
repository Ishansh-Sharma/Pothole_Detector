from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("/Users/ishanshsharma/Desktop/Pothole Detection/yolov8-roadpothole-detection-main/best.pt")
class_names = model.names

# Load video
cap = cv2.VideoCapture('/Users/ishanshsharma/Desktop/Pothole Detection/yolov8-roadpothole-detection-main/Original_Video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_pothole_detection.mp4', fourcc, 20.0, (1020, 500))

count = 0
while True:
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img, verbose=False)

    for r in results:
        boxes = r.boxes
        masks = r.masks

        if masks is not None:
            masks_data = masks.data.cpu().numpy()
            for seg, box in zip(masks_data, boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    class_id = int(box.cls)
                    class_name = class_names[class_id]
                    x, y, bw, bh = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show and save frame
    cv2.imshow('Pothole Detection', img)
    out.write(img)

    # Exit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
