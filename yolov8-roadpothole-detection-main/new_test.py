from ultralytics import YOLO
import cv2
import numpy as np

# Load the model
model = YOLO("/Users/ishanshsharma/Desktop/Pothole Detection/Final_Pothole_YolO_V8.pt")
class_names = model.names

# Load a single image
img = cv2.imread("/Users/ishanshsharma/Desktop/Pothole Detection/modified_images/2023-06-15_14-06-50-front_mp4_600_jpg.rf.cebe72676d6e66b482e5667e35b5d48d.jpg")  # Change this path
img = cv2.resize(img, (1020, 500))
h, w, _ = img.shape

# Predict
results = model.predict(img)[0]
boxes = results.boxes
masks = results.masks

# Draw detections
if masks is not None:
    masks = masks.data.cpu().numpy()
    for seg, box in zip(masks, boxes):
        seg = cv2.resize(seg, (w, h))
        contours, _ = cv2.findContours((seg > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cls_id = int(box.cls)
            cls_name = class_names[cls_id]
            x, y, w_box, h_box = cv2.boundingRect(contour)
            cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
            cv2.putText(img, cls_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Show result
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
