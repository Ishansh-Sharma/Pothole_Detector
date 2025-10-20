from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO("/Users/ishanshsharma/Desktop/Pothole Detection/Final_Pothole_YolO_V8.pt")
class_names = model.names

# Read image (not video)
img = cv2.imread('/Users/ishanshsharma/Desktop/Pothole Detection/modified_images/2023-06-15_14-06-50-front_mp4_1200_jpg.rf.643c16f9081cf35fda65764b62807275.jpg')

if img is None:
    print("Image not found or could not be loaded.")
else:
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img)

    for r in results:
        boxes = r.boxes
        masks = r.masks

    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, w_box, h_box = cv2.boundingRect(contour)
                cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Pothole Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
