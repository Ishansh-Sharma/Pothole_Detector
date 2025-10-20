import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from unet_model import UNet
from ultralytics import YOLO

# ------------------ UNET MODEL PREDICTION ------------------ #
# Load UNet model
model = UNet()
model.load_state_dict(torch.load(
    "/Users/ishanshsharma/Desktop/Pothole Detection/unet_best_model-2.pth",
    map_location="cpu"))
model.eval()

# Load image using PIL and convert to RGB
img_path = "/Users/ishanshsharma/Desktop/Pothole Detection/pothole_added.jpg"
pil_img = Image.open(img_path).convert("RGB")

# Convert PIL image to OpenCV format (for consistent size references)
img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Preprocess image for UNet
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(pil_img).unsqueeze(0)

# Predict UNet mask
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output)
    predicted_mask = prediction.squeeze().cpu().numpy()

# Threshold and resize mask to original image size
unet_mask = (predicted_mask > 0.1).astype(np.uint8) * 255
unet_mask_resized = cv2.resize(unet_mask, (img_cv.shape[1], img_cv.shape[0]))

# ------------------ YOLO MODEL PREDICTION ------------------ #
# Load YOLO model
yolo_model = YOLO("/Users/ishanshsharma/Desktop/Pothole Detection/Final_Pothole_YolO_V8.pt")
class_names = yolo_model.names

# Resize image for YOLO input
img_yolo = cv2.resize(img_cv.copy(), (1020, 500))
h, w, _ = img_yolo.shape
results = yolo_model.predict(img_yolo, verbose=False)

# Draw bounding boxes and labels from YOLO
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_yolo, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# Initialize YOLO mask
yolo_mask = np.zeros((h, w), dtype=np.uint8)

# Extract YOLO segmentation mask
for r in results:
    boxes = r.boxes
    masks = r.masks

if masks is not None:
    masks = masks.data.cpu().numpy()
    for seg in masks:
        seg_resized = cv2.resize(seg, (w, h))
        yolo_mask[seg_resized > 0.5] = 255

# Extract YOLO segmentation mask
for r in results:
    boxes = r.boxes
    masks = r.masks

if masks is not None:
    masks = masks.data.cpu().numpy()
    for seg in masks:
        seg_resized = cv2.resize(seg, (w, h))
        yolo_mask[seg_resized > 0.5] = 255

    # Draw boxes and labels onto the YOLO mask
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        label = f"{class_names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
        cv2.rectangle(yolo_mask, (x1, y1), (x2, y2), 255, 2)  # white box
        cv2.putText(yolo_mask, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 255, 2)


# Resize YOLO mask to match UNet mask size
yolo_mask_resized = cv2.resize(yolo_mask, (unet_mask_resized.shape[1], unet_mask_resized.shape[0]))

# ------------------ COMBINE MASKS ------------------ #
combined_mask = np.maximum(unet_mask_resized, yolo_mask_resized)

# ------------------ VISUALIZE ALL ------------------ #
plt.figure(figsize=(20, 12))

plt.subplot(1, 5, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

plt.subplot(1, 5, 2)
plt.title("UNet Mask")
plt.imshow(unet_mask_resized, cmap='gray')

plt.subplot(1, 5, 3)
plt.title("YOLO Mask")
plt.imshow(yolo_mask_resized, cmap='gray')

plt.subplot(1, 5, 4)
plt.title("Combined Mask")
plt.imshow(combined_mask, cmap='gray')

plt.subplot(1, 5, 5)
plt.title("Original + YOLO Boxes")
plt.imshow(cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()

# ----- UNET SEGMENTATION -----


# Load UNet model
#unet_model = UNet()
#unet_model.load_state_dict(torch.load(
 #   "/Users/ishanshsharma/Desktop/Pothole Detection/data_generation_for_pothole/Unet_model_me/unet_best_model copy.pth", 
  #  map_location="cpu"))
#unet_model.eval()

# Load image
#img_path = "/Users/ishanshsharma/Desktop/Pothole Detection/img87_png.rf.f2c41ea87da29278e58aa950decb4fb8.jpg"
#pil_img = Image.open(img_path).convert("RGB")
#img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Preprocess for UNet
#transform = transforms.Compose([
 #   transforms.Resize((256, 256)),
  #  transforms.ToTensor()
#])
#input_tensor = transform(pil_img).unsqueeze(0)

# Predict UNet mask
#with torch.no_grad():
 #   output = unet_model(input_tensor)
   # prediction = torch.sigmoid(output)
    #predicted_mask = prediction.squeeze().cpu().numpy()

# Threshold and convert to mask
#unet_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
#unet_mask_resized = cv2.resize(unet_mask, (img_cv.shape[1], img_cv.shape[0]))

# ----- YOLO MASK EXTRACTION -----
# Load YOLO model
#yolo_model = YOLO("/Users/ishanshsharma/Desktop/Pothole Detection/Final_Pothole_YolO_V8.pt")
#class_names = yolo_model.names

#img_yolo = cv2.resize(img_cv.copy(), (1020, 500))
#h, w, _ = img_yolo.shape
#results = yolo_model.predict(img_yolo, verbose=False)

#yolo_mask = np.zeros((h, w), dtype=np.uint8)

#for r in results:
 #   boxes = r.boxes
  #  masks = r.masks

#if masks is not None:
 #   masks = masks.data.cpu().numpy()
  #  for seg in masks:
     #   seg_resized = cv2.resize(seg, (w, h))
      ##  yolo_mask[seg_resized > 0.5] = 255

# Resize YOLO mask to match UNet mask
#yolo_mask_resized = cv2.resize(yolo_mask, (unet_mask_resized.shape[1], unet_mask_resized.shape[0]))

# Combine masks
#combined_mask = np.maximum(unet_mask_resized, yolo_mask_resized)

# ----- VISUALIZE ALL -----


