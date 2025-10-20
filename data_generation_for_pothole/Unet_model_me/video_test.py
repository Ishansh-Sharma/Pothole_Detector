import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from unet_model import UNet
from ultralytics import YOLO

# ------------------ Load Models ------------------ #
# Load UNet
unet_model = UNet()
unet_model.load_state_dict(torch.load(
    "/Users/ishanshsharma/Desktop/Pothole Detection/unet_best_model-2.pth",
    map_location="cpu"))
unet_model.eval()

# Load YOLO
yolo_model = YOLO("/Users/ishanshsharma/Desktop/Pothole Detection/Final_Pothole_YolO_V8.pt")
class_names = yolo_model.names

# UNet preprocessing
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

# ------------------ Video Setup ------------------ #
video_path = "/Users/ishanshsharma/Desktop/Pothole Detection/auto-nav good.MTS"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "/Users/ishanshsharma/Desktop/Pothole Detection/output_processed_video.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

# ------------------ Frame-by-Frame Processing ------------------ #
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make a copy for YOLO drawing
    frame_yolo = cv2.resize(frame.copy(), (1020, 500))
    h, w, _ = frame_yolo.shape

    # ----------- UNet Segmentation ----------- #
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_frame).unsqueeze(0)

    with torch.no_grad():
        output = unet_model(input_tensor)
        prediction = torch.sigmoid(output)
        predicted_mask = prediction.squeeze().cpu().numpy()

    unet_mask = (predicted_mask > 0.05).astype(np.uint8) * 255
    unet_mask_resized = cv2.resize(unet_mask, (frame.shape[1], frame.shape[0]))

    # ----------- YOLO Detection & Mask ----------- #
    results = yolo_model.predict(frame_yolo, verbose=False)
    yolo_mask = np.zeros((h, w), dtype=np.uint8)

    for r in results:
        boxes = r.boxes
        masks = r.masks

    if masks is not None:
        masks = masks.data.cpu().numpy()
        for seg in masks:
            seg_resized = cv2.resize(seg, (w, h))
            yolo_mask[seg_resized > 0.5] = 255

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = f"{class_names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
            cv2.rectangle(yolo_mask, (x1, y1), (x2, y2), 255, 2)
            cv2.putText(yolo_mask, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

    yolo_mask_resized = cv2.resize(yolo_mask, (frame.shape[1], frame.shape[0]))

    # ----------- Combine Masks ----------- #
    combined_mask = np.maximum(unet_mask_resized, yolo_mask_resized)

    # ----------- Overlay on Original Frame (Optional) ----------- #
    colored_mask = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

    # Save and show
    out.write(overlay)
    cv2.imshow('Overlayed Pothole Detection', overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
