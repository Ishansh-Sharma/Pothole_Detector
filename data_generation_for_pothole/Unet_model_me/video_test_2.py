import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from unet_model import UNet
from ultralytics import YOLO

# ------------------ Load Models ------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load UNet
unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load(
    "/Users/ishanshsharma/Desktop/Pothole Detection/unet_best_model-2.pth",
    map_location=device))
unet_model.eval()

# Load YOLO
yolo_model = YOLO("/Users/ishanshsharma/Desktop/Pothole Detection/Final_Pothole_YolO_V8.pt")
class_names = yolo_model.names

# UNet preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# ------------------ Video Setup ------------------ #
video_path = "/Users/ishanshsharma/Desktop/Pothole Detection/auto-nav good.MTS"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30  # Manually setting FPS for faster playback
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "/Users/ishanshsharma/Desktop/Pothole Detection/output_processed_video.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

# ------------------ Frame-by-Frame Processing ------------------ #
frame_skip = 2  # Skip every 2nd frame
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue

    # Resize frame for processing
    resized_frame = cv2.resize(frame, (512, 512))

    # ----------- UNet Segmentation ----------- #
    pil_frame = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = unet_model(input_tensor)
        prediction = torch.sigmoid(output)
        predicted_mask = prediction.squeeze().cpu().numpy()

    unet_mask = (predicted_mask > 0.05).astype(np.uint8) * 255
    unet_mask_resized = cv2.resize(unet_mask, (frame.shape[1], frame.shape[0]))

    # ----------- YOLO Detection & Mask ----------- #
    frame_yolo = resized_frame.copy()
    results = yolo_model.predict(frame_yolo, verbose=False)
    yolo_mask = np.zeros((512, 512), dtype=np.uint8)

    for r in results:
        boxes = r.boxes
        masks = r.masks

    if masks is not None:
        masks = masks.data.cpu().numpy()
        for seg in masks:
            seg_resized = cv2.resize(seg, (512, 512))
            yolo_mask[seg_resized > 0.5] = 255

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = f"{class_names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
            cv2.rectangle(yolo_mask, (x1, y1), (x2, y2), 255, 2)
            cv2.putText(yolo_mask, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

    yolo_mask_resized = cv2.resize(yolo_mask, (frame.shape[1], frame.shape[0]))

    # ----------- Combine Masks ----------- #
    combined_mask = np.maximum(unet_mask_resized, yolo_mask_resized)

    # ----------- Overlay on Original Frame ----------- #
    colored_mask = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

    # Save video
    out.write(overlay)

    #(Optional) Display live output (disabled for speed)
    cv2.imshow('Overlayed Pothole Detection', overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
