import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from unet_model import UNet

# ------------------ Load UNet Model ------------------ #
unet_model = UNet()
unet_model.load_state_dict(torch.load(
    "/Users/ishanshsharma/Desktop/Pothole Detection/data_generation_for_pothole/Unet_model_me/unet_best_model copy.pth",
    map_location="cpu"))
unet_model.eval()

# UNet preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ------------------ Video Setup ------------------ #
video_path = "/Users/ishanshsharma/Desktop/Pothole Detection/auto-nav good.MTS"
cap = cv2.VideoCapture(video_path)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 8  # Output video FPS

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "/Users/ishanshsharma/Desktop/Pothole Detection/unet_binary_mask_output.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h), isColor=False)

# ------------------ Frame-by-Frame UNet Processing ------------------ #
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL for UNet
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_frame).unsqueeze(0)

    # UNet Prediction
    with torch.no_grad():
        output = unet_model(input_tensor)
        prediction = torch.sigmoid(output)
        predicted_mask = prediction.squeeze().cpu().numpy()

    # Threshold + Resize to original frame size
    binary_mask = (predicted_mask > 0.1).astype(np.uint8) * 255
    binary_mask_resized = cv2.resize(binary_mask, (frame_w, frame_h))

    # Show only the binary mask
    cv2.imshow("UNet Binary Mask", binary_mask_resized)

    # Write to output video (grayscale)
    out.write(binary_mask_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
