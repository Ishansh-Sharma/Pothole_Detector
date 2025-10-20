import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet_model import UNet

# Load the model
model = UNet()
model.load_state_dict(torch.load("/Users/ishanshsharma/Desktop/Pothole Detection/data_generation_for_pothole/Unet_model_me/unet_best_model copy.pth", map_location="cpu"))
model.eval()

# Load and preprocess image
img_path = "/Users/ishanshsharma/Desktop/Pothole Detection/img87_png.rf.f2c41ea87da29278e58aa950decb4fb8.jpg"
img = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output)
    predicted_mask = prediction.squeeze().cpu().numpy()

# Visualize
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(predicted_mask, cmap="gray")

plt.tight_layout()
plt.show()
