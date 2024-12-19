from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Load the model
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.eval()  # Set model to evaluation mode

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the input image
image = Image.open("/Users/niranjanganesan/Pictures/Star_Trails/TIFs/Stars-20.tif")
original_image = image.copy()
input_images = transform_image(image).unsqueeze(0)  # No .to('cuda') for CPU

# Prediction
with torch.no_grad():
    preds = model(input_images)[-1].sigmoid()  # No .cpu() needed as it runs on CPU by default

# Process the prediction
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)

# Add alpha channel to the original image for transparency
image.putalpha(mask)

# Save the foreground image
image.save("foreground_image.png")

# Create the background using the inverted mask
mask_inverted = Image.eval(mask, lambda x: 255 - x)  # Invert the mask
background = Image.new("RGB", original_image.size, (255, 255, 255))  # White background
background = Image.composite(original_image, background, mask_inverted)

# Save the background image
background.save("background_image.png")

print("Foreground and background images saved successfully.")