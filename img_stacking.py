import cv2
import numpy as np

# Load the mask and original images
mask_path = "foreground_image.png"  # The mask image (background only)
original_path = "star_trails_with_pole_star.png"  # The original image

# Read the images
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read the mask as grayscale
original = cv2.imread(original_path)  # Read the original image

# Ensure the mask and original image have the same size
if mask.shape[:2] != original.shape[:2]:
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

# Normalize the mask to range [0, 1]
mask_normalized = mask / 255.0

# Invert the mask for foreground
mask_inverted = 1.0 - mask_normalized

# Apply the mask to the original image and background
background = cv2.imread(mask_path)  # Use the mask image as the background
if background.shape[:2] != original.shape[:2]:
    background = cv2.resize(background, (original.shape[1], original.shape[0]))

# Composite the two images
foreground = cv2.multiply(mask_inverted[:, :, None], original.astype(np.float32) / 255.0)
background = cv2.multiply(mask_normalized[:, :, None], background.astype(np.float32) / 255.0)
composite = cv2.add(foreground, background)

# Convert to 8-bit for saving or displaying
composite = (composite * 255).astype(np.uint8)

# Save or display the composite image
cv2.imwrite('superimposed_image.png', composite)

# Display the result (optional)
cv2.imshow('Superimposed Image', composite)
cv2.waitKey(0)
cv2.destroyAllWindows()