# Machine Learning
import torch
from transformers import AutoModel
from sklearn.decomposition import PCA
import torchvision.transforms.functional as TF

# Image handling
from PIL import Image
import matplotlib.pyplot as plt

# Debugging & Information
import time

# ----------------------------
# Configuration
# ----------------------------
PATCH_SIZE = 16 # Keep constant for DINOv3 ViT models
IMAGE_SIZE = 1024 # Desired size for the shorter side of the image (Consider a test with resolution/performance trade-offs)

# ImageNet normalization values
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ----------------------------
# Helper function
# ----------------------------
# Resize while preserving aspect ratio
def resize_transform_preserve_aspect_ratio(image: Image, image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Resize image so its dimensions are divisible by patch size while preserving aspect ratio."""
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(image, (h_patches * patch_size, w_patches * patch_size)))

# Alternative: Resize to square image (not preserving aspect ratio)
def resize_transform_square_img(image: Image, image_size: int = IMAGE_SIZE, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Resize image so its dimensions are divisible by patch size and the output is square."""
    # Resize shorter side to image_size
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    
    # Make divisible by patch size
    new_h = (h_patches * patch_size)
    new_w = (w_patches * patch_size)

    # Make square: take the larger of new_h and new_w, round up to nearest multiple of patch_size
    final_size = max(new_h, new_w)
    final_size = ((final_size + patch_size - 1) // patch_size) * patch_size  # round up

    return TF.to_tensor(TF.resize(image, (final_size, final_size)))

# ----------------------------
# Load and preprocess image
# ----------------------------
image_path = "geolocalization_dinov3/03_0747.JPG" # NOTE: Once we move to ROS2, find image without hardcoding path
pil_image = Image.open(image_path).convert("RGB")

# Format image for model (resize with or without aspect, to tensor, normalize)
image_resized = resize_transform_preserve_aspect_ratio(pil_image, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE) # Uses aspect ratio preserving resize
image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
image_tensor = image_resized_norm.unsqueeze(0)  # [1, 3, H, W]

# ----------------------------
# Setup model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/dinov3-vitl16-pretrain-sat493m" # DINOv3 ViT-Large with 16x16 patches, pretrained on satellite imagery (0.3B parameters)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()
image_tensor = image_tensor.to(device) # Move image to device

# ----------------------------
# Run model
# ----------------------------
start_time = time.time() # Measure inference time

with torch.no_grad(): # no gradients needed for inference, grad only used for training
    outputs = model(image_tensor, output_hidden_states=True)
    hidden_states = outputs.hidden_states

end_time = time.time()
print(f"Model inference time: {end_time - start_time:.2f} seconds")

# ----------------------------
# Extract last layer features and remove CLS token
# ----------------------------
x = hidden_states[-1][0, 1:, :]  # remove CLS token, take batch 0

# ----------------------------
# Compute patch grid
# ----------------------------
h_patches = image_resized.shape[1] // PATCH_SIZE # // is integer division, rounds down
w_patches = image_resized.shape[2] // PATCH_SIZE
num_patches = h_patches * w_patches
print(f"Image size: {image_resized.shape[1:]} -> Patch grid: {h_patches} x {w_patches} = {num_patches} patches")

# Keep only the first `num_patches` features in case of extra tokens
x = x[:num_patches, :]

# ----------------------------
# PCA to 3D for RGB visualization
# ----------------------------
x_cpu = x.cpu().numpy()  # move to CPU for PCA
pca = PCA(n_components=3, whiten=True)
projected_image = torch.from_numpy(pca.fit_transform(x_cpu)).view(h_patches, w_patches, 3)

# Multiply by 2 and apply sigmoid to scale values between 0 and 1 for visualization
projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)

# The outputs first 4 columns are misaligned, so we move them to the back, to restore the image
# This is a quick fix solution. If time allows, we should investigate the root cause.
projected_image = torch.cat((projected_image[:, :, 4:], projected_image[:, :, :4]), dim=2) 

# ----------------------------
# Visualize
# ----------------------------
plt.figure(figsize=(8, 8))
plt.imshow(projected_image.permute(1, 2, 0).cpu().numpy())
plt.axis("off")
plt.show()
