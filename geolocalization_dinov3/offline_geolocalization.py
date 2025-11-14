# Machine Learning
import torch
from transformers import AutoModel
from sklearn.decomposition import PCA
import torchvision.transforms.functional as TF

# Image handling
from PIL import Image

# Debugging & Information
import os
from pathlib import Path
import re

# ----------------------------
# Configuration
# ----------------------------
PATCH_SIZE = 16 # Keep constant for DINOv3 ViT models
IMAGE_SIZE = 512 # Desired size for the image (DINOv3 was trained on 512x512 images)

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

def preprocess_image(image: Image) -> torch.Tensor:
    """Preprocess image for model input."""
    image_resized = resize_transform_preserve_aspect_ratio(image, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE) # Uses aspect ratio preserving resize
    return image_resized.unsqueeze(0), image_resized  # [1, 3, H, W]


def process_image_with_dino(image_tensor: torch.Tensor, model: AutoModel, device: torch.device) -> torch.Tensor:
    """Process image with DINO model and return PCA projected features."""
    #start_time = time.time() # Measure inference time

    image_tensor = image_tensor.to(device)
    with torch.no_grad(): # no gradients needed for inference, grad only used for training
        outputs = model(image_tensor, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    #end_time = time.time()
    #print(f"Model inference time: {end_time - start_time:.2f} seconds")

    features = hidden_states[-1][0, 1:, :]  # remove CLS token, take batch 0

    return features

def pca_project_rgb(features: torch.Tensor, image_resized: torch.Tensor) -> torch.Tensor:
    """Project features to 3D RGB space using PCA."""
    x = features  # [num_patches, feature_dim]
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
    
    return projected_image


if __name__ == "__main__":
    # Setup model - This requires an internet connection, download model locally if needed.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m" # DINOv3 ViT-Large with 16x16 patches, pretrained on satellite imagery (0.3B parameters)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Empty cache prior to inference
    torch.cuda.empty_cache()

    os.makedirs("pca_projected_images", exist_ok=True)
    os.makedirs("geolocalization_dinov3/tile_features", exist_ok=True)

    tiles = [p for p in Path("geolocalization_dinov3/tiles_png_1km").iterdir() if p.is_file() and p.suffix.lower() == ".png"]

    # Loop through every image in 'tiles_png_uniform' folder
    for i in range(len(tiles)):
        print("Currently processing image: " + str(i) + " / " + str(len(tiles)))
        image_path = tiles[i]

        pil_image = Image.open(image_path).convert("RGB")

        # Split image into multiple tiles to minimize compression from DINOv3
        # Compute number of tiles
        desired_tile_height = 4; desired_tile_width = 3 # I used 5x4 for 2x2km so 4x3 should be fine for 1x1km
        num_tiles_width = max(round(pil_image.width / desired_tile_width), 1)
        num_tiles_height = max(round(pil_image.height / desired_tile_height), 1)
        print(pil_image.height, pil_image.width)

        # Compute actual tile size in pixels
        tile_width = max(pil_image.width // num_tiles_width, 1)
        tile_height = max(pil_image.height // num_tiles_height, 1)

        print(f"Tile count for satellite patch: {tile_height}x{tile_width}") # 3 tiles left, 4 tiles down
        print(f"Size of tiles pixels: {num_tiles_width}x{num_tiles_height}")

        sat_patch_feature = []
        for i in range(tile_height):
            for j in range(tile_width):
                top = i * num_tiles_height
                left = j * num_tiles_width
                bottom = top + num_tiles_height
                right = left + num_tiles_width

                # Slice the satellite image based on tiles
                tile_img = pil_image.crop([left, top, right, bottom])

                # Preprocess image
                image_tensor, image_resized = preprocess_image(tile_img)

                # Run model
                features = process_image_with_dino(image_tensor, model, device)

                # Add individually processed features to shared array.
                sat_patch_feature.append(features.cpu())

        # Figure out both x and y values for current satellite patch
        name = image_path.stem
        m = re.search(r"y(?P<y>\d+)_x(?P<x>\d+)", name)
        if not m:
            raise ValueError(f"Cannot parse offsets from '{pil_image.name}'. Expected '...y<Y>_x<X>...'")
        y = int(m.group("y"))
        x = int(m.group("x"))

        # Save DINOv3 features
        torch.save(sat_patch_feature, f"geolocalization_dinov3/tile_features/tile_y{y}_x{x}.pt")