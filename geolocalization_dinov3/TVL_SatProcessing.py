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

# ----------------------------
# Helper function
# ----------------------------
def process_image_with_dino(image: Image, model: AutoModel, device: torch.device) -> torch.Tensor:
    """Process image with DINO model and return PCA projected features."""
    #start_time = time.time() # Measure inference time
    image_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    image_tensor = image_tensor.to(device)
    with torch.no_grad(): # no gradients needed for inference, grad only used for training
        outputs = model(image_tensor, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    #end_time = time.time()
    #print(f"Model inference time: {end_time - start_time:.2f} seconds")

    features = hidden_states[-1][0, 1:, :]  # remove CLS token, take batch 0

    return features


if __name__ == "__main__":
    # Setup model - This requires an internet connection, download model locally if needed.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m" # DINOv3 ViT-Large with 16x16 patches, pretrained on satellite imagery (0.3B parameters)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Empty cache prior to inference
    torch.cuda.empty_cache()

    os.makedirs("geolocalization_dinov3/tile_features_uniform", exist_ok=True)

    tiles = [p for p in Path("geolocalization_dinov3/tiles_uniform").iterdir() if p.is_file() and p.suffix.lower() == ".png"]

    # Loop through every image in 'tiles_png_uniform' folder
    for i in range(len(tiles)):
        print("Currently processing image: " + str(i+1) + " / " + str(len(tiles)))
        image_path = tiles[i]

        pil_image = Image.open(image_path).convert("RGB")

        # Split image into multiple tiles to minimize compression from DINOv3
        # Compute number of tiles
        desired_tile_height = 6; desired_tile_width = 6
        num_tiles_width = max(round(pil_image.width / desired_tile_width), 1)
        num_tiles_height = max(round(pil_image.height / desired_tile_height), 1)

        # Compute actual tile size in pixels
        tile_width = max(pil_image.width // num_tiles_width, 1)
        tile_height = max(pil_image.height // num_tiles_height, 1)
        print(f"Tile count for satellite patch: {tile_width}x{tile_height}") # For my 1024px tiles this is 5 x 5, varies a bit at corners.
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

                # Run model
                features = process_image_with_dino(tile_img, model, device)

                # Add individually processed features to shared array.
                h_patches = tile_img.height // PATCH_SIZE; w_patches = tile_img.width // PATCH_SIZE # 10 x 10
                sat_patch_feature.append((features.cpu()))

        # Figure out both x and y values for current satellite patch
        name = image_path.stem
        m = re.search(r"y(?P<y>\d+)_x(?P<x>\d+)", name)
        y = int(m.group("y"))
        x = int(m.group("x"))

        # Save DINOv3 features
        torch.save(sat_patch_feature, f"geolocalization_dinov3/tile_features_uniform/tile_y{y}_x{x}.pt")