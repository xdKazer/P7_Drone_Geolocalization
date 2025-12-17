# Machine Learning
import torch
from transformers import AutoModel
from sklearn.decomposition import PCA
import torchvision.transforms.functional as TF

# Image handling
from PIL import Image

# Debugging & Information
from pathlib import Path

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


Base_dir = Path(__file__).parent.resolve() 
SAT_DIR   = Base_dir / "tiles"
output_dir = Base_dir / "dinov3_features"

def save_feats(img_path: Path):
    out = output_dir / (img_path.stem + ".pt")
    
    img = Image.open(img_path).convert("RGB")

    desired_tile_height = 6; desired_tile_width = 6
    num_tiles_width = max(round(img.width / desired_tile_width), 1)
    num_tiles_height = max(round(img.height / desired_tile_height), 1)

    # Compute actual tile size in pixels
    tile_width = max(img.width // num_tiles_width, 1)
    tile_height = max(img.height // num_tiles_height, 1)
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
            tile_img = img.crop([left, top, right, bottom])

            # Run model
            features = process_image_with_dino(tile_img, model, device)

            # Add individually processed features to shared array.
            sat_patch_feature.append((features.cpu()))

    torch.save(sat_patch_feature, out)
    print(f"[ok] saved {out.name}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m" # DINOv3 ViT-Large with 16x16 patches, pretrained on satellite imagery (0.3B parameters)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    # tiles
    tiles = [p for p in SAT_DIR.iterdir() if p.is_file() and p.suffix.lower()==".png"]
    for i, p in enumerate(tiles):
        print("Currently processing image: " + str(i+1) + " / " + str(len(tiles)))
        save_feats(p)