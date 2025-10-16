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
import os

# ----------------------------
# Configuration
# ----------------------------
PATCH_SIZE = 16 # Keep constant for DINOv3 ViT models
IMAGE_SIZE = 1024 # Desired size for the image (DINOv3 was trained on 512x512 images) - 900 is the image size. Preserve details.

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
    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m" # DINOv3 ViT-Large with 16x16 patches, pretrained on satellite imagery (0.3B parameters)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Empty cache prior to inference
    torch.cuda.empty_cache()

    os.makedirs("pca_projected_images", exist_ok=True)

    # Loop through every image in 'tiles_png_uniform' folder
    for i in range(len(os.listdir("tiles_png_uniform"))):
        print("Currently processing image: " + str(i) + " / " + str(len(os.listdir("tiles_png_uniform"))))
        image_path = f"tiles_png_uniform/tile_{i}.png"

        pil_image = Image.open(image_path).convert("RGB")

        # Preprocess image
        image_tensor, image_resized = preprocess_image(pil_image)

        # Run model
        features = process_image_with_dino(image_tensor, model, device)

        # Save DINOv3 features
        os.makedirs("dino_features", exist_ok=True)
        torch.save(features.cpu(), f"dino_features/features_{i}.pt")

        # --- Everything below is for visualization only ---
        # PCA project to RGB
        projected_image = pca_project_rgb(features, image_resized)

        # Save image to folder
        projected_image = projected_image.permute(1, 2, 0)  # (H, W, 3)
        plt.imsave(f"pca_projected_images/pca_projected_{i}.png", projected_image.numpy())

    pca_folder = "pca_projected_images"

    num_tiles_width = 18
    num_tiles_height = 18

    # Tile size read from first pca_projected_0.png
    check_tile_path = os.path.join(pca_folder, "pca_projected_0.png")
    check_tile_img = Image.open(check_tile_path).convert("RGB")

    tile_width, tile_height = check_tile_img.size
    print(f"Detected tile size: {tile_width}x{tile_height}")

    # Create empty canvas
    reconstructed_img = Image.new("RGB", (num_tiles_width * tile_width, num_tiles_height * tile_height))

    # Paste tiles
    tile_count = 0
    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            tile_path = os.path.join(pca_folder, f"pca_projected_{tile_count}.png")
            tile_img = Image.open(tile_path).convert("RGB")
            
            left = j * tile_width
            top = i * tile_height
            reconstructed_img.paste(tile_img, (left, top))
            
            tile_count += 1

    # Save and display
    reconstructed_img.save("geolocalization_dinov3/reconstructed_full_image.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(reconstructed_img)
    plt.axis("off")
    plt.show()