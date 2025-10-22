# Machine Learning
import torch
from transformers import AutoModel
from sklearn.decomposition import PCA
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# Image handling
from PIL import Image, ImageDraw
import cv2

# Math Stuff
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template

# Debugging & Information
import time
import os

# ----------------------------
# Configuration
# ----------------------------
PATCH_SIZE = 16 # Keep constant for DINOv3 ViT models
IMAGE_SIZE = 1024 # Desired size for the image (DINOv3 was trained on 512x512 images)

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
    """Preprocess image for model input"""
    image_resized = resize_transform_preserve_aspect_ratio(image, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE) # Uses aspect ratio preserving resize
    return image_resized.unsqueeze(0), image_resized  # [1, 3, H, W]


def process_image_with_dino(image_tensor: torch.Tensor, model: AutoModel, device: torch.device) -> torch.Tensor:
    """Process image with DINO model and return PCA projected features"""
    start_time = time.time() # Measure inference time

    image_tensor = image_tensor.to(device)
    with torch.no_grad(): # no gradients needed for inference, grad only used for backpropagation-training
        outputs = model(image_tensor, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    end_time = time.time() # About 0.3 seconds on RTX 4060 Laptop GPU (first run slower due to model load) - 0.03 seconds subsequently
    print(f"Model inference time: {end_time - start_time:.2f} seconds")

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


def locate_drone_position(drone_features: torch.Tensor,
                          sat_features_dir: str,
                          num_tiles_rows,
                          num_tiles_cols,
                          kernel_size,
                          device='cuda'):
    # Normalize drone features
    drone_features = F.normalize(drone_features.to(device), dim=-1)

    # Load satellite image
    sat_image_path = "geolocalization_dinov3/reconstructed_full_image_small.png"
    sat_image = Image.open(sat_image_path).convert("RGB")
    img_width, img_height = sat_image.size

    tile_width = img_width // num_tiles_cols
    tile_height = img_height // num_tiles_rows
    print(f"Tile size: {tile_width}x{tile_height} pixels")

    # Load all tile features
    num_tiles = num_tiles_rows * num_tiles_cols
    sat_features_list = [
        torch.load(os.path.join(sat_features_dir, f"features_{i}.pt"), weights_only=True)
        for i in range(num_tiles)
    ]

    # Compute valid kernel positions (only top-left positions where kernel fits entirely within tile grid)
    valid_rows = num_tiles_rows - kernel_size + 1
    valid_cols = num_tiles_cols - kernel_size + 1

    # Generate heatmap with -inf initial values
    heatmap = np.full((valid_rows, valid_cols), -np.inf, dtype=float)

    # Keep track of best score and features
    best_score = -float('inf')
    best_kernel_features = None

    # Slide kernel over tiles
    # Start at top-left, move by 1 tiles [0,0], [0,1], [1,0]. Done so it works for even kernel sizes too.
    for row in range(0, valid_rows, 1):
        for col in range(0, valid_cols, 1):
            kernel_tiles = []
            for r in range(row, row + kernel_size):
                for c in range(col, col + kernel_size):
                    idx = r * num_tiles_cols + c
                    kernel_tiles.append(sat_features_list[idx])
            
            # Combine kernel tile features and normalize
            kernel_features = torch.cat(kernel_tiles, dim=0).to(device)
            kernel_features = F.normalize(kernel_features, dim=-1)

            # Compute cosine similarity and mean of top-k similarities
            similarity = torch.matmul(drone_features, kernel_features.T)
            top_k = max(1, kernel_features.shape[0] // 20)
            topk_vals, _ = similarity.flatten().topk(top_k)
            mean_topk = topk_vals.mean().item()

            # Append to heatmap
            heatmap[row, col] = mean_topk

            # Update best score and features
            if mean_topk > best_score:
                best_score = mean_topk
                best_kernel_features = kernel_features.clone()

            del kernel_features, similarity
            torch.cuda.empty_cache()

    # Selecting top-3 from Heatmap (Highest mean cosine similarity positions)
    flat_idx = np.argpartition(-heatmap.flatten(), 3)[:3]
    top3_scores = heatmap.flatten()[flat_idx]
    top3_positions = [(idx // valid_cols, idx % valid_cols) for idx in flat_idx]
    top3_sorted = sorted(zip(top3_positions, top3_scores), key=lambda x: -x[1])

    # Map top-left kernel positions to center and clamp coords
    for rank, (pos, _) in enumerate(top3_sorted, 1): # _ is score -- was used for debugging, can be printed if needed
        row, col = pos  # these are top-left tile indices of the kernel
        
        # Compute kernel center in pixel coordinates
        center_y = row * tile_height + (kernel_size * tile_height) // 2
        center_x = col * tile_width  + (kernel_size * tile_width)  // 2

        # Compute the crop box
        crop_top = center_y - (kernel_size * tile_height) // 2
        crop_left = center_x - (kernel_size * tile_width)  // 2
        crop_bottom = crop_top + kernel_size * tile_height
        crop_right = crop_left + kernel_size * tile_width

        # Clamp to image bounds (so we don't get padded-black areas)
        crop_top_clamped    = max(0, crop_top)
        crop_left_clamped   = max(0, crop_left)
        crop_bottom_clamped = min(img_height, crop_bottom)
        crop_right_clamped  = min(img_width,  crop_right)

        patch = sat_image.crop((crop_left_clamped, crop_top_clamped,
                                crop_right_clamped, crop_bottom_clamped))

        patch.save(f"geolocalization_dinov3/fine_grained_candidates/candidate_{rank}_patch.png")

        # Save the features in each candidate for fine grained search later
        # This requires us to reload the tile features again (slightly inefficient, but simpler code)
        kernel_tiles = []
        for r in range(row, row + kernel_size):
            for c in range(col, col + kernel_size):
                idx = r * num_tiles_cols + c
                tile_features = sat_features_list[idx].to(device)
                kernel_tiles.append(tile_features)
                del tile_features # Free up memory

        kernel_features = torch.cat(kernel_tiles, dim=0)
        kernel_features = F.normalize(kernel_features, dim=-1)
        torch.save(kernel_features.cpu(), f"geolocalization_dinov3/fine_grained_candidates/candidate_{rank}_features.pt")
        
        del kernel_features # Free up memory

    # Fine grained search on the three candidates:
    for rank in range(1, 4): # Ranks 1, 2, 3
        # Load image patch
        patch_path = f"geolocalization_dinov3/fine_grained_candidates/candidate_{rank}_patch.png"
        patch_img = Image.open(patch_path).convert("RGB")

        # Load corresponding features
        patch_features = torch.load(f"geolocalization_dinov3/fine_grained_candidates/candidate_{rank}_features.pt", weights_only=True).to(device)

        # Divide image and features into 16x16 patches (256 total)
        patch_width, patch_height = patch_img.size

        # Compute the number of tiles along width and height, at least 1
        num_tiles_width  = max(round(patch_width  / (patch_width  / PATCH_SIZE)), 1)
        num_tiles_height = max(round(patch_height / (patch_height / PATCH_SIZE)), 1)

        # Compute actual tile size so that tiles fit exactly
        tile_width  = max(patch_width  // num_tiles_width, 1)
        tile_height = max(patch_height // num_tiles_height, 1)

        print(f"Adjusted tile size: {tile_width}x{tile_height}")
        print(f"Number of tiles: {num_tiles_width}x{num_tiles_height}")

        # Generate image patches
        patch_images = []
        for i in range(num_tiles_height):
            for j in range(num_tiles_width):
                left   = j * tile_width
                right  = min(left + tile_width, img_width)
                top    = i * tile_height
                bottom = min(top + tile_height, img_height)

                patch = patch_img.crop((left, top, right, bottom))
                patch_images.append(patch)

        # Extract features for each patch using the adjusted tile size
        # Number of tiles in feature grid along each axis
        num_feature_tiles = patch_features.shape[0]  # total tiles in kernel

        # Compute approximate number of tiles per patch
        tiles_per_patch_y = max(num_tiles_height, 1)
        tiles_per_patch_x = max(num_tiles_width, 1)

        # Determine how many tiles go into each patch along each axis
        tiles_per_patch = num_feature_tiles // (tiles_per_patch_y * tiles_per_patch_x)

        print(f"Tiles per patch: {tiles_per_patch} - Indicates how many feature tiles correspond to each image patch")

        # Split features roughly evenly across patches
        patch_features_list = []
        for patch_idx in range(len(patch_images)):
            start_idx = patch_idx * tiles_per_patch
            # Make sure we don't go out of bounds
            end_idx = min(start_idx + tiles_per_patch, num_feature_tiles)
            feature_patch = patch_features[start_idx:end_idx]
            patch_features_list.append(feature_patch)
        

        

    return patch, heatmap, best_kernel_features

# --- This is currently trash, as the satellite patch is too zoomed out to find good keypoints, do fine grained first ---
def drone_position_homography(sat_patch: Image.Image, drone_img: Image.Image):
    """ From the satellite patch and the drone image, find the drone position using feature matching and homography """


if __name__ == "__main__":
    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m" # DINOv3 ViT-Large with 16x16 patches, pretrained on satellite imagery (0.3B parameters)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Loop, continuously process images from drone camera feed
    while True: # <-- Replace with appropriate ROS2 subscriber callback or loop - to avoid infinite loop in real implementation
        # Capture image from drone camera feed
        #image = capture_image_from_drone()

        # For testing, load image from file (remove once drone image capture works)
        drone_image_path = "geolocalization_dinov3/03_0747.JPG" # NOTE: Once we move to ROS2, find image without hardcoding path
        pil_image = Image.open(drone_image_path).convert("RGB")

        # Preprocess image - Resize and convert to tensor
        image_tensor, image_resized = preprocess_image(pil_image)

        # Run DINOv3 model
        features = process_image_with_dino(image_tensor, model, device)

        # Locate drone tile position in satellite tiles
        patch, heatmap, patch_features = locate_drone_position(
        drone_features=features,            # Features from drone image
        sat_features_dir="dino_features",   # Point to folder with satellite tile features - Generated from offline_geolocalization.py
        num_tiles_rows=5,                   # Number of rows in satellite feature grid - Read from satellite_image_processing.py
        num_tiles_cols=4,                   # Number of columns in satellite feature grid - Read from satellite_image_processing.py
        kernel_size=2,                      # Expands kernel from top-left tile to N x N tiles
        device=device                       # Use same device as model (likely 'cuda' - GPU)
        )

        
        # drone_location = drone_position_homography(patch, pil_image)
        plt.figure(figsize=(8, 8))
        plt.imshow(patch)
        plt.axis("off")
        plt.title("Best Matching Satellite Patch")

        # --- Everything below is for visualization/report only ---
        # Visualize heatmap
        # Only the valid top-left positions (Removes padding zeros for 1-1 kernel positions)
        valid_rows = heatmap.any(axis=1)
        valid_cols = heatmap.any(axis=0)
        valid_heatmap = heatmap[np.ix_(valid_rows, valid_cols)]
        plt.figure(figsize=(8, 6))
        plt.imshow(valid_heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Mean Cosine Similarity')
        plt.title("Drone-to-Satellite Kernel Matching Heatmap")
        plt.xlabel("Kernel Column")
        plt.ylabel("Kernel Row")
        plt.show()

        # PCA project to RGB for visualization
        projected_image = pca_project_rgb(features, image_resized)
        plt.figure(figsize=(8, 8))
        plt.imshow(projected_image)
        plt.axis("off")
        plt.title("PCA Projected Drone Image")
        plt.show()
