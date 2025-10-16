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
    """Preprocess image for model input."""
    image_resized = resize_transform_preserve_aspect_ratio(image, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE) # Uses aspect ratio preserving resize
    return image_resized.unsqueeze(0), image_resized  # [1, 3, H, W]


def process_image_with_dino(image_tensor: torch.Tensor, model: AutoModel, device: torch.device) -> torch.Tensor:
    """Process image with DINO model and return PCA projected features."""
    start_time = time.time() # Measure inference time

    image_tensor = image_tensor.to(device)
    with torch.no_grad(): # no gradients needed for inference, grad only used for training
        outputs = model(image_tensor, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    end_time = time.time()
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
                          num_tiles_rows=18,
                          num_tiles_cols=18,
                          device='cuda',
                          kernel_size=3,
                          stride=1):
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

    # Initialize heatmap
    heatmap = np.zeros((num_tiles_rows, num_tiles_cols))

    best_score = -float('inf')
    best_tile_pos = (0, 0)
    best_kernel_features = None

    # Slide kernel over tiles (only valid top-left positions)
    for row in range(0, num_tiles_rows - kernel_size + 1, stride):
        for col in range(0, num_tiles_cols - kernel_size + 1, stride):
            kernel_tiles = []
            for r in range(row, row + kernel_size):
                for c in range(col, col + kernel_size):
                    idx = r * num_tiles_cols + c
                    kernel_tiles.append(sat_features_list[idx])
            print(f"Processing kernel at position ({row}, {col})")
            kernel_features = torch.cat(kernel_tiles, dim=0).to(device)
            kernel_features = F.normalize(kernel_features, dim=-1)

            similarity = torch.matmul(drone_features, kernel_features.T) # .T is transpose.
            top_k = 1 # max(1, kernel_features.shape[0] // 20)  # Keep only feature_count * 0.05 (5%) of the top values for averaging - 7169385
            topk_vals, _ = similarity.flatten().topk(top_k)  # take top k values across all drone and kernel features
            mean_topk = topk_vals.mean().item()

            # Fill heatmap
            heatmap[row, col] = mean_topk

            # Update best score
            if mean_topk > best_score:
                best_score = mean_topk
                best_tile_pos = (row, col)
                best_kernel_features = kernel_features.clone()

            del kernel_features, similarity
            torch.cuda.empty_cache()

    print(f"Finished sliding kernel. Best position: {best_tile_pos}, Best score: {best_score:.4f}")

    # Get top 3 candidates
    flat_idx = np.argpartition(-heatmap.flatten(), 3)[:3]  # negative for descending
    top3_scores = heatmap.flatten()[flat_idx]
    top3_positions = [ (idx // num_tiles_cols, idx % num_tiles_cols) for idx in flat_idx ]

    # Sort top3 by score descending
    top3_sorted = sorted(zip(top3_positions, top3_scores), key=lambda x: -x[1])

    for rank, (pos, score) in enumerate(top3_sorted, 1):
        row, col = pos
        center_row = row + kernel_size // 2
        center_col = col + kernel_size // 2

        center_y = center_row * tile_height + tile_height // 2
        center_x = center_col * tile_width + tile_width // 2

        # Crop a patch of the same size as the kernel (kernel_size tiles)
        crop_top = center_y - (kernel_size * tile_height) // 2
        crop_left = center_x - (kernel_size * tile_width) // 2
        crop_bottom = crop_top + kernel_size * tile_height
        crop_right = crop_left + kernel_size * tile_width

        patch = sat_image.crop((crop_left, crop_top, crop_right, crop_bottom))
        filename = f"geolocalization_dinov3/candidate_{rank}_patch.png"
        patch.save(filename)
        print(f"Saved Rank {rank} candidate patch at {pos} with score {score:.4f} -> {filename}")

    # Fine grained search on the three candidates:


    return patch, heatmap, best_kernel_features

# --- This is currently trash, as the satellite patch is too zoomed out to find good keypoints, do fine graind first ---
def drone_position_homography(sat_patch: Image.Image, drone_img: Image.Image):
    # Find drone location and orientation in the satellite patch using OpenCV keypoints and homography.
    # Convert to grayscale
    sat_cv = cv2.cvtColor(np.array(sat_patch), cv2.COLOR_RGB2GRAY)
    drone_cv = cv2.cvtColor(np.array(drone_img), cv2.COLOR_RGB2GRAY)

    # --- Step 1: Detect keypoints and descriptors ---
    sift = cv2.SIFT_create(nfeatures=2000) # Limit to 2000 keypoints
    kp1, des1 = sift.detectAndCompute(drone_cv, None)
    kp2, des2 = sift.detectAndCompute(sat_cv, None)

    # --- Step 2: Match descriptors using FLANN ---
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # --- Step 3: Lowe's ratio test to filter good matches ---
    good_matches = [m for m,n in matches if m.distance < 0.75 * n.distance]
    print(f"Found {len(good_matches)} good matches")

    if len(good_matches) < 4:
        raise ValueError("Not enough good matches to compute homography")

    # --- Step 4: Extract matched points ---
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # --- Step 5: Compute homography ---
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography could not be computed")

    # --- Step 6: Map drone image corners to satellite patch ---
    h_drone, w_drone = drone_cv.shape
    corners = np.float32([[0,0],[w_drone,0],[w_drone,h_drone],[0,h_drone]]).reshape(-1,1,2)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    # Draw rectangle
    sat_patch_vis = np.array(sat_patch).copy()
    cv2.polylines(sat_patch_vis, [np.int32(transformed_corners)], isClosed=True, color=(255,0,0), thickness=2)

    # Compute center
    center_x = int(np.mean(transformed_corners[:,0,0]))
    center_y = int(np.mean(transformed_corners[:,0,1]))
    print(f"Drone center at: ({center_x}, {center_y})")

    # Draw center point
    cv2.circle(sat_patch_vis, (center_x, center_y), radius=5, color=(0,255,0), thickness=-1)

    # Show result
    cv2.imshow("Drone Position via SIFT Homography", cv2.cvtColor(sat_patch_vis, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (center_x, center_y), sat_patch_vis


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
        drone_image_path = "geolocalization_dinov3/03_0565.JPG" # NOTE: Once we move to ROS2, find image without hardcoding path
        pil_image = Image.open(drone_image_path).convert("RGB")

        # Preprocess image
        image_tensor, image_resized = preprocess_image(pil_image)

        # Run model
        features = process_image_with_dino(image_tensor, model, device)

        patch, heatmap, patch_features = locate_drone_position(
        drone_features=features,
        sat_features_dir="dino_features",
        num_tiles_rows=18,
        num_tiles_cols=18,
        device=device,
        kernel_size=2,
        stride=1
    )

        # Visualize heatmap
        # Only the valid top-left positions (Removes padding zeros for 1-1 kernel positions)
        valid_rows = heatmap.any(axis=1)
        valid_cols = heatmap.any(axis=0)
        valid_heatmap = heatmap[np.ix_(valid_rows, valid_cols)]
        plt.figure(figsize=(8, 6))
        plt.imshow(valid_heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Mean Cosine Similarity')
        plt.title("Drone-to-Satellite Kernel Matching Heatmap (valid positions only)")
        plt.xlabel("Kernel Column")
        plt.ylabel("Kernel Row")
        plt.show()

        # drone_location = drone_position_homography(patch, pil_image)
        plt.figure(figsize=(8, 8))
        plt.imshow(patch)
        plt.axis("off")
        plt.title("Best Matching Satellite Patch")

        # PCA project to RGB
        projected_image = pca_project_rgb(features, image_resized)

        # Debug: visualize projected image
        plt.figure(figsize=(8, 8))
        plt.imshow(projected_image.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title("PCA Projected Image")
        plt.show()
