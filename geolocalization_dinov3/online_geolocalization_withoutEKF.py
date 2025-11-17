# Machine Learning
import torch
from transformers import AutoModel
from sklearn.decomposition import PCA
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from lightglue import LightGlue, SuperPoint         # type: ignore # For feature matching
from lightglue.utils import rbd                     # type: ignore # For robust backdoor matching

# Image handling
from PIL import Image, ImageDraw
import cv2

# Math Stuff
import numpy as np
import matplotlib.pyplot as plt

# Debugging & Information
import time
import os
import pandas as pd
from pathlib import Path
import re

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

def tile_offset_from_name(tile_path: Path):
    name = tile_path.stem
    m = re.search(r"y(?P<y>\d+)_x(?P<x>\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse offsets from '{tile_path.name}'. Expected '...y<Y>_x<X>...'")
    y_off = int(m.group("y"))
    x_off = int(m.group("x"))
    return x_off, y_off

def ellipse_from_cov(mu_xy, Sigma_xy, k=2.0):
    # mu_xy: (2,) ; Sigma_xy: (2,2)
    vals, vecs = np.linalg.eigh(Sigma_xy)       # vals ascending
    a = k * np.sqrt(vals[1])                    # major axis length
    b = k * np.sqrt(vals[0])                    # minor axis length
    angle_rad = np.arctan2(vecs[1,1], vecs[0,1])  # angle of major axis
    return a, b, angle_rad, mu_xy

def ellipse_bbox(mu_xy, Sigma_xy, k=2.0, n=72): # n is number of points to sample
    a,b,theta,mu = ellipse_from_cov(mu_xy, Sigma_xy, k)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.stack([a*np.cos(t), b*np.sin(t)], axis=0)  # 2×n
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    pts_world = (R @ pts).T + mu[None,:]
    x0,y0 = pts_world.min(axis=0)
    x1,y1 = pts_world.max(axis=0)
    return (x0,y0,x1,y1)  # bbox in ORIGINAL sat px

def tiles_in_bbox(bbox, TILE_W, TILE_H, all_tile_names):
    x_min, y_min, x_max, y_max = bbox
    selected_tiles = []
    for tile_path in all_tile_names:
        x_off, y_off = tile_offset_from_name(tile_path)
        tile_bbox = (x_off, y_off, x_off + TILE_W, y_off + TILE_H)
        # check intersection
        if not (tile_bbox[2] < x_min or tile_bbox[0] > x_max or
                tile_bbox[3] < y_min or tile_bbox[1] > y_max):
            selected_tiles.append(tile_path)
    return selected_tiles 

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


def locate_drone_position(drone_features: torch.Tensor, sat_patches, sat_features, kernel_size, fine_kernel_size, device='cuda'):
    # Normalize drone features
    drone_features = F.normalize(drone_features.to(device), dim=-1)

    num_tiles_rows = len(sat_features)
    num_tiles_cols = len(sat_features[0])
    tile_height = sat_patches.shape[0] // num_tiles_rows
    tile_width  = sat_patches.shape[1] // num_tiles_cols
    print(f"Tile size (pixels): {tile_width}x{tile_height}")
    print(f"Tile grid: {num_tiles_rows} rows x {num_tiles_cols} cols")

    # Compute valid kernel positions for coarse search
    valid_rows = num_tiles_rows - kernel_size + 1
    valid_cols = num_tiles_cols - kernel_size + 1
    heatmap = np.full((valid_rows, valid_cols), -np.inf, dtype=float)
    best_score = -float('inf')
    best_pos = None

    # Coarse search over feature_grid
    for row in range(valid_rows):
        for col in range(valid_cols):
            kernel_tiles = []
            for r in range(row, row + kernel_size):
                for c in range(col, col + kernel_size):
                    tile_feat = torch.cat([b.to(device) for b in sat_features[r][c]], dim=0)
                    kernel_tiles.append(tile_feat)
            kernel_features = torch.cat(kernel_tiles, dim=0)
            kernel_features = F.normalize(kernel_features, dim=-1)

            similarity = torch.matmul(drone_features, kernel_features.T)
            top_k = max(1, kernel_features.shape[0] // 1000) # top 0.1% matches
            mean_topk = similarity.flatten().topk(top_k)[0].mean().item()
            heatmap[row, col] = mean_topk

            if mean_topk > best_score:
                best_score = mean_topk
                best_pos = (row, col)

            del kernel_features, similarity
            torch.cuda.empty_cache()

    # Map best coarse kernel to patch coordinates
    row, col = best_pos
    crop_left   = col * tile_width
    crop_right  = (col + kernel_size) * tile_width
    crop_top    = (num_tiles_rows - (row + kernel_size)) * tile_height
    crop_bottom = crop_top + kernel_size * tile_height

    # NOTE: winner_x / winner_y should be passed in from metadata if available
    winner_y = ys[row]
    winner_x = xs[col]

    patch = Image.fromarray(sat_patches[crop_top:crop_bottom, crop_left:crop_right, :])

    # -----------------------------
    # Fine-grained processing
    # -----------------------------
    # Reconstruct true coarse patch grid from tiles
    tile_feature_blocks = []
    for r in range(row, row + kernel_size):
        row_blocks = []
        for c in range(col, col + kernel_size):
            tile_feat_list = sat_features[r][c]
            # Concatenate all feature blocks
            tile_feat = torch.cat([b.to(device) for b in tile_feat_list], dim=0)
            h_patch = int(np.sqrt(tile_feat.shape[0]))
            w_patch = tile_feat.shape[0] // h_patch
            D = tile_feat.shape[1]
            tile_grid = tile_feat[:h_patch*w_patch].view(h_patch, w_patch, D)
            row_blocks.append(tile_grid)
        # Concatenate tiles horizontally
        row_grid = torch.cat(row_blocks, dim=1)
        tile_feature_blocks.append(row_grid)
    # Concatenate rows vertically
    coarse_patch_grid = torch.cat(tile_feature_blocks, dim=0)  # shape: (Hbig, Wbig, D)

    # Split coarse_patch_grid into fine tiles
    patch_width, patch_height = patch.size
    num_tiles_width  = 4 * 4
    num_tiles_height = 3 * 4
    Hbig, Wbig, D = coarse_patch_grid.shape
    h_step = Hbig // num_tiles_height
    w_step = Wbig // num_tiles_width

    patch_features_list = []
    for i in range(num_tiles_height):
        h_start = i * h_step
        h_end = (i+1)*h_step if i < num_tiles_height-1 else Hbig
        for j in range(num_tiles_width):
            w_start = j * w_step
            w_end = (j+1)*w_step if j < num_tiles_width-1 else Wbig
            tile_feat = coarse_patch_grid[h_start:h_end, w_start:w_end, :].reshape(-1, D)
            patch_features_list.append(tile_feat)

    # Fine-grained heatmap
    heatmap_fine = np.full((num_tiles_height - fine_kernel_size + 1,
                            num_tiles_width - fine_kernel_size + 1), -np.inf, dtype=float)
    best_score_fine = -float('inf')
    best_kernel_features_fine = None
    best_pos_fine = None

    for row in range(heatmap_fine.shape[0]):
        for col in range(heatmap_fine.shape[1]):
            kernel_tiles = []
            for r in range(row, row + fine_kernel_size):
                for c in range(col, col + fine_kernel_size):
                    kernel_tiles.append(patch_features_list[r * num_tiles_width + c])
            kernel_features = torch.cat(kernel_tiles, dim=0).to(device)
            kernel_features = F.normalize(kernel_features, dim=-1)

            similarity = torch.matmul(drone_features, kernel_features.T)
            top_k = max(1, kernel_features.shape[0] // 20) # top 5% matches
            mean_topk = similarity.flatten().topk(top_k)[0].mean().item()
            heatmap_fine[row, col] = mean_topk

            if mean_topk > best_score_fine:
                best_score_fine = mean_topk
                best_kernel_features_fine = kernel_features.clone()
                best_pos_fine = (row, col)

            del kernel_features, similarity
            torch.cuda.empty_cache()

    # Map fine kernel to pixel coordinates
    row, col = best_pos_fine
    fine_tile_h = patch_height // num_tiles_height
    fine_tile_w = patch_width  // num_tiles_width
    center_y = row * fine_tile_h + (fine_kernel_size * fine_tile_h) // 2
    center_x = col * fine_tile_w  + (fine_kernel_size * fine_tile_w)  // 2

    crop_top_fine    = max(0, center_y - (fine_kernel_size * fine_tile_h) // 2)
    crop_left_fine   = max(0, center_x - (fine_kernel_size * fine_tile_w)  // 2)
    crop_bottom_fine = min(patch.height, crop_top_fine + fine_kernel_size * fine_tile_h)
    crop_right_fine  = min(patch.width,  crop_left_fine + fine_kernel_size * fine_tile_w)

    best_patch = patch.crop((crop_left_fine, crop_top_fine, crop_right_fine, crop_bottom_fine))
    return best_patch, heatmap_fine, best_kernel_features_fine, crop_left_fine, crop_top_fine, winner_x, winner_y, [kernel_size, 4, 3, fine_kernel_size, num_tiles_height, num_tiles_width]


def drone_position_homography(drone_img: Image.Image, sat_patch: Image.Image, drone_heading: float, kernel_info: list):
    """ From the satellite patch and the drone image, find the drone position using feature matching and homography """
    # Resize satellite patch to roughly match drone image
    # We use the meters per pixel from both the satellite and drone images to compute the scaling factors
    # The drone was reported to have a resolution of 0.1-0.2m per pixel, 0.15m per pixel seems to provide the right scale
    scale_hor =  2.167/0.15 * (kernel_info[0] / kernel_info[1]) * (kernel_info[3] / kernel_info[4])
    scale_ver =  2.234/0.15 * (kernel_info[0] / kernel_info[2]) * (kernel_info[3] / kernel_info[5])
    print("Resizing satellite patch by factors: ", scale_hor, scale_ver)
    new_width = int(sat_patch.width * scale_hor)
    new_height = int(sat_patch.height * scale_ver)

    # Resize satellite patch without cropping:
    sat_resized = sat_patch.resize((new_width, new_height), Image.LANCZOS)
    print("Image has been resized")

    # Convert to grayscale
    sat_gray = cv2.cvtColor(np.array(sat_resized), cv2.COLOR_RGB2GRAY)
    drone_gray = cv2.cvtColor(np.array(drone_img), cv2.COLOR_RGB2GRAY)

    h, w = drone_gray.shape[:2]
    cx, cy = w // 2, h // 2

    # Rotate the drone image using drone_heading
    M = cv2.getRotationMatrix2D((cx, cy), -drone_heading, 1.0)  # Negative angle for clockwise rotation
    
    # Compute new bounding size - to avoid cropping after rotation
    cos = abs(M[0,0])
    sin = abs(M[0,1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # update matrix so image is centered
    M[0,2] += new_w/2 - cx
    M[1,2] += new_h/2 - cy

    drone_gray_rotated = cv2.warpAffine(drone_gray, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)

    sat_tensor = TF.to_tensor(sat_gray).unsqueeze(0).float().to(device)  # Shape [1, C, H, W]
    drone_tensor = TF.to_tensor(drone_gray_rotated).unsqueeze(0).float().to(device)

    # --- SuperPoint keypoints and descriptors ---
    superpoint = SuperPoint().eval().to(device)
    
    # Increase preprocessing resolution
    superpoint.preprocess_conf["resize"] = 2048  # feed higher-res images
    superpoint.default_conf["detection_threshold"] = 0.0003  # keep more weak keypoints

    with torch.no_grad():
        kp_sat_data = superpoint.extract(sat_tensor)
        kp_drone_data = superpoint.extract(drone_tensor)

        # optional: wrap descriptors for LightGlue if needed
        kp_sat_data_r = rbd(kp_sat_data)
        kp_drone_data_r = rbd(kp_drone_data)

    print("Superpoint has finished extracting keypoints.")

    # --- LightGlue matching ---
    lightglue = LightGlue("superpoint").eval().to(device)

    # Increase pruning thresholds to consider more keypoints for matching
    lightglue.pruning_keypoint_thresholds["cuda"] = 8192
    lightglue.pruning_keypoint_thresholds["flash"] = 16384

    inputs = {
        "image0": kp_drone_data,
        "image1": kp_sat_data
    }

    with torch.no_grad():
        matches_r = lightglue(inputs)
        matches = rbd(matches_r)

    print("LightGlue has finished matching keypoints.")

    matches = matches.get("matches", None)

    # Extract matched points
    pts_drone = kp_drone_data_r["keypoints"][matches[:, 0]].detach().cpu().numpy()
    pts_sat   = kp_sat_data_r["keypoints"][matches[:, 1]].detach().cpu().numpy()

    # Convert to OpenCV KeyPoint and DMatch objects for visualization (if needed)
    kp_drone_cv = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in kp_drone_data_r["keypoints"].detach().cpu().numpy()]
    kp_sat_cv   = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in kp_sat_data_r["keypoints"].detach().cpu().numpy()]
    dmatches = [cv2.DMatch(int(i), int(j), 0) for i, j in matches]

    scale = 0.1

    cv2.imshow("Matched Keypoints", cv2.resize(cv2.drawMatches(drone_gray_rotated, kp_drone_cv, sat_gray, kp_sat_cv, dmatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS), (0,0), fx=scale, fy=scale))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    H = None
    sat_corners = None
    drone_center = None
    heading_vec = None
    heading_deg = None

    if pts_drone.shape[0] >= 4:
        H, mask = cv2.findHomography(pts_drone, pts_sat, cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.999)

        if len(pts_drone) < 1000: # Typically we get 3000+ for good matches
            print("Untrustworthy reprojection confidence due to low number of matches.")
            H = None
            return H, sat_corners, drone_center, [1, 1], drone_heading

        # Remember to remove the scale factor from satellite patch coordinates
        S = np.array([[1/scale_hor, 0, 0],
                      [0, 1/scale_ver, 0],
                      [0, 0, 1]])
        H = S @ H  # Adjust homography to original satellite patch size

        # Drone corners in drone image
        drone_corners = np.array([
            [0, 0],
            [drone_img.width, 0],
            [drone_img.width, drone_img.height],
            [0, drone_img.height]
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Convert corners back to original image using warp matrix
        drone_corners_rot = cv2.transform(drone_corners, M)

        # Map to satellite using Homography
        sat_corners = cv2.perspectiveTransform(drone_corners_rot, H)
        sat_corners = sat_corners.reshape(-1, 2) # From (4,1,2) -> (4,2)

        # Check convexity of the satellite corners. (Camera image is square, so mapped corners should also be convex)
        P = sat_corners
        signs = []
        for i in range(4):
            p0 = P[i]
            p1 = P[(i+1)%4]
            p2 = P[(i+2)%4]

            ab = np.array([p1[0]-p0[0], p1[1]-p0[1], 0])  # Make 3D to avoid deprecation warning
            bc = np.array([p2[0]-p1[0], p2[1]-p1[1], 0])

            cross_z = np.cross(ab, bc)[2] # Take z-component
            signs.append(float(np.sign(cross_z)))

        signs = np.array(signs)  # <-- Convert to NumPy array for comparisons

        # Convex if all turns are same direction, so check if not all positive or all negative
        if not (np.all(signs > 0) or np.all(signs < 0)):
            print("Warning: Homography produced non-convex quadrilateral.")
            H = None
            return H, sat_corners, drone_center, [1, 1], drone_heading # Return placeholder vector and prior heading if homography fails

        # Compute center
        drone_center = np.mean(sat_corners, axis=0)
        print(f"Estimated drone position in satellite patch pixels: x={drone_center[0]}, y={drone_center[1]}")

        # Compute drone heading from satellite corners
        top_center = (sat_corners[0] + sat_corners[1]) / 2
        bottom_center = (sat_corners[3] + sat_corners[2]) / 2

        heading_vec = top_center - bottom_center  # points forward
        dx, dy = heading_vec[0], heading_vec[1]

        heading_rad = np.arctan2(dx, -dy)  # north = up
        heading_deg = np.degrees(heading_rad)
        heading_deg = (heading_deg + 360) % 360  # normalize to [0,360)
        print(f"Estimated drone heading from homography: {heading_deg:.2f} degrees")

    else:
        print("Not enough good matches found for homography.")

    return H, sat_corners, drone_center, heading_vec, heading_deg
    

if __name__ == "__main__":
    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"     # Use GPU if available (Code made on 4060 Laptop GPU)
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m"      # DINOv3 ViT-Large with 16x16 patches, pretrained on satellite imagery (0.3B parameters)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Itteration variables (For testing with dataset images)
    i = 747                # Image index to process (For full set testing, start at 1)
    used_dataset = 3       # This is the dataset we are using (01, 02 ..., -> 11)
    failed_detections = 1  # Count failed homography detections - starts at 1 to avoid multi by zero

    # Initilize directories and paths
    sat_feature_dir = "geolocalization_dinov3/tile_features"

    # If working with a dataset, then we need to load their CSV file:
    csv_file_path = f"geolocalization_dinov3/dataset_data/csv_files/{used_dataset:02d}.csv"
    df = pd.read_csv(csv_file_path)
    # Structed as: num, filename, date, lat, lon, height, Omega, Kappa, Phi1, Phi2 (Phi1 is drone heading)

    # Get drone heading from CSV on 'take-off' (first image) - Approximate from homography afterwards
    curr_heading = df.loc[df['num'] == i, ['Phi1']].values[0][0]  # Drone heading from CSV

    curr_position = (df.loc[df['num'] == i, ['lat', 'lon']].values[0][0],
                    df.loc[df['num'] == i, ['lat', 'lon']].values[0][1]) # Known for first itteration -- Drone take-off
    
    lat_long_1 = (32.355491, 119.805926) # North, East
    lat_long_2 = (32.290290, 119.900052) # North, East
    geo_center = ((lat_long_1[0] + lat_long_2[0]) / 2, (lat_long_1[1] + lat_long_2[1]) / 2) # Center point
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(np.radians(geo_center[0]))
    geo_span = (abs(lat_long_1[0] - lat_long_2[0]), abs(lat_long_1[1] - lat_long_2[1])) # (degrees in lat, degrees in lon) - absolute value
    geo_span_meters = (geo_span[0] * meters_per_degree_lat, geo_span[1] * meters_per_degree_lon) #  (meters in lat, meters in lon)
    meters_per_pixel_lat = geo_span_meters[0] / 24308 # Height of full satellite image in pixels
    meters_per_pixel_lon = geo_span_meters[1] / 35092 # Width of full satellite image in pixels

    # Covert to pixels for first itteration, afterwards each update to curr, will be in pixels
    actual_drone_y = (lat_long_1[0] - curr_position[0]) * meters_per_degree_lat / meters_per_pixel_lat
    actual_drone_x = (curr_position[1] - lat_long_1[1]) * meters_per_degree_lon / meters_per_pixel_lon
    curr_position = np.array([actual_drone_x, actual_drone_y])

    # Loop, continuously process images from drone camera feed
    while True: # <-- Replace with loop that runs for each image in dataset (or ROS2 image callback for full onboard drone processing)
        start_time = time.time()
        
        # Due to file formatting, we need to fill in leading zeros for dataset and image index (The first drone image is 0001)
        drone_image_path = f"geolocalization_dinov3/dataset_data/drone_images/{used_dataset:02d}/{used_dataset:02d}_{i:04d}.JPG" # NOTE: Once we move to ROS2, find image without hardcoding path
        full_sat_image = cv2.imread("geolocalization_dinov3/full_satellite_image_small.png")
        pil_image = Image.open(drone_image_path).convert("RGB")

        # Read all relevant data from CSV for error calculation
        actual_drone_position = (df.loc[df['num'] == i, ['lat', 'lon']].values[0][0],
                                     df.loc[df['num'] == i, ['lat', 'lon']].values[0][1])
        print(f"Actual drone GPS position:    lat={actual_drone_position[0]}, lon={actual_drone_position[1]}")
        # Compute the actual drone location in pixels for plotting
        actual_drone_y = (lat_long_1[0] - actual_drone_position[0]) * meters_per_degree_lat / meters_per_pixel_lat
        actual_drone_x = (actual_drone_position[1] - lat_long_1[1]) * meters_per_degree_lon / meters_per_pixel_lon
        actual_drone_curr_position = np.array([actual_drone_x, actual_drone_y])

        # Preprocess image - Resize and convert to tensor
        image_tensor, image_resized = preprocess_image(pil_image)

        # Run DINOv3 model
        features = process_image_with_dino(image_tensor, model, device)

        # Generate an ellipse box around the current drone position estimate (Done to ensure drone position does not jump too far)
        sigma = np.array([[15000.0 * failed_detections, 0.0], [0.0, 15000.0 * failed_detections]])  # Uncertainty covariance in pixels (tune as needed)
        k = 2.0  # Scaling factor for ellipse size (e.g., 2 standard deviations)
        ellipse_bbox_coords = ellipse_bbox(curr_position[:2], sigma, k=k, n=72)

        TILE_W = 3963; TILE_H = 3349 # Currently width x height are just hardcoded from satellite_image_processing.py
        
        # Extract the tile names for the tiles within the ellipsis
        all_tile_names = [p for p in Path("geolocalization_dinov3/tiles_png_1km").iterdir() if p.is_file() and p.suffix.lower() == ".png"]
        selected_tiles = tiles_in_bbox(ellipse_bbox_coords, TILE_W, TILE_H, all_tile_names)
        print(f"Selected tiles in EKF ellipse: {[t.name for t in selected_tiles]}")

        # Extract the features for the tiles within the ellipsis
        all_feature_names = [p for p in Path("geolocalization_dinov3/tile_features").iterdir() if p.is_file() and p.suffix.lower() == ".pt"]
        selected_features_pts = tiles_in_bbox(ellipse_bbox_coords, TILE_W, TILE_H, all_feature_names)
        print(f"Selected tiles in EKF ellipse: {[t.name for t in selected_features_pts]}")

        # Construct full satellite image from touched tiles and reconstruct features
        tile_paths = {}         # (x, y) -> image tile path
        feature_paths = {}      # (x, y) -> feature tile path
        xs, ys = set(), set()

        # ---- Satellite image tiles ----
        for p in selected_tiles:
            m = re.search(r"y(?P<y>\d+)_x(?P<x>\d+)", p.stem)
            if not m:
                raise ValueError(f"Cannot parse tile name: {p.name}")
            x = int(m.group("x"))
            y = int(m.group("y"))
            tile_paths[(x, y)] = p
            xs.add(x)
            ys.add(y)

        # ---- Feature .pt tiles ----
        for p in selected_features_pts:
            m = re.search(r"y(?P<y>\d+)_x(?P<x>\d+)", p.stem)
            if not m:
                raise ValueError(f"Cannot parse feature name: {p.name}")
            x = int(m.group("x"))
            y = int(m.group("y"))
            feature_paths[(x, y)] = p

        xs = sorted(xs)
        ys = sorted(ys)

        # Determine stitched image output shape
        H_total = len(ys) * TILE_H
        W_total = len(xs) * TILE_W
        full_img = np.zeros((H_total, W_total, 3), dtype=np.uint8)

        # Get one existing feature tile to infer shapes
        sample_fpath = next(iter(feature_paths.values()))
        sample_tile_features = torch.load(sample_fpath, weights_only=True)

        N_blocks = len(sample_tile_features)

        BLOCK_SHAPE = sample_tile_features[0].shape  # e.g. (num_patches, feat_dim)

        # Empty tile feature is a list of zero tensors of same shape
        EMPTY_TILE_FEATURE = [torch.zeros(BLOCK_SHAPE) for _ in range(N_blocks)]

        # Construct the stitched feature grid and stitched satellite image
        feature_grid = []

        for yi, y in enumerate(ys):

            row_features = []

            for xi, x in enumerate(xs):
                # Image tile stitching
                img_path = tile_paths.get((x, y))
                if img_path is None:
                    tile_img = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
                else:
                    tile_img = np.array(Image.open(img_path))

                full_img[
                    yi*TILE_H : (yi+1)*TILE_H,
                    xi*TILE_W : (xi+1)*TILE_W
                ] = tile_img

                # Feature tile stitching
                fpath = feature_paths.get((x, y))
                if fpath is None:
                    tile_feature_list = EMPTY_TILE_FEATURE
                else:
                    tile_feature_list = torch.load(fpath, weights_only=True)  # list of tensors

                    # Safety check: ensure consistency
                    if len(tile_feature_list) != N_blocks:
                        print(f"WARNING: Tile ({x},{y}) has inconsistent block count: "
                            f"{len(tile_feature_list)} vs expected {N_blocks}")

                row_features.append(tile_feature_list)

            feature_grid.append(row_features)

        # I have done sanity checks here, which showed the satellite image and feature stitching works correctly. (Zero tiles are black, features are correct)
        # Do note that, due to overlap, visualising the stichted image, will look weird, but don't worry, only 1 tile is used for localisation.

        # Locate drone tile position in satellite tiles
        patch, heatmap, patch_features, crop_left, crop_top, crop_cand_left, crop_cand_top, info = locate_drone_position(
        drone_features=features,            # Features from drone image
        sat_patches=full_img,               # Pass the satellite images that are within ellipse
        sat_features=feature_grid,          # Point to the list of features corresponds to satellite image patches
        kernel_size=1,                      # Expands kernel from top-left tile to N x N tiles
        fine_kernel_size=9,                 # For fine grained search within each candidate
        device=device                       # Use same device as model (likely 'cuda' - GPU)
        )

        # From the satellite patch and the drone image, find the drone position using feature matching and homography
        H, sat_corners, location_in_crop, heading_vec, heading_deg = drone_position_homography(
        drone_img=pil_image,               # Drone image
        sat_patch=patch,                   # Satellite candidate patch
        drone_heading=curr_heading,        # Drone heading from CSV
        kernel_info=info                   # Information about kernel and tile sizes for scaling satellite image
        )
        curr_heading = heading_deg  # Update current heading for next iteration

        if H is not None:
            drone_position_candidate = location_in_crop.astype(np.float64) + np.array([crop_left, crop_top])
            drone_position_world = drone_position_candidate + np.array([crop_cand_left, crop_cand_top])

            # Convert to lat-long using inverse of earlier conversion (From satellite_image_processing.py)
            drone_position_lat = lat_long_1[0] - (drone_position_world[1] * meters_per_pixel_lat) / meters_per_degree_lat
            drone_position_lon = lat_long_1[1] + (drone_position_world[0] * meters_per_pixel_lon) / meters_per_degree_lon
            print(f"Estimated drone GPS position: lat={drone_position_lat}, lon={drone_position_lon}")

            # Compute position error in meters
            error_lat = abs(actual_drone_position[0] - drone_position_lat) * meters_per_degree_lat
            error_lon = abs(actual_drone_position[1] - drone_position_lon) * meters_per_degree_lon
            total_error = np.sqrt(error_lat**2 + error_lon**2)
            print(f"Position error: {total_error:.2f} meters (Lat error: {error_lat:.2f} m, Lon error: {error_lon:.2f} m)")
            
            curr_position = drone_position_world  # Update current position estimate for next itteration    

            # Added time to end of loop - measures total time for geolocalization
            end_time = time.time()
            print(f"Total drone geolocalization time: {end_time - start_time:.2f} seconds") # Roughly 11.5 seconds on RTX 4060 Laptop GPU

            # NOTE: Move the visualisation below out of the code, once we have nice images in the report
            # Do the same for sat_corners if needed for visualization
            sat_corners_pixel = sat_corners + np.array([crop_left, crop_top])
            sat_corners_pixel = sat_corners_pixel + np.array([crop_cand_left, crop_cand_top])

            scale = 0.3  # same scale used when resizing full image

            # Scale all coordinates
            sat_corners_pixel_scaled = sat_corners_pixel * scale
            drone_position_scaled = drone_position_world * scale
            heading_vec_scaled = heading_vec * scale

            # Draw rectangle, center, and heading
            a, b, angle, mu = ellipse_from_cov(curr_position[:2], sigma, k=2.0)
            center = tuple((mu * scale).astype(int))       # scale if needed
            axes   = (int(a * scale), int(b * scale))     # scale if needed
            angle_deg = np.degrees(angle)                 # rotation in degrees

            cv2.polylines(full_sat_image, [np.int32(sat_corners_pixel_scaled)], isClosed=True, color=(0, 0, 255), thickness=3) # Drone rectangle
            cv2.circle(full_sat_image, tuple((actual_drone_curr_position * scale).astype(int)), 2, (0, 255, 0), -1)            # Actual position
            cv2.circle(full_sat_image, tuple(drone_position_scaled.astype(int)), 1, (0, 0, 255), -1)                           # Estimated position
            cv2.ellipse(full_sat_image, center, axes, angle_deg, 0, 360, (255, 0, 0), 2)                                       # Uncertainty ellipse
            cv2.arrowedLine( # Draw the measurement predicted heading
                full_sat_image,
                tuple(drone_position_scaled.astype(int)),
                tuple((drone_position_scaled + heading_vec_scaled * 0.5).astype(int)),
                color=(0, 0, 255),
                thickness=1,
                tipLength=0.2
            )

            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(full_sat_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Drone Rectangle & Center on Full Satellite Image (scaled)")
            plt.show()

            failed_detections = 1  # Reset failed detections counter
            i += 1  # Move to next image in dataset
        else:
            print("Homography could not be computed for this frame. - Skipping localisation to avoid noise")

            failed_detections += 1

            scale = 0.3  # same scale used when resizing full image

            # Scale all coordinates
            sat_corners_pixel_scaled = sat_corners_pixel * scale

            # Draw rectangle, center, and heading
            a, b, angle, mu = ellipse_from_cov(curr_position[:2], sigma, k=2.0)
            center = tuple((mu * scale).astype(int))       # scale if needed
            axes   = (int(a * scale), int(b * scale))     # scale if needed
            angle_deg = np.degrees(angle)        

            # Draw drone position and ellipse
            cv2.circle(full_sat_image, tuple((actual_drone_curr_position * scale).astype(int)), 2, (0, 255, 0), -1)   # Actual position
            cv2.ellipse(full_sat_image, center, axes, angle_deg, 0, 360, (255, 0, 0), 2)                              # Uncertainty ellipse
            
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(full_sat_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Drone Rectangle & Center on Full Satellite Image (scaled)")
            plt.show()

            i += 1