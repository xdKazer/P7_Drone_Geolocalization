# Machine Learning
import torch
from transformers import AutoModel, AutoImageProcessor, SuperPointForKeypointDetection
from sklearn.decomposition import PCA
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

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

# -------------- Kalman Filter Class --------------
class EKF_ConstantVelHeading:
    """
    Extended Kalman Filter with constant speed + heading motion in 2D.

    State x = [x, y, v, phi, b_phi]^T
      - x, y  : position in your chosen map units (we'll use ORIGINAL sat pixels)
      - v     : speed (units/second)
      - phi   : heading in radians (image coords: +x right, +y down)
        b_phi : bias in heading

    Motion model over dt:
        x_{k+1}   = x_k + v_k * cos(phi_k) * dt
        y_{k+1}   = y_k + v_k * sin(phi_k) * dt
        v_{k+1}   = v_k + w_v          (random-walk on speed)
        phi_{k+1} = phi_k + w_phi      (random-walk on heading, then wrapped)
        b_phi{k+1}= b_phi + w_b_phi    (random-walk on bias)
    """

    def __init__(
        self,
        x0, P0,
        sigma_pos_proc=0.75,             # px/√s : baseline diffusion on x,y
        sigma_speed=0.5,                 # px/√s : how much v can wander
        sigma_phi=np.deg2rad(3.0),       # rad/√s: how much heading can wander
        sigma_b_phi=np.deg2rad(0.10),     # rad/√s: how much bias can wander (should be very small!!!)
    ):
        """
        Initialize the filter.

        Args:
          x0: (4,) initial state [x, y, v, phi]
          P0: (4,4) initial covariance
          sigma_*: process noise hyperparameters (tunable)
        """
        self.x = np.array(x0, dtype=float).reshape(-1)
        self.x[3] = (self.x[3] + np.pi) % (2*np.pi) - np.pi # keep phi in [-pi, pi]
        self.P = np.array(P0, dtype=float).reshape(5, 5)

        # Process noise scales (hyperparameters you tune)
        self.sigma_pos_proc = float(sigma_pos_proc)
        self.sigma_speed    = float(sigma_speed)
        self.sigma_phi      = float(sigma_phi)
        self.sigma_b_phi    = float(sigma_b_phi)

    # -------- PREDICT HELPERS USED BY predict() --------
    def _f(self, x, dt):
        """Nonlinear motion model f(x, dt): propagates the state mean."""
        X, Y, V, PHI, B = x
        c, s = np.cos(PHI), np.sin(PHI)
        Xn = X + V * c * dt
        Yn = Y + V * s * dt
        Vn = V                                      # random walk handled via Q
        PHIn = (PHI + np.pi) % (2*np.pi) - np.pi    # random walk handled via Q
        Bn = B                                      # random walk handled via Q
        return np.array([Xn, Yn, Vn, PHIn, Bn], dtype=float)

    def _F_jac(self, x, dt):
        """Jacobian F = ∂f/∂x at current state; propagates covariance."""
        _, _, V, PHI, _ = x
        c, s = np.cos(PHI), np.sin(PHI)
        F = np.eye(5)
        F[0, 2] = c * dt           # dX/dV
        F[0, 3] = -V * s * dt      # dX/dPHI
        F[1, 2] = s * dt           # dY/dV
        F[1, 3] =  V * c * dt      # dY/dPHI
        # rows for V, PHI, B are identity (random-walks handled via Q)
        return F

    def _Q_proc(self, dt):
        """
        Process noise covariance (how uncertainty grows during predict).
        - x,y get a small baseline diffusion: (sigma_pos_proc * sqrt(dt))^2
        - v    random-walk per step:         (sigma_speed           )^2
        - phi  random-walk:                   (sigma_phi * sqrt(dt) )^2
        """
        qx = (self.sigma_pos_proc * np.sqrt(dt))**2
        qy = (self.sigma_pos_proc * np.sqrt(dt))**2
        qv = (self.sigma_speed)**2
        qf = (self.sigma_phi * np.sqrt(dt))**2
        qb = (self.sigma_b_phi * np.sqrt(dt))**2
        return np.diag([qx, qy, qv, qf, qb])

    # -------------------- PREDICT (called every frame) --------------------
    def predict(self, dt):
        """
        Time update (a.k.a. 'predict'):
          1) propagate the state mean with the motion model over dt
          2) propagate covariance with the Jacobian and process noise

        Call this once per frame BEFORE any measurement updates for that frame.
        """
        dt = float(max(1e-6, dt))                 # guard tiny/zero dt
        # propagate mean
        self.x = self._f(self.x, dt)
        self.x[3] = (self.x[3] + np.pi) % (2*np.pi) - np.pi

        # propagate covariance
        F = self._F_jac(self.x, dt)
        Q = self._Q_proc(dt)
        self.P = F @ self.P @ F.T + Q

        return self.x.copy(), self.P.copy()
    
    #----------------- UPDATE (called when measurement is available) --------------------
    def update_pos_heading(self, z_xyphi, R_xyphi):
        """
        OBS: PHI must be in pixel heading and radians (image coords: +x right, +y down).
        Measurement update with z = [x_meas, y_meas, phi_meas].
        R_xyphi is 3x3 diag([var_x, var_y, var_phi]).
        """
        z = np.asarray(z_xyphi, dtype=float).reshape(3)
        # H maps state -> measurement: [x, y, phi]
        H = np.array([[1,0,0,0, 0],
                    [0,1,0,0, 0],
                    [0,0,0,1, 1]], dtype=float) # we want to measure phi + b_phi

        z_pred = np.array([self.x[0], self.x[1], (self.x[3] + self.x[4] + np.pi) % (2*np.pi) - np.pi], dtype=float)
        y = z - z_pred
        # wrap the heading innovation
        y[2] = (y[2] + np.pi) % (2*np.pi) - np.pi

        S = H @ self.P @ H.T + R_xyphi
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[3] = (self.x[3] + np.pi) % (2*np.pi) - np.pi # obs in rad and in image beskrivelse

        I = np.eye(5)
        self.P = (I - K @ H) @ self.P
        return self.x.copy(), self.P.copy()

    @staticmethod
    def R_from_conf(pos_base_std, heading_base_std_rad, overall_conf,
                    min_scale=0.3, max_scale=2.0): 
        """
        Build measurement covariance from a confidence in [0,1].
        Higher confidence -> smaller variance (clamped by min/max scale).
        """
        # liniar mapping from confidence to scale between min and max scale
        scale = np.interp(np.clip(overall_conf, 0.0, 1.0), [0.0, 1.0], [max_scale, min_scale]) 
        Rx = (pos_base_std * scale)**2
        Ry = (pos_base_std * scale)**2
        Rphi = (heading_base_std_rad * scale)**2
        return np.diag([Rx, Ry, Rphi]).astype(float)

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
                          fine_kernel_size,
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
    print(f"Tile grid: {num_tiles_rows} rows x {num_tiles_cols} cols")

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
            top_k = max(1, kernel_features.shape[0] // 1000) # top 0.1%
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

    # Selecting top-1 from Heatmap (Highest mean cosine similarity positions)
    flat_idx = np.argpartition(-heatmap.flatten(), 1)[:1]
    scores = heatmap.flatten()[flat_idx]
    positions = [(idx // valid_cols, idx % valid_cols) for idx in flat_idx]
    heatmaps_sorted = sorted(zip(positions, scores), key=lambda x: -x[1])

    # Map top-left kernel positions to center and clamp coords
    row, col = heatmaps_sorted[0][0]
    print(f"Best candidate at tile position (row={row}, col={col}) with score {heatmaps_sorted[0][1]:.4f}")

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
    
    """""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(patch)
    plt.title(f"Satellite Patch Candidate")
    plt.axis('off')
    """""

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
    patch_features = kernel_features.to(device)

    # Fine grained search:
    patch_width, patch_height = patch.size

    # Number of fine tiles along each axis (5x4 grid scaled up)
    num_tiles_width = 5*4 # PATCH_SIZE
    num_tiles_height = 4*4 # Keep aspect ratio similar to original tile grid

    information_for_scale = [kernel_size, num_tiles_rows, num_tiles_cols, fine_kernel_size, num_tiles_height, num_tiles_width]

    # Compute tile size
    tile_width = max(patch_width // num_tiles_width, 1)
    tile_height = max(patch_height // num_tiles_height, 1)

    print(f"Adjusted tile size: {tile_width}x{tile_height}")
    print(f"Number of tiles: {num_tiles_width}x{num_tiles_height}")

    # Generate image patches
    patch_images = []
    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            left   = j * tile_width
            right  = min(left + tile_width, patch_width)
            top    = i * tile_height
            bottom = min(top + tile_height, patch_height)
            patch_images.append(patch.crop((left, top, right, bottom)))

    # -------------------------------
    # Robustly split features into fine num_tiles_width x num_tiles_height patches
    # -------------------------------
    # Total number of ViT features in the patch
    total_patches = patch_features.shape[0]

    # Preserve aspect ratio from the original coarse satellite tile grid
    aspect_ratio = num_tiles_rows / num_tiles_cols  # e.g., 5/4
    w_patch_patches = int(np.sqrt(total_patches / aspect_ratio))
    h_patch_patches = total_patches // w_patch_patches
    D = patch_features.shape[1]

    print(f"ViT patch grid for this patch: {h_patch_patches} x {w_patch_patches} patches (D={D})")

    # Reshape features into spatial grid
    patch_features_grid = patch_features[:h_patch_patches * w_patch_patches].reshape(h_patch_patches, w_patch_patches, D)

    # Compute fine tile size in ViT patch units
    fine_tile_h = h_patch_patches // num_tiles_height
    fine_tile_w = w_patch_patches // num_tiles_width

    patch_features_list = []
    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            h_start = i * fine_tile_h
            h_end   = (i + 1) * fine_tile_h if i < num_tiles_height - 1 else h_patch_patches
            w_start = j * fine_tile_w
            w_end   = (j + 1) * fine_tile_w if j < num_tiles_width - 1 else w_patch_patches
            
            tile_feat = patch_features_grid[h_start:h_end, w_start:w_end, :].reshape(-1, D)
            patch_features_list.append(tile_feat)

    # -------------------------------
    # Fine-grained heatmap
    # -------------------------------
    fine_kernel = fine_kernel_size # Get fine kernel size from function argument
    valid_rows = num_tiles_height - fine_kernel + 1
    valid_cols = num_tiles_width - fine_kernel + 1
    
    heatmap_fine = np.full((valid_rows, valid_cols), -np.inf, dtype=float)
    best_score = -float('inf')
    best_kernel_features = None
    best_pos = None

    for row in range(valid_rows):
        for col in range(valid_cols):
            kernel_tiles = []
            for r in range(row, row + fine_kernel):
                for c in range(col, col + fine_kernel):
                    idx = r * num_tiles_width + c
                    kernel_tiles.append(patch_features_list[idx])

            kernel_features = torch.cat(kernel_tiles, dim=0).to(device)
            kernel_features = F.normalize(kernel_features, dim=-1)
            
            similarity = torch.matmul(drone_features, kernel_features.T) # (N, 1024) x (1024, M) -> (N, M)
            top_k = max(1, kernel_features.shape[0] // 20) # top 5%
            topk_vals, _ = similarity.flatten().topk(top_k)
            mean_topk = topk_vals.mean().item()

            heatmap_fine[row, col] = mean_topk

            if mean_topk > best_score:
                best_score = mean_topk
                best_kernel_features = kernel_features.clone()
                best_pos = (row, col)

            del kernel_features, similarity
            torch.cuda.empty_cache()
    
    # Crop the patch corresponding to the top-1 similarity
    row, col = best_pos
    center_y = row * tile_height + (fine_kernel * tile_height) // 2
    center_x = col * tile_width  + (fine_kernel * tile_width)  // 2

    crop_top = max(0, center_y - (fine_kernel * tile_height) // 2)
    crop_left = max(0, center_x - (fine_kernel * tile_width) // 2)
    crop_bottom = min(patch.height, crop_top + fine_kernel * tile_height)
    crop_right = min(patch.width,  crop_left + fine_kernel * tile_width)

    best_patch = patch.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    best_score, best_patch, heatmap_fine, best_kernel_features, crop_left, crop_top
    print(f"Best candidate has score {best_score:.4f}")
    
    """""
    plt.subplot(1, 2, 2)
    plt.imshow(best_patch)
    plt.title(f"Satellite Fine-Patch Candidate")
    plt.axis('off')

    # Show the best heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap_fine, cmap='hot', interpolation='nearest')
    plt.title(f"Best Fine-grained Heatmap")
    plt.colorbar()
    plt.show()
    """""

    return best_patch, heatmap_fine, best_kernel_features, crop_left, crop_top, crop_left_clamped, crop_top_clamped, information_for_scale

# --- This is currently trash, as the satellite patch is too zoomed out to find good keypoints, do fine grained first ---
def drone_position_homography(drone_img: Image.Image, sat_patch: Image.Image, drone_heading: float, kernel_info: list):
    """ From the satellite patch and the drone image, find the drone position using feature matching and homography """
    # Resize satellite patch to roughly match drone image
    # We use the meters per pixel from both the satellite and drone images to compute the scaling factors
    # The drone was reported to have a resolution of 0.1-0.2m per pixel, 0.15m per pixel seems to provide the right scale
    scale_hor = 1.0833/0.15 * (kernel_info[0] / kernel_info[1]) * (kernel_info[3] / kernel_info[4])
    scale_ver = 1.1171/0.15 * (kernel_info[0] / kernel_info[2]) * (kernel_info[3] / kernel_info[5])
    print("Resizing satellite patch by factors: ", scale_hor, scale_ver)
    new_width = int(sat_patch.width * scale_hor)
    new_height = int(sat_patch.height * scale_ver)

    # Resize satellite patch without cropping:
    sat_resized = sat_patch.resize((new_width, new_height), Image.LANCZOS)

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

    matches = matches.get("matches", None)

    # Extract matched points
    pts_drone = kp_drone_data_r["keypoints"][matches[:, 0]].detach().cpu().numpy()
    pts_sat   = kp_sat_data_r["keypoints"][matches[:, 1]].detach().cpu().numpy()

    # Convert to OpenCV KeyPoint and DMatch objects for visualization (if needed)
    # kp_drone_cv = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in kp_drone_data_r["keypoints"].detach().cpu().numpy()]
    # kp_sat_cv   = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in kp_sat_data_r["keypoints"].detach().cpu().numpy()]
    # dmatches = [cv2.DMatch(int(i), int(j), 0) for i, j in matches]

    H = None
    sat_corners = None
    drone_center = None
    heading_vec = None
    heading_deg = None

    if pts_drone.shape[0] >= 4:
        H, mask = cv2.findHomography(pts_drone, pts_sat, cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.999)

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

    # If working with a dataset, then we need to load their CSV file:
    csv_file_path = f"geolocalization_dinov3/dataset_data/csv_files/{used_dataset:02d}.csv"
    df = pd.read_csv(csv_file_path)
    # Structed as: num, filename, date, lat, lon, height, Omega, Kappa, Phi1, Phi2 (Phi1 is drone heading)

    # Get drone heading from CSV on 'take-off' (first image) - Approximate from homography afterwards
    curr_heading = df.loc[df['num'] == i, ['Phi1']].values[0][0]  # Drone heading from CSV

    # Loop, continuously process images from drone camera feed
    while True: # <-- Replace with loop that runs for each image in dataset (or ROS2 image callback for full onboard drone processing)
        start_time = time.time()
        
        # Capture image from drone camera feed
        #image = capture_image_from_drone()

        # For testing, load image from file (remove once drone image capture works)
        # Due to file formatting, we need to fill in leading zeros for dataset and image index (The first drone image is 0001)
        drone_image_path = f"geolocalization_dinov3/dataset_data/drone_images/{used_dataset:02d}/{used_dataset:02d}_{i:04d}.JPG" # NOTE: Once we move to ROS2, find image without hardcoding path
        pil_image = Image.open(drone_image_path).convert("RGB")

        # Preprocess image - Resize and convert to tensor
        image_tensor, image_resized = preprocess_image(pil_image)

        # Run DINOv3 model
        features = process_image_with_dino(image_tensor, model, device)

        # Locate drone tile position in satellite tiles
        patch, heatmap, patch_features, crop_left, crop_top, crop_cand_left, crop_cand_top, info = locate_drone_position(
        drone_features=features,            # Features from drone image
        sat_features_dir="dino_features",   # Point to folder with satellite tile features - Generated from offline_geolocalization.py
        num_tiles_rows=5,                   # Number of rows in satellite feature grid - Read from satellite_image_processing.py
        num_tiles_cols=4,                   # Number of columns in satellite feature grid - Read from satellite_image_processing.py
        kernel_size=3,                      # Expands kernel from top-left tile to N x N tiles
        fine_kernel_size=8,                 # For fine grained search within each candidate
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
            drone_position_world_crop = drone_position_candidate + np.array([crop_cand_left, crop_cand_top])

            # Go from cropped satellite image pixels to full satellite image pixels (Needs a .csv file in future from satellite_image_processing.py)
            crop_world_left, crop_world_top = 19925, 17338 # Full satellite image size in pixels (from satellite_image_processing.py)

            drone_position_world = drone_position_world_crop + np.array([crop_world_left, crop_world_top])
            print(f"Clipped drone position in full satellite image (pixels): x={drone_position_world[0]}, y={drone_position_world[1]}")

            # Convert to lat-long using inverse of earlier conversion (From satellite_image_processing.py)
            lat_long_1 = (32.355491, 119.805926) # North, East
            lat_long_2 = (32.290290, 119.900052) # North, East
            geo_center = ((lat_long_1[0] + lat_long_2[0]) / 2, (lat_long_1[1] + lat_long_2[1]) / 2) # Center point
            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * np.cos(np.radians(geo_center[0]))
            geo_span = (abs(lat_long_1[0] - lat_long_2[0]), abs(lat_long_1[1] - lat_long_2[1])) # (degrees in lat, degrees in lon) - absolute value
            geo_span_meters = (geo_span[0] * meters_per_degree_lat, geo_span[1] * meters_per_degree_lon) #  (meters in lat, meters in lon)
            meters_per_pixel_lat = geo_span_meters[0] / 24308 # Height of full satellite image in pixels
            meters_per_pixel_lon = geo_span_meters[1] / 35092 # Width of full satellite image in pixels
            drone_position_lat = lat_long_1[0] - (drone_position_world[1] * meters_per_pixel_lat) / meters_per_degree_lat
            drone_position_lon = lat_long_1[1] + (drone_position_world[0] * meters_per_pixel_lon) / meters_per_degree_lon
            print(f"Estimated drone GPS position: lat={drone_position_lat}, lon={drone_position_lon}")

            # Compute error from actual position
            actual_drone_position = (df.loc[df['num'] == i, ['lat', 'lon']].values[0][0],
                                     df.loc[df['num'] == i, ['lat', 'lon']].values[0][1])
            print(f"Actual drone GPS position:    lat={actual_drone_position[0]}, lon={actual_drone_position[1]}")

            error_lat = abs(actual_drone_position[0] - drone_position_lat) * meters_per_degree_lat
            error_lon = abs(actual_drone_position[1] - drone_position_lon) * meters_per_degree_lon
            total_error = np.sqrt(error_lat**2 + error_lon**2)
            print(f"Position error: {total_error:.2f} meters (Lat error: {error_lat:.2f} m, Lon error: {error_lon:.2f} m)")

            # NOTE: Move the visualisation below out of the code, once we have nice images in the report
            # Do the same for sat_corners if needed for visualization
            sat_corners_pixel = sat_corners + np.array([crop_left, crop_top])
            sat_corners_pixel = sat_corners_pixel + np.array([crop_cand_left, crop_cand_top])

            # Draw rectangle, center and heading vector on full satellite image
            full_sat_image = cv2.imread("geolocalization_dinov3/reconstructed_full_image_small.png")
            cv2.polylines(full_sat_image, [np.int32(sat_corners_pixel)], isClosed=True, color=(0, 0, 255), thickness=3)
            cv2.circle(full_sat_image, tuple(drone_position_world_crop.astype(int)), 6, (0, 0, 255), -1)
            cv2.arrowedLine(full_sat_image, tuple(drone_position_world_crop.astype(int)), 
                            tuple((drone_position_world_crop + heading_vec * 0.5).astype(int)), 
                            color=(0, 0, 255), thickness=3, tipLength=0.2)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(full_sat_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title("Drone Rectangle & Center on Full Satellite Image")
            plt.show()

            # Added time to end of loop - measures total time for geolocalization
            end_time = time.time()
            print(f"Total drone geolocalization time: {end_time - start_time:.2f} seconds") # Roughly 11.5 seconds on RTX 4060 Laptop GPU
            # Fine grained search is likely the cause of most of the time, optimizations possible. (5.7 seconds for single candidate search)
            # Depending on DINO performance, maybe just assume candidate 1 is correct?

            i += 1  # Move to next image in dataset
        else:
            print("Homography could not be computed for this frame. - Skipping localisation to avoid noise")
            i += 1