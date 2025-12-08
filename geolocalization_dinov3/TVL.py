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
import rasterio
import pymagsac 

# Math Stuff
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime  
import math
from scipy.stats import chi2                    # For confidence ellipse calculations

# Debugging & Information
import time
import pandas as pd
from pathlib import Path
import re

# ----------------------------
# Configuration
# ----------------------------
PATCH_SIZE = 16 # Keep constant for DINOv3 ViT models

# ----------------------------
# Helper function
# ----------------------------
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

def latlon_to_orig_xy(lat, lon, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META):
    """(lat, lon) -> ORIGINAL satellite pixel (u, v) as float64 (no downscale)."""
    with rasterio.open(SAT_DISPLAY_META) as src:
        sat_W = src.width
        sat_H = src.height
    with open(SAT_LONG_LAT_INFO_DIR, newline="") as f:
        for r in pd.read_csv(f).to_dict(orient="records"):
            if r["mapname"] == f"satellite{sat_number:02d}.tif":
                LT_lat = np.float64(r["LT_lat_map"])
                LT_lon = np.float64(r["LT_lon_map"])
                RB_lat = np.float64(r["RB_lat_map"])
                RB_lon = np.float64(r["RB_lon_map"])
                break
        else:
            raise FileNotFoundError(f"Bounds for satellite{sat_number}.tif not found")
    u = (np.float64(lon) - LT_lon) / (RB_lon - LT_lon) * sat_W
    v = (np.float64(lat) - LT_lat) / (RB_lat - LT_lat) * sat_H
    return u, v

def _wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]. Use anywhere you touch headings."""
    return (a + np.pi) % (2*np.pi) - np.pi

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
        x0, P0, Q0):
        """
        Initialize the filter.

        Args:
          x0: (5,) initial state [x, y, v, phi, bias_phi]
          P0: (5,5) initial covariance
          sigma_*: process noise hyperparameters (tunable)
        """
        self.x = np.array(x0, dtype=float).reshape(-1)
        self.x[3] = _wrap_pi(self.x[3])              # keep phi in [-pi, pi]
        self.P = np.array(P0, dtype=float).reshape(5, 5)

        # Process noise scales (hyperparameters you tune)
        self.sigma_pos_proc = Q0[0][0]
        self.sigma_speed    = Q0[1][1]
        self.sigma_phi      = Q0[2][2]
        self.sigma_b_phi    = Q0[3][3]

    # -------- PREDICT HELPERS USED BY predict() --------
    def _f(self, x, dt):
        """Nonlinear motion model f(x, dt): propagates the state mean."""
        X, Y, V, PHI, B = x
        c, s = np.cos(PHI), np.sin(PHI)
        Xn = X + V * c * dt
        Yn = Y + V * s * dt
        Vn = V                                   # random walk handled via Q
        PHIn = _wrap_pi(PHI)                     # random walk handled via Q
        Bn = B                                  # random walk handled via Q
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
        self.x[3] = _wrap_pi(self.x[3])

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

        z_pred = np.array([self.x[0], self.x[1], _wrap_pi(self.x[3] + self.x[4])], dtype=float)
        y = z - z_pred
        # wrap the heading innovation
        y[2] = _wrap_pi(y[2])

        S = H @ self.P @ H.T + R_xyphi
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[3] = _wrap_pi(self.x[3]) # obs in rad and in image beskrivelse

        I = np.eye(5)
        self.P = (I - K @ H) @ self.P
        return self.x.copy(), self.P.copy()

    @staticmethod
    def R_from_conf(pos_base_std, heading_base_std_rad, overall_conf,
                    pos_min_scale, pos_max_scale,
                    heading_min_scale, heading_max_scale ): 
        """
        Build measurement covariance from a confidence in [0,1].
        Higher confidence -> smaller variance (clamped by min/max scale).
        """
        # liniar mapping from confidence to scale between min and max scale
        scale_pose = np.interp(np.clip(overall_conf, 0.0, 1.0), [0.0, 1.0], [pos_max_scale, pos_min_scale]) 
        scale_heading = np.interp(np.clip(overall_conf, 0.0, 1.0), [0.0, 1.0], [heading_max_scale, heading_min_scale]) 
        Rx = (pos_base_std * scale_pose)**2
        Ry = (pos_base_std * scale_pose)**2
        Rphi = (heading_base_std_rad * scale_heading)**2
        return np.diag([Rx, Ry, Rphi]).astype(float)

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

def locate_drone_position(
    drone_features: torch.Tensor,
    sat_patches,       # stitched satellite image (H x W x 3)
    sat_features,      # feature_grid: 2D list of [row][col] tiles, each tile = list of feature blocks
    device='cuda'
):
    # Normalize drone features
    drone_features = F.normalize(drone_features.to(device), dim=-1)

    if sat_features is None or len(sat_features) == 0 or len(sat_features[0]) == 0:
        return None

    num_tiles_rows = len(sat_features)
    num_tiles_cols = len(sat_features[0])
    tile_height = sat_patches.shape[0] // num_tiles_rows
    tile_width  = sat_patches.shape[1] // num_tiles_cols

    # Prepare coordinate grid references
    coord_grid = [[(x, y) for x in xs] for y in ys]

    all_results = []  # ← store each result for ranking

    # Evaluate every tile
    for row in range(num_tiles_rows):
        for col in range(num_tiles_cols):

            tile_feat = torch.cat([b.to(device) for b in sat_features[row][col]], dim=0)
            kernel_features = F.normalize(tile_feat, dim=-1)

            similarity = torch.matmul(drone_features, kernel_features.T)

            # Weighting by softmax on a very small temperature (sharp attention)
            sim_weights = torch.softmax(similarity.flatten() / 1, dim=0)
            score = (similarity.flatten() * sim_weights).sum().item()

            # Extract image crop
            crop_left   = col * tile_width
            crop_right  = (col + 1) * tile_width
            crop_top    = row * tile_height
            crop_bottom = (row + 1) * tile_height
            patch_img   = Image.fromarray(sat_patches[crop_top:crop_bottom, crop_left:crop_right])

            winner_x, winner_y = coord_grid[row][col]
            normalized_score = (score + 1) / 2  # Map to [0,1]

            all_results.append({
                "patch": patch_img,
                "row": row,
                "col": col,
                "crop_left": crop_left,
                "crop_top": crop_top,
                "coord_x": winner_x,
                "coord_y": winner_y,
                "similarity": score,
                "cosine_confidence": normalized_score,
                "feature_file": Path("geolocalization_dinov3/tile_features_uniform") /
                                f"tile_y{winner_y}_x{winner_x}.pt"
            })

            del kernel_features, similarity
            torch.cuda.empty_cache()

    # Sort patches by similarity descending
    all_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)

    # Return is now full sorted list
    return all_results

def drone_position_homography(og_drone_img: Image.Image, drone_img_rotated: Image.Image, sat_patch: Image.Image, drone_heading: float, last_found_pos, last_found):
    """ From the satellite patch and the drone image, find the drone position using feature matching and homography """
    # Convert to grayscale
    sat_cv2 = cv2.cvtColor(np.array(sat_patch), cv2.COLOR_RGB2BGR)
    drone_cv2 = cv2.cvtColor(np.array(drone_img_rotated), cv2.COLOR_RGB2BGR)

    sat_tensor = TF.to_tensor(sat_cv2).unsqueeze(0).float().to(device)  # Shape [1, C, H, W]
    drone_tensor = TF.to_tensor(drone_cv2).unsqueeze(0).float().to(device)

    # --- SuperPoint keypoints and descriptors ---
    superpoint = SuperPoint().eval().to(device)

    with torch.no_grad():
        kp_sat_data = superpoint.extract(sat_tensor)
        kp_drone_data = superpoint.extract(drone_tensor)

        # optional: wrap descriptors for LightGlue if needed
        kp_sat_data_r = rbd(kp_sat_data)
        kp_drone_data_r = rbd(kp_drone_data)

    #print("Superpoint has finished extracting keypoints.")

    # --- LightGlue matching ---
    lightglue = LightGlue("superpoint").eval().to(device)

    inputs = {
        "image0": kp_drone_data,
        "image1": kp_sat_data
    }

    with torch.no_grad():
        matches_r = lightglue(inputs)
        matches = rbd(matches_r)

    matches_idx = matches.get("matches", None)
    
    # Extract matched points
    pts_drone = kp_drone_data_r["keypoints"][matches_idx[:, 0]].detach().cpu().numpy()
    pts_sat   = kp_sat_data_r["keypoints"][matches_idx[:, 1]].detach().cpu().numpy()

    # Update me:
    """""
    # Convert to OpenCV KeyPoint and DMatch objects for visualization (if needed)
    kp_drone_cv = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in kp_drone_data_r["keypoints"].detach().cpu().numpy()]
    kp_sat_cv   = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in kp_sat_data_r["keypoints"].detach().cpu().numpy()]
    dmatches = [cv2.DMatch(int(i), int(j), 0) for i, j in matches_idx]
    
    vis = cv2.drawMatches(
        drone_cv2, kp_drone_cv,
        sat_cv2, kp_sat_cv,
        dmatches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    scale = 0.7
    vis = cv2.resize(vis, (0,0), fx=scale, fy=scale)
    cv2.imshow("Feature Matches", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """""
    
    H = None
    sat_corners = None
    drone_center = None
    heading_vec = None
    heading_deg = None
    confidence = 0

    #print(f"Number of matches found: {len(pts_drone)}")
    if len(pts_drone) < 90:
        print(f"Warning: Only {len(pts_drone)} matches found, homography may be unreliable. Tossing to avoid noise")
        H = None
        return H, sat_corners, drone_center, heading_vec, heading_deg, confidence, pts_drone

    if pts_drone.shape[0] >= 4:
        H, mask = cv2.findHomography(pts_drone, pts_sat, cv2.USAC_MAGSAC, ransacReprojThreshold=5.0, confidence=0.9999)
        H, mask = cv2.findHomography(pts_drone[mask.ravel()==1], pts_sat[mask.ravel()==1], cv2.LMEDS) # Refine with LMedS on inliers only

        inlier_count = np.sum(mask)
        avg_inliers = 250  # Expected average inlier count for good homography
        inlier_confidence = min(1.0, inlier_count / avg_inliers) # Outputs between 0 and 1, if inlier_count >= avg_inliers then confidence = 1.0

        # Drone corners in drone image
        drone_corners = np.array([
            [0, 0],
            [og_drone_img.width, 0],
            [og_drone_img.width, og_drone_img.height],
            [0, og_drone_img.height]
        ], dtype=np.float32).reshape(-1, 1, 2)

        drone_corners_rot = cv2.transform(drone_corners.reshape(-1,1,2), M)
        drone_corners_rot = drone_corners_rot.reshape(-1,2)

        # Apply scaling
        S = np.array([[scale_hor, 0],
                      [0, scale_ver]], dtype=np.float32)

        drone_corners_scaled = (drone_corners_rot @ S.T).astype(np.float32)

        sat_corners = cv2.perspectiveTransform(drone_corners_scaled.reshape(-1,1,2), H)
        sat_corners = sat_corners.reshape(-1,2)

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
            return H, sat_corners, drone_center, [1, 1], drone_heading, confidence, pts_drone # Return placeholder vector and prior heading if homography fails

        # Check angles between edges to ensure they are roughly 90 degrees
        for i in range(4):
            v1 = P[(i+1)%4] - P[i]
            v2 = P[(i+2)%4] - P[(i+1)%4]
            angle = np.degrees(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
            if not 70 < angle < 110:
                print(f"Warning: Homography produced non-rectangular shape (angle {angle:.2f} degrees).")
                H = None
                return H, sat_corners, drone_center, [1, 1], drone_heading, confidence, pts_drone # Return placeholder vector and prior heading if homography fails

        # Compute center
        drone_center = np.mean(sat_corners, axis=0)

        if last_found_pos is not None:
            dist_moved = np.linalg.norm(drone_center - last_found_pos)
            if dist_moved > 1200 * last_found: # pixels Update Me
                print(f"Warning: Drone position jumped {dist_moved:.2f} pixels since last known position, allowed {1200 * last_found}. Tossing result.")
                H = None
                return H, sat_corners, drone_center, [1, 1], drone_heading, confidence, pts_drone # Return placeholder vector and prior heading if homography fails
        #print(f"Estimated drone position in satellite patch pixels: x={drone_center[0]}, y={drone_center[1]}")

        # Compute drone heading from satellite corners
        top_center = (sat_corners[0] + sat_corners[1]) / 2
        bottom_center = (sat_corners[3] + sat_corners[2]) / 2

        heading_vec = top_center - bottom_center  # points forward
        dx, dy = heading_vec[0], heading_vec[1]

        heading_rad = np.arctan2(dx, -dy)  # north = up
        heading_deg = np.degrees(heading_rad)
        heading_deg = (heading_deg + 360) % 360  # normalize to [0,360)
        #print(f"Estimated drone heading from homography: {heading_deg:.2f} degrees")

        # Compute confidence based on shape of resulting reshaped sat_corners
        # Remember that the drone image is square, so the mapped corners should also form a square (or close to it)
        side_lengths = np.array([])
        for i in range(4):
            p0 = sat_corners[i]
            p1 = sat_corners[(i+1)%4]
            length = np.linalg.norm(p1 - p0)
            side_lengths = np.append(side_lengths, length)

        # Compute Mu and STD for the side lengths  
        length_mean = np.mean(side_lengths)
        length_std = np.std(side_lengths)
        length_confidence = max(0.0, 1.0 - (length_std / length_mean))  # Higher stddev -> lower confidence

        confidence = (inlier_confidence + length_confidence) / 2.0
        #print(f"Homography confidence: {confidence:.3f} (Inlier: {inlier_confidence:.3f}, Side Lengths: {length_confidence:.3f})")

    else:
        print("Not enough good matches found for homography.")

    return H, sat_corners, drone_center, heading_vec, heading_deg, confidence, kp_drone_data_r["keypoints"].detach().cpu().numpy()
   

if __name__ == "__main__":
    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"     # Use GPU if available (Code made on 4060 Laptop GPU)
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m"      # DINOv3 ViT-Large with 16x16 patches, pretrained on satellite imagery (0.3B parameters)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Itteration variables (For testing with dataset images)  # NOTE: Update Me Testing
    i = 1                  # Image index to process (For full set testing, start at 1)
    used_dataset = 9       # This is the dataset we are using (01, 02 ..., -> 11)
    starting_drone_images = [f"{used_dataset:02d}_0001.JPG", f"{used_dataset:02d}_0129.JPG", f"{used_dataset:02d}_0256.JPG", 
                             f"{used_dataset:02d}_0384.JPG", f"{used_dataset:02d}_0512.JPG", f"{used_dataset:02d}_0640.JPG"] # the names of the drone images that starts a run

    # Initialize position and heading
    curr_position = None
    curr_heading = 0.0
    last_found_pos = None
    its_since_last_found = 1

    # Initialize EKF variables
    ekf = None
    t_last = None

    # Initilize directories and paths
    sat_feature_dir = "geolocalization_dinov3/tile_features_uniform"

    # If working with a dataset, then we need to load their CSV file:
    csv_file_path = f"geolocalization_dinov3/dataset_data/csv_files/{used_dataset:02d}.csv"
    df = pd.read_csv(csv_file_path)
    # Structed as: num, filename, date, lat, lon, height, Omega, Kappa, Phi1, Phi2 (Phi1 is drone heading)

    # Math for converting lat/long to pixels in satellite image    
    csv_lat_long_info_dir = "geolocalization_dinov3/dataset_data/csv_files/satellite_coordinates_range.csv"
    df_lat_long_info = pd.read_csv(csv_lat_long_info_dir)
    if df_lat_long_info.loc[df_lat_long_info['mapname'] == f"satellite{used_dataset:02d}.tif"].empty == False:
        lat_long_1 = (df_lat_long_info.loc[df_lat_long_info['mapname'] == f"satellite{used_dataset:02d}.tif", ['LT_lat_map', 'LT_lon_map']].values[0][0],
                      df_lat_long_info.loc[df_lat_long_info['mapname'] == f"satellite{used_dataset:02d}.tif", ['LT_lat_map', 'LT_lon_map']].values[0][1]) # North, East
        lat_long_2 = (df_lat_long_info.loc[df_lat_long_info['mapname'] == f"satellite{used_dataset:02d}.tif", ['RB_lat_map', 'RB_lon_map']].values[0][0],
                      df_lat_long_info.loc[df_lat_long_info['mapname'] == f"satellite{used_dataset:02d}.tif", ['RB_lat_map', 'RB_lon_map']].values[0][1]) # North, East
    geo_center = ((lat_long_1[0] + lat_long_2[0]) / 2, (lat_long_1[1] + lat_long_2[1]) / 2) # Center point
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(np.radians(geo_center[0]))
    geo_span = (abs(lat_long_1[0] - lat_long_2[0]), abs(lat_long_1[1] - lat_long_2[1])) # (degrees in lat, degrees in lon) - absolute value
    geo_span_meters = (geo_span[0] * meters_per_degree_lat, geo_span[1] * meters_per_degree_lon) #  (meters in lat, meters in lon)
    
    # NOTE: Update Me Testing
    meters_per_pixel_lat = geo_span_meters[0] / 33280 # Height of full satellite image in pixels
    meters_per_pixel_lon = geo_span_meters[1] / 44800 # Width of full satellite image in pixels

    # Loop, continuously process images from drone camera feed
    while i <= len(df): # <-- Replace with loop that runs for each image in dataset (or ROS2 image callback for full onboard drone processing)
        start_time = time.time()

        if df.loc[df['num'] == i, ['filename']].values[0][0] in starting_drone_images:
            # Reset position and heading at start of each dataset run
            ekf = None
            t_last = None
            last_found_pos = None
            its_since_last_found = 1
            curr_heading = df.loc[df['num'] == i, ['Phi1']].values[0][0]
            curr_position = (df.loc[df['num'] == i, ['lat', 'lon']].values[0][0],
                            df.loc[df['num'] == i, ['lat', 'lon']].values[0][1]) # Known for first itteration -- Drone take-off
            actual_drone_y = (lat_long_1[0] - curr_position[0]) * meters_per_degree_lat / meters_per_pixel_lat
            actual_drone_x = (curr_position[1] - lat_long_1[1]) * meters_per_degree_lon / meters_per_pixel_lon       
            curr_position = np.array([actual_drone_x, actual_drone_y])     
            print(f"\n--- Starting new dataset run at image {i}, resetting position and heading ---\n")

        # Due to file formatting, we need to fill in leading zeros for dataset and image index (The first drone image is 0001)
        drone_image_path = f"geolocalization_dinov3/dataset_data/drone_images/{used_dataset:02d}/{used_dataset:02d}_{i:04d}.JPG" # NOTE: Once we move to ROS2, find image without hardcoding path
        full_sat_image = cv2.imread("geolocalization_dinov3/full_satellite_image_small.png")
        pil_image = Image.open(drone_image_path).convert("RGB")

        # Read all relevant data from CSV for error calculation
        actual_drone_position = (df.loc[df['num'] == i, ['lat', 'lon']].values[0][0],
                                     df.loc[df['num'] == i, ['lat', 'lon']].values[0][1])
        #print(f"Actual drone GPS position:    lat={actual_drone_position[0]}, lon={actual_drone_position[1]}")
        # Compute the actual drone location in pixels for plotting
        actual_drone_y = (lat_long_1[0] - actual_drone_position[0]) * meters_per_degree_lat / meters_per_pixel_lat
        actual_drone_x = (actual_drone_position[1] - lat_long_1[1]) * meters_per_degree_lon / meters_per_pixel_lon
        actual_drone_curr_position = np.array([actual_drone_x, actual_drone_y])

        # Rotate drone image based on heading (Known for first itteration, estimated later)
        h, w = pil_image.size[1], pil_image.size[0]
        cx, cy = w // 2, h // 2

        # Rotate the drone image using drone_heading
        M = cv2.getRotationMatrix2D((cx, cy), -curr_heading, 1.0)  # Negative angle for clockwise rotation
        
        # Compute new bounding size - to avoid cropping after rotation
        cos = abs(M[0,0])
        sin = abs(M[0,1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Update matrix so image is centered
        M[0,2] += new_w/2 - cx
        M[1,2] += new_h/2 - cy

        # Perform the rotation
        drone_rotated = cv2.warpAffine(np.array(pil_image), M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)

        # Scale drone image to match satellite image scale (Based on pixels per meter) # NOTE: Update Me Testing
        scale_hor = 0.15/2.167 * 3
        scale_ver = 0.15/2.234 * 3
        #print("Resizing satellite patch by factors: ", scale_hor, scale_ver)
        new_width = int(drone_rotated.shape[1] * scale_hor)
        new_height = int(drone_rotated.shape[0] * scale_ver)

        # Resize satellite patch without cropping:
        rotated_drone_resized = Image.fromarray(drone_rotated).resize((new_width, new_height), Image.LANCZOS)
        # Currently drone is (329x306) while sub-tiles are (171x171) -- Consider getting these closer in size?

        # EKF initialisation (Only occours for first image)
        if ekf is None:
                rows = list(df.itertuples(index=False))
                for idx, row in enumerate(rows):
                    if row.filename == f"{used_dataset:02d}_{i:04d}.JPG":
                        starting_position_latlon = (np.float64(row.lat), np.float64(row.lon))
                        starting_position_xy = latlon_to_orig_xy(starting_position_latlon[0], starting_position_latlon[1], 
                                                                 f"geolocalization_dinov3/dataset_data/csv_files/satellite_coordinates_range.csv", used_dataset, 
                                                                 f"geolocalization_dinov3/dataset_data/satellite_images/satellite{used_dataset:02d}.tif")
                        t_current = datetime.fromisoformat(row.date)
                        t_last = t_current  # needed for next frame

                        # get phi
                        phi_deg0 = np.float64(row.Phi1)
                        phi0 = np.deg2rad(phi_deg0)
                        #print(f"[info] Initial heading from CSV: {phi_deg0:.2f} deg")
                        phi0_rad = np.deg2rad((((phi_deg0 - 90) + 180.0) % 360.0) - 180.0) # convert to image frame and rad
                        #print(f"[info] Initial heading in image frame: {np.rad2deg(phi0_rad):.2f} deg") # Seemingly just an offset of -90 degrees

                        # try reading the very next row in the file to get velocity estimate
                        next_row = rows[idx + 1]
                        lat1 = float(next_row.lat)
                        lon1 = float(next_row.lon)
                        k1_position_xy = latlon_to_orig_xy(lat1, lon1,
                                                           f"geolocalization_dinov3/dataset_data/csv_files/satellite_coordinates_range.csv", used_dataset, 
                                                           f"geolocalization_dinov3/dataset_data/satellite_images/satellite{used_dataset:02d}.tif")

                        # find dt between the two rows
                        t1 = datetime.fromisoformat(next_row.date)
                        dt = (t1 - t_current).total_seconds()
                        vel_x = (k1_position_xy[0] - starting_position_xy[0]) / dt
                        vel_y = (k1_position_xy[1] - starting_position_xy[1]) / dt
                        vel0 = np.sqrt(vel_x**2 + vel_y**2)
                        #print(f"[info] Initial velocity estimate from CSV: {vel0:.2f} px/s over dt={dt:.2f}s")
                        break

                # initial state: (x, y, v, phi, bias_phi) OBS: must be in image representation and phi:rad!!!
                x0 = np.array([starting_position_xy[0], starting_position_xy[1], vel0, phi0_rad, np.deg2rad(0.0)], dtype=np.float64)  # x,y in ORIGINAL sat pixels

                # P is the initial covariance for measurement uncertainty
                P0 = np.diag([(50.0)**2,            # σx = 50 px
                            (50.0)**2,              # σy = 50 px
                            (3.0)**2,               # σv = 3 px/s (since we have a rough estimate)
                            np.deg2rad(9.0)**2,     # σφ = 9° deg/s
                            np.deg2rad(9.0)**2      # at t0 we are unsure with around σbias_φ = 10.0 deg (This only affect us at start untill convergence)
                            ])  # this is something we only set for this first run it will be updated by EKF later. 
                
                # Process noise covariance Q (model uncertainty) Tune when ekf trust model to much or too little (high values=less trust in model)
                Q0 = np.diag([3.0,                  # px/√s : baseline diffusion on x,y
                            0.5,                  # px/√s : how much v can wander
                            np.deg2rad(0.5),      # rad/√s : how much phi can wander pr second. # we fly straight so expect small changes
                            np.deg2rad(0.0025)     # rad/√s : how much bias_phi can wander pr second
                            ])  # this is something we only set for this first run it will be updated by EKF later. 

                ekf = EKF_ConstantVelHeading(x0, P0, Q0) # initialize EKF
                
                i += 1  # Move to next image in dataset
                
                continue # EKF initialisation done, move to next image
        
        # Extract DINOv3 features from drone image (Using Native Drone Size!!!)
        features = process_image_with_dino(rotated_drone_resized, model, device)

        # Ensure that time difference is still computed, even if EKF is initialised
        for row in df.itertuples(index=False):
            if row.filename == f"{used_dataset:02d}_{i:04d}.JPG":
                t_current = datetime.fromisoformat(row.date)
                if t_last is not None:
                    dt = (t_current - t_last).total_seconds()
                    #print(f"dt between frames: {dt} seconds")
                t_last = t_current

        # EKF Prediction step - Done prior to measurement update
        x_pred, _ = ekf.predict(dt)
        x_pred[3] = np.deg2rad((((x_pred[3] + 90) + 180.0) % 360.0) - 180.0) # Shift frame by 90 degress and wrap.
        #print(f"[Predict-only] Predicted position: ({x_pred[0]:.2f}, {x_pred[1]:.2f}) px, heading={x_pred[3]:.2f}°, Bias_phi={np.rad2deg(x_pred[4]):.2f}°")

        P_pred = ekf.P
        sigma = P_pred[:2, :2]  # position covariance 2x2
        k = math.sqrt(chi2.ppf(0.85, df=2)) # TODO confidence scaling for 2D ellipse. so how conservative we are. the df is 2 since 2D.
        
        ellipse_bbox_coords = ellipse_bbox(x_pred[:2], sigma, k=k, n=72) # this uses SVD to determine viedest axes and orienation of ellipse to get bbx

        TILE_W = 1018; TILE_H = 1040 # Currently width x height are just hardcoded from satellite_image_processing.py # NOTE: Update Me Testing
        
        # Extract the tile names for the tiles within the ellipsis
        all_tile_names = [p for p in Path("geolocalization_dinov3/tiles_uniform").iterdir() if p.is_file() and p.suffix.lower() == ".png"]
        selected_tiles = tiles_in_bbox(ellipse_bbox_coords, TILE_W, TILE_H, all_tile_names)
        #print(f"Selected tiles in ellipse: {[t.name for t in selected_tiles]}")

        # Extract the features for the tiles within the ellipsis
        all_feature_names = [p for p in Path("geolocalization_dinov3/tile_features_uniform").iterdir() if p.is_file() and p.suffix.lower() == ".pt"]
        selected_features_pts = tiles_in_bbox(ellipse_bbox_coords, TILE_W, TILE_H, all_feature_names)
        #print(f"Selected tiles in ellipse: {[t.name for t in selected_features_pts]}")

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
        sample_tile_features = torch.load(sat_feature_dir + "/tile_y0_x0.pt", weights_only=True)

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

                row_features.append(tile_feature_list)

            feature_grid.append(row_features)

        # I have done sanity checks here, which showed the satellite image and feature stitching works correctly. (Zero tiles are black, features are correct)
        # Do note that, due to overlap, visualising the stichted image, will look weird, but don't worry, only 1 tile is used for localisation.

        # Locate drone tile position in satellite tiles
        all_results = locate_drone_position(
        drone_features=features,            # Features from drone image
        sat_patches=full_img,               # Pass the satellite images that are within ellipse
        sat_features=feature_grid,          # Point to the list of features corresponds to satellite image patches
        device=device                       # Use same device as model (likely 'cuda' - GPU)
        )

        if all_results is None:
            # Use predicted position as measurement
            drone_position_world = x_pred[:2]  # x_pred contains [x, y, v, heading]
            
            EKF_lat = lat_long_1[0] - (drone_position_world[1] * meters_per_pixel_lat) / meters_per_degree_lat
            EKF_lon = lat_long_1[1] + (drone_position_world[0] * meters_per_pixel_lon) / meters_per_degree_lon
            
            # Compute error from actual position using EKF updated position
            error_lat = abs(actual_drone_position[0] - EKF_lat) * meters_per_degree_lat
            error_lon = abs(actual_drone_position[1] - EKF_lon) * meters_per_degree_lon
            total_error = np.sqrt(error_lat**2 + error_lon**2)
            print(f"Position error: {total_error:.2f} meters (Lat error: {error_lat:.2f} m, Lon error: {error_lon:.2f} m) -- Using prediction")

            its_since_last_found += 1  # Increment counter for missed measurements

            end_time = time.time()

            # Generate CSV file:
            log_data = {
                'image_num': i,
                'estimated_lat': EKF_lat,
                'estimated_lon': EKF_lon,
                'actual_lat': actual_drone_position[0],
                'actual_lon': actual_drone_position[1],
                'error_meters': total_error,
                'processing_time_sec': end_time - start_time,
                'measurement_available': False,
                'feature_count': len(pts_drone)
            }
            log_df = pd.DataFrame([log_data])
            log_csv_path = f"geolocalization_dinov3/dataset_data/logs/geolocalization_log_dataset_TVL_EKF_{used_dataset:02d}.csv"

            # Write to CSV file for data logging:
            if i == 1:
                log_df.to_csv(log_csv_path, index=False, mode='w', header=True)
            else:
                log_df.to_csv(log_csv_path, index=False, mode='a', header=False)

            i += 1
            continue  # Move to next image

        highest_feature_count = 0
        for res in all_results:
            cosine_score = res['cosine_confidence']
            patch = res['patch']          # (H, W, 3) numpy array
            crop_cand_left = res['coord_x']
            crop_cand_top = res['coord_y']

            H, corners, center, heading_vec, heading_deg, confidence, pts_drone = drone_position_homography(
                    og_drone_img=pil_image,
                    drone_img_rotated=rotated_drone_resized,
                    sat_patch=patch,
                    drone_heading=curr_heading,
                    last_found_pos=last_found_pos,
                    last_found=its_since_last_found,
            )
            if len(pts_drone) > highest_feature_count:
                highest_feature_count = len(pts_drone)

            if H is not None:
                break  # Use the first successful homography

        if H is not None:
            curr_heading = heading_deg  # Update current heading for next iteration
            drone_position_world = center.astype(np.float64) + np.array([crop_cand_left, crop_cand_top])

            # Convert to lat-long using inverse of earlier conversion (From satellite_image_processing.py)
            drone_position_lat = lat_long_1[0] - (drone_position_world[1] * meters_per_pixel_lat) / meters_per_degree_lat
            drone_position_lon = lat_long_1[1] + (drone_position_world[0] * meters_per_pixel_lon) / meters_per_degree_lon
            #print(f"Estimated drone GPS position: lat={drone_position_lat}, lon={drone_position_lon}")

            # EKF Measurement update step
            print(f"Confidence measures - Feature: {cosine_score:.3f}, Reprojection: {confidence:.3f}")
            overall_confidence = 0.4 * cosine_score + 0.6 * confidence # Weights must sum to 1.0
            overall_confidence = np.clip(overall_confidence, 0.0, 1.0)
            #print(f"Overall measurement confidence for EKF: {overall_confidence:.3f}")
            
            R = ekf.R_from_conf(
            pos_base_std=50.0, # in px. we expect araound +-15 m ca= 50px error in position measurement
            heading_base_std_rad=np.deg2rad(15.0), # can jump due to bad matches
            overall_conf=overall_confidence,  # between 0 and 1
            pos_min_scale=0.3, # controls how much we trust low confidence measurements
            pos_max_scale=2.0,  # controls how much we trust high confidence measurements
            heading_min_scale=1, # same for heading
            heading_max_scale=4  # same for heading
            )

            # Perform EKF update with measured position and heading
            x_updated, P_estimated = ekf.update_pos_heading([drone_position_world[0], drone_position_world[1], np.deg2rad(curr_heading)], R) 

            x_updated[3] = ((np.rad2deg((x_updated[3] + 90.0)) + 180.0) % 360.0) - 180.0 # convert back to degrees for logging

            #print(f"[EKF] Updated state: x={x_updated[0]:.2f}, y={x_updated[1]:.2f}, v={x_updated[2]:.2f} px/s, phi={(x_updated[3]):.2f} deg")
            EKF_lat = lat_long_1[0] - (x_updated[1] * meters_per_pixel_lat) / meters_per_degree_lat
            EKF_lon = lat_long_1[1] + (x_updated[0] * meters_per_pixel_lon) / meters_per_degree_lon

            # Compute position error in meters
            error_lat = abs(actual_drone_position[0] - EKF_lat) * meters_per_degree_lat
            error_lon = abs(actual_drone_position[1] - EKF_lon) * meters_per_degree_lon
            total_error = np.sqrt(error_lat**2 + error_lon**2)
            print(f"Position error: {total_error:.2f} meters (Lat error: {error_lat:.2f} m, Lon error: {error_lon:.2f} m)")
            
            curr_position = drone_position_world  # Update current position estimate for next itteration   
            last_found_pos = center.astype(np.float64)  # Update last found position
            its_since_last_found = 1  # Reset counter

            end_time = time.time() 

            # NOTE: Move the visualisation below out of the code, once we have nice images in the report
            # Do the same for sat_corners if needed for visualization
            """""
            sat_corners_pixel = corners + np.array([patch[1], patch[2]])
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
            """""

            # Generate CSV file:
            log_data = {
                'image_num': i,
                'estimated_lat': EKF_lat,
                'estimated_lon': EKF_lon,
                'actual_lat': actual_drone_position[0],
                'actual_lon': actual_drone_position[1],
                'error_meters': total_error,
                'processing_time_sec': end_time - start_time,
                'measurement_available': True,
                'feature_count': highest_feature_count
            }
            log_df = pd.DataFrame([log_data])
            log_csv_path = f"geolocalization_dinov3/dataset_data/logs/geolocalization_log_dataset_TVL_EKF_{used_dataset:02d}.csv"

            # Write to CSV file for data logging:
            if i == 1:
                log_df.to_csv(log_csv_path, index=False, mode='w', header=True)
            else:
                log_df.to_csv(log_csv_path, index=False, mode='a', header=False)

            i += 1  # Move to next image in dataset

        else:
            #print("Homography could not be computed for this frame. - Skipping localisation to avoid noise")

            # Use predicted position as measurement
            drone_position_world = x_pred[:2]  # x_pred contains [x, y, v, heading]
            
            EKF_lat = lat_long_1[0] - (drone_position_world[1] * meters_per_pixel_lat) / meters_per_degree_lat
            EKF_lon = lat_long_1[1] + (drone_position_world[0] * meters_per_pixel_lon) / meters_per_degree_lon
            
            # Compute error from actual position using EKF updated position
            error_lat = abs(actual_drone_position[0] - EKF_lat) * meters_per_degree_lat
            error_lon = abs(actual_drone_position[1] - EKF_lon) * meters_per_degree_lon
            total_error = np.sqrt(error_lat**2 + error_lon**2)
            print(f"Position error: {total_error:.2f} meters (Lat error: {error_lat:.2f} m, Lon error: {error_lon:.2f} m) -- Using prediction")

            its_since_last_found += 1  # Increment counter for missed measurements

            end_time = time.time()

            """""
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
            """""

            # Generate CSV file:
            log_data = {
                'image_num': i,
                'estimated_lat': EKF_lat,
                'estimated_lon': EKF_lon,
                'actual_lat': actual_drone_position[0],
                'actual_lon': actual_drone_position[1],
                'error_meters': total_error,
                'processing_time_sec': end_time - start_time,
                'measurement_available': False,
                'feature_count': highest_feature_count
            }
            log_df = pd.DataFrame([log_data])
            log_csv_path = f"geolocalization_dinov3/dataset_data/logs/geolocalization_log_dataset_TVL_EKF_{used_dataset:02d}.csv"

            # Write to CSV file for data logging:
            if i == 1:
                log_df.to_csv(log_csv_path, index=False, mode='w', header=True)
            else:
                log_df.to_csv(log_csv_path, index=False, mode='a', header=False)

            i += 1