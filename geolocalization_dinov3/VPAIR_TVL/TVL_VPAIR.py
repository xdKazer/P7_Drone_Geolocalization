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
import csv

# ----------------------------
# Configuration
# ----------------------------
PATCH_SIZE = 16 # Keep constant for DINOv3 ViT models

DATASET_DIR = Path("geolocalization_dinov3/VPAIR_TVL")
TILES_INFO_CSV = DATASET_DIR / "poses_tiles.csv"
TILE_CENTERS_CSV = DATASET_DIR / "tile_centers_in_sat.csv"
DRONE_INFO_CSV = DATASET_DIR / "poses_drone.csv"

DRONE_DIR = DATASET_DIR / "drone"
TILES_DIR = DATASET_DIR / "tiles"
SX, SY = 0.08667013347200554, 0.08667013347200554
pixel_offset = [2981.38, 1362.62]

tile_meta = {}
with open(TILE_CENTERS_CSV, "r") as f:
    r = csv.DictReader(f)
    for row in r:
        name = row["tile_name"]
        tile_meta[name] = {
            "center_x": float(row["center_x"]),
            "center_y": float(row["center_y"]),
            "m_per_px_x": float(row["m_per_px_x"]),
            "m_per_px_y": float(row["m_per_px_y"]),
        }

# ----------------------------
# Helper function
# ----------------------------
def wrap_deg(a): # this is used to get between -180 and 180 which is used in campass based heading
    """Wrap degrees to [-180, 180)."""
    return ((a + 180.0) % 360.0) - 180.0

def img_to_compass(phi_img_deg): # 
    """Convert image angle -> compass heading."""
    # Check: φ_img= -90 (North) -> ψ = 0 ; φ_img=0 (East) -> ψ=90
    return wrap_deg(phi_img_deg + 90.0) # to make sure its between -180 and 180

def compass_to_img(psi_deg):
    """Convert compass heading -> image angle."""
    # Check: ψ=0 (North) -> φ_img=-90 ; ψ=90 (East) -> φ_img=0
    return wrap_deg(psi_deg - 90.0) # to make sure its between -180 and 180

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
    with open(TILE_CENTERS_CSV, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            tile_name = row["tile_name"]
            centre_x = float(row["center_x"])
            centre_y = float(row["center_y"])
            tile_bbox = (centre_x - TILE_W/2, centre_y - TILE_H/2, centre_x + TILE_W/2, centre_y + TILE_H/2)
            # check intersection
            if not (tile_bbox[2] < x_min or tile_bbox[0] > x_max or
                    tile_bbox[3] < y_min or tile_bbox[1] > y_max):
                selected_tiles.append(tile_name)
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

def project_pts(H, pts_xy):
    xy_h = cv2.convertPointsToHomogeneous(pts_xy).reshape(-1,3).T  # 3xN
    P = (H @ xy_h).T
    return (P[:, :2] / P[:, 2:3]).astype(np.float32)

def get_visualisation_parameters(H_orig2tile, DRONE_ORIGINAL_W, DRONE_ORIGINAL_H):
    """
    Given homography from original drone image to UPSCALED tile pixels (H_orig2tile), tile scale, original drone image size, 
    and tile offsets (x_off, y_off) in ORIGINAL satellite pixels, compute:
    - center_global: (x,y) center of drone image in ORIGINAL satellite pixels
    - corners_global: (4,2) corners of drone image in ORIGINAL satellite pixels
    - heading_unitvec_meas: unit vector of drone heading in ORIGINAL satellite pixels
    """
    w0, h0 = float(DRONE_ORIGINAL_W), float(DRONE_ORIGINAL_H)

    # 1) Points in the drone image (original frame)
    center0  = np.array([[w0/2.0, h0/2.0]], dtype=np.float32)
    forward0 = np.array([[w0/2.0, h0/2.0 - max(20.0, 0.10*h0)]], dtype=np.float32)

    # 2) Project to TILE (UPSCALED TILE PIXELS!)
    center_tile_px  = project_pts(H_orig2tile, center0)[0].astype(np.float64)
    forward_tile_px = project_pts(H_orig2tile, forward0)[0].astype(np.float64)

    # 4) LOCAL → GLOBAL ORIGINAL SATELLITE using tile offset
    center_global  = center_tile_px
    forward_global = forward_tile_px

    # 5) Heading vector
    v_pred = forward_global - center_global
    norm   = np.hypot(v_pred[0], v_pred[1]) or 1.0
    heading_unitvec_meas = v_pred / norm

    # 6) Corners for visualization
    corners0 = np.array([[0,0],[w0,0],[w0,h0],[0,h0]], dtype=np.float32)

    corners_tile_px = project_pts(H_orig2tile, corners0)

    corners_global = corners_tile_px

    return center_tile_px, corners_tile_px, heading_unitvec_meas

def get_measurements(center_global, heading_unitvector_from_homography):
    """
    Drone image size: DRONE_INPUT_W, DRONE_INPUT_H (original drone frame size).
    Get measurements: (meas_phi_rad, meas_phi_deg, (meas_x, meas_y)) in ORIGINAL satellite pixel coords.

    """
    # ---  extract position & angle ---
    meas_x, meas_y = float(center_global[0]), float(center_global[1])
    meas_phi_rad = float(np.arctan2(heading_unitvector_from_homography[1], heading_unitvector_from_homography[0]))
    meas_phi_deg = (((float(np.degrees(meas_phi_rad)) - 90) + 180) % 360 - 180)

    return meas_phi_deg, (meas_x, meas_y)

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
    selected_tiles,      # list of image paths
    selected_features,   # list of feature .pt paths
    device='cuda'
):
    # Normalize drone features
    drone_features = F.normalize(drone_features.to(device), dim=-1)

    all_results = []

    for img_path, feat_path in zip(selected_tiles, selected_features):
        # Load satellite image
        sat_img = np.array(Image.open(f"geolocalization_dinov3/VPAIR_TVL/tiles/{img_path}"))
        tile_img = Image.fromarray(sat_img)

        # Load features
        tile_features = torch.load(f"geolocalization_dinov3/VPAIR_TVL/dinov3_features/{feat_path}", weights_only=True)  # list of tensors
        tile_feat = torch.cat([b.to(device) for b in tile_features], dim=0)
        kernel_features = F.normalize(tile_feat, dim=-1)

        # Compute similarity
        similarity = torch.matmul(drone_features, kernel_features.T)
        sim_weights = torch.softmax(similarity.flatten(), dim=0)
        score = (similarity.flatten() * sim_weights).sum().item()
        normalized_score = (score + 1) / 2  # Map to [0,1]

        all_results.append({
            "patch": tile_img,
            "patch_name": img_path,
            "similarity": score,
            "cosine_confidence": normalized_score,
            "feature_file": feat_path
        })

        del kernel_features, similarity
        torch.cuda.empty_cache()

    # Sort tiles by similarity descending
    all_results = sorted(all_results, key=lambda x: x["similarity"], reverse=True)

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
        #print(f"Warning: Only {len(pts_drone)} matches found, homography may be unreliable. Tossing to avoid noise")
        H = None
        return H, sat_corners, drone_center, heading_vec, heading_deg, confidence, pts_drone, None, None, None, None, None

    if pts_drone.shape[0] >= 4:
        H, mask = cv2.findHomography(pts_drone, pts_sat, cv2.USAC_MAGSAC, ransacReprojThreshold=5.0, confidence=0.9999)
        H, mask = cv2.findHomography(pts_drone[mask.ravel()==1], pts_sat[mask.ravel()==1], cv2.LMEDS) # Refine with LMedS on inliers only

        inlier_count = np.sum(mask)
        avg_inliers = 250  # Expected average inlier count for good homography
        inlier_confidence = min(1.0, inlier_count / avg_inliers) # Outputs between 0 and 1, if inlier_count >= avg_inliers then confidence = 1.0

        # Drone corners in drone image
        drone_corners = np.array([
            [0, 0],
            [og_drone_img.shape[1], 0],
            [og_drone_img.shape[1], og_drone_img.shape[0]],
            [0, og_drone_img.shape[0]]
        ], dtype=np.float32).reshape(-1, 1, 2)

        drone_corners_rot = cv2.transform(drone_corners.reshape(-1,1,2), M)
        drone_corners_rot = drone_corners_rot.reshape(-1,2)

        sat_corners = cv2.perspectiveTransform(drone_corners_rot.reshape(-1,1,2), H)
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
            #print("Warning: Homography produced non-convex quadrilateral.")
            H = None
            return H, sat_corners, drone_center, [1, 1], drone_heading, confidence, pts_drone, None, None, None, None, None # Return placeholder vector and prior heading if homography fails

        # Check angles between edges to ensure they are roughly 90 degrees
        for i in range(4):
            v1 = P[(i+1)%4] - P[i]
            v2 = P[(i+2)%4] - P[(i+1)%4]
            angle = np.degrees(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
            if not 70 < angle < 110:
                #print(f"Warning: Homography produced non-rectangular shape (angle {angle:.2f} degrees).")
                H = None
                return H, sat_corners, drone_center, [1, 1], drone_heading, confidence, pts_drone, None, None, None, None, None # Return placeholder vector and prior heading if homography fails

        # Compute center
        drone_center = np.mean(sat_corners, axis=0)

        if last_found_pos is not None:
            dist_moved = np.linalg.norm(drone_center - last_found_pos)
            #print(f"Drone moved {dist_moved:.2f} pixels since last known position.")
            if dist_moved > 400 * last_found: # pixels Update Me
                #print(f"Warning: Drone position jumped {dist_moved:.2f} pixels since last known position, allowed {400 * last_found}. Tossing result.")
                H = None
                return H, sat_corners, drone_center, [1, 1], drone_heading, confidence, pts_drone, None, None, None, None, None # Return placeholder vector and prior heading if homography fails
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

        # Visualise UAV image in satellite patch
        #plt.figure(figsize=(8,8))
        #plt.imshow(sat_cv2)
        #plt.scatter(sat_corners[:, 0], sat_corners[:, 1], c='r', marker='o')  # Mark corners
        #plt.scatter(drone_center[0], drone_center[1], c='b', marker='x')  # Mark center
        #plt.title(f"Drone Image in Satellite Patch\nConfidence: {confidence:.2f}")
        #plt.show()

    else:
        pass
        #print("Not enough good matches found for homography.")

    return H, sat_corners, drone_center, heading_vec, heading_deg, confidence, kp_drone_data_r["keypoints"].detach().cpu().numpy(), length_std, length_mean, inlier_count, length_confidence, inlier_confidence
   

if __name__ == "__main__":
    # Setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"     # Use GPU if available (Code made on 4060 Laptop GPU)
    model_name = "facebook/dinov3-vitl16-pretrain-sat493m"      # DINOv3 ViT-Large with 16x16 patches, pretrained on satellite imagery (0.3B parameters)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Initialize position and heading
    curr_position = None
    curr_heading = 0.0
    last_found_pos = None
    its_since_last_found = 1
    missed_measurements_in_a_row = 0

    # Initialize EKF variables
    ekf = None
    t_last = None


    # Loop, continuously process images from drone camera feed
    for i, img_path in enumerate(sorted(DRONE_DIR.iterdir())):
        drone_img = img_path.name
        meta = tile_meta[drone_img]
        GT_centre_x_px_tile = meta["center_x"]
        GT_centre_y_px_tile = meta["center_y"]

        img_for_keypoint_debug = cv2.imread(str(DRONE_DIR / str(drone_img)), cv2.IMREAD_COLOR)
        debug_drone_tensor = TF.to_tensor(img_for_keypoint_debug).unsqueeze(0).float().to(device)
        superpoint_debugger = SuperPoint().eval().to(device)
        with torch.no_grad():
            kp_drone_data_debug = superpoint_debugger.extract(debug_drone_tensor)
            kp_drone_data_debug_r = rbd(kp_drone_data_debug)
        keypoint_length = len(kp_drone_data_debug_r["keypoints"].detach().cpu().numpy())

        start_time = time.time()

        with open(DRONE_INFO_CSV, "r") as f:
            rows = list(csv.DictReader(f))  # Load full CSV into list so we can check next row
            for i, row in enumerate(rows):
                if row["filename"] == img_path.name:  # match found
                    curr_heading = float(row["yaw"])  
                    curr_heading = np.rad2deg(curr_heading)

        # Rotate drone image based on heading (Known for first itteration, estimated later)
        drone_img = cv2.imread(str(DRONE_DIR / str(drone_img)), cv2.IMREAD_COLOR)
        h, w = drone_img.shape[0], drone_img.shape[1]
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
        drone_rotated = cv2.warpAffine(np.array(drone_img), M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)

        R_orig2rot = np.eye(3, dtype=np.float64)
        R_orig2rot[:2, :3] = M 

        # Currently drone is (329x306) while sub-tiles are (171x171) -- Consider getting these closer in size?
        if ekf is None:
            # EKF initialisation (Only occours for first image)
            with open(DRONE_INFO_CSV, "r") as f:
                rows = list(csv.DictReader(f))  # Load full CSV into list so we can check next row
                for i, row in enumerate(rows):
                    if row["filename"] == img_path.name:  # match found
                        x0 = float(row["x"])
                        y0 = float(row["y"])
                        yaw0_rad = float(row["yaw"])  
                        yaw0_rad_img = np.deg2rad(compass_to_img(np.rad2deg(yaw0_rad)))

                        # ----- velocity estimate from next frame if available -----
                        if i < len(rows) - 1:   # next row exists
                            next_row = rows[i+1]
                            x1 = float(next_row["x"])
                            y1 = float(next_row["y"])
                            dt = 1
                            vel0 = math.hypot(x1 - x0, y1 - y0)  # distance per frame (Δt = 1)
                        break

                # initial state: (x, y, v, phi, bias_phi) OBS: must be in image representation and phi:rad!!!
                x0 = np.array([0,0 , vel0, yaw0_rad_img, np.deg2rad(0.0)], dtype=np.float64)  # x,y in ORIGINAL sat pixels

                # P is the initial covariance for uncertainty
                P0 = np.diag([(20.0)**2,            # σx = 50 px
                            (20.0)**2,              # σy = 50 px
                            (3.0)**2,               # σv = 3 px/s 
                            np.deg2rad(9.0)**2,     # σφ = 9° deg/s
                            np.deg2rad(0)**2      # at t0 we are unsure with around σbias_φ = 10.0 deg (This only affect us at start untill convergence)
                            ])  # this is something we only set for this first run it will be updated by EKF later. 
                
                # Process noise covariance Q (model uncertainty) Tune when ekf trust model to much or too little (high values=less trust in model)
                Q0 = np.diag([3.0,                  # px/√s : baseline diffusion on x,y
                            0.5,                  # px/√s : how much v can wander
                            np.deg2rad(8),      # rad/√s : how much phi can wander pr second. 
                            np.deg2rad(0.0025)     # rad/√s : how much bias_phi can wander pr second
                            ])  # this is something we only set for this first run it will be updated by EKF later. 

                ekf = EKF_ConstantVelHeading(x0, P0, Q0) # initialize EKF

                continue # next drone image
        
        # Extract DINOv3 features from drone image (Using Native Drone Size!!!)
        features = process_image_with_dino(drone_rotated, model, device)
        
        # EKF Prediction step - Done prior to measurement update
        x_pred, _ = ekf.predict(dt)
        x_pred[3] = img_to_compass(np.rad2deg(x_pred[3]))
        #print(f"[Predict-only] Predicted position: ({x_pred[0]:.2f}, {x_pred[1]:.2f}) px, heading={x_pred[3]:.2f}°, Bias_phi={np.rad2deg(x_pred[4]):.2f}°")

        k = math.sqrt(chi2.ppf(0.99, df=2)) # TODO confidence scaling for 2D ellipse. so how conservative we are. the df is 2 since 2D.

        P_pred = ekf.P
        sigma = P_pred[:2, :2]  # position covariance 2x2

        if missed_measurements_in_a_row != 0:
            if missed_measurements_in_a_row == 1: # we want to keep searching around last known good position
                last_known_pose = x_updated.copy()

            search_pose = last_known_pose[:2]
            L = (last_known_pose[2] * 1)* 1.2 * missed_measurements_in_a_row  # (vel * dt) * 10% * number of fails = radius
            sigma = np.diag([L**2, L**2])
            k_temp = 1 # we do not want to inflate it further
        else:
            search_pose = x_pred[:2]
            k_temp = k
        
        ellipse_bbox_coords = ellipse_bbox(search_pose, sigma, k=k_temp, n=72) # this uses SVD to determine viedest axes and orienation of ellipse to get bbx

        TILE_W = 800; TILE_H = 600 # Currently width x height are just hardcoded from satellite_image_processing.py
        
        # Extract the tile names for the tiles within the ellipsis
        all_tile_names = [p for p in TILES_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
        selected_tiles = tiles_in_bbox(ellipse_bbox_coords, TILE_W, TILE_H, all_tile_names)

        selected_features_pts = [p.replace(".png", ".pt") for p in selected_tiles]

        # Locate drone tile position in satellite tiles
        all_results = locate_drone_position(
            drone_features=features,
            selected_tiles=selected_tiles,
            selected_features=selected_features_pts,
            device=device
        )

        if all_results is None:
            # Use predicted position as measurement
            drone_position_world = x_pred[:2]  # x_pred contains [x, y, v, heading]
            
            diff_px = drone_position_world[0] - GT_centre_x_px_tile
            diff_py = drone_position_world[1] - GT_centre_y_px_tile

            error_ekf = abs(math.hypot(diff_px, diff_py))

            its_since_last_found += 1  # Increment counter for missed measurements
            missed_measurements_in_a_row += 1

            end_time = time.time()

            # Generate CSV file:
            log_data = {
                'image_num': img_path.name,
                'estimated_x': drone_position_world[0],
                'estimated_y': drone_position_world[1],
                'actual_x': GT_centre_x_px_tile,
                'actual_y': GT_centre_y_px_tile,
                'error_meters': error_ekf,
                'processing_time_sec': end_time - start_time,
                'measurement_available': False,
                'feature_count': keypoint_length,
                'measurement_error': None,
                'length_std': None,
                'length_mean': None,
                'inlier_count': None,
                'length_confidence': None,
                'inlier_confidence': None,
                'cosine_confidence': None,
                'picked_tile': None
            }
            log_df = pd.DataFrame([log_data])
            log_csv_path = f"geolocalization_dinov3/dataset_data/logs/VPAIR_results_KH.csv"

            # Write to CSV file for data logging:
            if i == 1:
                log_df.to_csv(log_csv_path, index=False, mode='w', header=True)
            else:
                log_df.to_csv(log_csv_path, index=False, mode='a', header=False)

            continue  # Move to next image

        highest_feature_count = 0
        best_patch_name = None
        for res in all_results:
            cosine_score = res['cosine_confidence']
            patch = res['patch']          # (H, W, 3) numpy array
            patch_name = res['patch_name']

            H, corners, center, heading_vec, heading_deg, confidence, pts_drone, length_std, length_mean, inlier_count, length_confidence, inlier_confidence = drone_position_homography(
                    og_drone_img=drone_img,
                    drone_img_rotated=drone_rotated,
                    sat_patch=patch,
                    drone_heading=curr_heading,
                    last_found_pos=last_found_pos,
                    last_found=its_since_last_found,
            )
            if len(pts_drone) > highest_feature_count:
                highest_feature_count = len(pts_drone)

            if H is not None:
                best_patch_name = patch_name
                break  # Use the first successful homography

        if H is not None:
            #curr_heading = heading_deg  # Update current heading for next iteration
            drone_position_world = center.astype(np.float64)

            H_orig, W_orig = drone_img.shape[:2]

            # Compute H from ORIGINAL drone frame to TILE frame
            H_rot2tile = H # the best match
            H_orig2tile = H_rot2tile @ R_orig2rot 

            center_meas_in_tile_px, corners_meas_in_tile_px, heading_unitvector_measurement = get_visualisation_parameters(H_orig2tile, W_orig, H_orig)

            # ---- Build measurement (x, y, phi) from homography in ORIGINAL pixels. compass heading ----
            meas_phi_deg, (meas_x_px_in_tile, meas_y_px_in_tile) = get_measurements(center_meas_in_tile_px, heading_unitvector_measurement)
            #print(f"Measurement: x={meas_x_px:.2f}, y={meas_y_px:.2f}, phi={meas_phi_deg:.2f} deg")

            # Look up tile center & GSD (meters per image pixel)
            meta = tile_meta[best_patch_name]   # tile_name is like "00001.png"
            cx_global = meta["center_x"]        # global px of tile center
            cy_global = meta["center_y"]
            m_per_px_x = meta["m_per_px_x"]     # meters per image px (x)
            m_per_px_y = meta["m_per_px_y"]     # meters per image px (y)

            # Offset from tile center in IMAGE PIXELS
            dx_img_px = meas_x_px_in_tile - TILE_W / 2.0
            dy_img_px = meas_y_px_in_tile - TILE_H / 2.0

            # Convert that offset to METERS using image GSD
            dx_m = dx_img_px * m_per_px_x
            dy_m = dy_img_px * m_per_px_y   # sign matches y-down global frame

            # Convert meters to GLOBAL PIXELS (EKF map units)
            dx_global_px = dx_m / 1
            dy_global_px = dy_m / 1

            # Final measurement in global pixels
            meas_x_px_global = cx_global + dx_global_px
            meas_y_px_global = cy_global + dy_global_px
            #print(f"Final measurement in GLOBAL pixels: x={meas_x_px_global:.2f}, y={meas_y_px_global:.2f}, phi={meas_phi_deg:.2f} deg")

            # EKF Measurement update step
            #print(f"Confidence measures - Feature: {cosine_score:.3f}, Reprojection: {confidence:.3f}")
            overall_confidence = 0.3 * cosine_score + 0.7 * confidence # Weights must sum to 1.0
            overall_confidence = np.clip(overall_confidence, 0.0, 1.0)
            #print(f"Overall measurement confidence for EKF: {overall_confidence:.3f}")
            
            R = ekf.R_from_conf(
            pos_base_std=20.0, # in px. measured the difference of GT and meas for first run and found mean around 55 px, with 95 percentile around 10 and 123 px
            heading_base_std_rad=np.deg2rad(3.0), # a normal good measurement are seen to be within +-3 degrees
            overall_conf=overall_confidence,  # between 0 and 1
            pos_min_scale=0.5, # controls how much we trust low confidence measurements sets the unceartanty determined using 95 percentile
            pos_max_scale=2.0,  # controls how much we trust high confidence measurements sets the certainty -||-
            heading_min_scale=0.5, # same for heading -||- found through testing
            heading_max_scale=2.75  # same for heading -||-
            )  # this gives us R matrix for EKF update. R tells us how certain we are about the measurements.

            # Perform EKF update with measured position and heading
            x_updated, P_estimated = ekf.update_pos_heading([meas_x_px_global, meas_y_px_global, np.deg2rad(compass_to_img(meas_phi_deg))], R)
            
            x_updated[3] = img_to_compass(np.rad2deg(x_updated[3]))

            diff_px = x_updated[0] - GT_centre_x_px_tile
            diff_py = x_updated[1] - GT_centre_y_px_tile

            error_ekf = abs(math.hypot(diff_px, diff_py))

            diff_px_meas = meas_x_px_global - GT_centre_x_px_tile
            diff_py_meas = meas_y_px_global - GT_centre_y_px_tile
            error_meas = abs(math.hypot(diff_px_meas, diff_py_meas))
            
            curr_position = x_updated  # Update current position estimate for next itteration   
            last_found_pos = center.astype(np.float64)  # Update last found position
            its_since_last_found = 1  # Reset counter
            missed_measurements_in_a_row = 0

            end_time = time.time() 

            # NOTE: Move the visualisation below out of the code, once we have nice images in the report
            # Do the same for sat_corners if needed for visualization
            """""
            sat_patch_vis = np.array(patch.copy())        # Selected satellite tile
            sat_h, sat_w = sat_patch_vis.shape[:2]

            sat_corners_px = np.array(corners_meas_in_tile_px, dtype=int)
            cv2.polylines(
                sat_patch_vis,
                [sat_corners_px.reshape(-1,1,2)],
                isClosed=True,
                color=(0,0,255),    # red rectangle = location of the drone view
                thickness=3
            )

            cv2.circle(
                sat_patch_vis,
                (int(meas_x_px_in_tile), int(meas_y_px_in_tile)),
                6,(0,255,0),-1      # green dot = estimated drone center
            )

            head_scale = 120
            hx = int(meas_x_px_in_tile + heading_unitvector_measurement[0]* head_scale)
            hy = int(meas_y_px_in_tile + heading_unitvector_measurement[1]* head_scale)

            cv2.arrowedLine(
                sat_patch_vis,
                (int(meas_x_px_in_tile), int(meas_y_px_in_tile)),
                (hx,hy),
                color=(255,0,0),
                thickness=3, tipLength=0.25
            )

            gt_x_px = GT_centre_x_px_tile #- meta["center_x"] + sat_w/2
            gt_y_px = GT_centre_y_px_tile #- meta["center_y"] + sat_h/2
            print(gt_x_px, gt_y_px)

            cv2.circle(sat_patch_vis,(int(gt_x_px),int(gt_y_px)),8,(0,255,255),-1) # yellow = GT

            eigvals, eigvecs = np.linalg.eig(sigma)
            order = np.argsort(eigvals)[::-1]
            a,b = np.sqrt(eigvals[order]) * k    # scale to 99% confidence

            angle = np.degrees(np.arctan2(eigvecs[1,order[0]],eigvecs[0,order[0]]))

            cv2.ellipse(
                sat_patch_vis,
                (int(meas_x_px_in_tile),int(meas_y_px_in_tile)),
                (int(a),int(b)),
                angle,
                0,360,(255,128,0),2
            )

            plt.figure(figsize=(9,9))
            plt.imshow(cv2.cvtColor(sat_patch_vis,cv2.COLOR_BGR2RGB))
            plt.title(f"Satellite Match  →  Error {error_ekf:.1f}m")
            plt.axis("off")
            plt.show()
            """""

            # Generate CSV file:
            log_data = {
                'image_num': img_path.name,
                'estimated_x': x_updated[0],
                'estimated_y': x_updated[1],
                'actual_x': GT_centre_x_px_tile,
                'actual_y': GT_centre_y_px_tile,
                'error_meters': error_ekf,
                'processing_time_sec': end_time - start_time,
                'measurement_available': True,
                'feature_count': keypoint_length,
                'measurement_error': error_meas,
                'length_std': length_std,
                'length_mean': length_mean,
                'inlier_count': inlier_count,
                'length_confidence': length_confidence,
                'inlier_confidence': inlier_confidence,
                'cosine_confidence': cosine_score,
                'picked_tile': best_patch_name
            }
            log_df = pd.DataFrame([log_data])
            log_csv_path = f"geolocalization_dinov3/dataset_data/logs/VPAIR_results_KH.csv"

            # Write to CSV file for data logging:
            if i == 1:
                log_df.to_csv(log_csv_path, index=False, mode='w', header=True)
            else:
                log_df.to_csv(log_csv_path, index=False, mode='a', header=False)

        else:
            #print("Homography could not be computed for this frame. - Skipping localisation to avoid noise")

            # Use predicted position as measurement
            drone_position_world = x_pred[:2]  # x_pred contains [x, y, v, heading]
            
            diff_px = drone_position_world[0] - GT_centre_x_px_tile
            diff_py = drone_position_world[1] - GT_centre_y_px_tile

            error_ekf = abs(math.hypot(diff_px, diff_py))

            its_since_last_found += 1  # Increment counter for missed measurements
            missed_measurements_in_a_row += 1

            end_time = time.time()

            # Generate CSV file:
            log_data = {
                'image_num': img_path.name,
                'estimated_x': drone_position_world[0],
                'estimated_y': drone_position_world[1],
                'actual_x': GT_centre_x_px_tile,
                'actual_y': GT_centre_y_px_tile,
                'error_meters': error_ekf,
                'processing_time_sec': end_time - start_time,
                'measurement_available': False,
                'feature_count': keypoint_length,
                'measurement_error': None,
                'length_std': None,
                'length_mean': None,
                'inlier_count': None,
                'length_confidence': None,
                'inlier_confidence': None,
                'cosine_confidence': None,
                'picked_tile': None
            }
            log_df = pd.DataFrame([log_data])
            log_csv_path = f"geolocalization_dinov3/dataset_data/logs/VPAIR_results_KH.csv"

            # Write to CSV file for data logging:
            if i == 1:
                log_df.to_csv(log_csv_path, index=False, mode='w', header=True)
            else:
                log_df.to_csv(log_csv_path, index=False, mode='a', header=False)

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