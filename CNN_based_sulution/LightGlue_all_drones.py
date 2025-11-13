# run_batch_match_top3_constmem.py  (PT-features version)
import csv, math, gc, random, cv2, torch
from scipy.stats import chi2
from pathlib import Path
import shutil # to delete folders
from typing import Optional
import json, re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd

from datetime import datetime # to determine dt from csv

# -------------------- Config --------------------
FEATURES = "superpoint"       # 'superpoint' | 'disk' | 'sift' | 'aliked'
DISPLAY_LONG_SIDE = 1200      # only for visualization
MAX_KPTS = 4048

sat_number = "03"
visualisations_enabled = True
# --- EKF globals (top of file, before the big for-loop) ---
ekf = None
t_last = None   # timestamp of previous processed frame
x_updated = None # needed for rotation of drone img
starting_drone_images = ["03_0001.JPG", "03_0097.JPG", "03_0193.JPG", "03_0289.JPG", "03_0385.JPG", "03_0481.JPG", "03_0577.JPG", "03_0673.JPG", ] # the names of the drone images that starts a run


BASE = Path(__file__).parent.resolve()
DATASET_DIR = BASE / "UAV_VisLoc_dataset"
SAT_LONG_LAT_INFO_DIR = DATASET_DIR / "satellite_coordinates_range.csv"
DRONE_INFO_DIR = DATASET_DIR / sat_number / f"{sat_number}.csv"
DRONE_IMG_CLEAN = DATASET_DIR / sat_number / "drone"  
SAT_DIR   = DATASET_DIR / sat_number / "sat_tiles_overlap" 
OUT_DIR_CLEAN   = BASE / "outputs" / sat_number 

# delete folder if exists
folder = OUT_DIR_CLEAN

if folder.exists() and folder.is_dir():
    shutil.rmtree(folder)
    print(f"Deleted folder: {folder}")
else:
    print(f"Folder does not exist: {folder}")

OUT_DIR_CLEAN.mkdir(parents=True, exist_ok=True)
CSV_FINAL_RESULT_PATH = BASE / "outputs" / sat_number / f"results_{sat_number}.csv"
TILE_PT_DIR = DATASET_DIR / sat_number / f"{FEATURES}_features" / sat_number
SAT_DISPLAY_IMG  = DATASET_DIR / sat_number / "satellite03_small.png"
SAT_DISPLAY_META = SAT_DISPLAY_IMG.with_suffix(SAT_DISPLAY_IMG.suffix + ".json")  # {"scale": s, "original_size_hw":[H,W],...}
TILE_WH_DIR = DATASET_DIR / sat_number / "sat_tiles_overlap" / "a_tile_size.txt"  # optional, not used here

# Tile geometry (ORIGINAL satellite pixel units)
with open(TILE_WH_DIR) as f:
    w_str, h_str = f.read().strip().split()
    TILE_W, TILE_H = int(w_str), int(h_str)
STRIDE_X = TILE_W // 2
STRIDE_Y = TILE_H // 2

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
        self.x[3] = _wrap_pi(self.x[3])              # keep phi in [-pi, pi]
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



# -------------------- Angle wrapping --------------------
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

def _wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]. Use anywhere you touch headings."""
    return (a + np.pi) % (2*np.pi) - np.pi
# -------------------- Helpers --------------------
def geometric_confidence(H, pts0, pts1, inlier_mask, sigma=10.0):
    """Compute a scalar geometric confidence in [0,1]."""
    if H is None or inlier_mask is None or not inlier_mask.any():
        return 0.0

    idx = np.where(inlier_mask)[0]
    p0 = pts0[idx].astype(np.float64)
    p1 = pts1[idx].astype(np.float64)
    p0h = np.c_[p0, np.ones(len(p0))]
    q1h = (H @ p0h.T).T
    q1 = q1h[:, :2] / q1h[:, 2:3]
    reproj_error = np.linalg.norm(q1 - p1, axis=1)

    inlier_ratio = len(idx) / len(pts0)
    median_err = np.median(reproj_error)
    geom_conf = inlier_ratio * math.exp(-median_err / sigma)
    return geom_conf, inlier_ratio, median_err

def get_location_in_sat_img(drone_img_centre, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META):
    meta = json.loads(SAT_DISPLAY_META.read_text())
    if "original_size_hw" in meta:
        sat_H, sat_W = map(np.float64, meta["original_size_hw"])

    with open(SAT_LONG_LAT_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["mapname"] == f"satellite{sat_number}.tif":
                LT_lat = np.float128(r["LT_lat_map"])
                LT_lon = np.float128(r["LT_lon_map"])
                RB_lat = np.float128(r["RB_lat_map"])
                RB_lon = np.float128(r["RB_lon_map"])
                break
        else:
            raise FileNotFoundError(f"Bounds for satellite{sat_number}.tif not found")

    u, v = np.float128(drone_img_centre)
    lon = LT_lon + (u / sat_W) * (RB_lon - LT_lon)
    lat = LT_lat + (v / sat_H) * (RB_lat - LT_lat)
    return (lat, lon)

def determine_pos_error(pose, pose_ekf, heading_deg, heading_ekf_deg, DRONE_INFO_DIR, drone_img):
    """ Compute position error in meters between estimated pose and GT from CSV.
    pose: (lat, lon) as float128
    pose_ekf: in pixels
    heading_deg: float (compass heading in degrees)
    heading_ekf_deg: float (compass heading in degrees)
    output: total_error_m, dx, dy, total_error_m_ekf, dx_ekf, dy_ekf, dphi ,dphi_ekf, gt_pose_px
    """
    pose = np.array(pose, dtype=np.float128)
    pose_ekf = np.array(pose_ekf, dtype=np.float128) # is in pixels
    pose_ekf = get_location_in_sat_img(pose_ekf, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META) # convert to lat/lon

    gt_lat = gt_lon = None
    with open(DRONE_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["filename"] == f"{drone_img}":
                gt_lat = np.float128(r["lat"])
                gt_lon = np.float128(r["lon"])
                gt_heading = np.float128(r["Phi1"]) 
                break
    if gt_lat is None or gt_lon is None:
        raise ValueError(f"GT lat/lon not found for {drone_img}")
    
    # for debugging GT values
    gt_pose_px = latlon_to_orig_xy(gt_lat, gt_lon, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)  
    print(f"GT values for {drone_img}: x={gt_pose_px[0]}, y={gt_pose_px[1]}, phi={get_phi_deg(DRONE_INFO_DIR, drone_img)}")

    difference = pose - np.array([gt_lat, gt_lon], dtype=np.float128)
    difference_ekf = pose_ekf - np.array([gt_lat, gt_lon], dtype=np.float128)
    mean_lat = np.radians((pose[0] + gt_lat) / np.float128(2.0))
    mean_lat_ekf = np.radians((pose_ekf[0] + gt_lat) / np.float128(2.0))

    meters_per_degree_lat = np.float128(111_320.0)
    meters_per_degree_lon = meters_per_degree_lat * np.cos(mean_lat)
    meters_per_degree_lon_ekf = meters_per_degree_lat * np.cos(mean_lat_ekf)

    dy = difference[0] * meters_per_degree_lat
    dx = difference[1] * meters_per_degree_lon

    dy_ekf = difference_ekf[0] * meters_per_degree_lat
    dx_ekf = difference_ekf[1] * meters_per_degree_lon_ekf

    total_error_m = np.sqrt(dx**2 + dy**2)
    total_error_m_ekf = np.sqrt(dx_ekf**2 + dy_ekf**2)

    dphi = wrap_deg(float(heading_deg - gt_heading))
    dphi_ekf = wrap_deg(float(heading_ekf_deg - gt_heading))
    return total_error_m, dx, dy, total_error_m_ekf, dx_ekf, dy_ekf, dphi ,dphi_ekf, gt_pose_px

def get_R_rotated_by_phi1(phi_rad: float, W_orig: int, H_orig: int) -> np.ndarray:
    c, s = np.cos(phi_rad), np.sin(phi_rad)
    cx, cy = (W_orig - 1) * 0.5, (H_orig - 1) * 0.5
    T_to   = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], dtype=np.float64)
    R      = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
    T_back = np.array([[1,0,cx],[0,1,cy],[0,0,1]], dtype=np.float64)
    return T_back @ R @ T_to

def get_phi_deg(DRONE_INFO_DIR, drone_img,):
    """ Read Φ heading (deg) for this frame. Returns float or raises."""
    with open(DRONE_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["filename"] == f"{drone_img}":
                for key in ("Phi1", "heading", "yaw_deg"):
                    if key in r and r[key] not in (None, "", "nan"):
                        return float(r[key])
                raise ValueError(f"No heading (Phi1/heading/yaw_deg) for {drone_img}")
    raise FileNotFoundError(f"{drone_img} not found in {DRONE_INFO_DIR}")

def latlon_to_orig_xy(lat, lon, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META):
    """(lat, lon) -> ORIGINAL satellite pixel (u, v) as float64 (no downscale)."""
    meta = json.loads(SAT_DISPLAY_META.read_text())
    if "original_size_hw" not in meta:
        raise KeyError("SAT_DISPLAY_META missing 'original_size_hw'.")
    sat_H, sat_W = map(np.float64, meta["original_size_hw"])
    with open(SAT_LONG_LAT_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["mapname"] == f"satellite{sat_number}.tif":
                LT_lat = np.float128(r["LT_lat_map"])
                LT_lon = np.float128(r["LT_lon_map"])
                RB_lat = np.float128(r["RB_lat_map"])
                RB_lon = np.float128(r["RB_lon_map"])
                break
        else:
            raise FileNotFoundError(f"Bounds for satellite{sat_number}.tif not found")
    u = (np.float128(lon) - LT_lon) / (RB_lon - LT_lon) * sat_W
    v = (np.float128(lat) - LT_lat) / (RB_lat - LT_lat) * sat_H
    return u, v

def get_visualisation_parameters(H_orig2tile, DRONE_ORIGINAL_W, DRONE_ORIGINAL_H, x_off, y_off):
    """
    H_orig2tile: homography from ORIGINAL drone frame to TILE coords (ORIGINAL sat pixels).
    Drone image size: DRONE_ORIGINAL_W, DRONE_ORIGINAL_H (original drone frame size).
    Get center and forward points in TILE coordinates (ORIGINAL sat pixels).
    Get heading unit vector from homography (in image orientation)
    output: center_global, corners_global, forward_global, heading_unitvector_measurement
    """
    w0, h0 = float(DRONE_ORIGINAL_W), float(DRONE_ORIGINAL_H) # drone image size
    
    # --- 1. reference points in drone image ---
    center0  = np.array([[w0/2.0, h0/2.0]], dtype=np.float32)
    forward0 = np.array([[w0/2.0, h0/2.0 - max(20.0, 0.10*h0)]], dtype=np.float32)

    # --- 2. project both points into TILE coords via homography ---
    center_tile  = project_pts(H_orig2tile, center0)[0].astype(np.float64)
    forward_tile = project_pts(H_orig2tile, forward0)[0].astype(np.float64)

    # --- 3. lift to ORIGINAL satellite pixel coords using tile offsets ---
    center_global = center_tile  + np.array([x_off, y_off], dtype=np.float64)
    forward_global = forward_tile + np.array([x_off, y_off], dtype=np.float64)

    # --- 4. compute heading vector in ORIGINAL sat coordinates ---
    v_pred = forward_global - center_global
    norm = float(np.hypot(v_pred[0], v_pred[1])) or 1.0
    heading_unitvector_measurement = (v_pred / norm).astype(np.float64)

    corners0 = np.array([[0,0],[w0,0],[w0,h0],[0,h0]], dtype=np.float32)
    center0  = np.array([[w0/2, h0/2]], dtype=np.float32)

    # project to TILE frame for visualization
    corners_tile = project_pts(H_orig2tile, corners0)
    center_tile  = project_pts(H_orig2tile, center0)[0]

    # get tile offsets from name to get GLOBAL sat coords
    corners_global = corners_tile + np.array([x_off, y_off], np.float32)
    center_global  = center_tile  + np.array([x_off, y_off], np.float32)
            

    return center_global, corners_global, forward_global, heading_unitvector_measurement


def get_measurements(DRONE_INPUT_W, DRONE_INPUT_H, forward_global, center_global, heading_unitvector_from_homography):
    """
    Drone image size: DRONE_INPUT_W, DRONE_INPUT_H (original drone frame size).
    Get measurements: (meas_phi_rad, meas_phi_deg, (meas_x, meas_y)) in ORIGINAL satellite pixel coords.

    """
    # ---  extract position & angle ---
    meas_x, meas_y = float(center_global[0]), float(center_global[1])
    meas_phi_rad = float(np.arctan2(heading_unitvector_from_homography[1], heading_unitvector_from_homography[0]))
    meas_phi_deg = float(np.degrees(meas_phi_rad))
    meas_phi_deg = img_to_compass(meas_phi_deg)  # convert to compass heading

    return meas_phi_deg, (meas_x, meas_y)

def draw_cropped_pred_vs_gt_on_tile(
    tile_path, Hmat, x_off, y_off, heading_unitvector_measurement_in_pixel_heading, lat_long_pose_ekf, heading_ekf_deg,
    DRONE_INPUT_W, DRONE_INPUT_H,
    DRONE_INFO_DIR, drone_img,
    SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META, error, error_ekf,
    crop_radius_px=450, out_path=None
):
    """
    Draw Pred vs GT centers + heading arrows directly on the TILE image,
    cropped around the midpoint. Returns (crop, heading_diff_angle_deg, heading_diff_dot_prod).

    IMPORTANT: Hmat must map FROM the frame that matches (DRONE_INPUT_W, DRONE_INPUT_H).
    We are now passing H_orig2tile with (W_orig,H_orig).
    """
    tile = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
    if tile is None:
        raise FileNotFoundError(f"Cannot read tile: {tile_path}")
    TH, TW = tile.shape[:2]

    ray_len = np.float64(220.0)

    center0  = np.array([[DRONE_INPUT_W/2, DRONE_INPUT_H/2]], dtype=np.float32)

    # project to TILE frame for visualization
    center_tile  = project_pts(Hmat, center0)[0].astype(np.float64)

    # Pred center on TILE + heading
    pred_tip = center_tile + ray_len * heading_unitvector_measurement_in_pixel_heading # for drawing

    # GT center on TILE + heading
    gt_lat = gt_lon = None
    gt_heading_deg = None
    with open(DRONE_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["filename"] == f"{drone_img}":
                gt_lat = np.float128(r["lat"]); gt_lon = np.float128(r["lon"])
                for key in ("Phi1", "heading", "yaw_deg"):
                    if key in r and r[key] not in (None, "", "nan"):
                        gt_heading_deg = float(r[key]); break  
                break
    if gt_lat is None or gt_lon is None:
        raise ValueError("GT lat/lon not found.")

    # for plotting GT
    gt_u, gt_v = latlon_to_orig_xy(gt_lat, gt_lon, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
    gt_tile = np.array([gt_u - x_off, gt_v - y_off], dtype=np.float64)
    gt_heading_deg = compass_to_img(gt_heading_deg) 
    th = np.deg2rad(np.float64(gt_heading_deg))
    v_gt_unit = np.array([np.cos(th), np.sin(th)], np.float64)  
    gt_tip = gt_tile + ray_len * v_gt_unit

    # for EKF
    ekf_u, ekf_v = latlon_to_orig_xy(lat_long_pose_ekf[0], lat_long_pose_ekf[1], SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
    ekf_tile = np.array([ekf_u - x_off, ekf_v - y_off], dtype=np.float64)
    heading_ekf_deg = compass_to_img(heading_ekf_deg) 
    th = np.deg2rad(np.float64(heading_ekf_deg))
    v_ekf_unit = np.array([np.cos(th), np.sin(th)], np.float64)  
    ekf_tip = ekf_tile + ray_len * v_ekf_unit

    # Metrics
    dotp = float(np.clip(np.dot(heading_unitvector_measurement_in_pixel_heading, v_gt_unit), -1.0, 1.0))
    dtheta = float(np.degrees(np.arccos(dotp)))
    dotp_ekf = float(np.clip(np.dot(v_ekf_unit, v_gt_unit), -1.0, 1.0))
    dtheta_ekf = float(np.degrees(np.arccos(dotp_ekf)))

    # Crop & draw
    cx = float((center_tile[0] + gt_tile[0]) / 2.0)
    cy = float((center_tile[1] + gt_tile[1]) / 2.0)
    r  = int(crop_radius_px)
    x0 = max(0, int(round(cx - r))); y0 = max(0, int(round(cy - r)))
    x1 = min(TW, int(round(cx + r))); y1 = min(TH, int(round(cy + r)))
    if x1 - x0 < 40 or y1 - y0 < 40:
        x0 = max(0, min(TW-1, int(round(cx)) - r)); y0 = max(0, min(TH-1, int(round(cy)) - r))
        x1 = min(TW, x0 + 2*r); y1 = min(TH, y0 + 2*r)
    crop = tile[y0:y1, x0:x1].copy()
    def S(p): return (int(round(p[0]-x0)), int(round(p[1]-y0)))

    # draw the centers and heading vectors
    cv2.circle(crop, S(center_tile), 10, (0,255,0), -1)
    cv2.arrowedLine(crop, S(center_tile), S(pred_tip), (0,255,0), 3, tipLength=0.2)
    cv2.circle(crop, S(gt_tile), 10, (255,0,255), -1)
    cv2.arrowedLine(crop, S(gt_tile), S(gt_tip), (255,0,255), 3, tipLength=0.2)
    #cv2.circle(crop, S(ekf_tile), 10, (255,255,0), -1)
    #cv2.arrowedLine(crop, S(ekf_tile), S(ekf_tip), (255,255,0), 3, tipLength=0.2)

    cv2.rectangle(crop, (10,10), (500,130), (255,255,255), -1)
    cv2.putText(crop, "Pred (tile H)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,180,0), 2)
    cv2.putText(crop, "GT (Phi1)", (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170,0,170), 2)
    cv2.putText(crop, "EKF", (20,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,0), 2)
    cv2.putText(crop, f"angle={dtheta:.1f}, dot={dotp:.3f}, error={error:.3f}m", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(crop, f"angle_ekf={dtheta_ekf:.1f}, dot_ekf={dotp_ekf:.3f}, error_ekf={error_ekf:.3f}m", (20,115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    if out_path is not None:
        cv2.imwrite(str(out_path), crop)
    return dtheta, dotp, dtheta_ekf, dotp_ekf

def load_sat_display_and_scale():
    img = cv2.imread(str(SAT_DISPLAY_IMG), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {SAT_DISPLAY_IMG}")
    meta = json.loads(SAT_DISPLAY_META.read_text())
    if "scale" in meta:
        sx = sy = float(meta["scale"])
    elif "scale_xy" in meta:
        sx, sy = map(float, meta["scale_xy"])
    else:
        raise KeyError(f"{SAT_DISPLAY_META} missing 'scale' or 'scale_xy'")
    return img, sx, sy

def tile_offset_from_name(tile_path: Path):
    name = tile_path.stem
    m = re.search(r"y(?P<y>\d+)_x(?P<x>\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse offsets from '{tile_path.name}'. Expected '...y<Y>_x<X>...'")
    y_off = int(m.group("y"))
    x_off = int(m.group("x"))
    return x_off, y_off

def project_pts(H, pts_xy):
    xy_h = cv2.convertPointsToHomogeneous(pts_xy).reshape(-1,3).T  # 3xN
    P = (H @ xy_h).T
    return (P[:, :2] / P[:, 2:3]).astype(np.float32)

def draw_polygon(img_bgr, poly_xy, color=(0,255,0), thickness=2):
    pts = poly_xy.reshape(-1,1,2).astype(int)
    cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=thickness)

def draw_point(img_bgr, pt_xy, color=(0,0,255), r=4):
    cv2.circle(img_bgr, (int(pt_xy[0]), int(pt_xy[1])), r, color, -1)

def label_point(
    img, pt, text, color,
    offset=(20, -10), font_scale=0.5, thickness=1
):
    """
    Draw text near a point with a line connecting them.

    Args:
        img: target image (BGR)
        pt: (x, y) coordinates of the point
        text: label string
        color: (B, G, R)
        offset: (dx, dy) offset of text from the point
        font_scale: text size
        thickness: text thickness
        line_thickness: thickness of the connector line
    """
    x, y = int(pt[0]), int(pt[1])
    ox, oy = offset
    tx, ty = x + ox, y + oy

    # draw the text label
    cv2.putText(
        img, text, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, color, thickness, lineType=cv2.LINE_AA
    )

def draw_ellipse(img_bgr, center_xy, cov2x2, k_sigma=2.0, color=(255,0,0), thickness=2):
    vals, vecs = np.linalg.eigh(cov2x2)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    a = k_sigma * np.sqrt(vals[0])
    b = k_sigma * np.sqrt(vals[1])
    cv2.ellipse(img_bgr,
                (int(round(center_xy[0])), int(round(center_xy[1]))),
                (int(round(a)), int(round(b))),
                angle,
                0, 360,
                color,
                thickness)

def to_numpy_image(t: torch.Tensor):
    return t.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()

def resize_for_display(img_np, long_side=DISPLAY_LONG_SIDE):
    h, w = img_np.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0: return img_np, 1.0
    new_w, new_h = int(round(w * s)), int(round(h * s))
    img_small = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_small, s

def make_segments(p0, p1, x_offset):
    segs = np.zeros((len(p0), 2, 2), dtype=np.float32)
    segs[:, 0, :] = p0
    segs[:, 1, 0] = p1[:, 0] + x_offset
    segs[:, 1, 1] = p1[:, 1]
    return segs

def visualize_inliers(drone_path: Path, tile_path: Path, pts0, pts1, inlier_mask, out_png):
    I0 = to_numpy_image(load_image(str(drone_path)))
    I1 = to_numpy_image(load_image(str(tile_path)))
    I0d, s0 = resize_for_display(I0)
    I1d, s1 = resize_for_display(I1)
    p0d = pts0 * s0
    p1d = pts1 * s1

    H0, W0 = I0d.shape[:2]
    H1, W1 = I1d.shape[:2]
    canvas = np.ones((max(H0, H1), W0 + W1, 3), dtype=I0d.dtype)
    canvas[:H0, :W0] = I0d
    canvas[:H1, W0:W0 + W1] = I1d

    if inlier_mask is None or not inlier_mask.any():
        plt.figure(figsize=(14, 7))
        plt.imshow(canvas); plt.axis("off")
        plt.title(f"{tile_path.name}: No inliers")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        return

    idx = np.where(inlier_mask)[0]
    segs = make_segments(p0d[idx], p1d[idx], x_offset=W0)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(canvas); ax.axis("off")
    lc = LineCollection(segs, linewidths=0.9, alpha=0.95)
    lc.set_colors(np.array([[0.0, 0.8, 0.0]] * len(segs)))
    ax.add_collection(lc)
    ax.scatter(p0d[idx, 0],         p0d[idx, 1],       s=2, c="yellow", alpha=0.7)
    ax.scatter(p1d[idx, 0] + W0,    p1d[idx, 1],       s=2, c="cyan",   alpha=0.7)
    ax.set_title(f"{tile_path.name} | inliers={len(idx)}")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def make_feature_pt_path_for(image_path: Path) -> Path:
    """
    returns path for feature file for image (.pt) files
    """
    folder = TILE_PT_DIR if TILE_PT_DIR is not None else image_path.parent
    return folder / (image_path.stem + ".pt")

def load_feats_pt_batched(pt_path: Path, device: str):
    d = torch.load(str(pt_path), map_location="cpu")
    for k in ("keypoints", "descriptors", "keypoint_scores", "image_size"):
        if k not in d:
            raise KeyError(f"{pt_path} missing key '{k}'")
    kpts = d["keypoints"].to(dtype=torch.float32)
    desc = d["descriptors"].to(dtype=torch.float32)
    scrs = d["keypoint_scores"].to(dtype=torch.float32)
    isize = d["image_size"].to(dtype=torch.int64)
    feats_b = {
        "keypoints":   kpts.unsqueeze(0).to(device),
        "descriptors": desc.unsqueeze(0).to(device),
        "keypoint_scores": scrs.unsqueeze(0).to(device),
        "image_size":  isize.unsqueeze(0).to(device),
    }
    feats_r = rbd(feats_b)
    return feats_b, feats_r

def absolute_confidence(num_inliers, avg_conf, median_err_px,
                        s_inl=30.0, s_err=5.0, w=(0.65, 0.10, 0.25)):  
    """
    Compute an absolute confidence score in [0,1] based on:
    - num_inliers: number of inlier matches
    - avg_conf: average matching confidence [0,1]
    - median_err_px: median reprojection error in pixels

    s_inl: scaling factor for inliers
    s_err: scaling factor for reproj error
    w: weights for (inliers, avg_conf, err_score)
    """
    if avg_conf <= 0 or num_inliers <= 0 or not np.isfinite(median_err_px):
        return 0.0
    inlier_score = 1.0 - np.exp(-num_inliers / s_inl)
    err_score    = np.exp(- (median_err_px / s_err)**2)
    w_inl, w_avg_c, w_err = w
    return float(np.clip(w_inl*inlier_score + w_avg_c*avg_conf + w_err*err_score, 0.0, 1.0))

# ---------------------- search region ----------------------
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

def tiles_in_bbox(bbox, TILE_W, TILE_H, STRIDE_X, STRIDE_Y, all_tile_names):
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
###################################################################################################

################################## EKF Tuning parameters ##################################
"""
P0: initial state covariance matrix before we start
    Set it to the initial uncertainty in (x, y, phi, bias)
    This one converges over time
    Big P0 = you are open to being corrected quickly by the first good measurements.
    Small P0 = you start “confident” and resist early corrections.

    P: P carries our total uncertainty of the different states over timesteps.

Q: process noise covariance matrix (model uncertainty) (this one is defined in the EKF initialization)
    how much we trust our motion model
    high Q = we expect the state to change a lot between steps (less trust in model)
    low Q  = we expect the state to change little between steps (more trust in model

EKF initialization:
    state vector: [x, y, v, phi, bias_phi]

    P0 = diag([sigma_x0^2, sigma_y0^2, sigma_v0^2, sigma_phi0^2, sigma_b_phi0^2])

    sigma_pos_proc (px/√s): baseline random diffusion in (x,y).
        Bigger → looser ellipse / faster expansion of search; more responsive to measurements; noisier track.

    sigma_speed (px/√s in your code it’s used as absolute per step, i.e., Q uses (sigma_speed)**2): how much speed can wander.
        Bigger → speed adapts quickly; smaller → speed is sticky (constant-velocity).

    sigma_phi (rad/√s): how much true heading can meander per second.
        Bigger → allows turns/heading jitter; smaller → straight-line assumption.

    sigma_b_phi (rad/√s): how much heading bias can drift per second.
        Smaller → bias is nearly constant (good if offset is truly fixed).
        Bigger → bias can change quickly (good if offset drifts with time/conditions).

R: measurement noise covariance matrix (sensor noise)
    how much we trust our measurements
    Higher R → trust measurements less (updates are small).
    Lower R → trust measurements more (updates are big).
    This is controlled by our measurement confidence!!! We set some sigma values for the mean expected uncertainty

R AND Q WILL DEFINE HOW MUCH WE TRUST MEASUREMENTS VS MODEL PREDICTIONS, SO TUNING IS IMPORTANT. IF ONE IS VERY HIGH THE OTHER IS TRUSTED MORE!!


################################### Kalman steps #############################################

When using the kalman filter, we need to initialize it. We need known starting position, delta time and velocity.
We need to track dt between frames for the process model.

This is done by giving the following to the ekf class:
    dt: time difference between frames in seconds
    vel0: initial speed estimate in px/s
    phi0: initial heading estimate in rad
    initial_state (x0): [x0, y0, v0, phi0, bias_phi]
    initial_P (P0): covariance matrix for initial state
    process_noise_cov (Q): covariance matrix for process noise

For each step we call ekf.predict(dt) followed by ekf.update(measurement, R). R has to be set each run as it uses measurement confidence. 


#################### Bias #####################
P0 bias just let it converge.
If after convergence you still see bias hopping. either sigma_b_phi is too large (Q keeps re-inflating the bias uncertainty each predict), or
R_phi is too small (you are over-trusting noisy heading measurements).
"""
###################################################################################################
# -------------------- Device & models --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] device: {device}. Extracting features using {FEATURES}.")
# -------------------- Feature extractor & matcher --------------------

feat = FEATURES.lower()
if feat not in ("superpoint", "disk", "sift", "aliked"):
    raise ValueError("FEATURES must be 'superpoint', 'disk', 'sift', or 'aliked'.")

# Extractor as fallback when tile .pt is missing
extractor = None
if feat == "superpoint":
    extractor = SuperPoint(max_num_keypoints=MAX_KPTS).eval().to(device)
elif feat == "disk":
    extractor = DISK(max_num_keypoints=MAX_KPTS).eval().to(device)
elif feat == "sift":
    extractor = SIFT(max_num_keypoints=MAX_KPTS).eval().to("cpu")
elif feat == "aliked":
    extractor = ALIKED(max_num_keypoints=MAX_KPTS).eval().to(device)

matcher = LightGlue(features=feat).eval().to(device)

with open(CSV_FINAL_RESULT_PATH, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow(["drone_image", "tile", "total_matches", "inliers", "avg_confidence", 
                            "median_reproj_error_px", "overall_confidence", 
                            "x_meas", "y_meas", "phi_meas_deg",
                            "x_ekf", "y_ekf", "phi_ekf_deg",
                            "dx", "dy", 
                            "dx_ekf", "dy_ekf",
                            "error", "ekf_error", 
                            "heading_diff", "ekf_heading_diff", 
                            "time_s", "ekf_time_s"])

# -------------------- Load drone features --------------------
for i, img_path in enumerate(sorted(DRONE_IMG_CLEAN.iterdir())):
    if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
        continue
    print("---")
    drone_img = img_path.name

    if drone_img in starting_drone_images: # this indicates a new sequence so ekf reset
        meas_phi_deg = None #for rotation consistency check
        ekf = None
        t_last = None
        print(f"[info] Resetting EKF for new sequence starting at {drone_img}")

    print(f"[info] Processing drone image: {drone_img}")
    DRONE_IMG = DRONE_IMG_CLEAN / str(drone_img)
    OUT_DIR = OUT_DIR_CLEAN / str(drone_img)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_RESULT_PATH  = OUT_DIR / "results.csv" 
    if not DRONE_IMG.exists():
        raise FileNotFoundError(f"Missing drone image: {DRONE_IMG}")
    
    # -------------------- EKF initialisation--------------------
    if ekf is None: # first frame is skipped for EKF initialization
        # -------------------- EKF initialization (global across frames) --------------------
        #read CSV to get starting lat lon and velocity estimate
        with open(DRONE_INFO_DIR, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if row["filename"] == str(drone_img):
                    starting_position_latlon = (np.float128(row["lat"]), np.float128(row["lon"]))
                    starting_position_xy = latlon_to_orig_xy(starting_position_latlon[0], starting_position_latlon[1], SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
                    t_current = datetime.fromisoformat(row["date"])
                    t_last = t_current  # needed for next frame

                    # get phi
                    phi_deg0 = np.float64(row["Phi1"])
                    phi0 = np.deg2rad(phi_deg0)
                    print(f"[info] Initial heading from CSV: {phi_deg0:.2f} deg")
                    phi0_rad = np.deg2rad(compass_to_img(phi_deg0)) # convert to image frame and rad

                    # try reading the very next row in the file to get velocity estimate
                    next_row = next(r)
                    lat1 = float(next_row["lat"])
                    lon1 = float(next_row["lon"])
                    k1_position_xy = latlon_to_orig_xy(lat1, lon1, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)

                    # find dt between the two rows
                    t1 = datetime.fromisoformat(next_row["date"])
                    dt = (t1 - t_current).total_seconds()
                    if dt > 0 and not None:
                        vel_x = (k1_position_xy[0] - starting_position_xy[0]) / dt
                        vel_y = (k1_position_xy[1] - starting_position_xy[1]) / dt
                        vel0 = np.sqrt(vel_x**2 + vel_y**2)
                        print(f"[info] Initial velocity estimate from CSV: {vel0:.2f} px/s over dt={dt:.2f}s")
                    else:
                        vel0 = 50.0
                        print(f"[warning] Non-positive dt={dt:.2f}s between first two rows in CSV. Using default initial vel={vel0:.2f} px/s")
                    break

        # initial state: (x, y, v, phi, bias_phi) OBS: must be in image representation and phi:rad!!!
        x0 = np.array([starting_position_xy[0], starting_position_xy[1], vel0, phi0_rad, np.deg2rad(0.0)], dtype=np.float64)  # x,y in ORIGINAL sat pixels

        # P is the initial covariance for measurement uncertainty
        P0 = np.diag([(36.0)**2,            # σx = 36 px
                    (36.0)**2,              # σy = 36 px
                    (3.0)**2,               # σv = 3 px/s (since we have a rough estimate)
                    np.deg2rad(1.0)**2,     # σφ = 1° deg/s
                    np.deg2rad(9.0)**2      # at t0 we are unsure with around σbias_φ = 10.0 deg (This only affect us at start untill convergence)
                    ])  # this is something we only set for this first run it will be updated by EKF later. 
        ekf = EKF_ConstantVelHeading(x0, P0, 
                                     # the following are process noise sigmas (Q):
                                    sigma_pos_proc=0.75,            # px/√s : baseline diffusion on x,y
                                    sigma_speed=0.5,                # px/√s : how much v can wander
                                    sigma_phi=np.deg2rad(3.0),      # rad/√s : how much phi can wander pr second. # we fly straight so expect small changes
                                    sigma_b_phi=np.deg2rad(0.005)    # rad/√s : how much bias_phi can wander pr second
                                    ) # the sigmas here are for model process noise Q. TODO tune these!
        continue # next drone image

    # ---------------------------- EKF prediction + ellipse for search area ---------------
    #determine dt for model prediction: 
    with open(DRONE_INFO_DIR, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["filename"] == str(drone_img):
                t_current = datetime.fromisoformat(row["date"])
                if t_last is not None:
                    dt = (t_current - t_last).total_seconds()
                    print(f"dt between frames: {dt} seconds")
                t_last = t_current

    # -------------------- EKF predict step --------------------
    x_pred, _ = ekf.predict(dt)
    x_pred[3] = img_to_compass(np.rad2deg(x_pred[3]))
    print(f"[Predict-only] Predicted position: ({x_pred[0]:.2f}, {x_pred[1]:.2f}) px, heading={x_pred[3]:.2f}°, Bias_phi={np.rad2deg(x_pred[4]):.2f}°")

    # -------------------- EKF search area (2,-sigma ellipse) --------------------
    # create search area using Predicted covariance P:
    P_pred = ekf.P
    sigma = P_pred[:2, :2]  # position covariance 2x2
    k = math.sqrt(chi2.ppf(0.85, df=2)) # TODO confidence scaling for 2D ellipse. so how conservative we are. the df is 2 since 2D.
    
    ellipse_bbox_coords = ellipse_bbox(x_pred[:2], sigma, k=k, n=72) # this uses SVD to determine viedest axes and orienation of ellipse to get bbx

    # -------------------- Determine tiles in search area --------------------
    all_tile_names = [p for p in SAT_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    selected_tiles = tiles_in_bbox(ellipse_bbox_coords, TILE_W, TILE_H, STRIDE_X, STRIDE_Y, all_tile_names)

    # -------------------- Rotate drone image & extract features --------------------
    if feat == "sift":
        R_orig2rot = np.eye(3, dtype=np.float64)
        with torch.inference_mode():
            img0_t = load_image(str(DRONE_IMG)).to("cpu")
            extractor = SIFT(max_num_keypoints=MAX_KPTS).eval().to("cpu")
            feats0_batched = extractor.extract(img0_t)
            feats0_r = rbd(feats0_batched)
        _bgr = cv2.imread(str(DRONE_IMG), cv2.IMREAD_COLOR)
        drone_rot_size = (_bgr.shape[1], _bgr.shape[0])  # (W,H)
        DRONE_IMG_FOR_VIZ = DRONE_IMG
    else:
        # -------------------- additionally rotate drone image by csv heading --------------------
        # SuperPoint / DISK: rotate according to heading from CSV
        if meas_phi_deg is None:
            phi_deg_flip = get_phi_deg(DRONE_INFO_DIR, drone_img)
        else:
            phi_deg_flip = meas_phi_deg

        # Arbitrary-angle rotation with padding; save the exact affine
        img = cv2.imread(str(DRONE_IMG), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        a = -phi_deg_flip
        r = math.radians(a)
        c, s = abs(math.cos(r)), abs(math.sin(r))
        newW = int(math.ceil(W*c + H*s))
        newH = int(math.ceil(W*s + H*c))

        M = cv2.getRotationMatrix2D((W/2.0, H/2.0), a, 1.0)
        # shift so the rotated content is centered in the padded canvas
        M[0, 2] += (newW - W) / 2.0
        M[1, 2] += (newH - H) / 2.0

        bgr_rot = cv2.warpAffine(img, M, (newW, newH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        R_orig2rot = np.eye(3, dtype=np.float64)
        R_orig2rot[:2, :3] = M               

        hR, wR = bgr_rot.shape[:2]
        drone_rot_size = (wR, hR)
        DRONE_IMG_ROT_PATH = OUT_DIR / f"drone_rot_{a}.png"
        DRONE_IMG_FOR_VIZ = DRONE_IMG_ROT_PATH
        cv2.imwrite(str(DRONE_IMG_ROT_PATH), bgr_rot)

        # -------------------- Extract features from rotated drone image --------------------

        if feat == "superpoint":
            extractor_local = SuperPoint(max_num_keypoints=MAX_KPTS).eval().to(device)
        elif feat == "disk":
            extractor_local = DISK(max_num_keypoints=MAX_KPTS).eval().to(device)
        elif feat == "aliked":
            extractor_local = ALIKED(max_num_keypoints=MAX_KPTS).eval().to(device)
        else:
            raise ValueError("Unsupported feature type in rotated-drone branch.")

        with torch.inference_mode():
            img_t = load_image(str(DRONE_IMG_FOR_VIZ)).to(device)
            feats0_batched = extractor_local.extract(img_t)
            feats0_r = rbd(feats0_batched)

        hR, wR = bgr_rot.shape[:2]
        drone_rot_size = (wR, hR)
        
    # -------------------- Pass 1: score all tiles (does not compute performance, just finds best match) --------------------   
    if not selected_tiles:
        raise FileNotFoundError(f"No PNG tiles in {SAT_DIR}")

    scores_small = []
    with torch.inference_mode():
        for i, p in enumerate(selected_tiles):
            num_inliers = 0
            avg_conf = float("nan")
            median_err = float("inf")   
            K = 0
            H = None
            inlier_mask = None
            print(f"Scoring tile {i+1}/{len(selected_tiles)}")
            tile_pt = make_feature_pt_path_for(p)
            if tile_pt.exists():
                feats1_b, feats1_r = load_feats_pt_batched(tile_pt, device if feat != "sift" else "cpu")
            else:
                if extractor is None:
                    raise FileNotFoundError(f"Missing {tile_pt} and no extractor available.")
                img1_t  = load_image(str(p)).to(device if feat != "sift" else "cpu")
                feats1_b = extractor.extract(img1_t)
                feats1_r = rbd(feats1_b)

            matches01 = matcher({"image0": feats0_batched, "image1": feats1_b})
            matches01_r = rbd(matches01)

            matches = matches01_r.get("matches", None)
            K = int(matches.shape[0]) if (matches is not None and matches.numel() > 0) else 0

            num_inliers = 0
            avg_conf    = float("nan")
            if K >= 4:
                pts0_np = feats0_r["keypoints"][matches[:, 0]].detach().cpu().numpy()
                pts1_np = feats1_r["keypoints"][matches[:, 1]].detach().cpu().numpy()

                H, mask = cv2.findHomography(
                    pts0_np, pts1_np, method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=3.0, confidence=0.999
                ) 
                if mask is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    num_inliers = int(inlier_mask.sum())

                    scores_t = matches01_r.get("scores", None)
                    if scores_t is not None and num_inliers > 0:
                        scores_np = scores_t.detach().cpu().numpy()
                        avg_conf = float(np.mean(scores_np[inlier_mask]))
                
                #using DLT on inliers for better accuracy
                if H is not None and num_inliers >= 4:
                    H, _ = cv2.findHomography(pts0_np[inlier_mask], pts1_np[inlier_mask], method=0)

                if H is not None:
                    _, _, median_err = geometric_confidence(H, pts0_np, pts1_np, inlier_mask)

            scores_small.append({
                "tile": p,
                "inliers": num_inliers,
                "total_matches": K,
                "avg_conf": avg_conf,
                "median_err": median_err,  # describes quality of inliers based on 
                "sort_key": (num_inliers, - median_err), # prioritize inliers, then lower error 
            })

            del feats1_b, feats1_r, matches01, matches01_r
            if 'img1_t' in locals():
                del img1_t
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # -------------------- Rank them and save in CSV. There is gonna be one CSV for each drone image --------------------
    scores_small.sort(key=lambda d: d["sort_key"], reverse=True)
    with open(CSV_RESULT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "total_matches", "inliers", "avg_confidence", "median_reproj_error"
                    ])
        for r in scores_small:
            w.writerow([r["tile"].name, 
                        r["total_matches"],
                        r["inliers"],
                        "" if math.isnan(r["avg_conf"]) else f"{r['avg_conf']:.4f}", 
                        "" if math.isnan(r["median_err"]) else f"{r['median_err']:.4f}"
                        ])

    #### OBS this decides how many top tiles to visualize in detail ####
    top1 = scores_small[:1] # top-1 only. if you want to visualise for more than the top 1, change here
    rank_colors = [(0,255,0)]  # Green for top-1

    if visualisations_enabled:
        top1 = scores_small[:1]  # top-3 for visualizations TODO
        rank_colors = [(0,255,0), (255,0,0), (0,0,255)]  # Green, Red, Blue for top -1,2,3


    ########################################################################################################
    #- --------------------- Pass 2: Visualization and Kalman Filtering on top 1 candidate --------------------
    ##########################################################################################################
    #initialize satellite display
    sat_vis, SX, SY = load_sat_display_and_scale()
    sat_base = sat_vis.copy()

    # ORIGINAL drone size (for original-frame overlays / error)
    _bgr_orig = cv2.imread(str(DRONE_IMG), cv2.IMREAD_COLOR)
    H_orig, W_orig = _bgr_orig.shape[:2]

    ########## main loop in preprocessing each of the top-N tiles ##########
    for rank, r in enumerate(top1, 1): 
        # first step is to extract/load features for the tile and compute H for candidate tile
        tile_name = r["tile"]  # read tile name
        color = rank_colors[rank-1] # set color

        with torch.inference_mode(): # extracting precomputed features 
            tile_pt = make_feature_pt_path_for(tile_name)
            if tile_pt.exists():
                # use precomputed features instead of extracting on the fly
                feats1_b, feats1_r = load_feats_pt_batched(tile_pt, device if feat != "sift" else "cpu") 
            else: # no precomputed features extract on the fly
                img1_t  = load_image(str(tile_name)).to(device if feat != "sift" else "cpu")
                feats1_b = extractor.extract(img1_t)
                feats1_r = rbd(feats1_b)

            #call LightGlue matcher to get matches
            matches01 = matcher({"image0": feats0_batched, "image1": feats1_b}) 
            m_r = rbd(matches01) #Lightglue utility to make dimensions nice (possability of batches are removed)
            matches = m_r.get("matches", None)

            inlier_mask = None #initialise
            pts0_np = np.empty((0,2), np.float32) # initialise
            pts1_np = np.empty((0,2), np.float32) # initialise
            H_rot2tile = None   #initialise

            # Estimate homography using RANSAC + DLT refinement
            if matches is not None and matches.numel() > 0:
                pts0_np = feats0_r["keypoints"][matches[:, 0]].detach().cpu().numpy()
                pts1_np = feats1_r["keypoints"][matches[:, 1]].detach().cpu().numpy()
                if len(pts0_np) >= 4:
                    H_rot2tile, mask = cv2.findHomography(
                        pts0_np, pts1_np,
                        method=cv2.USAC_MAGSAC,
                        ransacReprojThreshold=3.0,
                        confidence=0.999
                    )  # TODO check convexity
                    if mask is not None:
                        inlier_mask = mask.ravel().astype(bool)

                    #using DLT on inliers for better accuracy
                    if H_rot2tile is not None and inlier_mask is not None and inlier_mask.sum() >= 4:
                        H_rot2tile, _ = cv2.findHomography(pts0_np[inlier_mask], pts1_np[inlier_mask], method=0)
                        # TODO: check convexity of projected drone corners in tile frame?
        # Now we have H_rot2tile mapping FROM ROTATED drone frame TO TILE frame

        # If homography is not reliable, skip overlay and error metrics TODO We schould still do EKF update?
        if H_rot2tile is None or inlier_mask is None or inlier_mask.sum() < 4:
            print(f"[warn] Homography not reliable for {tile_name.name}; skipping overlay.")
            # TODO: Implement EKF update even if overlay is skipped

            del feats1_b, feats1_r, matches01, m_r
            if 'img1_t' in locals():
                del img1_t
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue # EKFupdate schould be made
        
        # Compute H from ORIGINAL drone frame to TILE frame
        H_orig2tile = H_rot2tile @ R_orig2rot      

        # -------------------- Visualizations: plotting the matches side-by-side --------------------
        if visualisations_enabled:
            out_match_png = OUT_DIR / f"top{rank:02d}_{tile_name.stem}_matches.png"
            visualize_inliers(DRONE_IMG_FOR_VIZ, tile_name, pts0_np, pts1_np, inlier_mask, str(out_match_png))

        # -------------------- Get visualisation parameters --------------------
        x_off, y_off = tile_offset_from_name(tile_name) # Project ORIGINAL drone corners/center (for overlays & error)
        center_global, corners_global, forward_global, heading_unitvector_measurement = get_visualisation_parameters(H_orig2tile, W_orig, H_orig, x_off, y_off)

        ####################################### EKF + metrics for only top candidate #######################################
        if rank == 1:  # top-1 candidate
            # compute overall_confidence:
            with open(CSV_RESULT_PATH, newline="") as f:
                reader = csv.DictReader(f)
                first_row = next(reader)  # skip the overview line
                num_inliers = int(first_row["inliers"])
                avg_confidence = float(first_row["avg_confidence"])
                median_err_px = float(first_row["median_reproj_error"])
            overall_confidence = absolute_confidence(num_inliers, avg_confidence, median_err_px)

            #------------- extended kalman filter update step ------------
        
            # ---- Measurement covariance from confidence ---- must be computed each frame
            R = ekf.R_from_conf(
                    pos_base_std=36.0, # in px. we expect araound +-10 m = 36px error in position measurement
                    heading_base_std_rad=np.deg2rad(8.0), # can jump due to bad matches
                    overall_conf=overall_confidence  # between 0 and 1
                )  # this gives us R matrix for EKF update. R tells us how ceartain we are about the measurements.

            # ---- Build measurement (x, y, phi) from homography in ORIGINAL pixels. compass heading ----
            meas_phi_deg, (meas_x_px, meas_y_px) = get_measurements(W_orig, H_orig, forward_global, center_global, heading_unitvector_measurement)
            print(f"Measurement: x={meas_x_px:.2f}, y={meas_y_px:.2f}, phi={meas_phi_deg:.2f} deg")

            # EKF update with measurement (OBS: phi is in rad and in image representation!!! )
            # NOTE: the update function expects input heading in radians and image frame convention!
            x_updated, P_estimated = ekf.update_pos_heading([meas_x_px, meas_y_px, np.deg2rad(compass_to_img(meas_phi_deg))], R) 

            # To get phi estimated back to compass representation and degrees:
            x_updated[3] = img_to_compass(np.rad2deg(x_updated[3]))  # convert back to degrees for logging
            print(f"[EKF] Updated state: x={x_updated[0]:.2f}, y={x_updated[1]:.2f}, v={x_updated[2]:.2f} px/s, phi={(x_updated[3]):.2f} deg")
            #EKF done

            #------------- Position error computation --------------------
            # Error in meters (using ORIGINAL-frame center)
            lat_long_pose_estimated = get_location_in_sat_img(center_global, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
            error, dx, dy, error_ekf, dx_ekf, dy_ekf, dphi, dphi_ekf, gt_pose_px = determine_pos_error(lat_long_pose_estimated, (x_updated[0], x_updated[1]), meas_phi_deg, x_updated[3], DRONE_INFO_DIR, drone_img)

            print(f"[Metrics] Mean Error: {error}m, dx: {dx}m, dy: {dy}m")
            print(f"[Metrics] Mean Error (EKF): {error_ekf}m, dx: {dx_ekf}m, dy: {dy_ekf}m")

            if visualisations_enabled: 
                # ------------- Overlays on SATELLITE image -------------------- 
                # map to DOWNSCALED satellite coords for visualization
                corners_disp = corners_global * np.array([SX, SY], np.float32)
                center_measurement  = center_global  * np.array([SX, SY], np.float32)
                center_ekf = np.array([x_updated[0]*SX, x_updated[1]*SY], np.float32)
                center_gt = np.array([gt_pose_px[0]*SX, gt_pose_px[1]*SY], np.float32)
                center_pred = np.array([x_pred[0]*SX, x_pred[1]*SY], np.float32)

                for i in range(2):
                    if i == 0: # individual overlay for this tile
                        sat_individual = sat_base.copy()
                        out_overlay = OUT_DIR / f"top{rank:02d}_{tile_name.stem}_overlay_on_sat.png"
                        #BGR colors
                        color = [(255,255,255), (255, 0, 0), (0, 255, 0), (0, 0, 255),(0,255,255)]  # Blue: GT, Green: EKF, Red: Meas, Yellow: Pred, Cyan: overall

                        # same center as we use color to destingish
                        label_point(sat_individual, center_gt, "Pred", color[4], offset=(100, 40), font_scale=0.5, thickness=1)
                        label_point(sat_individual, center_gt, "GT", color[1], offset=(100, 20), font_scale=0.5, thickness=1)
                        label_point(sat_individual, center_gt, "EKF", color[2], offset=(100, 0), font_scale=0.5, thickness=1)
                        label_point(sat_individual, center_gt, "Meas", color[3], offset=(100, -20), font_scale=0.5, thickness=1)
                    else:
                        out_overlay = OUT_DIR.parent / f"overall_overlay_on_sat.png"
                        overlay_img = cv2.imread(str(out_overlay))
                        random_color = tuple(random.randint(128, 255) for _ in range(3))
                        color = [random_color, (255, 0, 0), (0, 255, 0), (0, 0, 255), (0,255,255)]   # Random for image. Red: GT, Green: EKF, Blue: Meas, Yellow: Pred

                        if overlay_img is None:
                            overlay_img = sat_base.copy()  # use base if file doesn't exist
                            # make label colors clear in img by text
                            label_point(overlay_img, [10,70], "Pred", color[4], offset=(0, 0), font_scale=3, thickness=3)
                            label_point(overlay_img, [300,70], "GT", color[1], offset=(0, 0), font_scale=3, thickness=3)
                            label_point(overlay_img, [450,70], "EKF", color[2], offset=(0, 0), font_scale=3, thickness=3)
                            label_point(overlay_img, [700,70], "Meas", color[3], offset=(0, 0), font_scale=3, thickness=3)
                            
                        sat_individual = overlay_img

                    # Overlay
                    draw_polygon(sat_individual, corners_disp, color=color[0], thickness=2)
                    draw_point(sat_individual, center_pred, color=color[4], r=2)
                    draw_point(sat_individual, center_measurement, color=color[3], r=2)
                    draw_point(sat_individual, center_ekf, color=color[2], r=2)
                    draw_point(sat_individual, center_gt, color=color[1], r=2)
                    draw_ellipse(sat_individual, center_measurement, P_estimated[:2, :2], k_sigma=k, color=color[0], thickness=1)

                    cv2.imwrite(str(out_overlay), sat_individual)
            
            #------------- save final results to overall CSV file --------------------
            # add to the bottom of results_{sat_number}.csv file:
            with open(CSV_FINAL_RESULT_PATH, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    drone_img,
                    first_row["tile"],
                    first_row["total_matches"],
                    num_inliers,
                    first_row["avg_confidence"],
                    f"{median_err_px:.4f}",
                    f"{overall_confidence:.4f}",
                    f"{meas_x_px:.4f}", f"{meas_y_px:.4f}", f"{meas_phi_deg:.4f}",
                    f"{x_updated[0]:.4f}", f"{x_updated[1]:.4f}", f"{x_updated[3]:.4f}",
                    f"{dx:.4f}", f"{dy:.4f}",
                    f"{dx_ekf:.4f}", f"{dy_ekf:.4f}",
                    f"{error:.4f}", f"{error_ekf:.4f}",
                    f"{dphi:.4f}", f"{dphi_ekf:.4f}",
                    # f"{time_total:.4f}",
                    # f"{time_total_ekf:.4f}",
                ])

        # empty memory
        del feats1_b, feats1_r, matches01, m_r
        if 'img1_t' in locals():
            del img1_t
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()