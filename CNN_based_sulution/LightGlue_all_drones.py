# run_batch_match_top3_constmem.py  (PT-features version)
import csv, math, gc
from pathlib import Path
import csv
from typing import Optional
import json, re

import numpy as np
import torch
import cv2
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
Heading_flip_180 = True  # flip GT heading by 180° if CSV stores view-direction (opposite of camera forward)
starting_position_latlon = (32.30462673, 119.8968847)  # TODO for ROI tile selection
confidence_start = 1.0  # initial confidence for ROI tile selection
last_dt = None # for finding dt between frames

# Extended Kalman Filter 
# we assume constant vel + heading brétween measurements
vel = 3.00505077959289e-06 #distance/sec measure in lat long - from determine_vel_from_csv

BASE = Path(__file__).parent.resolve()
DATASET_DIR = BASE / "UAV_VisLoc_dataset"
SAT_LONG_LAT_INFO_DIR = DATASET_DIR / "satellite_ coordinates_range.csv"
DRONE_INFO_DIR = DATASET_DIR / sat_number / f"{sat_number}.csv"
DRONE_IMG_CLEAN = DATASET_DIR / sat_number / "drone_test"  #TODO
SAT_DIR   = DATASET_DIR / sat_number / "test_signe" #TODO
OUT_DIR_CLEAN   = BASE / "outputs" / sat_number 
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
class EKF_ConstantSpeedHeading:
    """
    Extended Kalman Filter with constant speed + heading motion in 2D.
    State x = [x, y, s, phi]^T  (position in map units, speed in units/s, heading in rad)
    
    Process (nonlinear):
        x_{k+1}   = x_k + s_k * cos(phi_k) * dt
        y_{k+1}   = y_k + s_k * sin(phi_k) * dt
        s_{k+1}   = s_k + w_s          (random-walk on speed)
        phi_{k+1} = phi_k + w_phi      (random-walk on heading)   # or add turn-rate input if you have it

    Measurements supported:
      - pos only:       z = [x, y]
      - pos + heading:  z = [x, y, phi]
    """

    def __init__(
        self,
        x0, P0,
        sigma_pos_proc=0.5,      # process noise to diffuse position indirectly (units)
        sigma_speed=0.5,         # process noise on speed (units/sqrt(s))
        sigma_phi=np.deg2rad(5)  # process noise on heading (rad/sqrt(s))
    ):
        """
        x0: (4,) initial state [x,y,s,phi]
        P0: (4,4) initial covariance
        sigmas: process noise scales. Position process noise is approximated by
                injecting into x,y via the Jacobian (below we fold via Q).
        """
        self.x = np.array(x0, dtype=float).reshape(4)
        self.x[3] = _wrap_pi(self.x[3])
        self.P = np.array(P0, dtype=float).reshape(4,4)

        # Process noise scalars
        self.sigma_pos_proc = float(sigma_pos_proc)
        self.sigma_speed    = float(sigma_speed)
        self.sigma_phi      = float(sigma_phi)

    # ---------- Prediction ----------
    def _f(self, x, dt):
        """Nonlinear process model f(x, dt)."""
        X, Y, S, PHI = x
        c, s = np.cos(PHI), np.sin(PHI)
        Xn = X + S * c * dt
        Yn = Y + S * s * dt
        Sn = S  # random walk handled in Q
        PHIn = _wrap_pi(PHI)  # random walk handled in Q
        return np.array([Xn, Yn, Sn, PHIn], dtype=float)

    def _F_jac(self, x, dt):
        """Jacobian F = df/dx at current x, dt."""
        _, _, S, PHI = x
        c, s = np.cos(PHI), np.sin(PHI)
        F = np.eye(4)
        F[0,2] = c * dt                  # dX/dS
        F[0,3] = -S * s * dt             # dX/dphi
        F[1,2] = s * dt                  # dY/dS
        F[1,3] =  S * c * dt             # dY/dphi
        # S, PHI rows are identity (random walk)
        return F

    def _Q_proc(self, dt):
        """
        Process noise covariance (4x4) for the random-walk parts.
        We approximate position diffusion by a small baseline (sigma_pos_proc)
        to remain robust when S≈0 or dt varies.
        """
        qx = (self.sigma_pos_proc * np.sqrt(dt))**2
        qy = (self.sigma_pos_proc * np.sqrt(dt))**2
        qs = (self.sigma_speed           )**2     # already per-step variance
        qf = (self.sigma_phi * np.sqrt(dt))**2
        Q = np.diag([qx, qy, qs, qf])
        return Q

    def predict(self, dt):
        """EKF predict step."""
        # 1) Nonlinear propagation
        self.x = self._f(self.x, dt)
        self.x[3] = _wrap_pi(self.x[3])

        # 2) Linearize and propagate covariance
        F = self._F_jac(self.x, dt)
        Q = self._Q_proc(dt)
        self.P = F @ self.P @ F.T + Q
        return self.x.copy(), self.P.copy()

    # ---------- Update (two variants) ----------
    def update_pos(self, z_xy, R_pos):
        """
        Update with position-only measurement z = [x_meas, y_meas].
        R_pos: (2,2) measurement covariance in same units as x,y.
        """
        z = np.array(z_xy, dtype=float).reshape(2)
        H = np.array([[1,0,0,0],
                      [0,1,0,0]], dtype=float)
        z_pred = H @ self.x
        y = z - z_pred
        S = H @ self.P @ H.T + R_pos
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.x[3] = _wrap_pi(self.x[3])
        return self.x.copy(), self.P.copy()

    def update_pos_heading(self, z_xyphi, R_xyphi):
        """
        Update with position + heading measurement z = [x_meas, y_meas, phi_meas].
        R_xyphi: (3,3) diag([var_x, var_y, var_phi]) (rad^2 for phi).
        """
        z = np.array(z_xyphi, dtype=float).reshape(3)
        H = np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,0,1]], dtype=float)

        z_pred = np.array([self.x[0], self.x[1], self.x[3]])
        y = z - z_pred
        y[2] = _wrap_pi(y[2])  # wrap heading innovation

        S = H @ self.P @ H.T + R_xyphi
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = _wrap_pi(self.x[3])

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P
        return self.x.copy(), self.P.copy()

    # ---------- Utilities ----------
    def gating_mahalanobis2(self, z_xy, R_pos):
        """
        Mahalanobis distance^2 for a position-only measurement against current prediction.
        Useful if you want to gate candidate tiles before doing heavy work.
        """
        z = np.array(z_xy, dtype=float).reshape(2)
        H = np.array([[1,0,0,0],
                      [0,1,0,0]], dtype=float)
        z_pred = H @ self.x
        y = z - z_pred
        S = H @ self.P @ H.T + R_pos
        return float(y.T @ np.linalg.inv(S) @ y)

    def search_ellipse(self, k_sigma=2.0):
        """
        Returns ellipse (center, cov2x2) for position search region:
        { p | (p - mu)^T Σ^{-1} (p - mu) <= k_sigma^2 }.
        """
        mu = self.x[:2].copy()
        Sigma = self.P[:2,:2].copy()
        return mu, Sigma, float(k_sigma)

    def set_process_noise(self, sigma_pos_proc=None, sigma_speed=None, sigma_phi=None):
        if sigma_pos_proc is not None: self.sigma_pos_proc = float(sigma_pos_proc)
        if sigma_speed    is not None: self.sigma_speed    = float(sigma_speed)
        if sigma_phi      is not None: self.sigma_phi      = float(sigma_phi)

    @staticmethod
    def R_from_conf(pos_base_std, heading_base_std_rad=None, overall_conf=0.9,
                    min_scale=0.3, max_scale=3.0):
        """
        Build a measurement covariance from a [0..1] confidence.
        Higher confidence -> smaller variance (shrinks by ~overall_conf).
        We clamp scaling to avoid extremes.
        """
        # invert confidence to get a noise scale
        scale = np.clip(1.0 / max(1e-6, overall_conf), min_scale, max_scale)
        Rx = (pos_base_std * scale)**2
        Ry = (pos_base_std * scale)**2
        if heading_base_std_rad is None:
            return np.diag([Rx, Ry])
        else:
            Rphi = (heading_base_std_rad * scale)**2
            return np.diag([Rx, Ry, Rphi])



# -------------------- Helpers --------------------
def _wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

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
                LT_lat = np.float64(r["LT_lat_map"])
                LT_lon = np.float64(r["LT_lon_map"])
                RB_lat = np.float64(r["RB_lat_map"])
                RB_lon = np.float64(r["RB_lon_map"])
                break
        else:
            raise FileNotFoundError(f"Bounds for satellite{sat_number}.tif not found")

    u, v = np.float64(drone_img_centre)
    lon = LT_lon + (u / sat_W) * (RB_lon - LT_lon)
    lat = LT_lat + (v / sat_H) * (RB_lat - LT_lat)
    return (lat, lon)

def determine_pos_error(pose, DRONE_INFO_DIR, drone_img):
    pose = np.array(pose, dtype=np.float64)

    gt_lat = gt_lon = None
    with open(DRONE_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["filename"] == f"{drone_img}":
                gt_lat = np.float64(r["lat"])
                gt_lon = np.float64(r["lon"])
                break
    if gt_lat is None or gt_lon is None:
        raise ValueError(f"GT lat/lon not found for {drone_img}")

    difference = pose - np.array([gt_lat, gt_lon], dtype=np.float64)
    mean_lat = np.radians((pose[0] + gt_lat) / np.float64(2.0))

    meters_per_degree_lat = np.float64(111_320.0)
    meters_per_degree_lon = meters_per_degree_lat * np.cos(mean_lat)

    dy = difference[0] * meters_per_degree_lat
    dx = difference[1] * meters_per_degree_lon
    total_error_m = np.sqrt(dx**2 + dy**2)
    return total_error_m, dx, dy

def get_R_rotated_by_phi1(phi_rad: float, W_orig: int, H_orig: int) -> np.ndarray:
    c, s = np.cos(phi_rad), np.sin(phi_rad)
    cx, cy = (W_orig - 1) * 0.5, (H_orig - 1) * 0.5
    T_to   = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], dtype=np.float64)
    R      = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
    T_back = np.array([[1,0,cx],[0,1,cy],[0,0,1]], dtype=np.float64)
    return T_back @ R @ T_to

def get_phi_deg(DRONE_INFO_DIR, drone_img):
    """Read Φ heading (deg) for this frame. Returns float or raises."""
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

def draw_cropped_pred_vs_gt_on_tile(
    tile_path, Hmat, x_off, y_off,
    DRONE_INPUT_W, DRONE_INPUT_H,
    DRONE_INFO_DIR, drone_img,
    SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META, error, 
    crop_radius_px=450, out_path=None, heading_flip_180=False
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

    # Predicted center/forward in TILE coords
    w0, h0 = float(DRONE_INPUT_W), float(DRONE_INPUT_H)
    center0  = np.array([[w0/2.0, h0/2.0]], dtype=np.float32)
    forward0 = np.array([[w0/2.0, h0/2.0 - max(20.0, 0.10*h0)]], dtype=np.float32)  # forward = -y

    center_tile  = project_pts(Hmat, center0)[0].astype(np.float64)
    forward_tile = project_pts(Hmat, forward0)[0].astype(np.float64)

    v_pred = forward_tile - center_tile
    n_pred = float(np.linalg.norm(v_pred))
    ray_len = np.float64(220.0)
    if n_pred < 1e-12:
        v_pred_unit = np.array([1.0, 0.0], np.float64)
        pred_tip = center_tile + ray_len * v_pred_unit
    else:
        v_pred_unit = (v_pred / n_pred).astype(np.float64)
        pred_tip = center_tile + ray_len * v_pred_unit

    # GT center on TILE + heading
    gt_lat = gt_lon = None
    gt_heading_deg = None
    with open(DRONE_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["filename"] == f"{drone_img}":
                gt_lat = np.float64(r["lat"]); gt_lon = np.float64(r["lon"])
                for key in ("Phi1", "heading", "yaw_deg"):
                    if key in r and r[key] not in (None, "", "nan"):
                        gt_heading_deg = float(r[key]); break
                break
    if gt_lat is None or gt_lon is None:
        raise ValueError("GT lat/lon not found.")

    if (gt_heading_deg is not None) and heading_flip_180:
        gt_heading_deg = (gt_heading_deg + 180.0) % 360.0

    gt_u, gt_v = latlon_to_orig_xy(gt_lat, gt_lon, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
    gt_tile = np.array([gt_u - x_off, gt_v - y_off], dtype=np.float64)

    if gt_heading_deg is not None:
        th = np.deg2rad(np.float64(gt_heading_deg))
        v_gt_unit = np.array([np.cos(th), -np.sin(th)], np.float64)  # East=0, CCW+, y down
    else:
        v_gt_unit = np.array([np.nan, np.nan], np.float64)

    gt_tip = gt_tile + ray_len * (v_gt_unit if np.isfinite(v_gt_unit).all() else np.array([1.0, 0.0], np.float64))

    # Metrics
    if np.isfinite(v_gt_unit).all():
        dotp = float(np.clip(np.dot(v_pred_unit, v_gt_unit), -1.0, 1.0))
        dtheta = float(np.degrees(np.arccos(dotp)))
    else:
        dotp, dtheta = float("nan"), float("nan")

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

    cv2.circle(crop, S(center_tile), 10, (0,255,0), -1)
    cv2.arrowedLine(crop, S(center_tile), S(pred_tip), (0,255,0), 3, tipLength=0.2)
    cv2.circle(crop, S(gt_tile), 10, (255,0,255), -1)
    cv2.arrowedLine(crop, S(gt_tile), S(gt_tip), (255,0,255), 3, tipLength=0.2)

    cv2.rectangle(crop, (10,10), (420,90), (255,255,255), -1)
    cv2.putText(crop, "Pred (tile H)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,180,0), 2)
    cv2.putText(crop, "GT (Phi, CCW+)", (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170,0,170), 2)
    cv2.putText(crop, f"angle={dtheta:.1f}, dot={dotp:.3f}, error={error:.3f}m", (20,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    if out_path is not None:
        cv2.imwrite(str(out_path), crop)
    return crop, dtheta, dotp

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
    return img, sx, sy, meta

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

def absolute_confidence(num_inliers, total_matches, median_err_px,
                        s_inl=80.0, s_err=3.0, w=(0.6, 0.25, 0.15)):
    if total_matches <= 0 or num_inliers <= 0 or not np.isfinite(median_err_px):
        return 0.0
    inlier_score = 1.0 - np.exp(-num_inliers / s_inl)
    ratio_score  = np.clip(num_inliers / total_matches, 0.0, 1.0)
    err_score    = np.exp(- (median_err_px / s_err)**2)
    w_inl, w_ratio, w_err = w
    return float(np.clip(w_inl*inlier_score + w_ratio*ratio_score + w_err*err_score, 0.0, 1.0))


# -------------------- Device & models --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] device: {device}. Extracting features using {FEATURES}.")


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
                w.writerow(["drone_image", "tile", "inliers", "avg_confidence", "median_reproj_error", "error", "heading_diff", "overall_confidence"])

# -------------------- Load drone features --------------------
for img_path in sorted(DRONE_IMG_CLEAN.iterdir()):
    if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
        continue
    print("---")
    drone_img = img_path.name
    print(f"[info] Processing drone image: {drone_img}")
    DRONE_IMG = DRONE_IMG_CLEAN / str(drone_img)
    OUT_DIR = OUT_DIR_CLEAN / str(drone_img)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_RESULT_PATH  = OUT_DIR / "results.csv" 
    if not DRONE_IMG.exists():
        raise FileNotFoundError(f"Missing drone image: {DRONE_IMG}")

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
        # SuperPoint / DISK: rotate by k*90° based on heading 
        phi_deg = get_phi_deg(DRONE_INFO_DIR, drone_img)
           
        # Arbitrary-angle rotation with padding; save the exact affine
        img = cv2.imread(str(DRONE_IMG), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        a = -phi_deg                        
        r = math.radians(a)
        c, s = abs(math.cos(r)), abs(math.sin(r))
        newW = int(math.ceil(W*c + H*s))
        newH = int(math.ceil(W*s + H*c))

        M = cv2.getRotationMatrix2D((W/2.0, H/2.0), a, 1.0)
        # shift so the rotated content is centered in the padded canvas
        M[0, 2] += (newW - W) / 2.0
        M[1, 2] += (newH - H) / 2.0

        bgr_rot = cv2.warpAffine(img, M, (newW, newH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        DRONE_IMG_ROT_PATH = OUT_DIR / f"drone_rot_{a}.png"

        R_orig2rot = np.eye(3, dtype=np.float64)
        R_orig2rot[:2, :3] = M               

        hR, wR = bgr_rot.shape[:2]
        drone_rot_size = (wR, hR)
        DRONE_IMG_FOR_VIZ = DRONE_IMG_ROT_PATH
        cv2.imwrite(str(DRONE_IMG_ROT_PATH), bgr_rot)

        if feat == "superpoint":
            extractor_local = SuperPoint(max_num_keypoints=MAX_KPTS).eval().to(device)
        elif feat == "disk":
            extractor_local = DISK(max_num_keypoints=MAX_KPTS).eval().to(device)
        elif feat == "aliked":
            extractor_local = ALIKED(max_num_keypoints=MAX_KPTS).eval().to(device)
        else:
            raise ValueError("Unsupported feature type in rotated-drone branch.")

        with torch.inference_mode():
            img_t = load_image(str(DRONE_IMG_ROT_PATH)).to(device)
            feats0_batched = extractor_local.extract(img_t)
            feats0_r = rbd(feats0_batched)

        hR, wR = bgr_rot.shape[:2]
        drone_rot_size = (wR, hR)
        DRONE_IMG_FOR_VIZ = DRONE_IMG_ROT_PATH
        

    # -------------------- Pass 1: score all tiles --------------------
    tiles = [p for p in SAT_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    if not tiles:
        raise FileNotFoundError(f"No PNG tiles in {SAT_DIR}")

    scores_small = []
    with torch.inference_mode():
        for i, p in enumerate(tiles):
            num_inliers = 0
            avg_conf = float("nan")
            median_err = float("inf")   
            K = 0
            H = None
            inlier_mask = None
            print(f"Scoring tile {i+1}/{len(tiles)}")
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
                "median_err": median_err,
                #"geom_conf": geom_conf if 'geom_conf' in locals() else 0.0,
                #"inlier_ratio": inlier_ratio if 'inlier_ratio' in locals() else 0.0,
                "sort_key": (num_inliers, - median_err), # prioritize inliers, then lower error
            })

            del feats1_b, feats1_r, matches01, matches01_r
            if 'img1_t' in locals():
                del img1_t
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    # -------------------- Rank & write CSV --------------------
    scores_small.sort(key=lambda d: d["sort_key"], reverse=True)
    with open(CSV_RESULT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile", "total_matches", "inliers", "avg_confidence", "median_reproj_error", #"geom_confidence", "inlier_ratio"
                    ])
        for r in scores_small:
            w.writerow([r["tile"].name, 
                        r["total_matches"],
                        r["inliers"],
                        "" if math.isnan(r["avg_conf"]) else f"{r['avg_conf']:.4f}", 
                        "" if math.isnan(r["median_err"]) else f"{r['median_err']:.4f}",
                        #f"{r['geom_conf']:.4f}",
                        #f"{r['inlier_ratio']:.4f}"
                        ])
    print(f"[info] wrote CSV")

    top1 = scores_small[:1] # top-1 only for overlays chance this to visualise for more tiles

    # -------------------- Pass 2: matches viz + overlays --------------------
    sat_vis, SX, SY, _sat_meta = load_sat_display_and_scale()
    sat_base = sat_vis.copy()

    rank_colors = [(0,255,0), (255,0,0), (0,0,255)]  # BGR defined for the top 3

    #------------- extended kalman filter for search region estimation ------------
    #kf = KalmanCV2D(x0=[x0, y0, vx0, vy0])
    with open(DRONE_INFO_DIR, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["filename"] == str(drone_img):
                time = datetime.fromisoformat(row["date"])
                if last_dt is not None:
                    dt = (time -  last_dt).total_seconds()
                    print(f"dt between frames: {dt} seconds")
                last_dt = time

    # ORIGINAL drone size (for original-frame overlays / error)
    _bgr_orig = cv2.imread(str(DRONE_IMG), cv2.IMREAD_COLOR)
    H_orig, W_orig = _bgr_orig.shape[:2]

    for rank, r in enumerate(top1, 1):
        p = r["tile"]
        color = rank_colors[rank-1]

        with torch.inference_mode():
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
            m_r = rbd(matches01)
            matches = m_r.get("matches", None)

            inlier_mask = None
            pts0_np = np.empty((0,2), np.float32)
            pts1_np = np.empty((0,2), np.float32)
            H_rot2tile = None

            if matches is not None and matches.numel() > 0:
                pts0_np = feats0_r["keypoints"][matches[:, 0]].detach().cpu().numpy()
                pts1_np = feats1_r["keypoints"][matches[:, 1]].detach().cpu().numpy()
                if len(pts0_np) >= 4:
                    H_rot2tile, mask = cv2.findHomography(
                        pts0_np, pts1_np,
                        method=cv2.USAC_MAGSAC,
                        ransacReprojThreshold=3.0,
                        confidence=0.999
                    )
                    if mask is not None:
                        inlier_mask = mask.ravel().astype(bool)

                    #using DLT on inliers for better accuracy
                    if H_rot2tile is not None and num_inliers >= 4:
                        H_rot2tile, _ = cv2.findHomography(pts0_np[inlier_mask], pts1_np[inlier_mask], method=0)

        # classic side-by-side in the frame LightGlue saw (rotated)
        out_match_png = OUT_DIR / f"top{rank:02d}_{p.stem}_matches.png"
        visualize_inliers(DRONE_IMG_FOR_VIZ, p, pts0_np, pts1_np, inlier_mask, str(out_match_png))

        if H_rot2tile is None or inlier_mask is None or inlier_mask.sum() < 4:
            print(f"[warn] Homography not reliable for {p.name}; skipping overlay.")
            del feats1_b, feats1_r, matches01, m_r
            if 'img1_t' in locals():
                del img1_t
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        H_orig2tile = H_rot2tile @ R_orig2rot         

        # Project ORIGINAL drone corners/center (for overlays & error)
        corners0 = np.array([[0,0],[W_orig,0],[W_orig,H_orig],[0,H_orig]], dtype=np.float32)
        center0  = np.array([[W_orig/2, H_orig/2]], dtype=np.float32)

        corners_tile = project_pts(H_orig2tile, corners0)
        center_tile  = project_pts(H_orig2tile, center0)[0]

        # tile offsets are ORIGINAL sat pixels parsed from filename
        x_off, y_off = tile_offset_from_name(p)
        corners_global = corners_tile + np.array([x_off, y_off], np.float32)
        center_global  = center_tile  + np.array([x_off, y_off], np.float32)

        # map to DOWNSCALED satellite coords
        corners_disp = corners_global * np.array([SX, SY], np.float32)
        center_disp  = center_global  * np.array([SX, SY], np.float32)

        if rank == 1:
            # compute overall_confidence:
            with open(CSV_RESULT_PATH, newline="") as f:
                reader = csv.DictReader(f)
                first_row = next(reader) 
                num_inliers = int(first_row["inliers"])
                total_matches = int(first_row["total_matches"])
                median_err_px = float(first_row["median_reproj_error"])
            overall_confidence = absolute_confidence(num_inliers,total_matches,median_err_px)
        
            # Single overlay
            sat_individual = sat_base.copy()
            draw_polygon(sat_individual, corners_disp, color=color, thickness=3)
            draw_point(sat_individual, center_disp, color=color, r=5)
            out_overlay = OUT_DIR / f"top{rank:02d}_{p.stem}_overlay_on_sat.png"
            cv2.imwrite(str(out_overlay), sat_individual)

            # Error in meters (using ORIGINAL-frame center)
            pose_estimated = get_location_in_sat_img(center_global, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
            error, dx, dy = determine_pos_error(pose_estimated, DRONE_INFO_DIR, drone_img)
            print(f"[Metrics] Mean Error: {error}m, dx: {dx}m, dy: {dy}m")

            # Tight crop + heading metrics on TILE, using ORIGINAL-frame H and size
            out_cropped = OUT_DIR / f"top01_{p.stem}_pred_vs_gt_TILE_cropped.png"
            crop_img, dtheta_deg, dotp = draw_cropped_pred_vs_gt_on_tile(
                tile_path=p,
                Hmat=H_orig2tile,                    
                x_off=x_off, y_off=y_off,
                DRONE_INPUT_W=W_orig, DRONE_INPUT_H=H_orig,  
                DRONE_INFO_DIR=DRONE_INFO_DIR,
                drone_img=drone_img,
                SAT_LONG_LAT_INFO_DIR=SAT_LONG_LAT_INFO_DIR,
                sat_number=sat_number,
                SAT_DISPLAY_META=SAT_DISPLAY_META,
                error=error,
                crop_radius_px=450,
                out_path=out_cropped,
                heading_flip_180=Heading_flip_180
            )
            print(f"[Metrics] heading Δθ={dtheta_deg:.2f}°, dot={dotp:.4f}")

            # add to the bottom of results_{sat_number}.csv file:
            with open(CSV_FINAL_RESULT_PATH, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([drone_img,
                            first_row["tile"],
                            num_inliers,
                            first_row["avg_confidence"],
                            f"{median_err_px:.4f}",
                            f"{error:.4f}",
                            f"{dtheta_deg:.4f}",
                            f"{overall_confidence:.6f}",
            ])

        # empty memory
        del feats1_b, feats1_r, matches01, m_r
        if 'img1_t' in locals():
            del img1_t
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()



