# run_batch_match_top3_constmem.py  (PT-features version)
import csv, math, gc, random, cv2, torch, time
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

from get_metrics_from_csv import get_metrics

# -------------------- Config --------------------
FEATURES = "superpoint"       # 'superpoint' | 'disk' | 'sift' | 'aliked'
MAX_KPTS = None               # max keypoints to load from .pt files (None = all) (This controls memory usage) TODO also controls speed
MIN_CONFIDENCE_FOR_EKF_UPDATE = 0.2  # min confidence to use measurement update in EKF
dt_sec = 1
TILE_W = 800
TILE_H = 600
visualisations_enabled = True

# Global map scale: 1 global pixel = MAP_M_PER_PX meters
MAP_M_PER_PX_GLOBAL = 1.0


#OBS: ekf is tuned for straight flight. if working with more turning, increase sigma_phi and sigma_b_phi in EKF+ P0+ and tune R_from_conf
# OBS we assume top down view (important for shape terms, heading, and EKF motion model)
# --- EKF globals (top of file, before the big for-loop) ---
ekf = None
t_last = None   # timestamp of previous processed frame
x_updated = None # needed for rotation of drone img

BASE = Path(__file__).parent.resolve()
DATASET_DIR = BASE / "vpair_dataset"
TILES_INFO_CSV = DATASET_DIR / "poses_tiles.csv"
TILE_CENTERS_CSV = DATASET_DIR / "tile_centers_in_sat.csv"
DRONE_INFO_CSV = DATASET_DIR / "poses_drone.csv"

DRONE_DIR = DATASET_DIR / "drone"  
TILES_DIR   = DATASET_DIR /  "tiles" 
SAT_IMG = DATASET_DIR / "sat_mosaic_small.png"
SX, SY = 0.08667013347200554, 0.08667013347200554 
pixel_offset = [2980, 1365]


OUT_DIR_CLEAN   = BASE / "VPAIR_outputs"

# delete folder if exists
folder = OUT_DIR_CLEAN

if folder.exists() and folder.is_dir():
    shutil.rmtree(folder)

OUT_DIR_CLEAN.mkdir(parents=True, exist_ok=True)

CSV_FINAL_RESULT_PATH = OUT_DIR_CLEAN / "results_VPAIR.csv"
TILE_PT_DIR = DATASET_DIR / f"{FEATURES}_features"

# -------------------- Preload tile meta (global px + GSD) --------------------
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



########################################################################################
# --------------------- Helpers and EKF Class -------------------------------------------
########################################################################################

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
def get_median_projection_error(H, pts0, pts1, inlier_mask):
    """ Compute median projection error confidence based on reprojection error of inliers."""
    if H is None or inlier_mask is None or not inlier_mask.any():
        return np.inf # no homography or no inliers

    idx = np.where(inlier_mask)[0]
    p0 = pts0[idx].astype(np.float64)
    p1 = pts1[idx].astype(np.float64)
    p0h = np.c_[p0, np.ones(len(p0))]
    q1h = (H @ p0h.T).T
    q1 = q1h[:, :2] / q1h[:, 2:3]
    reproj_error = np.linalg.norm(q1 - p1, axis=1)
    median_err = np.median(reproj_error)
    return median_err

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

def resize_for_display(img_np, long_side=1200):
    h, w = img_np.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0: return img_np, 1.0
    new_w, new_h = int(round(w * s)), int(round(h * s))
    img_small = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_small, s

def visualize_inliers(drone_path: Path, tile_path: Path, pts0, pts1, inlier_mask, out_png):
    tile_path = TILES_DIR / tile_path
    I0 = to_numpy_image(load_image(str(drone_path)))
    I1 = to_numpy_image(load_image(str(tile_path)))
    I0d, s0 = resize_for_display(np.clip(I0, 0.0, 1.0))
    I1d, s1 = resize_for_display(np.clip(I1, 0.0, 1.0))
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
    meas_phi_deg = img_to_compass(float(np.degrees(meas_phi_rad)))

    return meas_phi_deg, (meas_x, meas_y)

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


def make_segments(p0, p1, x_offset):
    segs = np.zeros((len(p0), 2, 2), dtype=np.float32)
    segs[:, 0, :] = p0
    segs[:, 1, 0] = p1[:, 0] + x_offset
    segs[:, 1, 1] = p1[:, 1]
    return segs

def make_feature_pt_path_for(image_path: Path) -> Path:
    """
    returns path for feature file for image (.pt) files
    """
    folder = TILE_PT_DIR if TILE_PT_DIR is not None else image_path.parent
    return folder / (Path(image_path).stem + ".pt")

def load_feats_pt_batched(pt_path: Path, device: str):
    d = torch.load(str(pt_path), map_location="cpu", weights_only=True)
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

def classify_meas_quality(
    num_inliers,
    LG_conf,
    shape_conf,
    median_err_px,
    overall_conf,
):
    """
    Classify the measurement into { 'reject', 'low', 'medium', 'high' }.

    Based on:
      - inlier count
      - LG_conf (LightGlue)
      - shape_conf
      - median reprojection error
      - overall_conf (your combined score)

    Thresholds are tuned for your current dataset and can be tweaked.
    """

    # Basic guards
    if num_inliers is None or num_inliers <= 0:
        return "reject"
    if not np.isfinite(LG_conf) or not np.isfinite(shape_conf):
        return "reject"
    if overall_conf is None or not np.isfinite(overall_conf):
        overall_conf = 0.0

    # Very bad → reject
    if overall_conf < 0.2 or num_inliers < 10 or shape_conf < 0.2:
        return "reject"

    # HIGH quality: strong inliers, good shape, good LG conf, decent reproj error
    if (LG_conf > 0.65 and
        num_inliers > 80 and
        shape_conf > 0.65 and
        np.isfinite(median_err_px) and
        median_err_px < 2.5):
        return "high"

    # MEDIUM quality: usable but not great
    if (LG_conf > 0.45 and
        num_inliers > 20 and
        shape_conf > 0.4):
        return "medium"

    # LOW quality: we still update, but with a very large R
    return "low"

def get_confidence_meas(
    num_inliers,
    LG_conf,
    median_err_px,
    shape_conf,
    s_err=2,      # reproj error scaling
    n_err_min=80,   # only use reprojection error if inliers >= this
    alpha_err=0.2   # fixed weight for reprojection error when used
):
    """
    Confidence with:
      - ALWAYS equal weight between LightGlue and shape:
            c_prior = 0.5 * c_lg + 0.5 * c_shape
      - OPTIONAL small influence from reprojection error c_err
        when num_inliers >= n_err_min.

    Inputs:
      num_inliers   : number of inliers
      LG_conf      : LightGlue avg confidence (0..1)
      median_err_px : median reprojection error in pixels
      shape_conf    : shape/scale score (0..1)

    Behavior:
      - If LG_conf <= 0 or num_inliers <= 0 -> return 0.
      - Base confidence:
            c_prior = 0.5 * c_lg + 0.5 * c_shape
      - If num_inliers < n_err_min or median_err_px is not finite:
            conf = c_prior
      - Else:
            c_err = exp(-(median_err_px / s_err)^2)
            conf  = (1 - alpha_err) * c_prior + alpha_err * c_err
    """

    # basic guards
    if LG_conf <= 0 or num_inliers <= 0:
        return 0.0

    c_lg    = float(np.clip(LG_conf, 0.0, 1.0))
    c_shape = float(np.clip(shape_conf if np.isfinite(shape_conf) else 0.0, 0.0, 1.0))

    # --- always 50/50 LG & shape ---
    c_prior = 0.4 * c_lg + 0.6 * c_shape

    # --- if not enough inliers or bad error -> ignore reprojection term ---
    if num_inliers < n_err_min or not np.isfinite(median_err_px):
        return float(np.clip(c_prior, 0.0, 1.0))

    # --- small reprojection-error term as a nudge, not a dictator ---
    c_err = float(np.exp(- (median_err_px / s_err) ** 2))  # 0..1

    conf = (1.0 - alpha_err) * c_prior + alpha_err * c_err
    return float(np.clip(conf, 0.0, 1.0))

def fmt_shape_terms(terms) -> str:
    """
    Format shape term array as '(x,x,x,x,x)' with 3 decimals, or 'N/A' if missing.
    """
    if terms is None:
        return "N/A"
    arr = np.asarray(terms, dtype=np.float64).reshape(-1)
    parts = []
    for v in arr[:5]:
        if np.isfinite(v):
            parts.append(f"{v:.3f}")
        else:
            parts.append("nan")
    return f"({','.join(parts)})"

# ---------------------- homography convexity check ----------------------
def is_homography_convex_and_corners_warped(H, img_w, img_h):
    """
    Check if the homography H maps the drone image (0..w, 0..h) corners
    to a convex quadrilateral in the target (tile) image. Also compute side lengths.

    Args:
        H: 3x3 homography (drone -> tile)
        img_w, img_h: width and height of the drone image that H is defined for

    Returns:
        is_convex: bool, 
        sides: (4,) array of side lengths (top, right, bottom, left)
    """
    if H is None:
        return False, None
    
    # 4 corners of the drone image
    corners0 = np.array([
        [0,      0     ], # top-left
        [img_w,  0     ], # top-right
        [img_w,  img_h ], # bottom-right
        [0,      img_h ], # bottom-left
    ], dtype=np.float32)

    # warp them with your existing helper
    corners_warped = project_pts(H, corners0)  # (4, 2)

    # finite check (avoid NaNs / infs)
    if not np.isfinite(corners_warped).all():
        return False, None

    # use OpenCV convexity test
    cnt = corners_warped.reshape(-1, 1, 2).astype(np.float32)
    is_convex = bool(cv2.isContourConvex(cnt))

    return is_convex, corners_warped

def get_shape_score(
    corners,
    img_w,
    img_h,
    scale_drone_to_tile,   # drone_px per tile_px
    tau_pair=0.2,          # tolerance for opposite side length consistency
    tau_ar=0.2,            # tolerance for aspect ratio
    tau_angle=12.0,        # degrees tolerance for angles (no safe zone)
    tau_scale_abs=0.5,     # how fast we penalize scale error beyond dead-band
    scale_dead_band=0.10   # ONLY area has a dead zone (±10%)
):
    """
    Shape score in [0,1] using only the 4 warped corners (in tile coords).

    Checks:
      - s_w, s_h: opposite side length consistency (rectangularity)
      - s_ar: aspect ratio vs original drone W/H
      - s_angle: all 4 angles ~ 90°
      - s_scale_abs: absolute scale vs expected (with dead-band)

    Combination:
        score = 0.6 * min(terms) + 0.4 * mean(terms)
    (Strict wrt the worst term, but not as brittle as pure min.)
    """

    # ---------- Validate input ----------
    if corners is None:
        return 0.0, np.full(5, np.nan, dtype=np.float64)

    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return 0.0, np.full(5, np.nan, dtype=np.float64)

    # =====================================================================
    # 0) Compute side vectors & lengths from corners
    #     Order: [top-left, top-right, bottom-right, bottom-left]
    # =====================================================================
    vT = corners[1] - corners[0]  # top
    vR = corners[2] - corners[1]  # right
    vB = corners[3] - corners[2]  # bottom
    vL = corners[0] - corners[3]  # left

    lT = np.linalg.norm(vT)
    lR = np.linalg.norm(vR)
    lB = np.linalg.norm(vB)
    lL = np.linalg.norm(vL)

    sides = np.array([lT, lR, lB, lL], dtype=np.float64)
    if np.any(sides < 1e-3) or not np.isfinite(sides).all():
        return 0.0

    # =====================================================================
    # 1) OPPOSITE SIDE EQUALITY  (no safe zone)
    # =====================================================================
    d_w = abs(lT - lB) / (lT + lB + 1e-6)
    d_h = abs(lR - lL) / (lR + lL + 1e-6)

    s_w = float(np.exp(- (d_w / tau_pair) ** 2))
    s_h = float(np.exp(- (d_h / tau_pair) ** 2))

    # =====================================================================
    # 2) ASPECT RATIO  (no safe zone)
    # =====================================================================
    w_est = 0.5 * (lT + lB)
    h_est = 0.5 * (lR + lL)

    r_est = w_est / (h_est + 1e-6)
    r0    = img_w / float(img_h)

    ratio_err = np.log(r_est / (r0 + 1e-6))
    s_ar = float(np.exp(- (ratio_err / tau_ar) ** 2))

    # =====================================================================
    # 3) ANGLES  (no safe zone)
    # =====================================================================
    def angle(a, b, c):
        """Angle ABC in degrees, with B as corner."""
        BA = a - b
        BC = c - b
        den = (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9)
        cosang = np.dot(BA, BC) / den
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    ang = []
    for i in range(4):
        a = corners[i - 1]
        b = corners[i]
        c = corners[(i + 1) % 4]
        ang.append(angle(a, b, c))
    ang = np.asarray(ang, dtype=np.float64)

    ang_err = np.abs(ang - 90.0)  # want ~90° at all 4 corners
    mean_ang_err = float(np.mean(ang_err))

    s_angle = float(np.exp(- (mean_ang_err / tau_angle) ** 2))

    # (optional) super-hard rejection if angle is crazy:
    # if np.any(ang < 60.0) or np.any(ang > 120.0):
    #     return 0.0

    # =====================================================================
    # 4) ABSOLUTE SCALE (only term with dead-band)
    # =====================================================================
    area_now = w_est * h_est      # warped rect area in tile px²
    area0    = float(img_w * img_h)  # original drone rect area in drone px²
    if area_now <= 0 or area0 <= 0:
        return 0.0

    S = float(scale_drone_to_tile)     # drone_px per tile_px
    # expected: scale_now ~ 1
    scale_now = S * math.sqrt(area_now / area0)
    scale_err = abs(scale_now - 1.0)

    if scale_err <= scale_dead_band:
        s_scale_abs = 1.0
    else:
        eff = scale_err - scale_dead_band
        s_scale_abs = float(np.exp(- (eff / tau_scale_abs) ** 2))

    # =====================================================================
    # 5) Combine terms (strict wrt worst)
    # =====================================================================
    terms = np.array([s_w, s_h, s_ar, s_angle, s_scale_abs], dtype=np.float64)

    mean_t = float(terms.mean())
    min_t  = float(terms.min())

    shape_score = 0.6 * min_t + 0.4 * mean_t
    return float(np.clip(shape_score, 0.0, 1.0)), terms

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

    sigma_speed (px/√s in your code it is used as absolute per step, i.e., Q uses (sigma_speed)**2): how much speed can wander.
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
#print(f"[info] device: {device}. Extracting features using {FEATURES}.")
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
                w.writerow(["drone_image", "features_drone", "tile", "total_matches", "search_tiles", "inliers", "LG_confidence", 
                            "median_reproj_error_px", "shape_terms", "shape_score", "overall_confidence", 
                            "x_meas", "y_meas", "phi_meas_deg",
                            "x_ekf", "y_ekf", "phi_ekf_deg",
                            "dx", "dy", 
                            "dx_ekf", "dy_ekf",
                            "error", "error_pred", "ekf_error", 
                            "heading_diff", "ekf_heading_diff", 
                            "time_s", "SDS",])

# -------------------- Preload satellite tiles --------------------              
all_tile_names = [p for p in TILES_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
# -------------------- Confidence scaling for ellipse --------------------
k = math.sqrt(chi2.ppf(0.99, df=2)) # TODO confidence scaling for 2D ellipse. so how conservative we are. the df is 2 since 2D.

# -------------------- Load drone features --------------------
for j, img_path in enumerate(sorted(DRONE_DIR.iterdir())):
    if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
        continue
    drone_img = img_path.name
    meta = tile_meta[drone_img]
    GT_centre_x_px_tile = meta["center_x"]
    GT_centre_y_px_tile = meta["center_y"]

    t_start = time.perf_counter()
    t_end = None
    
    

    if j == 0: #first image
        meas_phi_deg = None #for rotation consistency check
        ekf = None
        t_last = None

        # clean cuda cache to avoid OOM on new sequences
        if device == "cuda":
            torch.cuda.empty_cache()

        if visualisations_enabled:
            # also reset global overlay for this satellite sequence
            OUT_OVERALL_SAT_VIS_PATH = OUT_DIR_CLEAN / f"overall_overlay_on_sat{drone_img}.png"
            overall_overlay = cv2.imread(str(SAT_IMG), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(OUT_OVERALL_SAT_VIS_PATH), overall_overlay)


    DRONE_IMG = DRONE_DIR / str(drone_img)
    if visualisations_enabled: 
        OUT_DIR = OUT_DIR_CLEAN / str(drone_img)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CSV_RESULT_PATH  = OUT_DIR / "results.csv" 
    if not DRONE_IMG.exists():
        raise FileNotFoundError(f"Missing drone image: {DRONE_IMG}")
    
    # -------------------- EKF initialisation--------------------
    if ekf is None: # first frame is skipped for EKF initialization
        # -------------------- EKF initialization (global across frames) --------------------
        #read CSV to get starting lat lon and velocity estimate
        with open(DRONE_INFO_CSV, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                if Path(row["filename"]).stem == Path(drone_img).stem:
                    x0 = float(row["x"])
                    y0 = float(row["y"])
                    yaw0_rad = float(row["yaw"])   # radians
                    yaw0_rad_img = np.deg2rad(compass_to_img(np.rad2deg(yaw0_rad)))

                    # Get next frame to estimate velocity
                    next_row = next(r)
                    x1 = float(next_row["x"])
                    y1 = float(next_row["y"])
                    vel0 = math.hypot(x1 - x0, y1 - y0) / dt_sec   # dt=1s
                    break

        # initial state: (x, y, v, phi, bias_phi) OBS: must be in image representation and phi:rad!!!
        x0 = np.array([0, 0, vel0, yaw0_rad_img, np.deg2rad(0.0)], dtype=np.float64)  # x,y in ORIGINAL sat pixels

        # P is the initial covariance for measurement uncertainty
        P0 = np.diag([(50.0)**2,            # σx = 50 px
                    (50.0)**2,              # σy = 50 px
                    (3.0)**2,               # σv = 3 px/s (since we have a rough estimate)
                    np.deg2rad(15.0)**2,     # σφ = 15° deg/s
                    np.deg2rad(3.0)**2      # at t0 we are unsure with around σbias_φ = 10.0 deg (This only affect us at start untill convergence)
                    ])  # this is something we only set for this first run it will be updated by EKF later. 
        
        # Process noise covariance Q (model uncertainty) Tune when ekf trust model to much or too little (high values=less trust in model)
        Q0 = np.diag([3.0,                  # px/√s : baseline diffusion on x,y
                      0.5,                  # px/√s : how much v can wander
                      np.deg2rad(5),      # rad/√s : how much phi can wander pr second. # we fly straight so expect small changes
                      np.deg2rad(0.0025)     # rad/√s : how much bias_phi can wander pr second
                    ])  # this is something we only set for this first run it will be updated by EKF later. 

        ekf = EKF_ConstantVelHeading(x0, P0, Q0) # initialize EKF
        if visualisations_enabled:  
            overlay_img = cv2.imread(str(OUT_OVERALL_SAT_VIS_PATH), cv2.IMREAD_COLOR)
            start_x, start_y = pixel_offset
            draw_point(overlay_img, (start_x, start_y), color=(155,155,155), r=6) # light grey
            label_point(overlay_img, (start_x, start_y), f"Start {drone_img}", color=(155,155,155),
                        offset=(10,-10), font_scale=0.5, thickness=1)
            draw_ellipse(overlay_img, (start_x, start_y), ekf.P[:2, :2]*(SX*SY), k_sigma=k, color=(255,0,0), thickness=2)
            cv2.imwrite(str(OUT_OVERALL_SAT_VIS_PATH), overlay_img)
        continue # next drone image

    # ---------------------------- EKF prediction + ellipse for search area ---------------

    # -------------------- EKF predict step --------------------
    x_pred, _ = ekf.predict(dt_sec)
    x_pred[3] = img_to_compass(np.rad2deg(x_pred[3]))
    #print(f"[Predict-only] Predicted position: ({x_pred[0]:.2f}, {x_pred[1]:.2f}) px, heading={x_pred[3]:.2f}°, Bias_phi={np.rad2deg(x_pred[4]):.2f}°")

    # -------------------- EKF search area (2,-sigma ellipse) --------------------
    # create search area using Predicted covariance P:
    P_pred = ekf.P
    sigma = P_pred[:2, :2]  # position covariance 2x2
    
    ellipse_bbox_coords = ellipse_bbox(x_pred[:2], sigma, k=k, n=72) # this uses SVD to determine viedest axes and orienation of ellipse to get bbx

    # -------------------- Determine tiles in search area --------------------
    selected_tiles = tiles_in_bbox(ellipse_bbox_coords, TILE_W, TILE_H, all_tile_names)
    print(f"[info] Drone image {drone_img}: selected {selected_tiles[0]} to {selected_tiles[-1]} tiles in search area.")

    # -------------------- Rotate drone image & extract features --------------------

    # SuperPoint / DISK: rotate according to heading from CSV
    if meas_phi_deg is None:
        phi_deg_flip = np.rad2deg(yaw0_rad)  # first frame use initial yaw
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

    if visualisations_enabled: 
        DRONE_IMG_ROT_PATH = OUT_DIR / f"drone_rot_{a}.png"
        DRONE_IMG_FOR_VIZ = DRONE_IMG_ROT_PATH
        cv2.imwrite(str(DRONE_IMG_ROT_PATH), bgr_rot)

    # -------------------- Extract features from rotated drone image --------------------
    with torch.inference_mode():
        # Convert BGR → RGB
        rgb = cv2.cvtColor(bgr_rot, cv2.COLOR_BGR2RGB)
        # Convert to float32 normalized [0,1]
        img_np = rgb.astype(np.float32) / 255.0
        # Convert to torch tensor: HWC → CHW → BCHW
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        feats_drone_b_scaled = extractor.extract(img_t)
        feats_drone_r_scaled = rbd(feats_drone_b_scaled)
        num_feats_in_drone = feats_drone_b_scaled["keypoints"].shape[1]

    hR, wR = bgr_rot.shape[:2]
    drone_rot_size = (wR, hR)
        
    # -------------------- Pass 1: score all tiles --------------------   
    if not selected_tiles:
        raise FileNotFoundError(f"No PNG tiles in {TILES_DIR} found for drone image {drone_img}.")

    scores_small = []
    # we need to restart variables for best match
    H_best = None
    best_shape_score = 0.0
    best_shape_terms = None
    best_conf_overall = 0.0

    with torch.inference_mode():
        for i, p in enumerate(selected_tiles):
            # the following is necessary to avoid "referenced before assignment" errors
            num_inliers = 0 
            LG_conf = float("nan")
            median_projection_err = float("inf")   
            overall_conf = float("-inf") 
            shape_conf = float("nan")  
            shape_terms = np.full(5, np.nan, dtype=np.float64)
            K = None
            H = None
            is_convex_dlt = False
            H_ransac = None
            H_dlt = None
            inlier_mask = None

            tile_pt = make_feature_pt_path_for(p)

            # -------------------- Load / extract features for tile --------------------
            if tile_pt.exists():
                feats_tile_b, feats_tile_r = load_feats_pt_batched(tile_pt, device if feat != "sift" else "cpu")
            else: # if none saved, extract on the fly
                if extractor is None:
                    raise FileNotFoundError(f"Missing {tile_pt} and no extractor available.")
                img1_t  = load_image(str(p)).to(device if feat != "sift" else "cpu")
                feats_tile_b = extractor.extract(img1_t)
                feats_tile_r = rbd(feats_tile_b) # TODO save the features from drone imge for future runs
                print(f"[info] Extracted features on-the-fly for tile {p.name}.")

            # -------------------- Match features --------------------
            matches01 = matcher({"image0": feats_drone_b_scaled, "image1": feats_tile_b})
            matches01_r = rbd(matches01)

            matches = matches01_r.get("matches", None)
            K = int(matches.shape[0]) if (matches is not None and matches.numel() > 0) else 0

            # -------------------- Estimate homography with RANSAC or DLT--------------------
            if K >= 4: # at least 4 matches to estimate homography 
                pts_drone_np = feats_drone_r_scaled["keypoints"][matches[:, 0]].detach().cpu().numpy()
                pts_tile_np = feats_tile_r["keypoints"][matches[:, 1]].detach().cpu().numpy()

                # RANSAC with MAGSAC
                H_ransac, mask = cv2.findHomography(
                    pts_drone_np, pts_tile_np, method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=3.0, confidence=0.999
                ) 
                # check for convexity of H_ransac
                is_convex, corners_ransac =is_homography_convex_and_corners_warped(H_ransac, drone_rot_size[0], drone_rot_size[1]) #original drone size for convexity check
                if is_convex == False:
                    continue # skip non-convex homographies
                # get inlier count and avg confidence from LG
                if mask is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    num_inliers = int(inlier_mask.sum())

                    scores_t = matches01_r.get("scores", None)
                    if scores_t is not None and num_inliers > 0:
                        scores_np = scores_t.detach().cpu().numpy()
                        LG_conf = float(np.mean(scores_np[inlier_mask]))

                # compute shape score and median reprojection error for RANSAC H
                shape_ransac, terms_ransac = get_shape_score(corners_ransac, drone_rot_size[0], drone_rot_size[1], 1)
                median_projection_err_ransac = get_median_projection_error(
                        H_ransac, pts_drone_np, pts_tile_np, inlier_mask
                    )

                #--------------- Optional DLT refinement ---------------
                if H_ransac is not None and num_inliers >= 8: # at least 8 inliers to use DLT otherwise unstable
                    H_dlt, _ = cv2.findHomography(pts_drone_np[inlier_mask], pts_tile_np[inlier_mask], method=0)
                    is_convex_dlt, corners_dlt = is_homography_convex_and_corners_warped(H_dlt, drone_rot_size[0], drone_rot_size[1])

                # ---------- Decide between RANSAC and DLT ----------------------
                #   We use both reprojection error and shape score to decide which one to keep.
                #   Reprojection error will always be better for DLT but if we have a good RANSAC
                if is_convex_dlt == False: # if DLT is not convex, use RANSAC
                    H = H_ransac
                    shape_conf = shape_ransac
                    shape_terms = terms_ransac
                    median_projection_err = median_projection_err_ransac
                else:
                    # compute shape score for DLT
                    shape_dlt, terms_dlt = get_shape_score(corners_dlt, drone_rot_size[0], drone_rot_size[1], 1)
                    median_projection_err_dlt = get_median_projection_error(
                        H_dlt, pts_drone_np, pts_tile_np, inlier_mask
                    )
                    # both exist, enough inliers: use DLT only if better in shape
                    if shape_dlt > shape_ransac: 
                        H = H_dlt
                        shape_conf = shape_dlt
                        shape_terms = terms_dlt
                        median_projection_err = median_projection_err_dlt
                    else: # keep RANSAC
                        H = H_ransac
                        shape_conf = shape_ransac
                        shape_terms = terms_ransac
                        median_projection_err = median_projection_err_ransac
                        
                # ---------- Overall Confidence + CHOSE BEST----------
                if H is not None:
                    overall_conf = get_confidence_meas(
                        num_inliers, LG_conf, median_projection_err, shape_conf
                    )

                    # ---------- Keep best homography across tiles ----------
                    if overall_conf > best_conf_overall:  
                        if overall_conf == best_conf_overall and shape_conf < best_shape_score:
                            continue  # keep previous best if overall_conf tie but shape_conf not better
                        H_best = H
                        best_shape_score = shape_conf
                        best_shape_terms = shape_terms
                        best_conf_overall = overall_conf
                        best_pts_drone_np = pts_drone_np
                        best_pts_tile_np = pts_tile_np
                        best_inlier_mask = inlier_mask

            # -------------------- Save per-tile score --------------------           
            scores_small.append({
                "tile": p,   
                "inliers": num_inliers,
                "total_matches": K,
                "LG_conf": LG_conf,
                "median_err": median_projection_err,  
                "shape_score": shape_conf,
                "shape_terms": shape_terms,
                "overall_conf": overall_conf
            })
    
    # -------------------- Rank them and save in CSV. There is gonna be one CSV for each drone image --------------------
    scores_small.sort(key=lambda d:(d["overall_conf"], 0.0 if not math.isfinite(d["shape_score"]) else d["shape_score"]),
                                    reverse=True)
    
    ########################################################################################################
    #- --------------------- Pass 2: Visualization and Kalman Filtering on top 1 candidate --------------------
    ##########################################################################################################

     # ----------------------------- IF needed, skip update and use predict only-------------------
    if (H_best is None or best_conf_overall < MIN_CONFIDENCE_FOR_EKF_UPDATE):
        t_end = time.perf_counter()
        tile_name = None
        num_inliers = None
        K = None
        LG_confidence = None
        median_reproj_error_px = None
        shape_conf = None
        shape_terms = None
        if H_best is not None:
            # -------------------- Read top-1 candidate scores --------------------
            top1 = scores_small[0] # top-1 only.
            tile_name = top1["tile"]           
            num_inliers = top1["inliers"]
            K = top1["total_matches"]
            LG_confidence = top1["LG_conf"]
            median_reproj_error_px = top1["median_err"]
            shape_conf = top1["shape_score"]
            shape_terms = top1.get("shape_terms")
            best_conf_overall = top1["overall_conf"]
            print(f"[info] Top-1 candidate for {drone_img} is tile {tile_name} but skipping EKF update.")

        print(f"[info] Skipping EKF update for {drone_img} due to no valid homography or low confidence ({best_conf_overall:.4f} < {MIN_CONFIDENCE_FOR_EKF_UPDATE})")
        meters_pr_pixel = MAP_M_PER_PX_GLOBAL
        pose_ekf_px = (x_pred[0], x_pred[1])
        diff_px = (pose_ekf_px[0] - GT_centre_x_px_tile,
                   pose_ekf_px[1] - GT_centre_y_px_tile)

        dx_ekf = diff_px[0] * meters_pr_pixel
        dy_ekf = diff_px[1] * meters_pr_pixel
        error_ekf = abs(math.hypot(dx_ekf, dy_ekf))

        dphi_ekf = 0.0
        shape_terms_str = fmt_shape_terms(shape_terms)
    
        with open(CSV_FINAL_RESULT_PATH, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([drone_img, #"drone_image",
                        num_feats_in_drone, # "drone_features",
                        "N/A" if tile_name is None else tile_name, # "tile",
                        "N/A" if K is None else K, # "total_matches",
                        len(selected_tiles), #search_tiles",
                        "N/A" if num_inliers is None else num_inliers, # "inliers",
                        "N/A" if LG_confidence is None else f"{LG_confidence:.3f}", #"LG_confidence", 
                        "N/A" if median_reproj_error_px is None else f"{median_reproj_error_px:.3f}", # median_reproj_error,
                        shape_terms_str,
                        "N/A" if shape_conf is None else f"{shape_conf:.3f}", # shape_score,
                        "N/A" if best_conf_overall == 0.0  else f"{best_conf_overall:.3f}", # best_conf_overall
                        "N/A", "N/A", "N/A", # x_meas, y_meas, phi_meas_deg
                        f"{x_pred[0]:.8f}", f"{x_pred[1]:.8f}", f"{x_pred[3]:.3f}", # x_ekf, y_ekf, phi_ekf_deg
                        "N/A", "N/A", # dx, dy
                        f"{dx_ekf:.4f}", f"{dy_ekf:.4f}", # dx_ekf, dy_ekf
                        "N/A",f"{error_ekf:.4f}", f"{error_ekf:.4f}", # error, error_pred, ekf_error
                        "N/A", f"{dphi_ekf:.4f}", # heading_diff, ekf_heading_diff
                        f"{t_end - t_start:.4f}",   # time_s
                        ])
        if visualisations_enabled:
            overlay_img = cv2.imread(str(OUT_OVERALL_SAT_VIS_PATH), cv2.IMREAD_COLOR)
            if overlay_img is None:
                # safety: initialize base image if file is missing
                overlay_img = cv2.imread(str(SAT_IMG), cv2.IMREAD_COLOR)

            pred_x, pred_y = x_pred[0], x_pred[1]       # ORIGINAL sat px
            pred_disp = (pred_x*SX + pixel_offset[0], pred_y*SY + pixel_offset[1])      # DISPLAY sat px

            draw_point(overlay_img, pred_disp, color=(255,255,0), r=1)

            # --- Correct covariance scaling (see next section) ---
            Sigma_orig = ekf.P[:2, :2]      # covariance in ORIGINAL px
            J = np.diag([SX, SY])           # scaling x,y → display coords
            Sigma_disp = J @ Sigma_orig @ J.T

            draw_ellipse(
                overlay_img,
                pred_disp,
                Sigma_disp,
                k_sigma=k,
                color=(255,255,0),
                thickness=2,
            )

            cv2.imwrite(str(OUT_OVERALL_SAT_VIS_PATH), overlay_img)
        continue # next drone image

    # -------------------- Read top-1 candidate scores needed for ekf --------------------
    top1 = scores_small[0] # top-1 only.
    tile_name = top1["tile"]
    best_conf_overall = top1["overall_conf"]
    shape_terms = best_shape_terms if best_shape_terms is not None else top1.get("shape_terms")
    shape_terms_str = fmt_shape_terms(shape_terms)
    print(f"[info] Top-1 candidate for {drone_img} is tile {tile_name}")

    if visualisations_enabled:
        out_match_png = OUT_DIR / f"top1_{Path(tile_name).stem}_matches.png"
        visualize_inliers(DRONE_IMG_FOR_VIZ, tile_name, best_pts_drone_np, best_pts_tile_np, best_inlier_mask, str(out_match_png))


    ##############################################################################################
    # ORIGINAL drone size (for original-frame overlays / error)
    _bgr_orig = cv2.imread(str(DRONE_IMG), cv2.IMREAD_COLOR)
    H_orig, W_orig = _bgr_orig.shape[:2]

    # Compute H from ORIGINAL drone frame to TILE frame
    H_rot2tile = H_best # the best match
    H_orig2tile = H_rot2tile @ R_orig2rot   

    # -------------------- Get visualisation parameters --------------------
    
    center_meas_in_tile_px, corners_meas_in_tile_px, heading_unitvector_measurement = get_visualisation_parameters(H_orig2tile, W_orig, H_orig)

    #################################################################
    #------------- extended kalman filter update step ------------
    #################################################################

    # ---- Measurement covariance from confidence ---- must be computed each frame
    R = ekf.R_from_conf(
            pos_base_std=40.0, #55 in px. measured the difference of GT and meas for first run and found mean around 55 px, with 95 percentile around 10 and 123 px
            heading_base_std_rad=np.deg2rad(3.0), # a normal good measurement are seen to be within +-3 degrees
            overall_conf=best_conf_overall,  # between 0 and 1
            pos_min_scale=0.1, # 0.5 controls how much we trust low confidence measurements sets the unceartanty determined using 95 percentile
            pos_max_scale=2.0,  # controls how much we trust high confidence measurements sets the certainty -||-
            heading_min_scale=0.1, # 0.5 same for heading -||- found through testing
            heading_max_scale=2.75  # same for heading -||-
        )  # this gives us R matrix for EKF update. R tells us how certain we are about the measurements.

    # ---- Build measurement (x, y, phi) from homography in ORIGINAL pixels. compass heading ----
    meas_phi_deg, (meas_x_px_in_tile, meas_y_px_in_tile) = get_measurements(center_meas_in_tile_px, heading_unitvector_measurement)
    #print(f"Measurement: x={meas_x_px:.2f}, y={meas_y_px:.2f}, phi={meas_phi_deg:.2f} deg")

    # ---- Build measurement (x, y, phi) from homography ----
    # center_meas_in_tile_px is in TILE pixels (800x600)
    meas_phi_deg, (meas_x_px_in_tile, meas_y_px_in_tile) = get_measurements(
        center_meas_in_tile_px, heading_unitvector_measurement
    )

    # Look up tile center & GSD (meters per image pixel)
    meta = tile_meta[tile_name]  # tile_name is like "00001.png"
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
    dx_global_px = dx_m / MAP_M_PER_PX_GLOBAL
    dy_global_px = dy_m / MAP_M_PER_PX_GLOBAL

    # Final measurement in global pixels
    meas_x_px_global = cx_global + dx_global_px
    meas_y_px_global = cy_global + dy_global_px

    # EKF update with measurement (OBS: phi is in rad and in image representation!!! )
    # NOTE: the update function expects input heading in radians and image frame convention!
    x_updated, _ = ekf.update_pos_heading([meas_x_px_global, meas_y_px_global, np.deg2rad(compass_to_img(meas_phi_deg))], R) 

    # To get phi estimated back to compass representation and degrees:
    x_updated[3] = img_to_compass(np.rad2deg(x_updated[3]))  # convert back to degrees for logging
    #print(f"[EKF] Updated state: x={x_updated[0]:.2f}, y={x_updated[1]:.2f}, v={x_updated[2]:.2f} px/s, phi={(x_updated[3]):.2f} deg")
    #EKF done
    t_end = time.perf_counter()

    meters_pr_pixel = MAP_M_PER_PX_GLOBAL
    pose_ekf_px = (x_updated[0], x_updated[1])

    diff_px_ekf = (pose_ekf_px[0] - GT_centre_x_px_tile,
                   pose_ekf_px[1] - GT_centre_y_px_tile)
    diff_px_meas = (meas_x_px_global - GT_centre_x_px_tile,
                    meas_y_px_global - GT_centre_y_px_tile)

    dx_ekf = diff_px_ekf[0] * meters_pr_pixel
    dy_ekf = diff_px_ekf[1] * meters_pr_pixel
    error_ekf = math.hypot(dx_ekf, dy_ekf)
    dphi_ekf = 0.0  # TODO if needed

    dx_meas = diff_px_meas[0] * meters_pr_pixel
    dy_meas = diff_px_meas[1] * meters_pr_pixel
    error_meas = math.hypot(dx_meas, dy_meas)
    dphi_meas = 0.0  # TODO if needed

    # predicted (before update) error relative to same tile center
    diff_px_pred = (x_pred[0] - GT_centre_x_px_tile,
                    x_pred[1] - GT_centre_y_px_tile)
    dx_pred = diff_px_pred[0] * meters_pr_pixel
    dy_pred = diff_px_pred[1] * meters_pr_pixel
    error_pred = math.hypot(dx_pred, dy_pred)


    #print(f"[Metrics] Mean Error: {error}m, dx: {dx}m, dy: {dy}m")
    #print(f"[Metrics] Mean Error (EKF): {error_ekf}m, dx: {dx_ekf}m, dy: {dy_ekf}m")

    if visualisations_enabled: 
        # ------------- Overlays on SATELLITE image -------------------- 
        # map to DOWNSCALED satellite coords for visualization
        corners_disp = (corners_meas_in_tile_px * SX) + np.array(pixel_offset)  # shift to display coords
        center_measurement  = round(meas_x_px_global) * SX + pixel_offset[0], round(meas_y_px_global) * SY + pixel_offset[1]
        center_ekf = np.array([x_updated[0] * SX + pixel_offset[0], x_updated[1] * SY + pixel_offset[1]])
        center_gt = np.array([GT_centre_x_px_tile * SX + pixel_offset[0], GT_centre_y_px_tile * SY + pixel_offset[1]])
        center_pred = np.array([x_pred[0] * SX + pixel_offset[0], x_pred[1] * SY + pixel_offset[1]])

        for i in range(2):
            if i == 0: # individual overlay for this tile
                sat_individual = cv2.imread(str(SAT_IMG), cv2.IMREAD_COLOR)
                out_overlay = OUT_DIR / f"top1_{tile_name}_overlay_on_sat.png"
                #BGR colors
                color = [(255,255,255), (255, 0, 0), (0, 255, 0), (0, 0, 255),(0,255,255)]  # Blue: GT, Green: EKF, Red: Meas, Yellow: Pred, Cyan: overall

                # same center as we use color to destingish
                label_point(sat_individual, center_gt, "Pred", color[4], offset=(100, 40), font_scale=0.5, thickness=1)
                label_point(sat_individual, center_gt, "GT", color[1], offset=(100, 20), font_scale=0.5, thickness=1)
                label_point(sat_individual, center_gt, "EKF", color[2], offset=(100, 0), font_scale=0.5, thickness=1)
                label_point(sat_individual, center_gt, "Meas", color[3], offset=(100, -20), font_scale=0.5, thickness=1)
            else:
                out_overlay = OUT_OVERALL_SAT_VIS_PATH
                overlay_img = cv2.imread(str(out_overlay))
                random_color = tuple(random.randint(128, 255) for _ in range(3))
                color = [random_color, (255, 0, 0), (0, 255, 0), (0, 0, 255), (0,255,255)]   # Random for image. Red: GT, Green: EKF, Blue: Meas, Yellow: Pred

                if overlay_img is None:
                    overlay_img = cv2.imread(str(SAT_IMG), cv2.IMREAD_COLOR)  # use base if file doesn't exist
                    # make label colors clear in img by text
                    label_point(overlay_img, [10,70], "Pred", color[4], offset=(0, 0), font_scale=3, thickness=3)
                    label_point(overlay_img, [300,70], "GT", color[1], offset=(0, 0), font_scale=3, thickness=3)
                    label_point(overlay_img, [450,70], "EKF", color[2], offset=(0, 0), font_scale=3, thickness=3)
                    label_point(overlay_img, [700,70], "Meas", color[3], offset=(0, 0), font_scale=3, thickness=3)
                    
                sat_individual = overlay_img

            # Overlay
            #draw_polygon(sat_individual, corners_disp, color=color[0], thickness=1)
            draw_point(sat_individual, center_pred, color=color[4], r=1)
            draw_point(sat_individual, center_measurement, color=color[3], r=1)
            draw_point(sat_individual, center_gt, color=color[1], r=1)
            draw_point(sat_individual, center_ekf, color=color[2], r=1)
            # P_pred is in ORIGINAL sat px, so we need to scale it for display sat px
            Sigma_orig = P_pred[:2, :2] * SX
            draw_ellipse(sat_individual, center_pred, Sigma_orig, k_sigma=k, color=color[0], thickness=1)

            cv2.imwrite(str(out_overlay), sat_individual)

    # -------------------- Read top-1 candidate scores needed for CSV --------------------            
    num_inliers = top1["inliers"]
    K = top1["total_matches"]
    LG_confidence = top1["LG_conf"]
    median_reproj_error_px = top1["median_err"]
    shape_conf = top1["shape_score"]

    dx_pred = x_pred[0] * meters_pr_pixel
    dy_pred = x_pred[1] * meters_pr_pixel
    error_pred = math.hypot(dx_pred, dy_pred)

    # pts0 = drone coordinates of inlier matches
    x = best_pts_drone_np[:, 0]
    y = best_pts_drone_np[:, 1]

    spread_x = (x.max() - x.min()) / img.shape[:2][1]  # normalize by width
    spread_y = (y.max() - y.min()) / img.shape[:2][0]  # normalize by height

    sds = np.sqrt(spread_x * spread_y)  # geometric mean

    #------------- save final results to overall CSV file --------------------
    # add to the bottom of results_{sat_number}.csv file:
    with open(CSV_FINAL_RESULT_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            drone_img,
            num_feats_in_drone,
            Path(tile_name).stem,
            K,
            len(selected_tiles),
            num_inliers,
            f"{LG_confidence:.3f}",
            f"{median_reproj_error_px:.3f}",
            shape_terms_str,
            f"{shape_conf:.3f}",
            f"{best_conf_overall:.3f}",
            f"{meas_x_px_global:.8f}", f"{meas_y_px_global:.8f}", f"{meas_phi_deg:.3f}",
            f"{x_updated[0]:.8f}", f"{x_updated[1]:.8f}", f"{x_updated[3]:.3f}",
            f"{dx_meas:.4f}", f"{dy_meas:.4f}",
            f"{dx_ekf:.4f}", f"{dy_ekf:.4f}",
            f"{error_meas:.4f}", f"{error_pred:.4f}", f"{error_ekf:.4f}",
            f"{dphi_meas:.4f}", f"{dphi_ekf:.4f}",
            f"{t_end-t_start:.4f}",
            f"{sds:.4f}"
        ])

results = get_metrics(CSV_FINAL_RESULT_PATH)
print(results)
