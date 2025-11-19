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
DISPLAY_LONG_SIDE = 1200      # only for visualization
MAX_KPTS = None               # max keypoints to load from .pt files (None = all) (This controls memory usage) TODO also controls speed
MAX_BATCH_TILES = 10          # max number of tiles to process in one batch for common-K trimming

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
SAT_DIR   = DATASET_DIR / sat_number / "sat_tiles_overlap_scaled" 
OUT_DIR_CLEAN   = BASE / "outputs" / sat_number

# delete folder if exists
folder = OUT_DIR_CLEAN

if folder.exists() and folder.is_dir():
    shutil.rmtree(folder)

OUT_DIR_CLEAN.mkdir(parents=True, exist_ok=True)
CSV_FINAL_RESULT_PATH = OUT_DIR_CLEAN / f"results_{sat_number}.csv"
TILE_PT_DIR = DATASET_DIR / sat_number / f"{FEATURES}_features" / sat_number
SAT_DISPLAY_IMG  = DATASET_DIR / sat_number / "satellite03_small.png"
SAT_DISPLAY_META = SAT_DISPLAY_IMG.with_suffix(SAT_DISPLAY_IMG.suffix + ".json")  # {"scale": s, "original_size_hw":[H,W],...}
TILE_WH_DIR = SAT_DIR / "a_tile_size.txt"  


############################ Read tile information ############################
# a_tile_size.txt format:
#   stride_h stride_w tile_h_sat tile_w_sat H_drone W_drone scale_sat_to_drone
with open(TILE_WH_DIR) as f:
    sh_str, sw_str, h_sat_str, w_sat_str, h_drone_str, w_drone_str, scale_str = f.read().strip().split()

STRIDE_Y = int(sh_str)         # stride in ORIGINAL sat px (vertical)
STRIDE_X = int(sw_str)         # stride in ORIGINAL sat px (horizontal)

TILE_H_ORIGINAL  = int(h_sat_str)       # tile height in ORIGINAL sat px
TILE_W_ORIGINAL  = int(w_sat_str)       # tile width  in ORIGINAL sat px

TILE_H_RESCALED = int(h_drone_str)  # expected drone / resized-tile height
TILE_W_RESCALED = int(w_drone_str)  # expected drone / resized-tile width

SCALE_SAT_TILE_ORG_TO_RESCALED = float(scale_str)  # tile_px = sat_original_px * SCALE_SAT_TILE_ORG_TO_RESCALED

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
    """

    def __init__(
        self,
        x0, P0, Q0):
        self.x = np.array(x0, dtype=float).reshape(-1)
        self.x[3] = _wrap_pi(self.x[3])
        self.P = np.array(P0, dtype=float).reshape(5, 5)

        # Process noise scales
        self.sigma_pos_proc = Q0[0][0]
        self.sigma_speed    = Q0[1][1]
        self.sigma_phi      = Q0[2][2]
        self.sigma_b_phi    = Q0[3][3]

    def _f(self, x, dt):
        X, Y, V, PHI, B = x
        c, s = np.cos(PHI), np.sin(PHI)
        Xn = X + V * c * dt
        Yn = Y + V * s * dt
        Vn = V
        PHIn = _wrap_pi(PHI)
        Bn = B
        return np.array([Xn, Yn, Vn, PHIn, Bn], dtype=float)

    def _F_jac(self, x, dt):
        _, _, V, PHI, _ = x
        c, s = np.cos(PHI), np.sin(PHI)
        F = np.eye(5)
        F[0, 2] = c * dt
        F[0, 3] = -V * s * dt
        F[1, 2] = s * dt
        F[1, 3] =  V * c * dt
        return F

    def _Q_proc(self, dt):
        qx = (self.sigma_pos_proc * np.sqrt(dt))**2
        qy = (self.sigma_pos_proc * np.sqrt(dt))**2
        qv = (self.sigma_speed)**2
        qf = (self.sigma_phi * np.sqrt(dt))**2
        qb = (self.sigma_b_phi * np.sqrt(dt))**2
        return np.diag([qx, qy, qv, qf, qb])

    def predict(self, dt):
        dt = float(max(1e-6, dt))
        self.x = self._f(self.x, dt)
        self.x[3] = _wrap_pi(self.x[3])

        F = self._F_jac(self.x, dt)
        Q = self._Q_proc(dt)
        self.P = F @ self.P @ F.T + Q

        return self.x.copy(), self.P.copy()
    
    def update_pos_heading(self, z_xyphi, R_xyphi):
        z = np.asarray(z_xyphi, dtype=float).reshape(3)
        H = np.array([[1,0,0,0, 0],
                      [0,1,0,0, 0],
                      [0,0,0,1, 1]], dtype=float)

        z_pred = np.array([self.x[0], self.x[1], _wrap_pi(self.x[3] + self.x[4])], dtype=float)
        y = z - z_pred
        y[2] = _wrap_pi(y[2])

        S = H @ self.P @ H.T + R_xyphi
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[3] = _wrap_pi(self.x[3])

        I = np.eye(5)
        self.P = (I - K @ H) @ self.P
        return self.x.copy(), self.P.copy()

    @staticmethod
    def R_from_conf(pos_base_std, heading_base_std_rad, overall_conf,
                    pos_min_scale, pos_max_scale,
                    heading_min_scale, heading_max_scale ): 
        overall_conf = float(np.clip(overall_conf, 0.0, 1.0))
        scale_pose = np.interp(overall_conf, [0.0, 1.0], [pos_max_scale, pos_min_scale]) 
        scale_heading = np.interp(overall_conf, [0.0, 1.0], [heading_max_scale, heading_min_scale]) 
        Rx = (pos_base_std * scale_pose)**2
        Ry = (pos_base_std * scale_pose)**2
        Rphi = (heading_base_std_rad * scale_heading)**2
        return np.diag([Rx, Ry, Rphi]).astype(float)


# -------------------- Angle wrapping --------------------
def wrap_deg(a):
    return ((a + 180.0) % 360.0) - 180.0

def img_to_compass(phi_img_deg):
    return wrap_deg(phi_img_deg + 90.0)

def compass_to_img(psi_deg):
    return wrap_deg(psi_deg - 90.0)

def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

# -------------------- Helpers --------------------
def get_median_projection_error(H, pts0, pts1, inlier_mask):
    if H is None or inlier_mask is None or not inlier_mask.any():
        return 0.0

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

def determine_pos_error(pose, heading_deg, DRONE_INFO_DIR, drone_img):
    pose = np.array(pose, dtype=np.float64)

    gt_lat = gt_lon = None
    with open(DRONE_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["filename"] == f"{drone_img}":
                gt_lat = np.float64(r["lat"])
                gt_lon = np.float64(r["lon"])
                gt_heading = np.float64(r["Phi1"]) 
                break
    if gt_lat is None or gt_lon is None:
        raise ValueError(f"GT lat/lon not found for {drone_img}")
    
    gt_pose_px = latlon_to_orig_xy(gt_lat, gt_lon, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)  

    difference = pose - np.array([gt_lat, gt_lon], dtype=np.float64)
    mean_lat = np.radians((pose[0] + gt_lat) / np.float64(2.0))

    meters_per_degree_lat = np.float64(111_320.0)
    meters_per_degree_lon = meters_per_degree_lat * np.cos(mean_lat)

    dy = difference[0] * meters_per_degree_lat
    dx = difference[1] * meters_per_degree_lon

    total_error_m = np.sqrt(dx**2 + dy**2)

    dphi = wrap_deg(float(heading_deg - gt_heading))
    return total_error_m, dx, dy, dphi, gt_pose_px

def get_R_rotated_by_phi1(phi_rad: float, W_orig: int, H_orig: int) -> np.ndarray:
    c, s = np.cos(phi_rad), np.sin(phi_rad)
    cx, cy = (W_orig - 1) * 0.5, (H_orig - 1) * 0.5
    T_to   = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], dtype=np.float64)
    R      = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
    T_back = np.array([[1,0,cx],[0,1,cy],[0,0,1]], dtype=np.float64)
    return T_back @ R @ T_to

def get_phi_deg(DRONE_INFO_DIR, drone_img,):
    with open(DRONE_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["filename"] == f"{drone_img}":
                for key in ("Phi1", "heading", "yaw_deg"):
                    if key in r and r[key] not in (None, "", "nan"):
                        return float(r[key])
                raise ValueError(f"No heading (Phi1/heading/yaw_deg) for {drone_img}")
    raise FileNotFoundError(f"{drone_img} not found in {DRONE_INFO_DIR}")

def latlon_to_orig_xy(lat, lon, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META):
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

def get_visualisation_parameters(H_orig2tile, TILE_SCALE, DRONE_ORIGINAL_W, DRONE_ORIGINAL_H, x_off, y_off):
    w0, h0 = float(DRONE_ORIGINAL_W), float(DRONE_ORIGINAL_H)

    center0  = np.array([[w0/2.0, h0/2.0]], dtype=np.float32)
    forward0 = np.array([[w0/2.0, h0/2.0 - max(20.0, 0.10*h0)]], dtype=np.float32)

    center_tile_px  = project_pts(H_orig2tile, center0)[0].astype(np.float64)
    forward_tile_px = project_pts(H_orig2tile, forward0)[0].astype(np.float64)

    center_original_tile  = center_tile_px  / TILE_SCALE
    forward_original_tile = forward_tile_px / TILE_SCALE

    offset = np.array([x_off, y_off], dtype=np.float64)
    center_global  = center_original_tile  + offset
    forward_global = forward_original_tile + offset

    v_pred = forward_global - center_global
    norm   = np.hypot(v_pred[0], v_pred[1]) or 1.0
    heading_unitvec_meas = v_pred / norm

    corners0 = np.array([[0,0],[w0,0],[w0,h0],[0,h0]], dtype=np.float32)
    center0c = np.array([[w0/2, h0/2]], dtype=np.float32)

    corners_tile_px = project_pts(H_orig2tile, corners0)
    center_tile_px2 = project_pts(H_orig2tile, center0c)[0]

    corners_local = corners_tile_px / TILE_SCALE
    corners_global = corners_local + offset

    return center_global, corners_global, heading_unitvec_meas


def get_measurements(center_global, heading_unitvector_from_homography):
    meas_x, meas_y = float(center_global[0]), float(center_global[1])
    meas_phi_rad = float(np.arctan2(heading_unitvector_from_homography[1], heading_unitvector_from_homography[0]))
    meas_phi_deg = float(np.degrees(meas_phi_rad))
    meas_phi_deg = img_to_compass(meas_phi_deg)
    return meas_phi_deg, (meas_x, meas_y)

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
    x, y = int(pt[0]), int(pt[1])
    ox, oy = offset
    tx, ty = x + ox, y + oy

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
    I0d, s0 = resize_for_display(np.clip(I0, 0.0, 1.0))
    I1d, s1 = resize_for_display(np.clip(I1, 0.0, 1.0))
    p0d = pts0 * s0
    p1d = pts1 * s1

    H0, W0 = I0d.shape[:2]
    H1, W1 = I1d.shape[:2]
    canvas = np.ones((max(H0, H1), W0 + W1, 3), dtype=I0d.dtype)
    canvas[:H0, :W0] = I0d
    canvas[:H1, W0:W0 + W1] = I1d
    canvas = np.clip(canvas, 0.0, 1.0)

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

def get_confidence_meas(num_inliers, avg_conf, median_err_px,
                        s_err=3, w=(0.5, 0.5)):
    if avg_conf <= 0 or num_inliers <= 0 or not np.isfinite(median_err_px):
        return 0.0
    err_score    = np.exp(- (median_err_px / s_err)**2)
    w_avg_c, w_err = w
    return float(np.clip(w_avg_c*avg_conf + w_err*err_score, 0.0, 1.0))

# ---------------------- homography convexity check ----------------------
def is_homography_convex(H, img_w, img_h):
    if H is None:
        return False
    
    corners0 = np.array([
        [0,      0     ],
        [img_w,  0     ],
        [img_w,  img_h ],
        [0,      img_h ],
    ], dtype=np.float32)

    corners_warped = project_pts(H, corners0)

    if not np.isfinite(corners_warped).all():
        return False

    cnt = corners_warped.reshape(-1, 1, 2).astype(np.float32)
    is_convex = bool(cv2.isContourConvex(cnt))

    return is_convex


# ---------------------- search region ----------------------
def ellipse_from_cov(mu_xy, Sigma_xy, k=2.0):
    vals, vecs = np.linalg.eigh(Sigma_xy)
    a = k * np.sqrt(vals[1])
    b = k * np.sqrt(vals[0])
    angle_rad = np.arctan2(vecs[1,1], vecs[0,1])
    return a, b, angle_rad, mu_xy

def ellipse_bbox(mu_xy, Sigma_xy, k=2.0, n=72):
    a,b,theta,mu = ellipse_from_cov(mu_xy, Sigma_xy, k)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.stack([a*np.cos(t), b*np.sin(t)], axis=0)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    pts_world = (R @ pts).T + mu[None,:]
    x0,y0 = pts_world.min(axis=0)
    x1,y1 = pts_world.max(axis=0)
    return (x0,y0,x1,y1)

def tiles_in_bbox(bbox, TILE_W, TILE_H, all_tile_names):
    x_min, y_min, x_max, y_max = bbox
    selected_tiles = []
    for tile_path in all_tile_names:
        x_off, y_off = tile_offset_from_name(tile_path)
        tile_bbox = (x_off, y_off, x_off + TILE_W, y_off + TILE_H)
        if not (tile_bbox[2] < x_min or tile_bbox[0] > x_max or
                tile_bbox[3] < y_min or tile_bbox[1] > y_max):
            selected_tiles.append(tile_path)
    return selected_tiles 

###################################################################################################
# --------- helper: trim TILE features to K_common by removing lowest-confidence keypoints -------
###################################################################################################
def trim_single_feats_to_K(feats_b, K_keep: int):
    """
    Trim TILE features to K_keep keypoints (top by keypoint_scores).

    feats_b: dict with keys
        'keypoints'       [1, K, 2]
        'descriptors'     [1, K, D]  OR  [1, D, K]
        'keypoint_scores' [1, K]
        'image_size'      [1, 2]

    We NEVER touch the drone features, only tiles.
    """
    scores = feats_b["keypoint_scores"]  # [1, K]
    B, K_full = scores.shape
    if K_keep >= K_full:
        # nothing to trim
        return {
            "keypoints":       feats_b["keypoints"].clone(),
            "descriptors":     feats_b["descriptors"].clone(),
            "keypoint_scores": feats_b["keypoint_scores"].clone(),
            "image_size":      feats_b["image_size"].clone(),
        }

    # top-K_keep by score (largest = best)
    # scores: [1, K_full]  -> idx: [1, K_keep]
    vals, idx = torch.topk(scores, k=K_keep, dim=1, largest=True, sorted=False)

    # ---- keypoints: [1, K, 2] -> gather along dim=1 ----
    keypoints = feats_b["keypoints"].gather(
        1, idx.unsqueeze(-1).expand(-1, -1, 2)
    )  # [1, K_keep, 2]

    # ---- descriptors: can be [1, K, D] OR [1, D, K] ----
    desc = feats_b["descriptors"]
    if desc.dim() != 3:
        raise ValueError(f"Unexpected descriptors shape: {desc.shape}")

    # case 1: [1, K, D]
    if desc.shape[1] == K_full:
        D = desc.shape[2]
        # gather along dim=1 (keypoint dim)
        descriptors = desc.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, D)
        )  # [1, K_keep, D]

    # case 2: [1, D, K]
    elif desc.shape[2] == K_full:
        D = desc.shape[1]
        # gather along dim=2 (keypoint dim)
        descriptors = desc.gather(
            2, idx.unsqueeze(1).expand(-1, D, -1)
        )  # [1, D, K_keep]

    else:
        raise ValueError(
            f"Descriptors shape {desc.shape} not compatible with scores shape {scores.shape}"
        )

    keypoint_scores = vals  # [1, K_keep]

    return {
        "keypoints":       keypoints,
        "descriptors":     descriptors,
        "keypoint_scores": keypoint_scores,
        "image_size":      feats_b["image_size"].clone(),
    }

###################################################################################################

# -------------------- Device & models --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} for feature extraction and matching using {FEATURES}.")

feat = FEATURES.lower()
if feat not in ("superpoint", "disk", "sift", "aliked"):
    raise ValueError("FEATURES must be 'superpoint', 'disk', 'sift', or 'aliked'.")

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
    w.writerow(["drone_image", "tile", "total_matches", "search_tiles", "inliers", "avg_confidence", 
                "median_reproj_error_px", "overall_confidence", 
                "x_meas", "y_meas", "phi_meas_deg",
                "x_ekf", "y_ekf", "phi_ekf_deg",
                "dx", "dy", 
                "dx_ekf", "dy_ekf",
                "error", "ekf_error", 
                "heading_diff", "ekf_heading_diff", 
                "time_s", "ekf_time_s"])

# -------------------- Preload satellite tiles --------------------              
all_tile_names = [p for p in SAT_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".png"]

k = math.sqrt(chi2.ppf(0.99, df=2))
sat_vis, SX, SY = load_sat_display_and_scale()

# -------------------- Main loop over drone images --------------------
for i, img_path in enumerate(sorted(DRONE_IMG_CLEAN.iterdir())):
    if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
        continue
    drone_img = img_path.name

    if drone_img in starting_drone_images:
        meas_phi_deg = None
        ekf = None
        t_last = None

    DRONE_IMG = DRONE_IMG_CLEAN / str(drone_img)
    if visualisations_enabled:
        OUT_DIR = OUT_DIR_CLEAN / str(drone_img)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CSV_RESULT_PATH  = OUT_DIR / "results.csv" 
    if not DRONE_IMG.exists():
        raise FileNotFoundError(f"Missing drone image: {DRONE_IMG}")
    
    # --------------- EKF initialisation -----------------
    if ekf is None:
        with open(DRONE_INFO_DIR, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if row["filename"] == str(drone_img):
                    starting_position_latlon = (np.float64(row["lat"]), np.float64(row["lon"]))
                    starting_position_xy = latlon_to_orig_xy(starting_position_latlon[0], starting_position_latlon[1], SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
                    t_current = datetime.fromisoformat(row["date"])
                    t_last = t_current

                    phi_deg0 = np.float64(row["Phi1"])
                    phi0_rad = np.deg2rad(compass_to_img(phi_deg0))

                    next_row = next(r)
                    lat1 = float(next_row["lat"])
                    lon1 = float(next_row["lon"])
                    k1_position_xy = latlon_to_orig_xy(lat1, lon1, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)

                    t1 = datetime.fromisoformat(next_row["date"])
                    dt = (t1 - t_current).total_seconds()
                    if dt > 0:
                        vel_x = (k1_position_xy[0] - starting_position_xy[0]) / dt
                        vel_y = (k1_position_xy[1] - starting_position_xy[1]) / dt
                        vel0 = np.sqrt(vel_x**2 + vel_y**2)
                    else:
                        vel0 = 50.0
                    break

        x0 = np.array([starting_position_xy[0], starting_position_xy[1], vel0, phi0_rad, np.deg2rad(0.0)], dtype=np.float64)

        P0 = np.diag([(50.0)**2,
                      (50.0)**2,
                      (3.0)**2,
                      np.deg2rad(9.0)**2,
                      np.deg2rad(9.0)**2])

        Q0 = np.diag([3.0,
                      0.5,
                      np.deg2rad(1),
                      np.deg2rad(0.0025)])

        ekf = EKF_ConstantVelHeading(x0, P0, Q0)
        continue

    # ---------------------------- EKF prediction + ellipse -------------------
    with open(DRONE_INFO_DIR, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["filename"] == str(drone_img):
                t_current = datetime.fromisoformat(row["date"])
                if t_last is not None:
                    dt = (t_current - t_last).total_seconds()
                t_last = t_current

    x_pred, _ = ekf.predict(dt)
    x_pred[3] = img_to_compass(np.rad2deg(x_pred[3]))

    P_pred = ekf.P
    sigma = P_pred[:2, :2]
    ellipse_bbox_coords = ellipse_bbox(x_pred[:2], sigma, k=k, n=72)

    selected_tiles = tiles_in_bbox(ellipse_bbox_coords, TILE_W_ORIGINAL, TILE_H_ORIGINAL, all_tile_names)

    # -------------------- Rotate drone & extract features (drone untouched later) -----------
    if feat == "sift":
        R_orig2rot = np.eye(3, dtype=np.float64)
        with torch.inference_mode():
            img0_t = load_image(str(DRONE_IMG)).to("cpu")
            feats_drone_b = extractor.extract(img0_t)
            feats_drone_r = rbd(feats_drone_b)
        bgr_rot = cv2.imread(str(DRONE_IMG), cv2.IMREAD_COLOR)
        drone_rot_size = (bgr_rot.shape[1], bgr_rot.shape[0])
        DRONE_IMG_FOR_VIZ = DRONE_IMG
    else:
        if meas_phi_deg is None:
            phi_deg_flip = get_phi_deg(DRONE_INFO_DIR, drone_img)
        else:
            phi_deg_flip = meas_phi_deg

        img = cv2.imread(str(DRONE_IMG), cv2.IMREAD_COLOR)
        H_img, W_img = img.shape[:2]
        a = -phi_deg_flip
        r_ang = math.radians(a)
        c, s = abs(math.cos(r_ang)), abs(math.sin(r_ang))
        newW = int(math.ceil(W_img*c + H_img*s))
        newH = int(math.ceil(W_img*s + H_img*c))

        M = cv2.getRotationMatrix2D((W_img/2.0, H_img/2.0), a, 1.0)
        M[0, 2] += (newW - W_img) / 2.0
        M[1, 2] += (newH - H_img) / 2.0

        bgr_rot = cv2.warpAffine(img, M, (newW, newH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        R_orig2rot = np.eye(3, dtype=np.float64)
        R_orig2rot[:2, :3] = M               

        hR, wR = bgr_rot.shape[:2]
        drone_rot_size = (wR, hR)
        if visualisations_enabled:
            DRONE_IMG_ROT_PATH = OUT_DIR / f"drone_rot_{a}.png"
            DRONE_IMG_FOR_VIZ = DRONE_IMG_ROT_PATH
            cv2.imwrite(str(DRONE_IMG_ROT_PATH), bgr_rot)

        with torch.inference_mode():
            rgb = cv2.cvtColor(bgr_rot, cv2.COLOR_BGR2RGB)
            img_np = rgb.astype(np.float32) / 255.0
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

            feats_drone_b = extractor.extract(img_t)
            feats_drone_r = rbd(feats_drone_b)

        hR, wR = bgr_rot.shape[:2]
        drone_rot_size = (wR, hR)

    if not selected_tiles:
        raise FileNotFoundError(f"No PNG tiles in {SAT_DIR}")

    scores_small = []
    H_best = None
    best_conf = 0.0
    best_pts_drone_np = None
    best_pts_tile_np = None
    best_inlier_mask = None

    with torch.inference_mode():
        # preload tile features
        tile_feats_list = []
        for p in selected_tiles:
            tile_pt = make_feature_pt_path_for(p)

            if tile_pt.exists():
                feats_tile_b, feats_tile_r = load_feats_pt_batched(tile_pt, device if feat != "sift" else "cpu")
            else:
                if extractor is None:
                    raise FileNotFoundError(f"Missing {tile_pt} and no extractor available.")
                img1_t  = load_image(str(p)).to(device if feat != "sift" else "cpu")
                feats_tile_b = extractor.extract(img1_t)
                feats_tile_r = rbd(feats_tile_b)

            tile_feats_list.append((p, feats_tile_b, feats_tile_r))

        # process tiles in batches for K_common (only tiles; drone untouched)
        for batch_start in range(0, len(tile_feats_list), MAX_BATCH_TILES):
            batch = tile_feats_list[batch_start:batch_start + MAX_BATCH_TILES]

            # find K_common over tiles in this batch
            K_values = [fb["keypoints"].shape[1] for (_, fb, _) in batch]
            K_common = int(min(K_values))

            for (p, feats_tile_b, feats_tile_r) in batch:
                num_inliers = 0
                avg_conf = float("nan")
                median_projection_err = float("inf")   
                overall_conf = 0.0
                K_matches = 0
                H = None
                H_ransac = None
                H_dlt = None
                inlier_mask = None

                # trim TILE only
                feats_tile_b_trim = trim_single_feats_to_K(feats_tile_b, K_common)
                feats_tile_r_trim = rbd(feats_tile_b_trim)

                # match single pair (drone untouched)
                matches01 = matcher({"image0": feats_drone_b, "image1": feats_tile_b_trim})
                matches01_r = rbd(matches01)

                matches = matches01_r.get("matches", None)
                if matches is not None and matches.numel() > 0:
                    K_matches = int(matches.shape[0])
                else:
                    scores_small.append({
                        "tile": p,
                        "inliers": 0,
                        "total_matches": 0,
                        "avg_conf": float("nan"),
                        "median_err": float("inf"),  
                        "overall_conf": 0.0,
                        "sort_key": (0.0, 0),
                    })
                    continue

                if K_matches >= 4:
                    pts_drone_np = feats_drone_r["keypoints"][matches[:, 0]].detach().cpu().numpy()
                    pts_tile_np = feats_tile_r_trim["keypoints"][matches[:, 1]].detach().cpu().numpy()

                    H_ransac, mask = cv2.findHomography(
                        pts_drone_np, pts_tile_np, method=cv2.USAC_MAGSAC,
                        ransacReprojThreshold=3.0, confidence=0.999
                    ) 

                    if not is_homography_convex(H_ransac, drone_rot_size[0], drone_rot_size[1]):
                        scores_small.append({
                            "tile": p,
                            "inliers": 0,
                            "total_matches": K_matches,
                            "avg_conf": float("nan"),
                            "median_err": float("inf"),  
                            "overall_conf": 0.0,
                            "sort_key": (0.0, 0),
                        })
                        continue

                    if mask is not None:
                        inlier_mask = mask.ravel().astype(bool)
                        num_inliers = int(inlier_mask.sum())

                        scores_t = matches01_r.get("scores", None)
                        if scores_t is not None and num_inliers > 0:
                            scores_np = scores_t.detach().cpu().numpy()
                            avg_conf = float(np.mean(scores_np[inlier_mask]))
                    
                    if H_ransac is not None and num_inliers >= 4:
                        H_dlt, _ = cv2.findHomography(pts_drone_np[inlier_mask], pts_tile_np[inlier_mask], method=0)
                        
                    H_candidates = [H_ransac, H_dlt]
                    best_median_err = float("inf")
                    for H_cand in H_candidates:
                        median_err_cand = get_median_projection_error(H_cand, pts_drone_np, pts_tile_np, inlier_mask)
                        if median_err_cand < best_median_err:
                            best_median_err = median_err_cand
                            H = H_cand

                    if H is not None:
                        median_projection_err = best_median_err
                        overall_conf = get_confidence_meas(num_inliers, avg_conf if not math.isnan(avg_conf) else 0.0, median_projection_err)
                        if overall_conf > best_conf:
                            H_best = H
                            best_conf = overall_conf
                            best_pts_drone_np = pts_drone_np
                            best_pts_tile_np = pts_tile_np
                            best_inlier_mask = inlier_mask
                            
                scores_small.append({
                    "tile": p,
                    "inliers": num_inliers,
                    "total_matches": K_matches,
                    "avg_conf": avg_conf,
                    "median_err": median_projection_err,  
                    "overall_conf": overall_conf,
                    "sort_key": (overall_conf, num_inliers),
                })

    # -------------------- Rank tiles --------------------
    scores_small.sort(key=lambda d: d["sort_key"], reverse=True)
    if visualisations_enabled:
        with open(CSV_RESULT_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tile", "total_matches", "inliers", "avg_confidence", "median_reproj_error", "overall_conf"])
            for r in scores_small:
                w.writerow([
                    r["tile"].name, 
                    r["total_matches"],
                    r["inliers"],
                    "" if (isinstance(r["avg_conf"], float) and math.isnan(r["avg_conf"])) else f"{r['avg_conf']:.4f}", 
                    "" if (isinstance(r["median_err"], float) and (math.isnan(r["median_err"]) or math.isinf(r["median_err"]))) else f"{r['median_err']:.4f}", 
                    "" if (isinstance(r["overall_conf"], float) and math.isnan(r["overall_conf"])) else f"{r['overall_conf']:.4f}"
                ])

    top1 = scores_small[:1]
    rank_colors = [(0,255,0)]

    sat_base = sat_vis.copy()

    _bgr_orig = cv2.imread(str(DRONE_IMG), cv2.IMREAD_COLOR)
    H_orig, W_orig = _bgr_orig.shape[:2]

    if H_best is None:
        pose_ekf = get_location_in_sat_img((x_pred[0], x_pred[1]), SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
        error_ekf, dx_ekf, dy_ekf, dphi_ekf, _ = determine_pos_error(pose_ekf, x_pred[3], DRONE_INFO_DIR, drone_img)
        with open(CSV_FINAL_RESULT_PATH, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([drone_img, "N/A", "N/A", len(selected_tiles), "N/A", "N/A",
                        "N/A", "N/A",
                        "N/A", "N/A", "N/A",
                        x_pred[0], x_pred[1], x_pred[3],
                        "N/A", "N/A",
                        dx_ekf, dy_ekf,
                        "N/A", error_ekf,
                        "N/A", dphi_ekf,
                        "N/A", "N/A"])
        continue

    for rank, r in enumerate(top1, 1): 
        tile_name = r["tile"]
        color = rank_colors[rank-1]
        
        H_rot2tile = H_best
        H_orig2tile = H_rot2tile @ R_orig2rot      

        if visualisations_enabled and best_pts_drone_np is not None:
            out_match_png = OUT_DIR / f"top{rank:02d}_{tile_name.stem}_matches.png"
            visualize_inliers(DRONE_IMG_FOR_VIZ, tile_name, best_pts_drone_np, best_pts_tile_np, best_inlier_mask, str(out_match_png))

        x_off, y_off = tile_offset_from_name(tile_name)
        center_global, corners_global, heading_unitvector_measurement = get_visualisation_parameters(
            H_orig2tile, SCALE_SAT_TILE_ORG_TO_RESCALED, W_orig, H_orig, x_off, y_off
        )

        if rank == 1:
            num_inliers = r["inliers"]
            K_matches = r["total_matches"]
            avg_confidence = r["avg_conf"]
            median_reproj_error_px = r["median_err"]
            overall_confidence = r["overall_conf"]

            R_meas = ekf.R_from_conf(
                pos_base_std=50.0,
                heading_base_std_rad=np.deg2rad(15.0),
                overall_conf=overall_confidence,
                pos_min_scale=0.3,
                pos_max_scale=2.0,
                heading_min_scale=1,
                heading_max_scale=4
            )

            meas_phi_deg, (meas_x_px, meas_y_px) = get_measurements(center_global, heading_unitvector_measurement)

            x_updated, _ = ekf.update_pos_heading(
                [meas_x_px, meas_y_px, np.deg2rad(compass_to_img(meas_phi_deg))],
                R_meas
            )

            x_updated[3] = img_to_compass(np.rad2deg(x_updated[3]))

            lat_long_pose_estimated = get_location_in_sat_img(center_global, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
            error, dx, dy, dphi, _ = determine_pos_error(lat_long_pose_estimated, meas_phi_deg, DRONE_INFO_DIR, drone_img)

            ekf_pose_lat_long = get_location_in_sat_img((x_updated[0], x_updated[1]), SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
            error_ekf, dx_ekf, dy_ekf, dphi_ekf, gt_pose_px = determine_pos_error(ekf_pose_lat_long, x_updated[3], DRONE_INFO_DIR, drone_img)

            if visualisations_enabled: 
                corners_disp = corners_global * np.array([SX, SY], np.float32)
                center_measurement  = center_global  * np.array([SX, SY], np.float32)
                center_ekf = np.array([x_updated[0]*SX, x_updated[1]*SY], np.float32)
                center_gt = np.array([gt_pose_px[0]*SX, gt_pose_px[1]*SY], np.float32)
                center_pred = np.array([x_pred[0]*SX, x_pred[1]*SY], np.float32)

                for i_vis in range(2):
                    if i_vis == 0:
                        sat_individual = sat_base.copy()
                        out_overlay = OUT_DIR / f"top{rank:02d}_{tile_name.stem}_overlay_on_sat.png"
                        color_set = [(255,255,255), (255, 0, 0), (0, 255, 0), (0, 0, 255),(0,255,255)]

                        label_point(sat_individual, center_gt, "Pred", color_set[4], offset=(100, 40), font_scale=0.5, thickness=1)
                        label_point(sat_individual, center_gt, "GT", color_set[1], offset=(100, 20), font_scale=0.5, thickness=1)
                        label_point(sat_individual, center_gt, "EKF", color_set[2], offset=(100, 0), font_scale=0.5, thickness=1)
                        label_point(sat_individual, center_gt, "Meas", color_set[3], offset=(100, -20), font_scale=0.5, thickness=1)
                    else:
                        out_overlay = OUT_DIR.parent / f"overall_overlay_on_sat.png"
                        overlay_img = cv2.imread(str(out_overlay))
                        random_color = tuple(random.randint(128, 255) for _ in range(3))
                        color_set = [random_color, (255, 0, 0), (0, 255, 0), (0, 0, 255), (0,255,255)]

                        if overlay_img is None:
                            overlay_img = sat_base.copy()
                            label_point(overlay_img, [10,70], "Pred", color_set[4], offset=(0, 0), font_scale=3, thickness=3)
                            label_point(overlay_img, [300,70], "GT", color_set[1], offset=(0, 0), font_scale=3, thickness=3)
                            label_point(overlay_img, [450,70], "EKF", color_set[2], offset=(0, 0), font_scale=3, thickness=3)
                            label_point(overlay_img, [700,70], "Meas", color_set[3], offset=(0, 0), font_scale=3, thickness=3)
                            
                        sat_individual = overlay_img

                    draw_polygon(sat_individual, corners_disp, color=color_set[0], thickness=2)
                    draw_point(sat_individual, center_pred, color=color_set[4], r=2)
                    draw_point(sat_individual, center_measurement, color=color_set[3], r=2)
                    draw_point(sat_individual, center_gt, color=color_set[1], r=2)
                    draw_point(sat_individual, center_ekf, color=color_set[2], r=2)

                    Sigma_orig = P_pred[:2, :2]
                    J = np.diag([SX, SY])
                    Sigma_disp = J @ Sigma_orig @ J.T
                    draw_ellipse(sat_individual, center_pred, Sigma_disp, k_sigma=k, color=color_set[0], thickness=1)

                    cv2.imwrite(str(out_overlay), sat_individual)
            
            with open(CSV_FINAL_RESULT_PATH, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    drone_img,
                    r["tile"].name,
                    K_matches,
                    len(selected_tiles),
                    num_inliers,
                    f"{avg_confidence:.4f}",
                    f"{median_reproj_error_px:.4f}",
                    f"{overall_confidence:.4f}",
                    f"{meas_x_px:.4f}", f"{meas_y_px:.4f}", f"{meas_phi_deg:.4f}",
                    f"{x_updated[0]:.4f}", f"{x_updated[1]:.4f}", f"{x_updated[3]:.4f}",
                    f"{dx:.4f}", f"{dy:.4f}",
                    f"{dx_ekf:.4f}", f"{dy_ekf:.4f}",
                    f"{error:.4f}", f"{error_ekf:.4f}",
                    f"{dphi:.4f}", f"{dphi_ekf:.4f}",
                    "N/A",
                    "N/A",
                ])

results = get_metrics(CSV_FINAL_RESULT_PATH)
print(results)
