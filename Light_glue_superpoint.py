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

from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import load_image, rbd

# -------------------- Config --------------------
FEATURES = "superpoint"       # precomputed .pt 
DISPLAY_LONG_SIDE = 1200      # only for visualization 
MAX_KPTS = 4048               # not used if precomputed

# Tile geometry used during tiling (in ORIGINAL satellite pixel units)
TILE_W  = 1205                # width of each tile when you cut them 
TILE_H  = 1807                # height of each tile
STRIDE_X = TILE_W // 2        # stride in x used during tiling
STRIDE_Y = TILE_H // 2        # stride in y

sat_number = "03"
drone_img = "03_0010.JPG"  
Heading_flip_180 = True  # whether to add 180 deg to GT heading from CSV

BASE = Path(__file__).parent.resolve()
DATASET_DIR = BASE / "UAV_VisLoc_dataset"
SAT_LONG_LAT_INFO_DIR = DATASET_DIR / "satellite_ coordinates_range.csv"
DRONE_INFO_DIR = DATASET_DIR / sat_number / f"{sat_number}.csv"
DRONE_IMG = DATASET_DIR / sat_number / "drone" / str(drone_img)
SAT_DIR   = DATASET_DIR / sat_number / "test_signe"
OUT_DIR   = BASE / "outputs" / sat_number / str(drone_img)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH  = OUT_DIR / "results.csv"
TILE_PT_DIR = DATASET_DIR / sat_number / f"{FEATURES}_features" / sat_number
SAT_DISPLAY_IMG  = DATASET_DIR / sat_number / "satellite03_small.png"
SAT_DISPLAY_META = SAT_DISPLAY_IMG.with_suffix(SAT_DISPLAY_IMG.suffix + ".json")  # contains {"scale": s, ...}



# -------------------- Helpers --------------------
# ---------- Float64-safe geo <-> pixel mapping ----------
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
    return u, v  # ORIGINAL pixels (float64)

def draw_cropped_pred_vs_gt_on_tile(
    tile_path,            # Path to the top-match tile PNG
    Hmat,                 # homography image->tile
    x_off, y_off,         # tile origin in ORIGINAL pixels
    DRONE_IMG,
    DRONE_INFO_DIR, drone_img,
    SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META,
    crop_radius_px=450,   # in TILE pixels
    out_path=None,
    heading_flip_180=False  # flip GT heading by 180° if your CSV stores view-dir
):
    """
    Draw Pred vs GT centers + heading arrows directly on the TILE image,
    cropped around the midpoint.

    Orientation metrics returned:
      - heading_diff_angle_deg  : angle(pred, gt) in degrees in [0, 180]
      - heading_diff_dot_prod   : unit dot product ∈ [-1, 1]

    GT heading convention: East=0°, CCW positive (North=+90°, South=-90°).
    """
    # --- Load tile image (BGR) ---
    tile = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
    if tile is None:
        raise FileNotFoundError(f"Cannot read tile: {tile_path}")
    TH, TW = tile.shape[:2]

    # --- Predicted center/forward in TILE coords via H ---
    drone_np = cv2.imread(str(DRONE_IMG), cv2.IMREAD_UNCHANGED)
    if drone_np is None:
        raise FileNotFoundError(f"Cannot read {DRONE_IMG}")
    h0, w0 = drone_np.shape[:2]

    center0  = np.array([[w0/2, h0/2]], dtype=np.float32)
    forward0 = np.array([[w0/2, h0/2 - max(20.0, 0.10*h0)]], dtype=np.float32)  # “forward” = −y in image

    center_tile  = project_pts(Hmat, center0)[0].astype(np.float64)   # (x,y) in TILE pixels
    forward_tile = project_pts(Hmat, forward0)[0].astype(np.float64)

    # Predicted heading vector (unit) in tile pixels
    v_pred = forward_tile - center_tile
    n_pred = float(np.linalg.norm(v_pred))
    ray_len = np.float64(220.0)
    if n_pred < 1e-12:
        v_pred_unit = np.array([1.0, 0.0], dtype=np.float64)  # fallback: point east
        pred_tip = center_tile + ray_len * v_pred_unit
    else:
        v_pred_unit = (v_pred / n_pred).astype(np.float64)
        pred_tip = center_tile + ray_len * v_pred_unit

    # --- Ground truth center on TILE ---
    # 1) GT lat/lon (+ heading Φ)
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
        raise ValueError("GT lat/lon not found for this drone image.")

    if (gt_heading_deg is not None) and heading_flip_180:
        gt_heading_deg = (gt_heading_deg + 180.0) % 360.0

    # 2) GT original pixel -> TILE pixel
    gt_u, gt_v = latlon_to_orig_xy(gt_lat, gt_lon, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
    gt_tile = np.array([gt_u - x_off, gt_v - y_off], dtype=np.float64)  # (x,y) in TILE pixels

    # 3) GT heading vector (unit). East=0°, CCW positive => dx=cosθ, dy=-sinθ (image y down)
    if gt_heading_deg is not None:
        th = np.deg2rad(np.float64(gt_heading_deg))
        v_gt_unit = np.array([np.cos(th), -np.sin(th)], dtype=np.float64)
        # normalize (defensive)
        n_gt = float(np.linalg.norm(v_gt_unit))
        if n_gt < 1e-12:
            v_gt_unit = np.array([1.0, 0.0], dtype=np.float64)
    else:
        v_gt_unit = np.array([np.nan, np.nan], dtype=np.float64)  # flag missing

    gt_tip = gt_tile + ray_len * (v_gt_unit if np.isfinite(v_gt_unit).all() else np.array([1.0, 0.0], np.float64))

    # --- Orientation metrics ---
    if np.isfinite(v_gt_unit).all():
        # dot product ∈ [-1, 1]
        heading_diff_dot_prod = float(np.clip(np.dot(v_pred_unit, v_gt_unit), -1.0, 1.0))
        # angle in degrees ∈ [0, 180]
        heading_diff_angle_deg = float(np.degrees(np.arccos(heading_diff_dot_prod)))
    else:
        heading_diff_dot_prod = float("nan")
        heading_diff_angle_deg = float("nan")

    # --- Crop around midpoint (in TILE pixels), clamped to tile ---
    cx = float((center_tile[0] + gt_tile[0]) / 2.0)
    cy = float((center_tile[1] + gt_tile[1]) / 2.0)
    r  = int(crop_radius_px)

    x0 = max(0, int(round(cx - r)))
    y0 = max(0, int(round(cy - r)))
    x1 = min(TW, int(round(cx + r)))
    y1 = min(TH, int(round(cy + r)))
    if x1 - x0 < 40 or y1 - y0 < 40:
        x0 = max(0, min(TW-1, int(round(cx)) - r))
        y0 = max(0, min(TH-1, int(round(cy)) - r))
        x1 = min(TW, x0 + 2*r)
        y1 = min(TH, y0 + 2*r)

    crop = tile[y0:y1, x0:x1].copy()

    def S(p):  # shift to crop coords
        return (int(round(p[0] - x0)), int(round(p[1] - y0)))

    # Draw: Pred (green), GT (magenta)
    cv2.circle(crop, S(center_tile), 10, (0,255,0), -1)
    cv2.arrowedLine(crop, S(center_tile), S(pred_tip), (0,255,0), 3, tipLength=0.2)

    cv2.circle(crop, S(gt_tile), 10, (255,0,255), -1)
    cv2.arrowedLine(crop, S(gt_tile), S(gt_tip), (255,0,255), 3, tipLength=0.2)

    # small legend
    cv2.rectangle(crop, (10,10), (360,90), (255,255,255), -1)
    cv2.putText(crop, "Pred (tile H)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,180,0), 2)
    cv2.putText(crop, "GT (Phi, CCW+)", (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170,0,170), 2)

    # Optional: print metrics on the crop
    txt = f"angle={heading_diff_angle_deg:.1f}°, dot={heading_diff_dot_prod:.3f}"
    cv2.putText(crop, txt, (20,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    if out_path is not None:
        cv2.imwrite(str(out_path), crop)

    return crop, heading_diff_angle_deg, heading_diff_dot_prod


def get_location_in_sat_img(drone_img_centre, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META):
    meta = json.loads(SAT_DISPLAY_META.read_text())
    if "original_size_hw" in meta:
        sat_H, sat_W = map(np.float64, meta["original_size_hw"])

    # Load satellite corner coordinates
    with open(SAT_LONG_LAT_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["mapname"] == f"satellite{sat_number}.tif":
                LT_lat = np.float64(r["LT_lat_map"])
                LT_lon = np.float64(r["LT_lon_map"])
                RB_lat = np.float64(r["RB_lat_map"])
                RB_lon = np.float64(r["RB_lon_map"])
                break

    # Ensure drone center is float64
    u, v = np.float64(drone_img_centre)

    # --- Compute high-precision lat/lon ---
    lon = LT_lon + (u / sat_W) * (RB_lon - LT_lon)
    lat = LT_lat + (v / sat_H) * (RB_lat - LT_lat)

    return (lat, lon)

def determine_pos_error(pose, DRONE_INFO_DIR, drone_img):
    # Ensure pose is float64
    pose = np.array(pose, dtype=np.float64)

    # --- Read ground-truth drone coordinates ---
    with open(DRONE_INFO_DIR, newline="") as f:
        for r in csv.DictReader(f):
            if r["filename"] == f"{drone_img}":
                lat = np.float64(r["lat"])
                lon = np.float64(r["lon"])
                break

    # --- Difference in degrees ---
    difference = pose - np.array([lat, lon], dtype=np.float64)

    # --- Mean latitude (radians) ---
    mean_lat = np.radians((pose[0] + lat) / np.float64(2.0))

    # --- Meters per degree (float64 constants) ---
    meters_per_degree_lat = np.float64(111_320.0)
    meters_per_degree_lon = meters_per_degree_lat * np.cos(mean_lat)

    # --- Convert to meters ---
    dy = difference[0] * meters_per_degree_lat   # north-south (Δlat)
    dx = difference[1] * meters_per_degree_lon   # east-west (Δlon)

    total_error_m = np.sqrt(dx**2 + dy**2)

    return total_error_m, dx, dy

def load_sat_display_and_scale():
    img = cv2.imread(str(SAT_DISPLAY_IMG), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {SAT_DISPLAY_IMG}")
    meta = json.loads(SAT_DISPLAY_META.read_text())
    # Prefer single uniform scale ("scale"); else allow non-uniform "scale_xy"
    if "scale" in meta:
        s = float(meta["scale"])
        sx = sy = s
    elif "scale_xy" in meta:
        sx, sy = map(float, meta["scale_xy"])
    else:
        raise KeyError(f"{SAT_DISPLAY_META} missing 'scale' or 'scale_xy'")
    return img, sx, sy, meta

def tile_offset_from_name(tile_path: Path):
    """
    Parse ORIGINAL satellite pixel offsets from filenames like:
      sat_tile_y16856_x29799.png
      foo_sat_tile_y16254_x30702_extra.png   (works too)
    Returns (x_off, y_off) in ORIGINAL satellite pixels.
    """
    name = tile_path.stem  # no extension
    m = re.search(r"y(?P<y>\d+)_x(?P<x>\d+)", name)
    if not m:
        raise ValueError(
            f"Cannot parse offsets from '{tile_path.name}'. "
            f"Expected '...y<Y>_x<X>...'"
        )
    y_off = int(m.group("y"))
    x_off = int(m.group("x"))
    return x_off, y_off

def project_pts(H, pts_xy):
    """Project Nx2 points with 3x3 homography H."""
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
    # load images just for viz
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
    folder = TILE_PT_DIR if TILE_PT_DIR is not None else image_path.parent
    return folder / (image_path.stem + ".pt")

def load_feats_pt_batched(pt_path: Path, device: str):
    d = torch.load(str(pt_path), map_location="cpu")
    # Ensure required keys and dtypes
    for k in ("keypoints", "descriptors", "keypoint_scores", "image_size"):
        if k not in d:
            raise KeyError(f"{pt_path} missing key '{k}'")
    # Cast to expected types
    kpts = d["keypoints"].to(dtype=torch.float32)
    desc = d["descriptors"].to(dtype=torch.float32)
    scrs = d["keypoint_scores"].to(dtype=torch.float32)
    isize = d["image_size"].to(dtype=torch.int64)
    # Add batch dim and move to device
    feats_b = {
        "keypoints":   kpts.unsqueeze(0).to(device),
        "descriptors": desc.unsqueeze(0).to(device),
        "keypoint_scores":      scrs.unsqueeze(0).to(device),
        "image_size":  isize.unsqueeze(0).to(device),
    }
    feats_r = rbd(feats_b)  # non-batched convenience view
    return feats_b, feats_r

# -------------------- Device & models --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] device: {device}")

feat = FEATURES.lower()
if feat not in ("superpoint", "disk", "sift"):
    raise ValueError("FEATURES must be 'superpoint', 'disk', or 'sift'.")

# We instantiate an extractor ONLY as a fallback when a .pt is missing.
extractor = None
if feat == "superpoint":
    extractor = SuperPoint(max_num_keypoints=MAX_KPTS).eval().to(device)
elif feat == "disk":
    extractor = DISK(max_num_keypoints=MAX_KPTS).eval().to(device)
elif feat == "sift":
    extractor = SIFT(max_num_keypoints=MAX_KPTS).eval().to("cpu")

matcher = LightGlue(features=feat).eval().to(device)

# -------------------- Load drone features (prefer .pt; fallback to extract) --------------------
if not DRONE_IMG.exists():
    raise FileNotFoundError(f"Missing drone image: {DRONE_IMG}")

drone_pt = make_feature_pt_path_for(DRONE_IMG)
if drone_pt.exists():
    feats0_batched, feats0_r = load_feats_pt_batched(drone_pt, device if feat != "sift" else "cpu")
else:
    print(f"[warn] No precomputed drone .pt found for {DRONE_IMG.name}; extracting on-the-fly.")
    with torch.inference_mode():
        img0_t = load_image(str(DRONE_IMG)).to(device if feat != "sift" else "cpu")
        feats0_batched = extractor.extract(img0_t)  # B=1
        feats0_r = rbd(feats0_batched)

# -------------------- Pass 1: score all tiles using precomputed .pt --------------------
tiles = [p for p in SAT_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
if not tiles:
    raise FileNotFoundError(f"No PNG tiles in {SAT_DIR}")

scores_small = []

with torch.inference_mode():
    for i, p in enumerate(tiles):
        print(f"Scoring tile {i+1}/{len(tiles)}: {p.name}")

        # 1) Load tile features from .pt (fallback to extract if missing)
        tile_pt = make_feature_pt_path_for(p)
        if tile_pt.exists():
            feats1_b, feats1_r = load_feats_pt_batched(tile_pt, device if feat != "sift" else "cpu")
        else:
            if extractor is None:
                raise FileNotFoundError(f"Missing {tile_pt} and no extractor available.")
            # Fallback extraction
            img1_t  = load_image(str(p)).to(device if feat != "sift" else "cpu")
            feats1_b = extractor.extract(img1_t)                  # B=1
            feats1_r = rbd(feats1_b)

        # 2) Match with batched feats
        matches01 = matcher({"image0": feats0_batched, "image1": feats1_b})

        # 3) Drop batch dims for metrics
        matches01_r = rbd(matches01)

        matches = matches01_r.get("matches", None)  # Kx2 (idx0, idx1)
        K = int(matches.shape[0]) if (matches is not None and matches.numel() > 0) else 0

        num_inliers = 0
        avg_conf    = float("nan")

        if K >= 4:
            # gather matched keypoints
            pts0_np = feats0_r["keypoints"][matches[:, 0]].detach().cpu().numpy()
            pts1_np = feats1_r["keypoints"][matches[:, 1]].detach().cpu().numpy()

            # RANSAC
            _, mask = cv2.findHomography(
                pts0_np, pts1_np, method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=3.0, confidence=0.999
            )
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                num_inliers = int(inlier_mask.sum())

                # LightGlue avg scores 
                scores_t = matches01_r.get("keypoint_scores", None)
                if scores_t is not None and num_inliers > 0:
                    scores_np = scores_t.detach().cpu().numpy()
                    avg_conf = float(np.mean(scores_np[inlier_mask]))

        sort_avg = avg_conf if not math.isnan(avg_conf) else -1e9
        scores_small.append({
            "tile": p,
            "inliers": num_inliers,
            "avg_conf": avg_conf,
            "sort_key": (num_inliers, sort_avg),
        })

        # free per-tile tensors
        del feats1_b, feats1_r, matches01, matches01_r
        if 'img1_t' in locals():
            del img1_t
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

# -------------------- Rank & write CSV --------------------
scores_small.sort(key=lambda d: d["sort_key"], reverse=True)
with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tile", "inliers", "avg_confidence"])
    for r in scores_small:
        w.writerow([r["tile"].name, r["inliers"],
                    "" if math.isnan(r["avg_conf"]) else f"{r['avg_conf']:.4f}"])
print(f"[info] wrote CSV")

top3 = scores_small[:3]

# -------------------- Pass 2: matches viz (per-rank) + colored top-3 overlays on sat --------------------
sat_vis, SX, SY, _sat_meta = load_sat_display_and_scale()
sat_base = sat_vis.copy()

# BGR colors for cv2: rank1=green, rank2=blue, rank3=red
rank_colors = [(0,255,0), (255,0,0), (0,0,255)]

for rank, r in enumerate(top3, 1):
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

        # default values if no matches
        inlier_mask = None
        pts0_np = np.empty((0,2), np.float32)
        pts1_np = np.empty((0,2), np.float32)
        Hmat = None

        if matches is not None and matches.numel() > 0:
            pts0_np = feats0_r["keypoints"][matches[:, 0]].detach().cpu().numpy()
            pts1_np = feats1_r["keypoints"][matches[:, 1]].detach().cpu().numpy()
            if len(pts0_np) >= 4:
                Hmat, mask = cv2.findHomography(
                    pts0_np, pts1_np,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=3.0,
                    confidence=0.999
                )
                if mask is not None:
                    inlier_mask = mask.ravel().astype(bool)

    # --- (A) Save classic side-by-side match visualization (like earlier) ---
    out_match_png = OUT_DIR / f"top{rank:02d}_{p.stem}_matches.png"
    visualize_inliers(DRONE_IMG, p, pts0_np, pts1_np, inlier_mask, str(out_match_png))

    # --- (B) If H was estimated, also draw overlay on downscaled satellite ---
    if Hmat is None or inlier_mask is None or inlier_mask.sum() < 4:
        print(f"[warn] Homography not reliable for {p.name}; skipping overlay.")
        # free and continue
        del feats1_b, feats1_r, matches01, m_r
        if 'img1_t' in locals():
            del img1_t
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        continue

    # Project drone footprint
    drone_np = cv2.imread(str(DRONE_IMG), cv2.IMREAD_UNCHANGED)
    if drone_np is None:
        raise FileNotFoundError(f"Cannot read {DRONE_IMG}")
    h0, w0 = drone_np.shape[:2]
    corners0 = np.array([[0,0],[w0,0],[w0,h0],[0,h0]], dtype=np.float32)
    center0  = np.array([[w0/2, h0/2]], dtype=np.float32)

    corners_tile = project_pts(Hmat, corners0)
    center_tile  = project_pts(Hmat, center0)[0]

    # tile offsets are ORIGINAL sat pixels parsed from filename
    x_off, y_off = tile_offset_from_name(p)
    corners_global = corners_tile + np.array([x_off, y_off], np.float32)
    center_global  = center_tile  + np.array([x_off, y_off], np.float32)

    # map to DOWNSCALED satellite coords
    corners_disp = corners_global * np.array([SX, SY], np.float32)
    center_disp  = center_global  * np.array([SX, SY], np.float32)

    # Per-rank overlay
    if rank == 1:
        sat_individual = sat_base.copy()
        draw_polygon(sat_individual, corners_disp, color=color, thickness=3)
        draw_point(sat_individual, center_disp, color=color, r=5)
        out_overlay = OUT_DIR / f"top{rank:02d}_{p.stem}_overlay_on_sat.png"
        cv2.imwrite(str(out_overlay), sat_individual)

        #estimate error:
        pose = get_location_in_sat_img(center_global, SAT_LONG_LAT_INFO_DIR, sat_number, SAT_DISPLAY_META)
        error, dx, dy = determine_pos_error(pose, DRONE_INFO_DIR, drone_img)
        print(f"Mean Error: {error}m, dx: {dx}m, dy: {dy}m")

        out_cropped = OUT_DIR / f"top01_{p.stem}_pred_vs_gt_TILE_cropped.png"
        crop_img, dtheta_deg, dotp = draw_cropped_pred_vs_gt_on_tile(
            tile_path=p,
            Hmat=Hmat,
            x_off=x_off, y_off=y_off,
            DRONE_IMG=DRONE_IMG,
            DRONE_INFO_DIR=DRONE_INFO_DIR,
            drone_img=drone_img,
            SAT_LONG_LAT_INFO_DIR=SAT_LONG_LAT_INFO_DIR,
            sat_number=sat_number,
            SAT_DISPLAY_META=SAT_DISPLAY_META,
            crop_radius_px=450,
            out_path=out_cropped,
            heading_flip_180=True  # set True if your CSV stores view-direction
        )
        print(f"[metrics] heading Δθ={dtheta_deg:.2f}°, dot={dotp:.4f}")

    # Accumulate into combined overlay
    draw_polygon(sat_vis, corners_disp, color=color, thickness=3)
    draw_point(sat_vis, center_disp, color=color, r=5)

    # free per-tile stuff
    del feats1_b, feats1_r, matches01, m_r
    if 'img1_t' in locals():
        del img1_t
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# Save combined overlay (all top-3)
combined_path = OUT_DIR / "top3_combined_overlay_on_sat.png"
cv2.imwrite(str(combined_path), sat_vis)