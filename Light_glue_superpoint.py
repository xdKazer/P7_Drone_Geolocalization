# run_batch_match_top3_constmem.py  (PT-features version)
import csv, math, gc
from pathlib import Path
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

drone_img = "03_0010.JPG"  
BASE = Path(__file__).parent.resolve()
DRONE_IMG = BASE / "UAV_VisLoc_dataset" / "03" / "drone" / str(drone_img)
SAT_DIR   = BASE / "UAV_VisLoc_dataset" / "03" / "test_signe"
OUT_DIR   = BASE / "outputs" / "03" / str(drone_img)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH  = OUT_DIR / "results.csv"
TILE_PT_DIR = BASE / "UAV_VisLoc_dataset" / "03" / f"{FEATURES}_features" / "03"
SAT_DISPLAY_IMG  = BASE / "UAV_VisLoc_dataset" / "03" / "satellite03_small.png"
SAT_DISPLAY_META = SAT_DISPLAY_IMG.with_suffix(SAT_DISPLAY_IMG.suffix + ".json")  # contains {"scale": s, ...}



# -------------------- Helpers --------------------
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
print(f"[info] wrote {CSV_PATH}")

top3 = scores_small[:3]

# -------------------- Pass 2: matches viz (per-rank) + colored top-3 overlays on sat --------------------
sat_vis, SX, SY, _sat_meta = load_sat_display_and_scale()
sat_base = sat_vis.copy()

# BGR colors for cv2: rank1=green, rank2=blue, rank3=red
rank_colors = [(0,255,0), (255,0,0), (0,0,255)]

for rank, r in enumerate(top3, 1):
    p = r["tile"]
    color = rank_colors[rank-1]
    print(f"[info] rank{rank}: {p.name}")

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
    sat_individual = sat_base.copy()
    draw_polygon(sat_individual, corners_disp, color=color, thickness=3)
    draw_point(sat_individual, center_disp, color=color, r=5)
    out_overlay = OUT_DIR / f"top{rank:02d}_{p.stem}_overlay_on_sat.png"
    cv2.imwrite(str(out_overlay), sat_individual)

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
print(f"[ok] Saved matches + overlays. Combined: {combined_path}")
