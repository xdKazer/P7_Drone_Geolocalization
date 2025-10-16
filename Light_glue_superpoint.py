# run_batch_match_top3_constmem.py
import csv, math, gc
from pathlib import Path
from typing import Optional

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
FEATURES = "superpoint"       # "superpoint", "disk", or "sift"
DISPLAY_LONG_SIDE = 1200      # only for visualization (not matching)
MAX_KPTS = 4048               # reduce if RAM/CPU is tight (e.g., 1024)

drone_img = "03_0087.JPG"  # which drone image to use (in "03/drone")
BASE = Path(__file__).parent.resolve()
DRONE_IMG = BASE / "UAV_VisLoc_dataset" / "03" / "drone" / str(drone_img)
SAT_DIR   = BASE / "UAV_VisLoc_dataset" / "03" / "satellite_tiles"
OUT_DIR   = BASE / "outputs" /"outputs_" / str(drone_img)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH  = OUT_DIR / "results.csv"

# -------------------- Helpers --------------------
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

# -------------------- Device & models --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] device: {device}")

feat = FEATURES.lower()
if feat == "superpoint":
    extractor = SuperPoint(max_num_keypoints=MAX_KPTS).eval().to(device)
elif feat == "disk":
    extractor = DISK(max_num_keypoints=MAX_KPTS).eval().to(device)
elif feat == "sift":
    extractor = SIFT(max_num_keypoints=MAX_KPTS).eval().to("cpu")
else:
    raise ValueError("FEATURES must be 'superpoint', 'disk', or 'sift'.")

matcher = LightGlue(features=feat).eval().to(device)

# -------------------- Load fixed drone features (batched for matcher) --------------------
if not DRONE_IMG.exists():
    raise FileNotFoundError(f"Missing drone image: {DRONE_IMG}")

with torch.inference_mode():
    img0_t = load_image(str(DRONE_IMG)).to(device if feat != "sift" else "cpu")
    feats0_batched = extractor.extract(img0_t)  # KEEP batch (B=1)

# Also keep a non-batched view for keypoint indexing
feats0_r = rbd(feats0_batched)

# -------------------- Pass 1: score all tiles (constant memory) --------------------
tiles = [p for p in SAT_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
if not tiles:
    raise FileNotFoundError(f"No PNG tiles in {SAT_DIR}")

scores_small = []

with torch.inference_mode():
    for i, p in enumerate(tiles):
        print(f"Scoring number {i} of {len(tiles)} ...")

        # 1) load tile & extract (batched)
        img1_t  = load_image(str(p)).to(device if feat != "sift" else "cpu")
        feats1  = extractor.extract(img1_t)                  # B=1

        # 2) match with batched feats
        matches01 = matcher({"image0": feats0_batched, "image1": feats1})

        # 3) drop batch dims for metrics
        feats1_r    = rbd(feats1)
        matches01_r = rbd(matches01)

        matches = matches01_r.get("matches", None)
        K = int(matches.shape[0]) if (matches is not None and matches.numel() > 0) else 0

        num_inliers = 0
        avg_conf    = float("nan")

        if K >= 4:
            # gather matched keypoints
            pts0_np = feats0_r["keypoints"][matches[:, 0]].detach().cpu().numpy()
            pts1_np = feats1_r["keypoints"][matches[:, 1]].detach().cpu().numpy()

            # RANSAC
            Hmat, mask = cv2.findHomography(
                pts0_np, pts1_np, method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=3.0, confidence=0.999
            )
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                num_inliers = int(inlier_mask.sum())

                # LightGlue scores are optional
                scores_t = matches01_r.get("scores", None)
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
        del img1_t, feats1, feats1_r, matches01, matches01_r
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

# -------------------- Pass 2: reload and visualize only the top-5 --------------------
for rank, r in enumerate(top3, 1):
    p = r["tile"]
    print(f"[info] visualizing top{rank}: {p.name} ...")
    with torch.inference_mode():
        img1_t = load_image(str(p)).to(device if feat != "sift" else "cpu")
        feats1 = extractor.extract(img1_t)
        matches01 = matcher({"image0": feats0_batched, "image1": feats1})

    f0_r, f1_r, m_r = feats0_r, rbd(feats1), rbd(matches01)
    matches = m_r.get("matches", None)
    if matches is None or matches.numel() == 0:
        inlier_mask = None
        pts0_np = np.empty((0,2), np.float32)
        pts1_np = np.empty((0,2), np.float32)
    else:
        pts0_np = f0_r["keypoints"][matches[:, 0]].detach().cpu().numpy()
        pts1_np = f1_r["keypoints"][matches[:, 1]].detach().cpu().numpy()
        inlier_mask = None
        if len(pts0_np) >= 4:
            Hmat, mask = cv2.findHomography(
                pts0_np, pts1_np, method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=3.0, confidence=0.999
            )
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)

    out_png = OUT_DIR / f"top{rank:02d}_{p.stem}.png"
    visualize_inliers(DRONE_IMG, p, pts0_np, pts1_np, inlier_mask, str(out_png))

    # free per-tile stuff
    del img1_t, feats1, matches01, f1_r, m_r
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

print("[done] Top-5 visualizations saved; see CSV for ranking.")
