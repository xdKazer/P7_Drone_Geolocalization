# run_match_and_viz.py
import os
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")  # headless render on the cluster
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from lightglue import LightGlue, SuperPoint, DISK  # add others if you want
from lightglue.utils import load_image, rbd

# --------------------
# Config
# --------------------
# Choose: "superpoint" or "disk"
FEATURES = "superpoint"   # change to "superpoint" if you want SP+LG

# Your image paths
im0_path = "/ceph/home/student.aau.dk/zn23sc/P7_Drone_Geolocalization/UAV_VisLoc_dataset/03/drone/03_0010.JPG"
im1_path = "/ceph/home/student.aau.dk/zn23sc/P7_Drone_Geolocalization/UAV_VisLoc_dataset/03/satellite_tiles/sat_tile_4_11.png"

# Optional: set a display long-side to keep images manageable for vis (LightGlue resizes internally for extraction)
DISPLAY_LONG_SIDE = 1600  # only affects the rendered figure; matching uses original tensors

out_png = "matches.png"

# --------------------
# Helpers
# --------------------
def to_numpy_image(t: torch.Tensor):
    # t: (3,H,W) in [0,1] on GPU/CPU; return (H,W,3) float in [0,1]
    return t.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()

def make_segments(p0, p1, x_offset):
    segs = np.zeros((len(p0), 2, 2), dtype=np.float32)
    segs[:, 0, :] = p0
    segs[:, 1, 0] = p1[:, 0] + x_offset
    segs[:, 1, 1] = p1[:, 1]
    return segs

def resize_for_display(img_np, long_side=1600):
    h, w = img_np.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0:
        return img_np, 1.0
    new_w, new_h = int(round(w * s)), int(round(h * s))
    img_small = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_small, s

# --------------------
# Device
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] device: {device}")

# --------------------
# Models
# --------------------
if FEATURES.lower() == "superpoint":
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
elif FEATURES.lower() == "disk":
    extractor = DISK(max_num_keypoints=2048).eval().to(device)
else:
    raise ValueError("FEATURES must be 'superpoint' or 'disk'.")

matcher = LightGlue(features=FEATURES.lower()).eval().to(device)

# --------------------
# Load images as torch (3,H,W) in [0,1]
# --------------------
image0 = load_image(im0_path).to(device)
image1 = load_image(im1_path).to(device)

# --------------------
# Extract + match
# --------------------
with torch.inference_mode():
    feats0 = extractor.extract(image0)  # LightGlue utils handle resizing internally for features
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})

# Remove batch dimension
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
matches = matches01["matches"]  # (K,2) long indices

# Gather matched keypoints
if matches.numel() == 0:
    print("[warn] No matches found.")
    # still dump a side-by-side for debugging
points0 = feats0["keypoints"][matches[..., 0]]  # (K,2)
points1 = feats1["keypoints"][matches[..., 1]]  # (K,2)

p0 = points0.detach().cpu().numpy()
p1 = points1.detach().cpu().numpy()

# Convert images to numpy for viz
I0 = to_numpy_image(image0)
I1 = to_numpy_image(image1)

# Optionally shrink for display (does NOT change p0/p1 because those are in original coords)
I0_disp, s0 = resize_for_display(I0, DISPLAY_LONG_SIDE)
I1_disp, s1 = resize_for_display(I1, DISPLAY_LONG_SIDE)

# Scale keypoints to display coords (so lines land in the right places)
p0_disp = p0 * s0
p1_disp = p1 * s1

H0, W0 = I0_disp.shape[:2]
H1, W1 = I1_disp.shape[:2]

# --------------------
# RANSAC (homography) to highlight inliers (planar-ish scenes)
# --------------------
inlier_mask = None
if len(p0) >= 4:
    Hmat, mask = cv2.findHomography(p0, p1, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0, confidence=0.999)
    if mask is not None:
        inlier_mask = mask.ravel().astype(bool)

# --------------------
# Build side-by-side canvas
# --------------------
canvas = np.ones((max(H0, H1), W0 + W1, 3), dtype=I0_disp.dtype)
canvas[:H0, :W0] = I0_disp
canvas[:H1, W0:W0 + W1] = I1_disp

segs = make_segments(p0_disp, p1_disp, x_offset=W0)

# --- keep ONLY inliers (if any); else show "no matches" ---
if inlier_mask is not None and inlier_mask.any():
    in_idx = np.where(inlier_mask)[0]
    segs_in = segs[in_idx]
    p0_in   = p0_disp[in_idx]
    p1_in   = p1_disp[in_idx]
else:
    # no inliers -> save the side-by-side only and exit
    plt.figure(figsize=(14, 7))
    plt.imshow(canvas); plt.axis("off")
    plt.title("No inlier matches")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[info] saved {out_png} (no inliers)")
    raise SystemExit(0)

# --- plot only inliers ---
fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(canvas); ax.axis("off")

lc = LineCollection(segs_in, linewidths=0.9, alpha=0.95)
lc.set_colors(np.array([[0.0, 0.8, 0.0]] * len(segs_in)))  # all green
ax.add_collection(lc)

# optional: scatter endpoints for inliers
ax.scatter(p0_in[:, 0],        p0_in[:, 1],        s=2, c="yellow", alpha=0.7)
ax.scatter(p1_in[:, 0] + W0,   p1_in[:, 1],        s=2, c="cyan",   alpha=0.7)

title = f"{FEATURES.upper()} + LightGlue  |  inliers={len(segs_in)}"
ax.set_title(title)

plt.tight_layout()
plt.savefig(out_png, dpi=150)
print(f"[info] saved {out_png}")


