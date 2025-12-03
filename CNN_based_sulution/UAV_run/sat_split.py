import cv2
import math
from pathlib import Path
import numpy as np

"""
This script tiles the satellite image into overlapping patches that:

- Cover 1.3 × the drone-image diagonal (in meters) at any rotation.
- Use ORIGINAL satellite resolution (NO UPSCALING).
- Use 50% overlap.
- Produce tiles: UAV_VisLoc_dataset/<sat_number>/sat_tiles_overlap_native
- Produce a_tile_size.txt with metadata.

This avoids destroying features, unlike the previous rescaled version.
"""

# ----------------- Config -----------------
BASE        = Path(__file__).parent.resolve()
dataset_dir = BASE / "UAV_VisLoc_dataset"
sat_number  = "03"

tif_path = dataset_dir / sat_number / f"satellite{sat_number}.tif"
out_dir  = dataset_dir / sat_number / "sat_tiles_overlap"
out_dir.mkdir(parents=True, exist_ok=True)

# Reference drone image
drone_ref_path = dataset_dir / sat_number / "drone" / f"{sat_number}_0010.JPG"

# Drone GSD (meters per pixel)
drone_m_per_px = 0.125

# Geographic bounds of the satellite map
coordinate_range_lat_lon_sat = [
    (32.355491, 119.805926),  # top-left
    (32.290290, 119.900052),  # bottom-right
]

# Tile margin factor
TILE_DIAG_MARGIN = 1.3
OVERLAP = 0.5

# ----------------- Helpers -----------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_starts_full_tiles(size, tile_size, stride):
    if tile_size >= size:
        return [0]

    starts = [0]
    while True:
        nxt = starts[-1] + stride
        if nxt + tile_size >= size:
            starts.append(size - tile_size)
            break
        starts.append(nxt)
    return sorted(set(starts))

# ----------------- Load images -----------------
sat = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
if sat is None:
    raise FileNotFoundError(f"Cannot load satellite TIFF: {tif_path}")

drone = cv2.imread(str(drone_ref_path), cv2.IMREAD_UNCHANGED)
if drone is None:
    raise FileNotFoundError(f"Cannot load drone image: {drone_ref_path}")

H_sat, W_sat = sat.shape[:2]
H_drone, W_drone = drone.shape[:2]

print(f"Satellite size: {H_sat} × {W_sat}")
print(f"Drone size:     {H_drone} × {W_drone}")

# ----------------- Satellite GSD -----------------
(lat_min, lon_min), (lat_max, lon_max) = coordinate_range_lat_lon_sat

sat_height_m = haversine_m(lat_min, lon_min, lat_max, lon_min)
sat_width_m  = haversine_m(lat_min, lon_min, lat_min, lon_max)

sat_m_px_h = sat_height_m / H_sat
sat_m_px_w = sat_width_m  / W_sat
sat_m_per_px = math.sqrt(sat_m_px_h * sat_m_px_w)

print(f"Satellite GSD: {sat_m_per_px:.6f} m/px")
print(f"Drone GSD:     {drone_m_per_px:.6f} m/px")

# ----------------- Tile Size (NO RESCALING) -----------------
drone_diag_px = math.hypot(H_drone, W_drone)
drone_diag_m  = drone_diag_px * drone_m_per_px
tile_diag_m   = TILE_DIAG_MARGIN * drone_diag_m

# satellite tile side in px (square tile)
tile_side_sat = int(round(tile_diag_m / sat_m_per_px))
tile_side_sat = max(1, min(tile_side_sat, min(H_sat, W_sat)))

tile_h_sat = tile_side_sat
tile_w_sat = tile_side_sat

print(f"Tile size (sat px): {tile_h_sat} × {tile_w_sat}")

# ----------------- Overlap -----------------
stride_h = int(round(tile_h_sat * (1 - OVERLAP)))
stride_w = int(round(tile_w_sat * (1 - OVERLAP)))
stride_h = max(stride_h, 1)
stride_w = max(stride_w, 1)

ys = build_starts_full_tiles(H_sat, tile_h_sat, stride_h)
xs = build_starts_full_tiles(W_sat, tile_w_sat, stride_w)

print(f"Generated grid: {len(ys)} rows × {len(xs)} cols → {len(ys)*len(xs)} tiles")

# ----------------- Write tiles -----------------
count = 0
for y0 in ys:
    for x0 in xs:
        y1 = y0 + tile_h_sat
        x1 = x0 + tile_w_sat

        tile = sat[y0:y1, x0:x1]
        out_path = out_dir / f"sat_tile_y{y0}_x{x0}.png"
        cv2.imwrite(str(out_path), tile)
        print(f"Tile {count} done")
        count += 1

print(f"Saved {count} tiles to: {out_dir}")

# ----------------- Metadata -----------------
meta_path = out_dir / "a_tile_size.txt"
with open(meta_path, "w") as f:
    f.write(
        f"{stride_h} {stride_w} "
        f"{tile_h_sat} {tile_w_sat} "
        f"{np.float64(sat_m_per_px / drone_m_per_px)}\n"
    )

print(f"Wrote metadata: {meta_path}")
