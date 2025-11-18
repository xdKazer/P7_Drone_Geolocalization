import cv2
import math
from pathlib import Path
import numpy as np

"""
This code makes sure that each pixel in the drone images corresponds to approximately the same
real-world distance as each pixel in the satellite tiles. It does this by calculating the ground
sampling distance (GSD) of the satellite image using geo-coordinates, then rescaling the satellite
tiles to match the drone image GSD. The tiles are generated with overlap to ensure full coverage.   

"""

# ---- Paths ----
BASE = Path(__file__).parent.resolve()
dataset_path = BASE / "UAV_VisLoc_dataset"
sat_number   = "03"

tif_path = dataset_path / sat_number / f"satellite{sat_number}.tif"
out_dir  = dataset_path / sat_number / "sat_tiles_overlap_scaled"
out_dir.mkdir(parents=True, exist_ok=True)

# ---- Read images ----
sat = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
drone = cv2.imread(str(dataset_path / sat_number / "drone" / f"{sat_number}_0738.JPG"),
                   cv2.IMREAD_UNCHANGED)
if sat is None:
    raise FileNotFoundError(tif_path)
if drone is None:
    raise FileNotFoundError(dataset_path / sat_number / "drone" / f"{sat_number}_0738.JPG")

# ---- Dataset / geo information ----
# from UAV-VisLoc dataset description
drone_altityde_m = 466        # not directly used here, but kept for completeness
drone_m_px       = 0.125       # meters per pixel at given altitude

# (lat_min, lon_min), (lat_max, lon_max)
coordinate_range_lat_long_sat_map = [
    (32.355491, 119.805926),
    (32.290290, 119.900052)
]

# satellite image size (should match sat.shape)
sat_pixels_h, sat_pixels_w = sat.shape[:2]

# ---- Helpers ----
def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points (lat/lon in degrees) in meters."""
    R = 6371000.0  # Earth radius [m]
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def build_starts(size, tile_size, stride):
    """Compute tile start indices so that we cover the full extent including far edge."""
    starts = list(range(0, max(1, size - tile_size + 1), stride))
    if starts[-1] + tile_size < size:
        starts.append(size - tile_size)
    return starts

# ---- Compute satellite meters-per-pixel (GSD) ----
(lat_min, lon_min), (lat_max, lon_max) = coordinate_range_lat_long_sat_map

# Physical dimensions of satellite map
sat_height_m = haversine_m(lat_min, lon_min, lat_max, lon_min)  # north–south
sat_width_m  = haversine_m(lat_min, lon_min, lat_min, lon_max)  # east–west

sat_m_per_px_h = sat_height_m / sat_pixels_h
sat_m_per_px_w = sat_width_m  / sat_pixels_w

# Single isotropic value (geometric mean)
sat_m_per_px = math.sqrt(sat_m_per_px_h * sat_m_per_px_w)

print(f"Satellite GSD (m/px): vertical={sat_m_per_px_h:.4f}, "
      f"horizontal={sat_m_per_px_w:.4f}, iso={sat_m_per_px:.4f}")
print(f"Drone GSD (m/px): {drone_m_px:.4f}")

# ---- Scale factor from satellite to drone ----
# (how much you must upscale satellite pixels so they match drone resolution)
scale_sat_to_drone = sat_m_per_px / drone_m_px
print(f"Scale factor sat -> drone: {scale_sat_to_drone:.4f}")

# ---- Tile sizes ----
H_sat, W_sat = sat.shape[:2]
H_drone, W_drone = drone.shape[:2]

# Tile size in ORIGINAL satellite image, before upscaling
tile_h_sat = int(round(H_drone / scale_sat_to_drone))
tile_w_sat = int(round(W_drone / scale_sat_to_drone))

# Safety in case of extreme rounding
tile_h_sat = max(1, min(tile_h_sat, H_sat))
tile_w_sat = max(1, min(tile_w_sat, W_sat))

print(f"Satellite image size: {H_sat} x {W_sat}")
print(f"Drone image size:     {H_drone} x {W_drone}")
print(f"Satellite tile size (pre-resize): {tile_h_sat} x {tile_w_sat}")

# ---- Overlap / strides (based on satellite tile size) ----
stride_h = max(1, int(tile_h_sat * 0.7))  # 30% overlap
stride_w = max(1, int(tile_w_sat * 0.7))

ys = build_starts(H_sat, tile_h_sat, stride_h)
xs = build_starts(W_sat, tile_w_sat, stride_w)

print(f"rows: {len(ys)}  cols: {len(xs)}  -> total tiles: {len(ys) * len(xs)}")


# ---- Generate tiles: crop from original sat, then upscale to drone size ----
count = 0
for y0 in ys:
    y1 = min(y0 + tile_h_sat, H_sat)
    for x0 in xs:
        x1 = min(x0 + tile_w_sat, W_sat)
        tile = sat[y0:y1, x0:x1]
        if tile.size == 0:
            continue

        # Resize each tile to the drone image size before feature extraction
        tile_rescaled = cv2.resize(
            tile,
            (W_drone, H_drone),
            interpolation=cv2.INTER_LINEAR,
        )

        # Encode original satellite pixel start indices in filename
        out_path = out_dir / f"sat_tile_y{y0}_x{x0}.png"
        cv2.imwrite(str(out_path), tile_rescaled)
        count += 1


print(f"Wrote {count} overlapping, rescaled tiles to: {out_dir}")

# Save the *rescaled* tile size (what downstream code will see)
with open(out_dir / "a_tile_size.txt", "w") as f:
    f.write(f"{stride_h} {stride_w} {tile_h_sat} {tile_w_sat} {H_drone} {W_drone} {np.float64(scale_sat_to_drone)}\n")
print(f"Wrote rescaled tile size to: {out_dir / 'a_tile_size.txt'}")
