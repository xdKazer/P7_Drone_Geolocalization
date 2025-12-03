
import csv
import math
from pathlib import Path

"""
Preprocess VPAIR poses into:
  - poses_drone.csv        (for EKF: global pixel coords + yaw + m_per_px)
  - poses_tiles.csv        (tile centers in meters, for error in meters)
  - tile_centers_in_sat.csv (tile centers in global "sat pixels")
  
Assumptions:
  - One CSV with: filename, lat, lon, altitude, roll, pitch, yaw
  - Drone and tile images share the same filenames (00001.png etc.).
  - Camera intrinsics are for the 800x600 images.
"""

# ----------------------- CONFIG -----------------------
BASE         = Path(__file__).parent.resolve()
DATASET_DIR  = BASE / "vpair_dataset"

# CSV with lat/lon/alt for each frame 
POSES_LATLON_CSV = DATASET_DIR / "poses_lat_long.csv"   

DRONE_DIR    = DATASET_DIR / "drone"
TILES_DIR    = DATASET_DIR / "tiles"

# Outputs 
DRONE_INFO_CSV     = DATASET_DIR / "poses_drone.csv"
TILES_INFO_CSV     = DATASET_DIR / "poses_tiles.csv"
TILE_CENTERS_CSV   = DATASET_DIR / "tile_centers_in_sat.csv"

# Camera intrinsics for 800x600 (from cam0 snippet)
FX = 750.62614972
FY = 750.26301185

# Size of each tile/drone image
IMG_W = 800
IMG_H = 600

# Global map scale: how many meters per 1 "global pixel"
# EKF will live in this coordinate system.
MAP_M_PER_PX = 1.0   # 1 px = 1 meter (simple & intuitive)

EARTH_R = 6378137.0  # meters


# ------------------ Helpers -------------------------

def ll_to_xy_m(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat  = math.radians(lat_deg)
    lon  = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    dlat = lat - lat0
    dlon = lon - lon0

    x_east  = dlon * math.cos(lat0) * EARTH_R   # east
    y_north = dlat * EARTH_R                    # north

    # EKF & images use y-down, so define +y = south
    x = x_east
    y = -y_north
    return x, y



def meters_to_global_px(x_m, y_m, x0_m, y0_m, m_per_px=MAP_M_PER_PX):
    """
    Convert metric ENU (x_m, y_m) to global pixel coords.
    """
    x_px = (x_m - x0_m) / m_per_px
    y_px = (y_m - y0_m) / m_per_px
    return x_px, y_px


# ------------------ Main processing ------------------

def main():
    if not POSES_LATLON_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV with poses: {POSES_LATLON_CSV}")

    rows = []
    with open(POSES_LATLON_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise RuntimeError("Input poses CSV is empty.")

    # Reference lat/lon for ENU origin = first row
    lat0 = float(rows[0]["lat"])
    lon0 = float(rows[0]["lon"])

    # First, compute ENU (meters) for all, and remember origin in meters
    xy_m_list = []
    for r in rows:
        lat = float(r["lat"])
        lon = float(r["lon"])
        x_m, y_m = ll_to_xy_m(lat, lon, lat0, lon0)
        xy_m_list.append((x_m, y_m))

    # Use first ENU position as (0,0)
    x0_m, y0_m = xy_m_list[0]

    # Prepare writers
    DRONE_INFO_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(DRONE_INFO_CSV, "w", newline="") as f_drone, \
         open(TILES_INFO_CSV, "w", newline="") as f_tiles, \
         open(TILE_CENTERS_CSV, "w", newline="") as f_centers:

        # ------------- poses_drone.csv -------------
        # This is what your EKF init reads (x,y,yaw etc.)
        drone_fieldnames = [
            "filename",
            "lat", "lon", "altitude",
            "x", "y",         # global pixel coordinates for EKF
            "yaw",            # rad, from CSV
            "m_per_px_x", "m_per_px_y"
        ]
        w_drone = csv.DictWriter(f_drone, fieldnames=drone_fieldnames)
        w_drone.writeheader()

        # ------------- poses_tiles.csv -------------
        # Here we store tile centers in METERS (for error computations)
        tiles_fieldnames = [
            "tile_name",
            "lat", "lon", "altitude",
            "x", "y"          # ENU meters
        ]
        w_tiles = csv.DictWriter(f_tiles, fieldnames=tiles_fieldnames)
        w_tiles.writeheader()

        # ------------- tile_centers_in_sat.csv -------------
        # This is the "global pixel" center for each tile
        centers_fieldnames = [
            "tile_name",
            "center_x", "center_y",  # global pixel coords
            "lat", "lon", "altitude",
            "m_per_px_x", "m_per_px_y"
        ]
        w_centers = csv.DictWriter(f_centers, fieldnames=centers_fieldnames)
        w_centers.writeheader()

        # ------------- loop over all images -------------
        for (r, (x_m, y_m)) in zip(rows, xy_m_list):
            filename  = r["filename"]  # e.g. "00001.png"
            lat       = float(r["lat"])
            lon       = float(r["lon"])
            altitude  = float(r["altitude"])
            yaw_rad   = float(r["yaw"])   # your snippet already in radians

            # global pixel coordinates (what EKF uses)
            x_px, y_px = meters_to_global_px(x_m, y_m, x0_m, y0_m, MAP_M_PER_PX)

            # image-based meters per pixel using pinhole & altitude
            # assuming nadir-ish and flat ground
            m_per_px_x = altitude / FX
            m_per_px_y = altitude / FY

            # ----------------- DRONE CSV -----------------
            w_drone.writerow({
                "filename": filename,
                "lat": lat,
                "lon": lon,
                "altitude": altitude,
                "x": f"{x_px:.8f}",
                "y": f"{y_px:.8f}",
                "yaw": f"{yaw_rad:.8f}",
                "m_per_px_x": f"{m_per_px_x:.8f}",
                "m_per_px_y": f"{m_per_px_y:.8f}",
            })

            # ----------------- TILES (meters) -----------------
            # for VPAIR, tile & drone share same center pose & filename
            tile_name = filename  # or f"{Path(filename).stem}.png" if needed
            w_tiles.writerow({
                "tile_name": tile_name,
                "lat": lat,
                "lon": lon,
                "altitude": altitude,
                "x": f"{x_m:.8f}",
                "y": f"{y_m:.8f}",
            })

            # ----------------- TILE CENTERS (global px) -----------------
            w_centers.writerow({
                "tile_name": tile_name,
                "center_x": f"{x_px:.8f}",
                "center_y": f"{y_px:.8f}",
                "lat": lat,
                "lon": lon,
                "altitude": altitude,
                "m_per_px_x": f"{m_per_px_x:.8f}",
                "m_per_px_y": f"{m_per_px_y:.8f}",
            })

    print("Wrote:")
    print(f"  {DRONE_INFO_CSV}")
    print(f"  {TILES_INFO_CSV}")
    print(f"  {TILE_CENTERS_CSV}")


if __name__ == "__main__":
    main()
