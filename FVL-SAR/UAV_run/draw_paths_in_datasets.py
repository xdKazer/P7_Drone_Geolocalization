# draw_gt_paths_on_overlays.py
import csv
from pathlib import Path
import json
import cv2
import numpy as np

# -------------------- Config --------------------
# Adjust this list to the datasets you want to process
SAT_LIST = [f"{i:02d}" for i in range(0, 12) if i not in (6, 7)]

# Whether to overwrite the original overlay image or create a new one
OVERWRITE = False

BASE = Path(__file__).parent.resolve()
DATASET_DIR = BASE / "UAV_VisLoc_dataset"
SAT_LONG_LAT_INFO_PATH = DATASET_DIR / "satellite_coordinates_range.csv"
output_path = BASE / "UAV_VisLoc_dataset" / "paths_overlay"
output_path.mkdir(exist_ok=True)

# Starting images copied from your main script
STARTING_DRONE_IMAGES = {
    "01": ["01_0001.JPG", "01_0080.JPG", "01_0162.JPG", "01_0241.JPG",
           "01_0323.JPG", "01_0403.JPG", "01_0486.JPG", "01_0567.JPG",
           "01_0651.JPG", "01_0732.JPG"],
    "02": ["02_0001.JPG", "02_0102.JPG", "02_0207.JPG", "02_0310.JPG",
           "02_0416.JPG", "02_0521.JPG", "02_0629.JPG", "02_0736.JPG",
           "02_0847.JPG", "02_0958.JPG"],
    "03": ["03_0001.JPG", "03_0097.JPG", "03_0193.JPG", "03_0289.JPG",
           "03_0385.JPG", "03_0481.JPG", "03_0577.JPG", "03_0673.JPG"],
    "04": ["04_0001.JPG", "04_0090.JPG", "04_0179.JPG", "04_0270.JPG",
           "04_0361.JPG", "04_0455.JPG", "04_0549.JPG", "04_0644.JPG"],
    "05": ["05_0001.JPG", "05_0041.JPG", "05_0052.JPG", "05_0076.JPG",
           "05_0116.JPG", "05_0156.JPG", "05_0196.JPG", "05_0236.JPG",
           "05_0275.JPG", "05_0315.JPG", "05_0355.JPG", "05_0395.JPG",
           "05_0434.JPG"],
    "08": ["08_0215.JPG", "08_0312.JPG", "08_0409.JPG", "08_0509.JPG",
           "08_0609.JPG", "08_0713.JPG", "08_0818.JPG", "08_0926.JPG"],
    "09": ["09_0001.JPG", "09_0129.JPG", "09_0256.JPG", "09_0384.JPG",
           "09_0512.JPG", "09_0640.JPG"],
    "10": ["10_0001.JPG", "10_0019.JPG", "10_0037.JPG", "10_0055.JPG",
           "10_0073.JPG", "10_0091.JPG", "10_0109.JPG", "10_0127.JPG"],
    "11": ["11_0003.JPG", "11_0052.JPG", "11_0101.JPG", "11_0150.JPG",
           "11_0199.JPG", "11_0248.JPG", "11_0297.JPG", "11_0346.JPG",
           "11_0395.JPG", "11_0444.JPG", "11_0493.JPG", "11_0542.JPG"],
}


# -------------------- Helpers --------------------
def load_sat_meta(sat_number: str):
    """
    Load the satellite small image scale and original size.
    Returns (sx, sy, sat_H_orig, sat_W_orig).
    """
    sat_small = DATASET_DIR / sat_number / f"satellite{sat_number}_small.png"
    sat_meta = sat_small.with_suffix(sat_small.suffix + ".json")

    if not sat_meta.exists():
        raise FileNotFoundError(f"Missing meta JSON: {sat_meta}")

    meta = json.loads(sat_meta.read_text())

    if "original_size_hw" not in meta:
        raise KeyError(f"{sat_meta} missing 'original_size_hw'")

    sat_H_orig, sat_W_orig = map(float, meta["original_size_hw"])

    if "scale" in meta:
        sx = sy = float(meta["scale"])
    elif "scale_xy" in meta:
        sx, sy = map(float, meta["scale_xy"])
    else:
        raise KeyError(f"{sat_meta} missing 'scale' or 'scale_xy'")

    return sx, sy, sat_H_orig, sat_W_orig


def latlon_to_orig_xy(lat, lon, sat_number: str, sat_H, sat_W):
    """(lat, lon) -> ORIGINAL satellite pixel (u,v) as float64."""
    with open(SAT_LONG_LAT_INFO_PATH, newline="") as f:
        for r in csv.DictReader(f):
            if r["mapname"] == f"satellite{sat_number}.tif":
                LT_lat = float(r["LT_lat_map"])
                LT_lon = float(r["LT_lon_map"])
                RB_lat = float(r["RB_lat_map"])
                RB_lon = float(r["RB_lon_map"])
                break
        else:
            raise FileNotFoundError(
                f"Bounds for satellite{sat_number}.tif not found in {SAT_LONG_LAT_INFO_PATH}"
            )

    u = (float(lon) - LT_lon) / (RB_lon - LT_lon) * sat_W
    v = (float(lat) - LT_lat) / (RB_lat - LT_lat) * sat_H
    return u, v


def draw_polylines(img, polylines, color=(0, 255, 0), thickness=8):
    """Draw multiple open polylines on img."""
    for pts in polylines:
        if len(pts) < 2:
            continue
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts_arr], isClosed=False, color=color, thickness=thickness)


# -------------------- Main --------------------
def main():
    for sat_number in SAT_LIST:
        print(f"\nProcessing satellite {sat_number}...")

        # Where the existing visualisation lives
        vis_dir = BASE / "UAV_VisLoc_dataset" / sat_number
        overlay_path = vis_dir / f"satellite{sat_number}_small.png"
        if not overlay_path.exists():
            print(f"  [skip] overlay image not found: {overlay_path}")
            continue

        # Load overlay image
        img = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] failed to read overlay: {overlay_path}")
            continue

        # Load satellite scale + original size
        try:
            sx, sy, sat_H_orig, sat_W_orig = load_sat_meta(sat_number)
        except Exception as e:
            print(f"  [skip] error loading sat meta for {sat_number}: {e}")
            continue

        # GT path data from CSV
        drone_csv = DATASET_DIR / sat_number / f"{sat_number}.csv"
        if not drone_csv.exists():
            print(f"  [skip] drone CSV missing: {drone_csv}")
            continue

        start_imgs = set(STARTING_DRONE_IMAGES.get(sat_number, []))

        polylines = []
        current_poly = []

        # NEW: flag so we IGNORE all points before the first starting image
        started = False

        with open(drone_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "")
                lat_str = row.get("lat", "")
                lon_str = row.get("lon", "")

                if not lat_str or not lon_str:
                    continue  # skip rows without position

                lat = float(lat_str)
                lon = float(lon_str)

                # original satellite pixel coords
                u_orig, v_orig = latlon_to_orig_xy(
                    lat, lon, sat_number, sat_H_orig, sat_W_orig
                )
                # display coords for the overlay image
                x_disp = u_orig * sx
                y_disp = v_orig * sy
                pt = (int(round(x_disp)), int(round(y_disp)))

                # restart polyline when we hit a starting image
                if fname in start_imgs:
                    # from now on we want to draw
                    started = True

                    if len(current_poly) >= 2:
                        polylines.append(current_poly)
                    current_poly = [pt]
                else:
                    # ONLY add points after we've hit the first starting image
                    if started:
                        current_poly.append(pt)
                    # if not started yet, we ignore these points entirely

        # flush last polyline
        if len(current_poly) >= 2:
            polylines.append(current_poly)

        if not polylines:
            print("  [info] no GT points / polylines found.")
            continue

        # NEW: thickness scaled by image resolution so the visual width looks similar
        h, w = img.shape[:2]
        ref_size = 2000  # "reference" smaller dimension in pixels
        base_thickness = 10
        thickness = max(2, int(round(base_thickness * min(h, w) / ref_size)))
        # you can tweak `ref_size` and `base_thickness` to taste

        # Draw GT path in green
        draw_polylines(img, polylines, color=(0, 255, 0), thickness=thickness)

        # Save result
        if OVERWRITE:
            out_path = output_path
        else:
            out_path = output_path / f"{sat_number}_paths_overlay.png"

        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        print(f"  Saved GT path overlay to: {out_path} (thickness={thickness})")


if __name__ == "__main__":
    main()
