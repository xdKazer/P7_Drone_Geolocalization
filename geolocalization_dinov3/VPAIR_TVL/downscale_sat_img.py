import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

# -------------------- CONFIG --------------------

BASE_DIR = Path(__file__).parent.resolve()
print(f"[info] Base dir: {BASE_DIR}")

TILES_DIR = BASE_DIR / "tiles"
CENTERS_CSV = BASE_DIR / "tile_centers_in_sat.csv"   # your file with center_x, center_y, m_per_px_x, m_per_px_y
OUT_DIR = BASE_DIR / "sat_mosaic"
RESULTS_CSV = BASE_DIR / "VPAIR_outputs" / "results_VPAIR.csv"

MAX_LONG_SIDE_SMALL = 6000  # longest side for downscaled mosaic

# -------------------- UTILS --------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_centers_and_meta(csv_path: Path):
    """
    Load tile centers and GSD from CSV.

    Expected columns:
        tile_name, center_x, center_y, ..., m_per_px_x, m_per_px_y

    Returns:
        rows: list of dicts with keys:
              'tile_name', 'center_x', 'center_y', 'm_per_px_x', 'm_per_px_y'
    """
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "tile_name": r["tile_name"],
                "center_x": float(r["center_x"]),
                "center_y": float(r["center_y"]),
                "m_per_px_x": float(r["m_per_px_x"]),
                "m_per_px_y": float(r["m_per_px_y"]),
            })
    if not rows:
        raise RuntimeError(f"No rows read from {csv_path}")
    return rows


def compute_mosaic_bounds(rows, tiles_dir: Path):
    """
    Compute mosaic bounds in global meters/pixels (1 px = 1 m).

    For each tile:
      - load image to know original px size
      - scale to metric size using m_per_px_x/y
      - compute bounding box in global coords
    """
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for r in rows:
        tile_path = tiles_dir / r["tile_name"]
        img = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read tile image: {tile_path}")

        h, w = img.shape[:2]
        mppx = r["m_per_px_x"]
        mppy = r["m_per_px_y"]

        # new tile size in global-pixel units (1 px = 1 m)
        new_w = w * mppx      # meters
        new_h = h * mppy      # meters

        cx = r["center_x"]
        cy = r["center_y"]

        x0 = cx - new_w / 2.0
        y0 = cy - new_h / 2.0
        x1 = cx + new_w / 2.0
        y1 = cy + new_h / 2.0

        min_x = min(min_x, x0)
        min_y = min(min_y, y0)
        max_x = max(max_x, x1)
        max_y = max(max_y, y1)

    # width/height in mosaic pixels (1 px = 1 m)
    width = int(math.ceil(max_x - min_x))
    height = int(math.ceil(max_y - min_y))

    return min_x, min_y, max_x, max_y, width, height


def build_mosaic(rows, tiles_dir: Path, out_dir: Path):
    """
    Build the full-resolution mosaic.

    Returns:
        mosaic: np.ndarray (H, W, 3) uint8
        meta: dict with min_x, min_y, max_x, max_y, width, height
    """
    ensure_dir(out_dir)

    min_x, min_y, max_x, max_y, width, height = compute_mosaic_bounds(rows, tiles_dir)
    print(f"[info] Mosaic bounds: x=[{min_x:.2f},{max_x:.2f}], y=[{min_y:.2f},{max_y:.2f}]")
    print(f"[info] Mosaic size: {width} x {height} px (1px = 1m)")

    # initialize mosaic as black
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)

    for r in rows:
        tile_name = r["tile_name"]
        tile_path = tiles_dir / tile_name
        img = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] Skipping missing tile {tile_path}")
            continue

        h, w = img.shape[:2]
        mppx = r["m_per_px_x"]
        mppy = r["m_per_px_y"]
        cx = r["center_x"]
        cy = r["center_y"]

        # new tile size in mosaic pixels (meters)
        new_w = int(round(w * mppx))
        new_h = int(round(h * mppy))
        if new_w <= 0 or new_h <= 0:
            print(f"[warn] Non-positive scaled size for tile {tile_name}, skipping.")
            continue

        img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        half_w = new_w / 2.0
        half_h = new_h / 2.0

        # global coords (meters/pixels) -> mosaic indices
        x0_global = cx - half_w
        y0_global = cy - half_h

        # shift by min_x, min_y so everything is >= 0
        x0 = int(round(x0_global - min_x))
        y0 = int(round(y0_global - min_y))

        x1 = x0 + new_w
        y1 = y0 + new_h

        # clip to mosaic bounds, just in case
        x0_cl = max(0, x0)
        y0_cl = max(0, y0)
        x1_cl = min(width, x1)
        y1_cl = min(height, y1)

        if x1_cl <= x0_cl or y1_cl <= y0_cl:
            print(f"[warn] Scaled tile {tile_name} is fully outside mosaic bounds, skipping.")
            continue

        tile_x0 = x0_cl - x0
        tile_y0 = y0_cl - y0
        tile_x1 = tile_x0 + (x1_cl - x0_cl)
        tile_y1 = tile_y0 + (y1_cl - y0_cl)

        mosaic[y0_cl:y1_cl, x0_cl:x1_cl] = img_scaled[tile_y0:tile_y1, tile_x0:tile_x1]

    meta = {
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
        "width": int(width),
        "height": int(height),
        "meters_per_mosaic_px": 1.0,  # by design
    }

    full_path = out_dir / "sat_mosaic_full.png"
    cv2.imwrite(str(full_path), mosaic)
    print(f"[info] Saved full-resolution mosaic to {full_path}")

    meta_path = out_dir / "sat_mosaic_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[info] Saved mosaic meta to {meta_path}")

    return mosaic, meta


def build_downscaled_mosaic(mosaic: np.ndarray, meta: dict, out_dir: Path, max_long_side=3000):
    """
    Build a downscaled mosaic where the longest side is <= max_long_side.
    Saves small image and a meta JSON including the scale factor.
    """
    h, w = mosaic.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        # no need to resize
        small = mosaic.copy()
        scale = 1.0
    else:
        scale = max_long_side / float(long_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        small = cv2.resize(mosaic, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"[info] Downscaled mosaic to {new_w} x {new_h} (scale={scale:.6f})")

    small_path = out_dir / "sat_mosaic_small.png"
    cv2.imwrite(str(small_path), small)
    print(f"[info] Saved downscaled mosaic to {small_path}")

    small_meta = dict(meta)
    small_meta["downscale_factor"] = scale

    small_meta_path = out_dir / "sat_mosaic_small_meta.json"
    small_meta_path.write_text(json.dumps(small_meta, indent=2))
    print(f"[info] Saved small mosaic meta to {small_meta_path}")

    return small, small_meta


# -------------------- OVERLAY: PREDICTIONS + MEASUREMENTS + SEARCH AREA --------------------

def draw_ellipse(img, center_xy, cov2x2, k_sigma=2.0, color=(0, 255, 255), thickness=2):
    """
    Draw a k-sigma ellipse for a 2x2 covariance matrix.
    center_xy: (x, y) in image pixel coords
    cov2x2: 2x2 covariance
    """
    vals, vecs = np.linalg.eigh(cov2x2)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    # axis lengths
    a = k_sigma * math.sqrt(max(vals[0], 0.0))
    b = k_sigma * math.sqrt(max(vals[1], 0.0))
    # angle of major axis
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))

    center = (int(round(center_xy[0])), int(round(center_xy[1])))
    axes = (int(round(a)), int(round(b)))

    if axes[0] <= 0 or axes[1] <= 0:
        return

    cv2.ellipse(
        img,
        center,
        axes,
        angle,
        0,
        360,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def overlay_results_on_mosaic(
    mosaic: np.ndarray,
    meta: dict,
    results_csv: Path,
    out_path: Path,
    scale_for_small: float = 1.0,
):
    """
    Overlay EKF predictions, measurements, and optional search ellipses on a mosaic.

    Assumes results CSV has (any subset of) columns:
        x_meas, y_meas,
        x_ekf, y_ekf,
        x_pred, y_pred,
        P_xx, P_xy, P_yy   (for ellipse)

    All coordinates are in the same global frame as center_x/center_y (meters),
    which we map to mosaic pixel coords using:
        x_img = (x_global - min_x) * scale_for_small
        y_img = (y_global - min_y) * scale_for_small
    """
    img = mosaic.copy()
    min_x = meta["min_x"]
    min_y = meta["min_y"]

    if not results_csv.exists():
        print(f"[warn] Results CSV {results_csv} not found, skipping overlay.")
        return img

    with results_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        has_meas = "x_meas" in fieldnames and "y_meas" in fieldnames
        has_ekf = "x_ekf" in fieldnames and "y_ekf" in fieldnames
        has_pred = "x_pred" in fieldnames and "y_pred" in fieldnames  # optional
        has_cov = all(fn in fieldnames for fn in ("P_xx", "P_xy", "P_yy"))

        meas_color = (0, 0, 255)      # red
        ekf_color = (0, 255, 0)       # green
        pred_color = (0, 255, 255)    # yellow
        search_color = (255, 0, 255)  # magenta

        for row in reader:
            # Measurements
            if has_meas:
                try:
                    xm = float(row["x_meas"])
                    ym = float(row["y_meas"])
                    x_img = (xm - min_x) * scale_for_small
                    y_img = (ym - min_y) * scale_for_small
                    cv2.circle(img, (int(round(x_img)), int(round(y_img))), 3, meas_color, -1)
                except ValueError:
                    pass

            # EKF posterior
            if has_ekf:
                try:
                    xe = float(row["x_ekf"])
                    ye = float(row["y_ekf"])
                    x_img = (xe - min_x) * scale_for_small
                    y_img = (ye - min_y) * scale_for_small
                    cv2.circle(img, (int(round(x_img)), int(round(y_img))), 3, ekf_color, -1)
                except ValueError:
                    pass

            # EKF prediction (if saved)
            if has_pred:
                try:
                    xp = float(row["x_pred"])
                    yp = float(row["y_pred"])
                    x_img = (xp - min_x) * scale_for_small
                    y_img = (yp - min_y) * scale_for_small
                    cv2.circle(img, (int(round(x_img)), int(round(y_img))), 3, pred_color, -1)
                except ValueError:
                    pass

            # Search ellipse from covariance (if provided)
            if has_cov and has_pred:
                try:
                    P_xx = float(row["P_xx"])
                    P_xy = float(row["P_xy"])
                    P_yy = float(row["P_yy"])
                    cov = np.array([[P_xx, P_xy], [P_xy, P_yy]], dtype=np.float64)

                    xp = float(row["x_pred"])
                    yp = float(row["y_pred"])
                    cx_img = (xp - min_x) * scale_for_small
                    cy_img = (yp - min_y) * scale_for_small

                    # scale covariance to image pixels (since 1 px = 1 m before scaling)
                    S = scale_for_small
                    J = np.diag([S, S])  # simple isotropic scaling
                    cov_img = J @ cov @ J.T

                    draw_ellipse(img, (cx_img, cy_img), cov_img, k_sigma=2.0, color=search_color, thickness=1)
                except ValueError:
                    pass

    cv2.imwrite(str(out_path), img)
    print(f"[info] Saved overlay mosaic to {out_path}")
    return img


# -------------------- MAIN --------------------

def main():
    ensure_dir(OUT_DIR)

    # 1) Build full-resolution mosaic
    rows = load_centers_and_meta(CENTERS_CSV)
    mosaic_full, meta_full = build_mosaic(rows, TILES_DIR, OUT_DIR)

    # 2) Build downscaled mosaic
    mosaic_small, meta_small = build_downscaled_mosaic(
        mosaic_full, meta_full, OUT_DIR, max_long_side=MAX_LONG_SIDE_SMALL
    )


if __name__ == "__main__":
    main()
