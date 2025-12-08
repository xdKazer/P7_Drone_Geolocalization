import os
import glob
import csv
import cv2 as cv
import numpy as np
from pathlib import Path

# ===================== CONFIG =====================
BASE = Path(__file__).parent.resolve()
DATASET_DIR = BASE / "vpair_dataset"
TILES_DIR = DATASET_DIR / "tiles"
PATCH_HALF = 100                     # 200x200 patch around center
OUT_MOSAIC = DATASET_DIR / "sat_img.png"    # Mosaic image output
OUT_CENTERS = DATASET_DIR / "tile_centers_in_sat.csv"   # CSV storing center pixel positions
# ===================== HELPERS =====================

def load_tile_paths(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        raise RuntimeError(f"No PNG images found in: {folder}")
    return paths

# ===================== STEP 1: OFFSET COMPUTATION =====================

def compute_offsets(tile_paths):
    """
    Compute cumulative offsets for all tiles relative to the first one.
    Return:
      offsets: list[(off_x, off_y)]
      tile_size: (W, H)
      bbox: (min_x, min_y, max_x, max_y)
    """
    first_color = cv.imread(tile_paths[0])
    if first_color is None:
        raise RuntimeError("Could not read first tile.")

    H, W = first_color.shape[:2]

    ph = min(PATCH_HALF, H//4)
    pw = min(PATCH_HALF, W//4)
    cx, cy = W//2, H//2

    offsets = [(0.0, 0.0)]  # First tile at (0,0)

    min_x = 0.0
    min_y = 0.0
    max_x = W
    max_y = H

    prev_gray = cv.cvtColor(first_color, cv.COLOR_BGR2GRAY)

    for i in range(1, len(tile_paths)):
        print(f"[{i+1}/{len(tile_paths)}] Matching: "
              f"{os.path.basename(tile_paths[i-1])} -> {os.path.basename(tile_paths[i])}")

        curr_color = cv.imread(tile_paths[i])
        if curr_color is None:
            print("  WARNING: Could not read image.")
            offsets.append(offsets[-1])
            continue

        curr_gray = cv.cvtColor(curr_color, cv.COLOR_BGR2GRAY)

        # Extract patch from previous image
        patch = prev_gray[cy-ph:cy+ph, cx-pw:cx+pw]

        # Template matching
        res = cv.matchTemplate(curr_gray, patch, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(res)
        px, py = max_loc
        print(f"  NCC = {max_val:.4f} at {max_loc}")

        # Centers
        prev_center = np.array([cx, cy])
        curr_center = np.array([px + pw, py + ph])

        # Required shift
        dx, dy = prev_center - curr_center
        dx, dy = float(dx), float(dy)

        # Cumulative offset
        last_x, last_y = offsets[-1]
        off_x = last_x + dx
        off_y = last_y + dy
        offsets.append((off_x, off_y))

        # Update bounding box
        min_x = min(min_x, off_x)
        min_y = min(min_y, off_y)
        max_x = max(max_x, off_x + W)
        max_y = max(max_y, off_y + H)

        prev_gray = curr_gray

    return offsets, (W, H), (min_x, min_y, max_x, max_y)

# ===================== STEP 2: CREATE MOSAIC + SAVE CENTERS =====================

def create_mosaic_and_center_csv(tile_paths, offsets, tile_size, bbox,
                                 mosaic_out, centers_out):

    W, H = tile_size
    min_x, min_y, max_x, max_y = bbox

    mosaic_w = int(np.ceil(max_x - min_x))
    mosaic_h = int(np.ceil(max_y - min_y))

    print(f"Allocating mosaic: {mosaic_w} x {mosaic_h}")

    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    # CSV init
    with open(centers_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "center_x", "center_y"])

        for (path, (off_x, off_y)) in zip(tile_paths, offsets):
            img = cv.imread(path)
            if img is None:
                print("Could not read:", path)
                continue

            # Top-left inside mosaic
            x = int(round(off_x - min_x))
            y = int(round(off_y - min_y))

            # Paste image
            mosaic[y:y+H, x:x+W] = img

            # Center pixel in mosaic
            center_x = x + W/2
            center_y = y + H/2

            writer.writerow([
                os.path.basename(path),
                center_x,
                center_y
            ])

    cv.imwrite(mosaic_out, mosaic)
    print(f"Saved mosaic to: {mosaic_out}")
    print(f"Saved tile centers to: {centers_out}")

# ===================== MAIN =====================

if __name__ == "__main__":
    tile_paths = load_tile_paths(TILES_DIR)
    print(f"Found {len(tile_paths)} tiles.")

    offsets, tile_size, bbox = compute_offsets(tile_paths)

    create_mosaic_and_center_csv(
        tile_paths,
        offsets,
        tile_size,
        bbox,
        OUT_MOSAIC,
        OUT_CENTERS
    )
