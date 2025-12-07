import cv2
import math
from pathlib import Path
import numpy as np
import csv
import rasterio
from rasterio.merge import merge


# ----------------- Config -----------------
BASE        = Path(__file__).parent.resolve()
dataset_dir = BASE / "UAV_VisLoc_dataset"
sat_number  = "09"
tif_path = dataset_dir / sat_number / f"satellite{sat_number}"

# Specifically for satellite 09, This code is needed to stitch the 4 provided tiles into one large image
tiles = [
    f"{tif_path}_01-01.tif",
    f"{tif_path}_01-02.tif",
    f"{tif_path}_02-01.tif",
    f"{tif_path}_02-02.tif"
]

src_files = [rasterio.open(t) for t in tiles]
mosaic, transform = merge(src_files)

# Use metadata of first tile but update to mosaic size
meta = src_files[0].meta.copy()
meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": transform
})

stitched_file = f"{tif_path}.tif"


with rasterio.open(stitched_file, "w", **meta) as dst:
    dst.write(mosaic)

for s in src_files: s.close()
print(f"Mosaic created: {stitched_file}")

# -------- TILE THE MOSAIC --------