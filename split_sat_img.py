import cv2
import pathlib

# ---- Input path ----
dataset_path = pathlib.Path(__file__).parent.resolve() / "UAV_VisLoc_dataset"
tif_path = dataset_path / "03" / "satellite03.tif"
output_dir = dataset_path / "03" / "satellite_tiles"
output_dir.mkdir(parents=True, exist_ok=True)

 
# ---- Read the image ----
sattelite_img = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
drone_img = cv2.imread(str(dataset_path / "03" / "drone" / "03_0738.JPG"), cv2.IMREAD_UNCHANGED)

import cv2, numpy as np
from pathlib import Path



img = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(tif_path)

# If the TIFF is 16-bit or float, scale to 8-bit for quick viewing
def to_8bit(x):
    if x is None: return None
    if x.dtype == np.uint8: return x
    if x.dtype == np.uint16:
        x8 = (x / 256).astype(np.uint8)         # simple downscale
        return x8
    if x.dtype in (np.float32, np.float64):
        mn, mx = np.min(x), np.max(x)
        x8 = ((x - mn) / (mx - mn + 1e-12) * 255).astype(np.uint8)
        return x8
    return x

preview = to_8bit(img)
ok = cv2.imwrite("sat_preview.png", preview)
print("wrote:", ok, "to sat_preview.png")

 
#split the image into tiles 
tile_height, tile_width, _ = drone_img.shape
tile_height = tile_height // 2
tile_width = tile_width // 2
sat_height, sat_width, _ = sattelite_img.shape
print(f"sat size: {sat_height} x {sat_width}")
print(f"drone size: {tile_height} x {tile_width}")
print(f"sat ratio: {sat_height // tile_height} x {sat_width // tile_width}")
print(f"sat rest: {sat_height % tile_height} x {sat_width % tile_width}")

    
 
for i in range(sat_height // tile_height):
    for j in range(sat_width // tile_width):
        if j == (sat_width // tile_width)-1 and i == (sat_height // tile_height)-1:
            tile = sattelite_img[i*tile_height:(i+1)*tile_height + (sat_height % tile_height), j*tile_width:(j+1)*tile_width + (sat_width % tile_width) ]
            cv2.imwrite( str(output_dir / f"sat_tile_{i}_{j}.png"), tile)
        if j == (sat_width // tile_width)-1:
            tile = sattelite_img[i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width + (sat_width % tile_width) ]
            cv2.imwrite( str(output_dir / f"sat_tile_{i}_{j}.png"), tile)
        elif i == (sat_height // tile_height)-1:
            tile = sattelite_img[i*tile_height:(i+1)*tile_height + (sat_height % tile_height), j*tile_width:(j+1)*tile_width]
            cv2.imwrite( str(output_dir / f"sat_tile_{i}_{j}.png"), tile)
        else: 
            tile = sattelite_img[i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width]
            print(str(output_dir / f"sat_tile_{i}_{j}.png"))
            cv2.imwrite( str(output_dir / f"sat_tile_{i}_{j}.png"), tile)