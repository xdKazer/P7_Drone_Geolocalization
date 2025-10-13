import cv2
import pathlib
 
 
# 3976 x 2652
# ---- Input path ----
dataset_path = pathlib.Path(__file__).parent.resolve() / "UAV_VisLoc_dataset"
tif_path = dataset_path / "03" / "satellite03.tif"
output_dir = dataset_path / "03" / "satellite_tiles"
output_dir.mkdir(parents=True, exist_ok=True)
 
# ---- Read the image ----
sattelite_img = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
 
#split the image into tiles of H 3976 x W 2652
tile_height = 3899
tile_width = 2700
sat_height, sat_width, _ = sattelite_img.shape
print(f"sat size: {sat_height} x {sat_width}")
clean_tiles_height = sat_height % tile_height
clean_tiles_width = sat_width % tile_width
num_tiles_height = sat_height // tile_height
num_tiles_width = sat_width // tile_width
if clean_tiles_width != 0 or clean_tiles_height != 0:
    print("Image size is not divisible by tile size")
    print(f"Image size: {clean_tiles_height} x {clean_tiles_width}")
    
 
for i in range(num_tiles_height):
    for j in range(num_tiles_width):
        tile = sattelite_img[i*tile_height:(i+1)*tile_height, j*tile_width:(j+1)*tile_width]
        cv2.imwrite( str(output_dir) / f"sat_tile_{i}_{j}.png", tile)