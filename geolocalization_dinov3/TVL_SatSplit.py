import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from PIL import Image
import numpy as np
from pathlib import Path
import math

# Specifically for satellite 09, This code is needed to stitch the 4 provided tiles into one large image
tiles = [
    "geolocalization_dinov3/dataset_data/satellite_images/satellite09_01-01.tif",
    "geolocalization_dinov3/dataset_data/satellite_images/satellite09_01-02.tif",
    "geolocalization_dinov3/dataset_data/satellite_images/satellite09_02-01.tif",
    "geolocalization_dinov3/dataset_data/satellite_images/satellite09_02-02.tif"
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

stitched_file = "geolocalization_dinov3/dataset_data/satellite_images/satellite09.tif"


with rasterio.open(stitched_file, "w", **meta) as dst:
    dst.write(mosaic)

for s in src_files: s.close()
print(f"Mosaic created: {stitched_file}")

# -------- TILE THE MOSAIC --------

input_satellite = "geolocalization_dinov3/dataset_data/satellite_images/satellite08.tif"
output_folder = Path("geolocalization_dinov3/tiles_uniform")
output_folder.mkdir(parents=True, exist_ok=True)

target = 1024          # desired tile size
overlap = 0.25         # 25% overlap

with rasterio.open(stitched_file) as src:  # Change input_satellite to stitched_file for satellite 09
    H, W = src.height, src.width
    print(f"Stitched Image size: {W} x {H}")

    nx = round(W / target)
    ny = round(H / target)

    tile_w = W // nx
    tile_h = H // ny

    stride_w = int(tile_w * (1 - overlap))
    stride_h = int(tile_h * (1 - overlap))

    tiles_x = math.ceil((W - tile_w) / stride_w) + 1
    tiles_y = math.ceil((H - tile_h) / stride_h) + 1

    print(f"Tile size: {tile_w} x {tile_h}")
    print(f"Stride: {stride_w} x {stride_h}")
    print(f"Creating {tiles_x} x {tiles_y} tiles with 25% overlap")

    count = 0
    for i in range(tiles_y):
        for j in range(tiles_x):

            x0 = j * stride_w
            y0 = i * stride_h

            if x0 + tile_w > W: x0 = W - tile_w
            if y0 + tile_h > H: y0 = H - tile_h

            window = Window(x0, y0, tile_w, tile_h)
            tile = src.read(window=window)
            tile = np.moveaxis(tile, 0, -1)

            Image.fromarray(tile).save(output_folder / f"tile_y{y0}_x{x0}.png")

            count += 1
            print(f"Saved tile {count}/{tiles_x * tiles_y}")


    # Save downscaled visualization
    scale = 0.3
    full_img = src.read()
    full_img = np.moveaxis(full_img, 0, -1)
    full_img_pil = Image.fromarray(full_img)
    new_size = (int(full_img_pil.width * scale), int(full_img_pil.height * scale))
    full_img_pil = full_img_pil.resize(new_size, Image.LANCZOS)
    full_img_pil.save("geolocalization_dinov3/full_satellite_image_small.png")

print(f"\nWrote {count} tiles to: {output_folder}")