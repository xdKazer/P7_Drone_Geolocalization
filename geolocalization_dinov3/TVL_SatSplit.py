import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
from pathlib import Path
import math

input_file = "geolocalization_dinov3/dataset_data/satellite_images/satellite03.tif"
output_folder = Path("geolocalization_dinov3/tiles_uniform")
output_folder.mkdir(parents=True, exist_ok=True)

target = 2048           # desired tile size
overlap = 0.25          # 25% overlap

with rasterio.open(input_file) as src:
    H, W = src.height, src.width
    print(f"Image size: {W} x {H}")

    # Number of tiles without overlap (as baseline)
    nx = round(W / target)
    ny = round(H / target)

    # Uniform tile size
    tile_w = W // nx
    tile_h = H // ny

    # Overlap stride
    stride_w = int(tile_w * (1 - overlap))
    stride_h = int(tile_h * (1 - overlap))

    # Total tiles needed to cover the image with overlap
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

            # Clip windows at borders
            if x0 + tile_w > W:
                x0 = W - tile_w
            if y0 + tile_h > H:
                y0 = H - tile_h

            window = Window(col_off=x0, row_off=y0,
                            width=tile_w, height=tile_h)

            img = src.read(window=window)
            img = np.moveaxis(img, 0, -1)

            tile = Image.fromarray(img)
            out_path = output_folder / f"tile_y{y0}_x{x0}.png"
            tile.save(out_path)

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