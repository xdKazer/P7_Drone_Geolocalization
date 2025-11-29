import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
from pathlib import Path
import math

input_file = "geolocalization_dinov3/dataset_data/satellite_images/satellite03.tif"
output_folder = Path("geolocalization_dinov3/tiles_uniform")
output_folder.mkdir(parents=True, exist_ok=True)

target = 2048  # desired tile size

with rasterio.open(input_file) as src:
    H, W = src.height, src.width
    print(f"Image size: {W} x {H}")

    # Compute number of tiles to get closest size to target
    nx = round(W / target)
    ny = round(H / target)

    # Compute exact tile size to fill the image
    tile_w = W // nx
    tile_h = H // ny

    print(f"Dividing into {nx} x {ny} tiles")
    print(f"Uniform tile size: {tile_w} x {tile_h}")

    count = 0
    for i in range(ny):
        for j in range(nx):
            x0 = j * tile_w
            y0 = i * tile_h

            # All tiles use the same size
            window = Window(col_off=x0, row_off=y0,
                            width=tile_w, height=tile_h)

            img = src.read(window=window)
            img = np.moveaxis(img, 0, -1)

            tile = Image.fromarray(img)
            out_path = output_folder / f"tile_y{y0}_x{x0}.png"
            tile.save(out_path)

            count += 1
            print(f"Saved tile {count}/{nx*ny}")

    scale = 0.3
    full_img = src.read()
    full_img = np.moveaxis(full_img, 0, -1)
    full_img_pil = Image.fromarray(full_img)
    new_size = (int(full_img_pil.width * scale), int(full_img_pil.height * scale))
    full_img_pil = full_img_pil.resize(new_size, Image.LANCZOS)
    full_img_pil.save("geolocalization_dinov3/full_satellite_image_small.png")

print(f"\nWrote {count} tiles to: {output_folder}")