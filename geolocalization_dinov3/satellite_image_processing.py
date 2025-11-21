import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
import os
from pathlib import Path

# ===============================================================
# Create 1x1 km tiles from a georeferenced satellite image (.tif)
# ===============================================================

# ---- Paths ----
input_file = "geolocalization_dinov3/dataset_data/satellite_images/satellite03.tif"
output_folder = Path("geolocalization_dinov3/tiles_png_1km")
output_folder.mkdir(parents=True, exist_ok=True)

# ---- Tile size (in meters) ----
tile_size_m = 1000  # 1 km x 1 km

# ---- Reference geographic bounds (to compute meters per pixel) ----
lat_long_1 = (32.355491, 119.805926)
lat_long_2 = (32.290290, 119.900052)

geo_center = (
    (lat_long_1[0] + lat_long_2[0]) / 2,
    (lat_long_1[1] + lat_long_2[1]) / 2
)

# ---- Convert geographic span to meters ----
meters_per_degree_lat = 111320
meters_per_degree_lon = 111320 * np.cos(np.radians(geo_center[0]))

geo_span = (
    abs(lat_long_1[0] - lat_long_2[0]),
    abs(lat_long_1[1] - lat_long_2[1])
)
geo_span_meters = (
    geo_span[0] * meters_per_degree_lat,
    geo_span[1] * meters_per_degree_lon
)
print(f"Geographic span (m): {geo_span_meters}")

# ---- Open image ----
with rasterio.open(input_file) as src:
    print(f"Image size: {src.width} x {src.height}")

    # Compute resolution (meters per pixel)
    res_m_per_pix_lat = geo_span_meters[0] / src.height
    res_m_per_pix_lon = geo_span_meters[1] / src.width
    print(f"Resolution: {res_m_per_pix_lat:.3f} m/px (lat), {res_m_per_pix_lon:.3f} m/px (lon)")

    # ---- Convert desired tile size to pixels ----
    tile_height_px = int(round(tile_size_m / res_m_per_pix_lat))
    tile_width_px = int(round(tile_size_m / res_m_per_pix_lon))
    print(f"Tile size (pixels): {tile_height_px} x {tile_width_px} - height x width")
    print(f"Resolution per tile: {geo_span_meters[0]/(tile_height_px):.3f} m/px (lat), {geo_span_meters[1]/(tile_width_px):.3f} m/px (lon)")

    def build_starts_with_fractional_overlap(size, tile_size, overlap_frac):
        overlap = int(tile_size * overlap_frac)
        step = tile_size - overlap
        starts = list(range(0, max(1, size - tile_size + 1), step))
        if starts[-1] + tile_size < size:
            starts.append(size - tile_size)
        return starts

    ys = build_starts_with_fractional_overlap(src.height, tile_height_px, 0.1)
    xs = build_starts_with_fractional_overlap(src.width, tile_width_px, 0.1)

    print(f"rows: {len(ys)}  cols: {len(xs)}  -> total tiles: {len(ys) * len(xs)}")

    count = 0
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            print(f"Processing tile {count + 1} / {len(ys) * len(xs)} at pixel (y: {y0}, x: {x0})")
            window = Window(col_off=x0, row_off=y0,
                            width=tile_width_px, height=tile_height_px)
            img = src.read(window=window)
            img = np.moveaxis(img, 0, -1)

            if img.size == 0:
                continue

            tile_img = Image.fromarray(img)
            out_path = output_folder / f"sat_tile_y{y0}_x{x0}.png"
            tile_img.save(out_path)

            count += 1

    print(f"Wrote {count} 1x1 km tiles to: {output_folder}")

    # Reconstruct full satellite image and save as png
    scale = 0.3
    full_img = src.read()
    full_img = np.moveaxis(full_img, 0, -1)
    full_img_pil = Image.fromarray(full_img)
    new_size = (int(full_img_pil.width * scale), int(full_img_pil.height * scale))
    full_img_pil = full_img_pil.resize(new_size, Image.LANCZOS)
    full_img_pil.save("geolocalization_dinov3/full_satellite_image_small.png")