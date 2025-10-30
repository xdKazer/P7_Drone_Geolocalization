import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Script to crop a 2km x 2km area around a drone position from a satellite image, Will be used for datasets gather online --- #

# WARNING:
# --- If the drone position is close to corners of the satellite image, the resulting cropped image may be smaller than 2km x 2km --- #

# NOTE:
# --- We may need to change this code later, to produce roughly 2x2 areas of the entire image, instead of just cropping around the drone position --- #
# --- So that when the drone leaves the cropped area, we still have satellite images to match it with, in the new area --- #

input_file = "geolocalization_dinov3/dataset_data/satellite_images/satellite03.tif"
output_folder = "tiles_png_uniform"
os.makedirs(output_folder, exist_ok=True)

desired_tile_width = 3976 / 2
desired_tile_height = 2652 / 2

lat_long_1 = (32.355491, 119.805926) # North, East
lat_long_2 = (32.290290, 119.900052) # North, East
# Compute the geographic span and center
geo_span = (abs(lat_long_1[0] - lat_long_2[0]), abs(lat_long_1[1] - lat_long_2[1])) # (degrees in lat, degrees in lon) - absolute value
geo_center = ((lat_long_1[0] + lat_long_2[0]) / 2, (lat_long_1[1] + lat_long_2[1]) / 2) # 

# Initial take of position of drone - Used to crop satellite image to 2x2 km area around drone
drone_position = (32.3, 119.87) # (lat, lon)

# Covert the span to meters using the flat earth approximation
meters_per_degree_lat = 111320
meters_per_degree_lon = 111320 * np.cos(np.radians(geo_center[0]))

geo_span_meters = (geo_span[0] * meters_per_degree_lat, geo_span[1] * meters_per_degree_lon) #  (meters in lat, meters in lon)
print("Geographic span in meters: ", geo_span_meters)

# Use information to crop a 2km x 2km area around the drone position
crop_size_meters = 2000  # 2 km
drone_position_meters = (drone_position[0] * meters_per_degree_lat, drone_position[1] * meters_per_degree_lon)

# Compute the window to crop
left = drone_position_meters[1] - crop_size_meters / 2
right = drone_position_meters[1] + crop_size_meters / 2
top = drone_position_meters[0] - crop_size_meters / 2
bottom = drone_position_meters[0] + crop_size_meters / 2

with rasterio.open(input_file) as src:
    img_crs = src.crs
    img_transform = src.transform

    print("image size is: ", src.width, src.height)

    # --- Convert meters to degrees
    meters_per_degree_lat = 111320
    meters_per_degree_lon = 111320 * np.cos(np.radians(drone_position[0]))

    crop_size_deg_lat = crop_size_meters / meters_per_degree_lat
    crop_size_deg_lon = crop_size_meters / meters_per_degree_lon

    # Bounding box in degrees
    lat_min = drone_position[0] - crop_size_deg_lat / 2
    lat_max = drone_position[0] + crop_size_deg_lat / 2
    lon_min = drone_position[1] - crop_size_deg_lon / 2
    lon_max = drone_position[1] + crop_size_deg_lon / 2

    # --- Convert bounding box to pixel indices using affine transform
    # col = (lon - c) / a
    # row = (lat - f) / e  (e usually negative)
    # Compute raw row/col indices from drone bounding box
    row1 = (lat_max - img_transform.f) / img_transform.e
    row2 = (lat_min - img_transform.f) / img_transform.e
    col1 = (lon_min - img_transform.c) / img_transform.a
    col2 = (lon_max - img_transform.c) / img_transform.a

    # Convert to integer indices
    row_min = int(np.floor(min(row1, row2)))
    row_max = int(np.ceil(max(row1, row2)))
    col_min = int(np.floor(min(col1, col2)))
    col_max = int(np.ceil(max(col1, col2)))

    # --- Clamp to image bounds ---
    row_min = max(row_min, 0)
    row_max = min(row_max, src.height)
    col_min = max(col_min, 0)
    col_max = min(col_max, src.width)

    print(col_min, row_min)

    # --- Ensure positive width/height ---
    height = max(row_max - row_min, 1)  # at least 1 row
    width = max(col_max - col_min, 1)   # at least 1 column

    # Create the window
    window = Window(col_off=col_min, row_off=row_min, width=width, height=height)
    print("Pixel window:", window)

    # --- Read cropped image
    img_cropped = src.read(window=window)
    img_cropped = np.moveaxis(img_cropped, 0, -1)

    # Get dimensions of the cropped image
    img_height, img_width = img_cropped.shape[:2]

    # Compute number of tiles (at least 1)
    num_tiles_width = max(round(img_width / desired_tile_width), 1)
    num_tiles_height = max(round(img_height / desired_tile_height), 1)

    # Compute actual tile size (at least 1)
    tile_width = max(img_width // num_tiles_width, 1)
    tile_height = max(img_height // num_tiles_height, 1)

    print(f"Adjusted tile size: {tile_width}x{tile_height}")
    print(f"Number of tiles: {num_tiles_width}x{num_tiles_height}")

    tile_count = 0

    print("Total image count should be: " + str(num_tiles_width * num_tiles_height))

    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            top = i * tile_height
            left = j * tile_width
            bottom = top + tile_height
            right = left + tile_width
            print(f"Processing tile {tile_count}/{num_tiles_width * num_tiles_height}: ({left}, {top}) to ({right}, {bottom})")

            # Slice the cropped image
            tile_data = img_cropped[top:bottom, left:right]

            # Convert to PIL Image
            tile_img = Image.fromarray(tile_data)

            # Save
            tile_filename = os.path.join(output_folder, f"tile_{tile_count}.png")
            tile_img.save(tile_filename)

            tile_count += 1

    print(f"Created {tile_count} uniform PNG tiles.")

    pca_folder = "tiles_png_uniform"

    # Create empty canvas
    reconstructed_img = Image.new("RGB", (num_tiles_width * tile_width, num_tiles_height * tile_height))

    # Paste tiles
    tile_count = 0
    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            tile_path = os.path.join(pca_folder, f"tile_{tile_count}.png")
            tile_img = Image.open(tile_path).convert("RGB")
            
            left = j * tile_width
            top = i * tile_height
            reconstructed_img.paste(tile_img, (left, top))
            
            tile_count += 1

    # Save and display
    reconstructed_img.save("geolocalization_dinov3/reconstructed_full_image_small.png")
    plt.figure(figsize=(12, 12))
    plt.imshow(reconstructed_img)
    plt.axis("off")
    plt.show()
