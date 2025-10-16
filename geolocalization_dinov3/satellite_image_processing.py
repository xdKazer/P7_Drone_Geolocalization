import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

input_file = "geolocalization_dinov3/satellite03.tif"
output_folder = "tiles_png_uniform"
os.makedirs(output_folder, exist_ok=True)

desired_tile_width = 3976 / 2
desired_tile_height = 2652 / 2

with rasterio.open(input_file) as src:
    img_width = src.width
    img_height = src.height

    # Compute number of tiles along each dimension
    num_tiles_width = round(img_width / desired_tile_width)
    num_tiles_height = round(img_height / desired_tile_height)

    # Compute actual tile size to divide image exactly
    tile_width = img_width // num_tiles_width
    tile_height = img_height // num_tiles_height

    print(f"Adjusted tile size: {tile_width}x{tile_height}")
    print(f"Number of tiles: {num_tiles_width}x{num_tiles_height}")

    tile_count = 0

    print("Total image count should be: " + str(num_tiles_width * num_tiles_height))

    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            print("Currently processing tile: " + str(tile_count) + " / " + str(num_tiles_width * num_tiles_height))
            left = j * tile_width
            top = i * tile_height

            window = Window(left, top, tile_width, tile_height)

            # Read the window as (bands, height, width)
            data = src.read(window=window)

            # Convert to (height, width, bands) for PIL
            data = np.transpose(data, (1, 2, 0))

            # If single-band, remove the extra dimension
            if data.shape[2] == 1:
                data = data[:, :, 0]

            # Convert to PIL Image
            img = Image.fromarray(data)

            # Save as PNG
            tile_filename = os.path.join(output_folder, f"tile_{tile_count}.png")
            img.save(tile_filename)
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

    scale_factor = 0.1  # e.g., reduce to 10% of original
    new_width = int(reconstructed_img.width * scale_factor)
    new_height = int(reconstructed_img.height * scale_factor)

    reconstructed_img_small = reconstructed_img.resize((new_width, new_height), Image.BILINEAR)

    # Save and display
    reconstructed_img_small.save("geolocalization_dinov3/reconstructed_full_image_small.png")
    plt.figure(figsize=(12, 12))
    plt.imshow(reconstructed_img_small)
    plt.axis("off")
    plt.show()
