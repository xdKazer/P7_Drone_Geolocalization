import cv2
from pathlib import Path

# ---- Paths ----
BASE = Path(__file__).parent.resolve()
dataset_path = BASE / "UAV_VisLoc_dataset"
tif_path     = dataset_path / "03" / "satellite03.tif"
out_dir      = dataset_path / "03" / "sat_tiles_overlap"
out_dir.mkdir(parents=True, exist_ok=True)

# ---- Read images ----
sat = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
drone = cv2.imread(str(dataset_path / "03" / "drone" / "03_0738.JPG"), cv2.IMREAD_UNCHANGED)
if sat is None:
    raise FileNotFoundError(tif_path)
if drone is None:
    raise FileNotFoundError(dataset_path / "03" / "drone" / "03_0738.JPG")

# ---- Tile size (from drone, scaled like before) ----
tile_h, tile_w = drone.shape[:2]
tile_h = int(tile_h // 2.2) # this needs to be tuned based on altityde and satelite zoom level !!!OBSS!!! TODO
tile_w = int(tile_w // 2.2)
H, W = sat.shape[:2]
print(f"sat size:  {H} x {W}")
print(f"tile size: {tile_h} x {tile_w}")

# ---- Half-stride (50% overlap) ----
stride_h = max(1, tile_h // 2) # this schould be finetuned as well !!!OBSS!!! TODO
stride_w = max(1, tile_w // 2)

# Build start indices so we also cover the far edges
def build_starts(size, tile_size, stride):
    starts = list(range(0, max(1, size - tile_size + 1), stride))
    # ensure the last tile touches the edge
    if starts[-1] + tile_size < size:
        starts.append(size - tile_size)
    return starts

ys = build_starts(H, tile_h, stride_h)
xs = build_starts(W, tile_w, stride_w)

print(f"rows: {len(ys)}  cols: {len(xs)}  -> total tiles: {len(ys)*len(xs)}")

count = 0
for i, y0 in enumerate(ys):
    y1 = min(y0 + tile_h, H)
    for j, x0 in enumerate(xs):
        x1 = min(x0 + tile_w, W)
        tile = sat[y0:y1, x0:x1]
        # sanity: skip any empty slice
        if tile.size == 0:
            continue
        # encode the *pixel start* indices in the filename (clearer for overlaps)
        out_path = out_dir / f"sat_tile_y{y0}_x{x0}.png"
        cv2.imwrite(str(out_path), tile)
        count += 1

print(f"Wrote {count} overlapping tiles to: {out_dir}")
# save the sat tile size for later use
with open(out_dir / "tile_size.txt", "w") as f:
    f.write(f"{tile_h} {tile_w}\n") 
print(f"Wrote tile size to: {out_dir / 'tile_size.txt'}")
