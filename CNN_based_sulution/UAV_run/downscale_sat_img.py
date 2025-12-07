# downscale_satellite_vscode.py
# Run directly in VSCode (F5). All settings are below.

from pathlib import Path
import json
import datetime
import cv2
import numpy as np

# =========================
# CONFIG (edit these)
# =========================
sat_num = "06"  # e.g., "01", "02", "03", ...
BASE_PATH         = Path(__file__).parent.resolve()
INPUT_TIF         = BASE_PATH / f"UAV_VisLoc_dataset/{sat_num}/satellite{sat_num}.tif"
OUTPUT_IMAGE      = BASE_PATH / f"UAV_VisLoc_dataset/{sat_num}/satellite{sat_num}_small.png"
TARGET_LONG_SIDE  = 3000     # pixels for the longest side of the output (e.g., 3000)
CONVERT_TO_UINT8  = False     # Set True if your TIF is 16-bit/float and you want smaller files
UINT8_P_LOW       = 1.0       # Percentile low for contrast stretch when converting to 8-bit
UINT8_P_HIGH      = 99.0      # Percentile high
WRITE_META_JSON   = True      # Writes <OUTPUT_IMAGE>.json with scale & sizes
JPG_QUALITY       = 92        # Used only if OUTPUT_IMAGE is .jpg

# Optional: if you prefer exact output size, set FIXED_SIZE = (width, height) instead of TARGET_LONG_SIDE.
# Leave FIXED_SIZE = None to use TARGET_LONG_SIDE.
FIXED_SIZE = None  # e.g., (4096, 3072) or None

# =========================
# Helpers
# =========================
def to_uint8(img: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    """Optional 16-bit/float -> 8-bit with percentile stretch for compact files."""
    if img.dtype == np.uint8:
        return img
    lo = float(np.percentile(img, p_low))
    hi = float(np.percentile(img, p_high))
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((img - lo) / (hi - rail_guard(hi - lo)), 0, 1)
    out = (out * 255.0 + 0.5).astype(np.uint8)
    return out

def rail_guard(x: float, eps: float = 1e-12) -> float:
    return x if x > eps else eps

def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2))

# =========================
# Main logic
# =========================
def main():
    if not INPUT_TIF.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_TIF}")

    # Read unchanged to preserve bit depth/channels
    img = cv2.imread(str(INPUT_TIF), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"OpenCV failed to read {INPUT_TIF}")

    H, W = img.shape[:2]

    # Determine target size
    if FIXED_SIZE is not None:
        new_w, new_h = int(FIXED_SIZE[0]), int(FIXED_SIZE[1])
        s_w = new_w / float(W)
        s_h = new_h / float(H)
        # For metadata we keep a single scale if aspect preserved; otherwise store both
        same_aspect = abs((W / H) - (new_w / new_h)) < 1e-6
        scale_meta = float(s_w) if same_aspect else None
    else:
        # Keep aspect ratio, set longest side = TARGET_LONG_SIDE
        if TARGET_LONG_SIDE <= 0:
            raise ValueError("TARGET_LONG_SIDE must be > 0")
        s = float(TARGET_LONG_SIDE) / float(max(H, W))
        new_w = max(1, int(round(W * s)))
        new_h = max(1, int(round(H * s)))
        s_w = s_h = s
        scale_meta = float(s)

    # Resize using area interpolation (best for downscaling)
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Optional: convert to 8-bit for compact output
    if CONVERT_TO_UINT8:
        if small.ndim == 2:
            small = to_uint8(small, UINT8_P_LOW, UINT8_P_HIGH)
        else:
            chans = []
            for c in range(small.shape[2]):
                chans.append(to_uint8(small[..., c], UINT8_P_LOW, UINT8_P_HIGH))
            small = cv2.merge(chans)

    # Save image
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    ext = OUTPUT_IMAGE.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        ok = cv2.imwrite(str(OUTPUT_IMAGE), small, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY)])
    else:
        ok = cv2.imwrite(str(OUTPUT_IMAGE), small)
    if not ok:
        raise RuntimeError(f"Failed to write {OUTPUT_IMAGE}")

    # Save metadata
    if WRITE_META_JSON:
        meta = {
            "source_path": str(INPUT_TIF.resolve()),
            "output_path": str(OUTPUT_IMAGE.resolve()),
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "original_size_hw": [int(H), int(W)],
            "downscaled_size_hw": [int(new_h), int(new_w)],
        }
        if scale_meta is not None:
            # Single uniform scale (aspect preserved)
            meta["scale"] = float(scale_meta)  # p_small = p_orig * scale
            meta["note"] = "Uniform scale (aspect preserved). For overlays: p_small = p_orig * scale."
        else:
            # Non-uniform scaling (aspect changed)
            meta["scale_xy"] = [float(s_w), float(s_h)]
            meta["note"] = "Non-uniform scaling. For overlays: p_small = p_orig * [sx, sy] elementwise."
        save_json(OUTPUT_IMAGE.with_suffix(OUTPUT_IMAGE.suffix + ".json"), meta)

    print(f"[ok] Saved downscaled image: {OUTPUT_IMAGE}")
    if WRITE_META_JSON:
        print(f"[ok] Saved metadata: {OUTPUT_IMAGE.with_suffix(OUTPUT_IMAGE.suffix + '.json')}")
    print(f"[info] Original (H,W)=({H},{W}) -> Downscaled (H,W)=({new_h},{new_w})")

if __name__ == "__main__":
    main()
