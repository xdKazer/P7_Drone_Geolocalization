# precompute_sp_features.py
from pathlib import Path
import torch
from lightglue import SuperPoint
from lightglue.utils import load_image, rbd

Base_dir = Path(__file__).parent.resolve() 
SAT_DIR   = Base_dir / "UAV_VisLoc_dataset/03/sat_tiles_overlap"
output_dir = Base_dir / "UAV_VisLoc_dataset/03/SuperPoint_features/03"


MAX_KPTS = 4048
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_feats(img_path: Path):
    out = output_dir / (img_path.stem + ".pt")
    if out.exists():
        print(f"[skip] {out.name} exists")
        return
    img = load_image(str(img_path)).to(device)
    with torch.inference_mode():
        feats = extractor.extract(img)     # keep batch here
    feats = rbd(feats)                     # remove batch before saving
    # keep essential fields only
    keep = {k: v.cpu() for k, v in feats.items() if isinstance(v, torch.Tensor)}
    torch.save(keep, out)
    print(f"[ok] saved {out.name}")

if __name__ == "__main__":
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = SuperPoint(max_num_keypoints=MAX_KPTS).eval().to(device)

    # tiles
    tiles = [p for p in SAT_DIR.iterdir() if p.is_file() and p.suffix.lower()==".png"]
    for p in tiles:
        save_feats(p)
