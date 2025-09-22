import json
from pathlib import Path

import numpy as np
from skimage import io
from tqdm import tqdm

def read_from_json(path):
    with open(path, "r") as f:
        return json.load(f)

def to_rgb(img):
    # img: HxW, HxWx3, or HxWx4
    if img.ndim == 2:                     # gray -> RGB
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:  # RGBA -> RGB
        img = img[..., :3]
    return img

def pad_to_size(img, H, W):
    out = np.zeros((H, W, 3), dtype=img.dtype)
    h, w = img.shape[:2]
    out[:h, :w] = img
    return out

def stack_images_from_json_file(json_path: str, images_root_candidates=("images", ".")):
    """
    Load images listed as keys in the JSON, pad to the max H/W, and save a single .npy stack
    next to the JSON: <json_stem>_images.npy  with shape [N, H, W, 3] (uint8).
    """
    json_path = Path(json_path)
    data_root = json_path.parent

    anno_json = read_from_json(json_path)
    _ = anno_json.pop("classes", None)  # not needed for stacking
    img_keys = list(anno_json.keys())

    # Resolve each key to an actual file path
    resolved = []
    for k in img_keys:
        p = Path(k)
        if p.is_absolute() and p.exists():
            resolved.append(p)
            continue
        found = None
        for sub in images_root_candidates:
            cand = (data_root / sub / k).resolve()
            if cand.exists():
                found = cand
                break
        if found is None:
            cand = (data_root / k).resolve()
            if cand.exists():
                found = cand
        if found is None:
            raise FileNotFoundError(
                f"Could not find image for key '{k}' under {data_root} "
                f"(tried {images_root_candidates} and '.')"
            )
        resolved.append(found)

    # Load and track max size
    imgs, maxH, maxW = [], 0, 0
    for p in tqdm(resolved, desc=f"read {json_path.stem}"):
        img = io.imread(str(p))
        img = to_rgb(img).astype(np.uint8)
        h, w = img.shape[:2]
        maxH, maxW = max(maxH, h), max(maxW, w)
        imgs.append(img)

    if not imgs:
        raise RuntimeError(f"No images loaded from {json_path}")

    # Pad and stack
    stacked = np.stack([pad_to_size(im, maxH, maxW) for im in imgs], axis=0)  # [N,H,W,3]
    out_file = data_root / f"{json_path.stem}_images.npy"
    np.save(out_file, stacked)
    print(f"[OK] {json_path.stem}: saved {stacked.shape} to {out_file}")

# ---- Run for the two specific files ----
stack_images_from_json_file("/home/pwojcik/PromptNucSeg/prompter/datasets/freiburg/train.json")
stack_images_from_json_file("/home/pwojcik/PromptNucSeg/prompter/datasets/freiburg/test.json")
