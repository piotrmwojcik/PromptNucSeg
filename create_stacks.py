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
    if img.ndim == 2:  # gray -> RGB
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:  # RGBA -> RGB (drop alpha)
        img = img[..., :3]
    return img

def pad_to_size(img, H, W):
    # pad with zeros (black) to target HxW
    h, w = img.shape[:2]
    out = np.zeros((H, W, 3), dtype=img.dtype)
    out[:h, :w] = img
    return out

def stack_images_from_json(cfg, mode: str = "train",
                           images_root_candidates=("images", ".")):
    """
    Reads datasets/{cfg.data.name}/{mode}.json, loads all images listed as keys,
    pads them to a common size, and saves a stacked .npy array:

      datasets/{name}/{mode}_images.npy  (shape: [N, H, W, 3], dtype=uint8)
    """
    json_path = Path(f"datasets/{cfg.data.name}/{mode}.json")
    data_root = json_path.parent

    anno_json = read_from_json(json_path)
    _ = anno_json.pop("classes", None)  # not used
    img_keys = list(anno_json.keys())

    # resolve image path for each key
    resolved_paths = []
    for k in img_keys:
        p = Path(k)
        if p.is_absolute() and p.exists():
            resolved_paths.append(p)
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
        resolved_paths.append(found)

    # load & collect sizes
    imgs = []
    maxH = maxW = 0
    print(f"[INFO] Loading {len(resolved_paths)} images…")
    for p in tqdm(resolved_paths, desc=f"read {mode}"):
        img = io.imread(str(p))
        img = to_rgb(img).astype(np.uint8)
        h, w = img.shape[:2]
        maxH, maxW = max(maxH, h), max(maxW, w)
        imgs.append(img)

    if not imgs:
        raise RuntimeError("No images loaded; nothing to stack.")

    # pad to common size & stack
    print(f"[INFO] Padding to common size H={maxH}, W={maxW} and stacking…")
    stacked = np.stack([pad_to_size(im, maxH, maxW) for im in imgs], axis=0)  # [N,H,W,3], uint8

    out_images = data_root / f"{mode}_images.npy"
    np.save(out_images, stacked)

    print(f"[OK] Saved images to: {out_images}  (shape={stacked.shape}, dtype={stacked.dtype})")

# ---------- usage ----------
from mmengine.config import Config
cfg = Config.fromfile('/home/pwojcik/PromptNucSeg/prompter/datasets/freiburg/')  # adjust to your config path
stack_images_from_json(cfg, mode="train")
# stack_images_from_json(cfg, mode="val")
stack_images_from_json(cfg, mode="test")
