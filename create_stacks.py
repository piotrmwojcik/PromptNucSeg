import json
from pathlib import Path
import numpy as np

def read_from_json(path):
    with open(path, "r") as f:
        return json.load(f)

def stack_paths_from_json_file(json_path: str, images_root_candidates=("images", ".")):
    """
    Read keys from the JSON and resolve them to existing image file paths.
    Save them (as strings) to <json_stem>_images.npy next to the JSON.
    """
    json_path = Path(json_path)
    data_root = json_path.parent

    anno_json = read_from_json(json_path)
    _ = anno_json.pop("classes", None)  # ignore classes
    img_keys = list(anno_json.keys())

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

    paths_array = np.array([str(p) for p in resolved], dtype=object)
    out_file = data_root / f"{json_path.stem}_images.npy"  # same filename as before
    np.save(out_file, paths_array)
    print(f"[OK] {json_path.stem}: saved {paths_array.shape} paths to {out_file}")

# ---- Run for the two specific files ----
stack_paths_from_json_file("/home/pwojcik/PromptNucSeg/prompter/datasets/freiburg/train.json")
stack_paths_from_json_file("/home/pwojcik/PromptNucSeg/prompter/datasets/freiburg/test.json")
