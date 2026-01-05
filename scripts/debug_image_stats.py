import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from datasets.real_video_radar import ACTIONS


def _collect_paths(root: str, per_class: int, allow_nested: bool) -> Dict[str, List[str]]:
    samples: Dict[str, List[str]] = {a: [] for a in ACTIONS}
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    if allow_nested:
        for dirpath, _, filenames in os.walk(root):
            label = None
            for part in dirpath.split(os.sep):
                if part in ACTIONS:
                    label = part
            if label is None:
                continue
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts:
                    samples[label].append(os.path.join(dirpath, fname))
    else:
        for action in ACTIONS:
            action_dir = os.path.join(root, action)
            if not os.path.isdir(action_dir):
                continue
            for fname in os.listdir(action_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts:
                    samples[action].append(os.path.join(action_dir, fname))
    # Trim / sample
    for action in ACTIONS:
        paths = samples[action]
        if len(paths) > per_class:
            samples[action] = random.sample(paths, per_class)
    return samples


def _describe_image(path: str) -> Dict:
    with Image.open(path) as img:
        arr = np.array(img)
    stats = {
        "path": path,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Print basic stats for radar images (generated vs real)")
    parser.add_argument("--root_gen", type=str, required=True, help="Root directory of generated samples (actions subfolders)")
    parser.add_argument("--root_real", type=str, required=True, help="Root directory of real radar images for comparison")
    parser.add_argument("--per_class", type=int, default=5, help="Number of samples per class to inspect")
    parser.add_argument("--allow_nested", action="store_true", help="Allow nested subject folders")
    args = parser.parse_args()

    random.seed(0)
    print("== Generated samples ==")
    gen_paths = _collect_paths(args.root_gen, args.per_class, allow_nested=args.allow_nested)
    for action in ACTIONS:
        print(f"[{action}]")
        for p in gen_paths[action]:
            print(_describe_image(p))
    print("\n== Real samples ==")
    real_paths = _collect_paths(args.root_real, args.per_class, allow_nested=args.allow_nested)
    for action in ACTIONS:
        print(f"[{action}]")
        for p in real_paths[action]:
            print(_describe_image(p))


if __name__ == "__main__":
    main()
