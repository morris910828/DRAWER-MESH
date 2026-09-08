"""
Convert ns-render accumulation outputs to binary PNG masks and update transforms.json.

Usage:
    python scripts/bake_accumulation_masks.py \
        --accum_dir  <data_dir>/accumulation_render/accumulation \
        --mask_dir   <data_dir>/masks \
        --transforms <data_dir>/transforms.json \
        --threshold 0.5 --dilate 5
"""

import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--accum_dir",  required=True, help="Directory with accumulation PNGs from ns-render")
parser.add_argument("--mask_dir",   required=True, help="Output directory for binary masks")
parser.add_argument("--transforms", required=True, help="transforms.json path")
parser.add_argument("--threshold",  type=float, default=0.5, help="Binarisation threshold [0-1]")
parser.add_argument("--dilate",     type=int,   default=5,   help="Dilation kernel size (0 = skip)")
args = parser.parse_args()

os.makedirs(args.mask_dir, exist_ok=True)

with open(args.transforms) as f:
    meta = json.load(f)

# ns-render saves files named after the image stem, e.g. 0001.png
accum_files = sorted(glob.glob(os.path.join(args.accum_dir, "*.png")))
print(f"Found {len(accum_files)} accumulation images in {args.accum_dir}")

# Build stem → accum_path lookup
accum_map = {os.path.splitext(os.path.basename(p))[0]: p for p in accum_files}

kernel = np.ones((args.dilate, args.dilate), np.uint8) if args.dilate > 0 else None

matched = 0
for frame in tqdm(meta["frames"], desc="Baking masks"):
    file_path = frame["file_path"]
    stem = os.path.splitext(os.path.basename(file_path))[0]

    if stem not in accum_map:
        continue

    acc = np.array(Image.open(accum_map[stem]).convert("L"), dtype=np.float32) / 255.0
    mask = (acc >= args.threshold).astype(np.uint8) * 255

    if kernel is not None:
        mask = cv2.dilate(mask, kernel)

    mask_fname = stem + ".png"
    out_path = os.path.join(args.mask_dir, mask_fname)
    Image.fromarray(mask).save(out_path)
    frame["mask_path"] = f"./masks/{mask_fname}"
    matched += 1

print(f"Written {matched} masks → {args.mask_dir}")

# Backup and overwrite transforms.json
backup = args.transforms + ".bak"
if not os.path.exists(backup):
    shutil.copy2(args.transforms, backup)

with open(args.transforms, "w") as f:
    json.dump(meta, f, indent=2)

print(f"transforms.json updated  (backup → {backup})")
