import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--skip_depth', action='store_true',
                    help='Skip depth conversion (use when Metric3D handles depth)')
args = parser.parse_args()

source_dir = args.data_dir

# ── depth ─────────────────────────────────────────────────────────────────────
depth_npy_dir = os.path.join(source_dir, "depth_npy")
if not args.skip_depth:
    if os.path.isdir(depth_npy_dir):
        target_depth_dir = os.path.join(source_dir, "depth")
        os.makedirs(target_depth_dir, exist_ok=True)
        for name in tqdm(sorted(os.listdir(depth_npy_dir)), desc="depth"):
            depth = np.load(os.path.join(depth_npy_dir, name))
            np.save(os.path.join(target_depth_dir, name.replace("_pred", "")), depth)
    else:
        print(f"[read_marigold] depth_npy/ not found at {depth_npy_dir}, skipping depth")

# ── normals ───────────────────────────────────────────────────────────────────
normal_colored_dir = os.path.join(source_dir, "normal_colored")
if os.path.isdir(normal_colored_dir):
    target_normal_dir = os.path.join(source_dir, "normal")
    os.makedirs(target_normal_dir, exist_ok=True)
    for name in tqdm(sorted(os.listdir(normal_colored_dir)), desc="normals"):
        normal_im = np.array(
            Image.open(os.path.join(normal_colored_dir, name)), dtype=np.float32
        ) / 255.0
        normal_im = 1.0 - normal_im
        normal_im = Image.fromarray(np.clip(normal_im * 255.0, 0, 255).astype(np.uint8))
        normal_im.save(os.path.join(target_normal_dir, name.replace("_pred_colored", "")))
else:
    print(f"[read_marigold] normal_colored/ not found at {normal_colored_dir}, skipping normals")
