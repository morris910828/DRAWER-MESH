"""
Convert Lotus-2 raw output → SDF dataparser format.

Lotus-2 emits:
  depth  : disparity in [0,1]  (near = high value)
  normal : unit vector in [0,1] encoding, arbitrary axis convention

This script produces:
  <lotus_depth>/<stem>.npy   — float32, depth = 1/(disparity + eps)   (near = small)
  <lotus_normal>/<stem>.png  — uint8 RGB, same axis convention as Marigold/omnidata

Normal axis convention is determined automatically by testing all 48 signed
permutations against a reference normal map (Marigold) via cosine similarity.
If no reference is available, the identity transform is used (with a warning).

Usage:
    python marigold/lotus_convert.py \
        --img_dir      <DATA_DIR>/images \
        --lotus2_dir   <DATA_DIR>/lotus2 \
        --out_depth    <DATA_DIR>/lotus_depth \
        --out_normal   <DATA_DIR>/lotus_normal \
        [--ref_normal  <DATA_DIR>/marigold_ft/normal] \
        [--disp_eps    0.02]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def apply_T(n: np.ndarray, p: tuple, s: tuple) -> np.ndarray:
    return n[..., list(p)] * np.array(s)


def find_normal_transform(
    lotus_npy_dir: str,
    ref_normal_dir: str,
    stems: list[str],
) -> tuple[tuple, tuple]:
    """Test all 48 signed permutations; return the one with highest cosine vs reference."""
    perms = list(itertools.permutations(range(3)))
    signs = list(itertools.product([1, -1], repeat=3))

    ref_first = os.path.join(ref_normal_dir, stems[0] + ".png")
    if not (os.path.isdir(ref_normal_dir) and os.path.exists(ref_first)):
        print(
            f"[lotus_convert] WARNING: no Marigold reference at {ref_normal_dir} → "
            "using identity normal transform. Verify normals or hardcode perm/signs."
        )
        return (0, 1, 2), (1, 1, 1)

    sample_idx = [0, len(stems) // 4, len(stems) // 2, 3 * len(stems) // 4, len(stems) - 1]
    LA, MA = [], []
    for i in sample_idx:
        st = stems[i]
        ln = np.load(os.path.join(lotus_npy_dir, st + ".npy")).squeeze().astype(np.float64) * 2 - 1
        mp = (
            np.asarray(Image.open(os.path.join(ref_normal_dir, st + ".png")).convert("RGB"))
            .astype(np.float64)
            / 255.0
            * 2 - 1
        )
        if mp.shape[:2] != ln.shape[:2]:
            mp = (
                np.asarray(
                    Image.fromarray(((mp + 1) / 2 * 255).astype(np.uint8)).resize(
                        (ln.shape[1], ln.shape[0])
                    )
                ).astype(np.float64)
                / 255.0
                * 2 - 1
            )
        ln /= np.linalg.norm(ln, axis=2, keepdims=True) + 1e-9
        mp /= np.linalg.norm(mp, axis=2, keepdims=True) + 1e-9
        LA.append(ln.reshape(-1, 3)[::200])
        MA.append(mp.reshape(-1, 3)[::200])

    LA = np.concatenate(LA)
    MA = np.concatenate(MA)
    best_score = -9.0
    best_p, best_s = (0, 1, 2), (1, 1, 1)
    for p in perms:
        for s in signs:
            score = (apply_T(LA, p, s) * MA).sum(1).mean()
            if score > best_score:
                best_score, best_p, best_s = score, p, s

    print(f"[lotus_convert] best normal transform perm={best_p} signs={best_s} mean_cos={best_score:.3f}")
    return best_p, best_s


def convert(
    img_dir: str,
    lotus2_dir: str,
    out_depth: str,
    out_normal: str,
    ref_normal: str,
    disp_eps: float,
    depth_norm: str = "robust",
    clip_pct: float = 2.0,
    normal_perm: tuple = (0, 1, 2),
    normal_signs: tuple = (-1, -1, -1),
    search_normal: bool = False,
) -> None:
    # ── collect image stems ───────────────────────────────────────────────────
    imgs: list[str] = []
    for pat in IMAGE_EXTENSIONS:
        imgs.extend(glob.glob(os.path.join(img_dir, pat)))
    imgs = sorted(imgs)
    if not imgs:
        sys.exit(f"[lotus_convert] ERROR: no images found in {img_dir}")

    stems = [os.path.splitext(os.path.basename(f))[0] for f in imgs]

    lotus_depth_npy = os.path.join(lotus2_dir, "depth", "depth_npy")
    lotus_normal_npy = os.path.join(lotus2_dir, "normal", "normal_npy")
    os.makedirs(out_depth, exist_ok=True)
    os.makedirs(out_normal, exist_ok=True)

    # ── normal axis transform ─────────────────────────────────────────────────
    # Lotus-2 vs Marigold normal convention is a FIXED signed-axis map (a model
    # property, dataset-independent): perm=(0,1,2) signs=(-1,-1,-1), i.e. a global
    # negation (empirically mean_cos 0.936 on gopro_2fps). Hardcoded by default;
    # pass --search_normal (with --ref_normal) to re-detect for a different model.
    if search_normal:
        perm, sign = find_normal_transform(lotus_normal_npy, ref_normal, stems)
    else:
        perm, sign = tuple(normal_perm), tuple(normal_signs)
        print(f"[lotus_convert] using hardcoded normal transform perm={perm} signs={sign}")

    # ── convert each frame ────────────────────────────────────────────────────
    for i, st in enumerate(stems):
        # depth: disparity [0,1] (near=high) → depth = 1/(disp+eps) (near=small)
        disp = np.load(os.path.join(lotus_depth_npy, st + ".npy")).squeeze().astype(np.float32)
        depth = 1.0 / (disp + disp_eps)
        # 1/(disp+eps) is heavily right-skewed: on large scenes far pixels (disp≈0)
        # pile up at the 1/eps cap (e.g. 50), giving a heavy-tailed distribution
        # (std ~13) that makes the scale+shift-invariant depth loss ill-conditioned
        # → NaN once geometry sharpens (~50k). Robust per-image normalize to [0,1]
        # (like Marigold's affine depth) fixes the conditioning; the loss is
        # scale/shift-invariant so absolute values are irrelevant, only the shape.
        if depth_norm == "robust":
            lo, hi = np.percentile(depth, [clip_pct, 100.0 - clip_pct])
            depth = np.clip(depth, lo, hi)
            depth = (depth - lo) / (hi - lo + 1e-9)
        np.save(os.path.join(out_depth, st + ".npy"), depth.astype(np.float32))

        # normal: [0,1]→dir, apply axis transform, re-encode as Marigold-style PNG
        ln = np.load(os.path.join(lotus_normal_npy, st + ".npy")).squeeze().astype(np.float64) * 2 - 1
        ln /= np.linalg.norm(ln, axis=2, keepdims=True) + 1e-9
        png = ((apply_T(ln, perm, sign) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(png).save(os.path.join(out_normal, st + ".png"))

        if i % 80 == 0:
            print(f"  {i}/{len(stems)}", flush=True)

    print(f"[lotus_convert] done: {len(stems)} depth + {len(stems)} normal")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Lotus-2 output to SDF dataparser format")
    parser.add_argument("--img_dir",    required=True,  help="Image directory (to determine stems)")
    parser.add_argument("--lotus2_dir", required=True,  help="Lotus-2 raw output root (contains depth/ and normal/)")
    parser.add_argument("--out_depth",  required=True,  help="Output directory for converted depth .npy")
    parser.add_argument("--out_normal", required=True,  help="Output directory for converted normal .png")
    parser.add_argument("--ref_normal", default="",     help="Marigold normal dir for auto axis detection (optional)")
    parser.add_argument("--disp_eps",   type=float, default=0.02,
                        help="Epsilon added before inverting disparity (default: 0.02)")
    parser.add_argument("--depth_norm", choices=("robust", "raw"), default="robust",
                        help="'robust' = per-image clip+normalize depth to [0,1] (default, "
                             "prevents heavy-tail NaN on large scenes); 'raw' = keep 1/(disp+eps)")
    parser.add_argument("--clip_pct",   type=float, default=2.0,
                        help="Percentile to clip at each end before normalizing (default: 2.0)")
    parser.add_argument("--normal_perm",  default="0,1,2",
                        help="Hardcoded normal axis permutation (default '0,1,2')")
    parser.add_argument("--normal_signs", default="-1,-1,-1",
                        help="Hardcoded normal axis signs (default '-1,-1,-1' = global negation)")
    parser.add_argument("--search_normal", action="store_true",
                        help="Brute-force the 48 signed permutations vs --ref_normal instead of the hardcoded transform")
    args = parser.parse_args()

    convert(
        img_dir=args.img_dir,
        lotus2_dir=args.lotus2_dir,
        out_depth=args.out_depth,
        out_normal=args.out_normal,
        ref_normal=args.ref_normal,
        disp_eps=args.disp_eps,
        depth_norm=args.depth_norm,
        clip_pct=args.clip_pct,
        normal_perm=tuple(int(x) for x in args.normal_perm.split(",")),
        normal_signs=tuple(int(x) for x in args.normal_signs.split(",")),
        search_normal=args.search_normal,
    )


if __name__ == "__main__":
    main()
