#!/usr/bin/env python
"""
Build the images_N/ and masks_N/ folders that --downscale_factor N reads.

    python scripts/make_downscaled.py --data data/table --factor 4

nerfstudio's dataparser does not generate these. With --downscale_factor N it rewrites
every path to <data>/images_N/<name> and <data>/masks_N/<name>, and if that folder is
missing it silently skips every frame and then asserts "No image files found" -- which is
what a run with factor 4 hits when only images_2/ exists.

Filtering: images are resampled with LANCZOS, masks with NEAREST. A mask must stay binary;
resampling it smoothly would produce intermediate values along every silhouette edge, and
those become partial foreground weights in the loss rather than a clean cut.

Mask location: transforms.json's mask_path is used first, and if it does not resolve the
same basename is looked for under gs_masks/masks/. Both layouts exist in this project and
which one is present has differed between machines.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # these are 23 MP; PIL's decompression-bomb guard trips


def resolve(data: Path, rel: str, fallbacks) -> Path:
    p = (data / rel.replace("\\", "/")).resolve()
    if p.exists():
        return p
    name = os.path.basename(rel)
    for fb in fallbacks:
        cand = (data / fb / name)
        if cand.exists():
            return cand
    return p  # non-existent; caller reports it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir holding transforms.json")
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--transforms", default="transforms.json")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite files that already exist (default: skip them)")
    args = ap.parse_args()

    data = Path(args.data).resolve()
    meta = json.loads((data / args.transforms).read_text(encoding="utf-8"))
    frames = meta["frames"]
    f = args.factor
    if f < 2:
        print("factor must be >= 2 (factor 1 is the original folder)")
        return 1

    img_out = data / f"images_{f}"
    msk_out = data / f"masks_{f}"
    img_out.mkdir(exist_ok=True)
    has_mask = any("mask_path" in fr for fr in frames)
    if has_mask:
        msk_out.mkdir(exist_ok=True)

    print(f"{data}  ->  images_{f}/" + (f" and masks_{f}/" if has_mask else ""))
    n_img = n_msk = n_skip = 0
    missing = []
    size = None

    for fr in frames:
        src = resolve(data, fr["file_path"], ["images"])
        if not src.exists():
            missing.append(str(src))
            continue
        dst = img_out / src.name
        if dst.exists() and not args.overwrite:
            n_skip += 1
        else:
            im = Image.open(src)
            size = (im.width // f, im.height // f)
            im.resize(size, Image.LANCZOS).save(dst, quality=95)
            n_img += 1

        if "mask_path" in fr:
            msrc = resolve(data, fr["mask_path"], ["mask", "gs_masks/masks", "masks"])
            if not msrc.exists():
                missing.append(str(msrc))
                continue
            mdst = msk_out / msrc.name
            if mdst.exists() and not args.overwrite:
                continue
            mi = Image.open(msrc)
            # Sized from the mask's own dimensions, not the image's: they are the same here,
            # but a mask stored at a different resolution must still land on the image grid.
            mi.resize((mi.width // f, mi.height // f), Image.NEAREST).save(mdst)
            n_msk += 1

    print(f"  wrote {n_img} images, {n_msk} masks" + (f", skipped {n_skip} existing" if n_skip else ""))
    if size:
        print(f"  output size {size[0]} x {size[1]}")
    if missing:
        print(f"  MISSING {len(missing)} source file(s); first few:")
        for m in missing[:5]:
            print("    ", m)
        print("  Those frames were not written -- the dataparser will skip them, so fix the")
        print("  paths in transforms.json (or the folder layout) before training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
