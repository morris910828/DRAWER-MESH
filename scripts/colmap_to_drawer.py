#!/usr/bin/env python
"""Convert a COLMAP-preprocessed dataset (sparse/0/*.bin + images/) into the
nerfstudio format DRAWER's panoptic-data dataparser expects:

  <data_dir>/transforms.json
  <data_dir>/images/             (full-res, already exists)
  <data_dir>/images_2/           (created here, 2x downscaled)

Single-camera datasets only. Accepts any COLMAP camera model whose distortion maps
onto the OpenCV k1/k2/k3/p1/p2 convention that transforms.json carries and the
panoptic dataparser consumes (see _SUPPORTED_MODELS) -- the distortion coefficients
are passed through rather than discarded, so undistorting the images beforehand with
`colmap image_undistorter` is optional, not required. Fisheye models are rejected:
their distortion function is a different one and the coefficients would be silently
misread as radial/tangential.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.timer import StepTimer

from nerfstudio.data.utils.colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    qvec2rotmat,
)


# COLMAP camera model -> its parameter names, in COLMAP's fixed order. Restricted to
# models whose distortion is the OpenCV radial/tangential polynomial, since that is what
# transforms.json's k1/k2/k3/p1/p2 fields mean to the panoptic dataparser. Fisheye
# variants (OPENCV_FISHEYE, RADIAL_FISHEYE, ...) are deliberately absent: their
# coefficients index a different distortion function, so accepting them would produce a
# plausible-looking but wrong camera.
_SUPPORTED_MODELS = {
    "SIMPLE_PINHOLE": ("f", "cx", "cy"),
    "PINHOLE":        ("fx", "fy", "cx", "cy"),
    "SIMPLE_RADIAL":  ("f", "cx", "cy", "k1"),
    "RADIAL":         ("f", "cx", "cy", "k1", "k2"),
    "OPENCV":         ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "FULL_OPENCV":    ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2",
                       "k3", "k4", "k5", "k6"),
}


def colmap_to_transforms(cameras_bin: Path, images_bin: Path) -> dict:
    cameras = read_cameras_binary(str(cameras_bin))
    images = read_images_binary(str(images_bin))

    if len(cameras) != 1:
        raise SystemExit(f"Expected 1 camera, got {len(cameras)}")
    cam = next(iter(cameras.values()))
    if cam.model not in _SUPPORTED_MODELS:
        raise SystemExit(
            f"Unsupported COLMAP camera model: {cam.model}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_MODELS))}. "
            f"For fisheye models, run `colmap image_undistorter` first."
        )
    names = _SUPPORTED_MODELS[cam.model]
    if len(cam.params) != len(names):
        raise SystemExit(
            f"{cam.model} expects {len(names)} parameters {names}, got {len(cam.params)}"
        )
    P = dict(zip(names, (float(v) for v in cam.params)))
    # SIMPLE_* models share one focal length across both axes.
    fx = P.get("fx", P.get("f"))
    fy = P.get("fy", P.get("f"))
    cx, cy = P["cx"], P["cy"]
    k1, k2, k3 = P.get("k1", 0.0), P.get("k2", 0.0), P.get("k3", 0.0)
    p1, p2 = P.get("p1", 0.0), P.get("p2", 0.0)

    # FULL_OPENCV's rational terms have nowhere to go in transforms.json. Silently
    # dropping them would change the camera, so say so loudly instead.
    dropped = {n: P[n] for n in ("k4", "k5", "k6") if abs(P.get(n, 0.0)) > 1e-12}
    if dropped:
        print(f"[colmap_to_drawer] WARNING: {cam.model} rational coefficients "
              f"{dropped} cannot be represented in transforms.json and are dropped. "
              f"Undistort with `colmap image_undistorter` if this matters.")

    # Report how much the distortion actually bends the image, measured at the far
    # corner (where it is largest). A number of a few pixels means undistorting the
    # images first would have changed almost nothing; a large one means the pass-through
    # coefficients are doing real work and the images must NOT be pre-undistorted.
    xn, yn = (cam.width - cx) / fx, (cam.height - cy) / fy
    r2 = xn * xn + yn * yn
    radial = k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    dx = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    dy = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    corner_shift = float(np.hypot(dx * fx, dy * fy))
    print(f"[colmap_to_drawer] camera model {cam.model}, {cam.width}x{cam.height}, "
          f"fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}")
    print(f"[colmap_to_drawer] distortion k1={k1:.3e} k2={k2:.3e} k3={k3:.3e} "
          f"p1={p1:.3e} p2={p2:.3e}  -> corner displacement {corner_shift:.2f} px "
          f"(passed through, images stay as-is)")

    frames = []
    for im in images.values():
        R = qvec2rotmat(im.qvec)
        t = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([R, t], 1),
                              np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # COLMAP -> nerfstudio convention (flip y/z, swap x/y)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[[1, 0, 2, 3], :]
        c2w[2, :] *= -1
        frames.append({
            "file_path": f"./images/{im.name}",
            "transform_matrix": c2w.tolist(),
        })

    frames.sort(key=lambda f: f["file_path"])

    return {
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx":   float(cx),
        "cy":   float(cy),
        "w":    int(cam.width),
        "h":    int(cam.height),
        "camera_model": "OPENCV",
        # Real coefficients from the COLMAP camera, not zeros: the panoptic dataparser
        # feeds these straight into camera_utils.get_distortion_params, so the images can
        # stay in their original (distorted) form. All zeros for a PINHOLE input, which
        # reproduces the previous behavior exactly.
        "k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2,
        "frames": frames,
    }


def downscale_images(src_dir: Path, dst_dir: Path, factor: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for src in tqdm(files, desc=f"downscale x{factor} -> {dst_dir.name}"):
        dst = dst_dir / src.name
        if dst.exists():
            continue
        with Image.open(src) as img:
            w, h = img.size
            img.resize((w // factor, h // factor),
                       Image.LANCZOS).save(dst, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path,
                    help="Dataset root containing sparse/0/ and images/")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--timing", default=None, metavar="JSON",
                    help="Timing output JSON path (optional)")
    args = ap.parse_args()

    timer = StepTimer(args.timing)

    @timer.record("step_0_colmap_to_drawer")
    def run():
        data_dir = args.data_dir.resolve()
        cameras_bin = data_dir / "sparse" / "0" / "cameras.bin"
        images_bin = data_dir / "sparse" / "0" / "images.bin"
        images_dir = data_dir / "images"

        for p in (cameras_bin, images_bin, images_dir):
            if not p.exists():
                raise SystemExit(f"missing: {p}")

        print(f"[1/2] writing transforms.json")
        transforms = colmap_to_transforms(cameras_bin, images_bin)
        out = data_dir / "transforms.json"
        out.write_text(json.dumps(transforms, indent=2))
        print(f"      {out}: {len(transforms['frames'])} frames, "
              f"{transforms['w']}x{transforms['h']}, fl=({transforms['fl_x']:.1f},{transforms['fl_y']:.1f})")

        if args.downscale > 1:
            print(f"[2/2] building images_{args.downscale}/")
            downscale_images(images_dir, data_dir / f"images_{args.downscale}", args.downscale)
        else:
            print("[2/2] downscale=1, using images/ directly (no copy)")
        print("done.")

    run()


if __name__ == "__main__":
    main()
