#!/usr/bin/env python3
"""Full-resolution masked-PSNR comparison across gaussian-splat-on-mesh runs.

For every experiment given, this renders each train + eval view at the native
photo resolution (5568x4176) and reports PSNR

    * inside the mask   (the table -- the number comparable to vanilla's 26.46 dB)
    * outside the mask   (background the mesh model structurally cannot draw)
    * over the whole frame

against the SAME ground-truth photos and the SAME masks, so the three models are
judged on one ruler.

gs19 was trained at downscale_factor 4 (1392 px). It is force-rendered at
downscale 1 here; its Gaussians only carry 1392 px of colour detail, so a low
inside-mask PSNR for gs19 is the expected, honest result -- that is the point of
the comparison.

RUN ON THE DGX (needs the checkpoints and a CUDA GPU):

    cd /workspace/DRAWER-MESH/splat
    python ../scripts/compare_masked_psnr.py \
        --exp table_gs12 table_gs19 table_gs20 \
        --data /workspace/DRAWER-MESH/data/table \
        --out  /workspace/DRAWER-MESH/data/table/psnr_compare

Add --exp table_gs24 to fold in the latest run. --save-renders writes a
half-size preview and an error heatmap per view under --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


def log10(x: float) -> float:
    return math.log10(max(x, 1e-20))


def psnr_from_mse(mse: float) -> float:
    return -10.0 * log10(mse)


def preflight_mask_dir(data_dir: Path) -> None:
    """The dataparser resolves ./mask/<name> at downscale 1. Make sure it exists
    and is full resolution, the same check run_table_gs20.sh does before training."""
    mask_dir = data_dir / "mask"
    if not mask_dir.exists():
        src = data_dir / "gs_masks" / "masks"
        if not src.is_dir():
            sys.exit(f"ERROR: no masks. Expected {src} (50 files, 5568x4176).")
        os.symlink(src, mask_dir)
        print(f"[pre-flight] linked {mask_dir} -> {src}")
    n_mask = len(list(mask_dir.glob("*.png")))
    n_img = len(list((data_dir / "images").glob("*.JPG")))
    if n_mask != n_img:
        sys.exit(f"ERROR: {n_mask} masks vs {n_img} images.")
    print(f"[pre-flight] {n_mask} masks resolved at {mask_dir}")


def load_pipeline(config_path: Path, cache_device: str):
    from nerfstudio.utils.eval_utils import eval_setup

    def cb(cfg):
        dp = cfg.pipeline.datamanager.dataparser
        dp.downscale_factor = 1                       # force full res (only bites gs19)
        cfg.pipeline.datamanager.camera_res_scale_factor = 1.0
        cfg.pipeline.datamanager.cache_images = cache_device
        cfg.pipeline.datamanager.cache_images_type = "uint8"
        return cfg

    config, pipeline, ckpt, step = eval_setup(
        config_path, test_mode="test", update_config_callback=cb
    )
    pipeline.eval()
    return config, pipeline, ckpt, step


def iter_views(dm, split: str):
    """Yield (name, camera[1], image_uint8[H,W,3], mask_or_None) for a split."""
    if split == "train":
        dataset, cached = dm.train_dataset, dm.cached_train
    else:
        dataset, cached = dm.eval_dataset, dm.cached_eval
    cameras = dataset.cameras
    for i in range(len(dataset)):
        name = Path(dataset.image_filenames[i]).stem
        cam = cameras[i : i + 1]
        batch = cached[i]
        yield name, cam, batch["image"], batch.get("mask", None)


@torch.no_grad()
def render_rgb(model, camera) -> torch.Tensor:
    out = model.get_outputs_for_camera(camera.to(model.device))
    return out["rgb"].clamp(0.0, 1.0)  # [H,W,3] float on device


def sq_err(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (pred - gt) ** 2  # [H,W,3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", required=True, help="experiment dir names under --data")
    ap.add_argument("--data", type=Path, required=True, help="dataset root, e.g. .../data/table")
    ap.add_argument("--out", type=Path, required=True, help="output dir for csv / renders")
    ap.add_argument("--splits", nargs="+", default=["eval", "train"], choices=["eval", "train"])
    ap.add_argument("--cache-device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--save-renders", action="store_true", help="dump half-size preview + error map per view")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    preflight_mask_dir(args.data)

    rows = []            # per-image
    summary = []         # per exp/split
    for exp in args.exp:
        config_path = args.data / exp / "config.yml"
        if not config_path.exists():
            print(f"!! {config_path} missing, skipping")
            continue
        print(f"\n==================== {exp} ====================")
        config, pipeline, ckpt, step = load_pipeline(config_path, args.cache_device)
        model, dm = pipeline.model, pipeline.datamanager
        n_gauss = int(getattr(model, "num_points", -1))
        print(f"loaded {ckpt.name} (step {step}), {n_gauss:,} gaussians")

        for split in args.splits:
            acc_in = acc_out = acc_full = 0.0
            px_in = px_out = px_full = 0
            per_img_psnr_in = []
            names = []
            for name, cam, img_u8, mask in iter_views(dm, split):
                names.append(name)
                gt = model.get_gt_img(img_u8)                     # [H,W,3] float, device
                rgb = render_rgb(model, cam)
                if rgb.shape != gt.shape:
                    gt = torch.nn.functional.interpolate(
                        gt.permute(2, 0, 1)[None], size=rgb.shape[:2], mode="bilinear", align_corners=False
                    )[0].permute(1, 2, 0)
                se = sq_err(rgb, gt)                              # [H,W,3]
                e_full = se.mean().item()
                acc_full += se.sum().item()
                px_full += se.numel()

                img_psnr_in = img_psnr_out = float("nan")
                mask_frac = float("nan")
                if mask is not None:
                    m = (mask.to(rgb.device).float().reshape(mask.shape[0], mask.shape[1], -1)[..., 0] > 0.5)
                    m3 = m[..., None].expand_as(se)
                    n_in = int(m3.sum().item())
                    n_out = int((~m3).sum().item())
                    mask_frac = float(m.float().mean().item())
                    if n_in:
                        s_in = se[m3].sum().item()
                        acc_in += s_in
                        px_in += n_in
                        img_psnr_in = psnr_from_mse(s_in / n_in)
                        per_img_psnr_in.append(img_psnr_in)
                    if n_out:
                        s_out = se[~m3].sum().item()
                        acc_out += s_out
                        px_out += n_out
                        img_psnr_out = psnr_from_mse(s_out / n_out)

                rows.append(dict(
                    exp=exp, split=split, image=name,
                    psnr_in=img_psnr_in, psnr_out=img_psnr_out,
                    psnr_full=psnr_from_mse(e_full), mask_frac=mask_frac,
                ))

                if args.save_renders:
                    _save_preview(args.out, exp, split, name, rgb, gt, se)

            dpsnr_in = psnr_from_mse(acc_in / px_in) if px_in else float("nan")
            dpsnr_out = psnr_from_mse(acc_out / px_out) if px_out else float("nan")
            dpsnr_full = psnr_from_mse(acc_full / px_full) if px_full else float("nan")
            mean_img_in = float(np.mean(per_img_psnr_in)) if per_img_psnr_in else float("nan")
            summary.append(dict(
                exp=exp, split=split, n=len(names), gaussians=n_gauss,
                psnr_in_dataset=dpsnr_in, psnr_in_meanimg=mean_img_in,
                psnr_out_dataset=dpsnr_out, psnr_full_dataset=dpsnr_full,
                images=",".join(names),
            ))
            print(f"  [{split:5s}] n={len(names):2d}  "
                  f"PSNR_in={dpsnr_in:6.2f} dB (mean-img {mean_img_in:6.2f})  "
                  f"PSNR_out={dpsnr_out:6.2f}  PSNR_full={dpsnr_full:6.2f}")

        del pipeline
        torch.cuda.empty_cache()

    # ---- write outputs ----
    csv_path = args.out / "per_image_psnr.csv"
    with open(csv_path, "w") as f:
        f.write("exp,split,image,psnr_in,psnr_out,psnr_full,mask_frac\n")
        for r in rows:
            f.write(f"{r['exp']},{r['split']},{r['image']},"
                    f"{r['psnr_in']:.4f},{r['psnr_out']:.4f},{r['psnr_full']:.4f},{r['mask_frac']:.4f}\n")
    json_path = args.out / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print("\n================ SUMMARY (dataset PSNR = -10log10(mean MSE over all pixels of the split)) ================")
    hdr = f"{'exp':<12} {'split':<6} {'n':>3} {'gaussians':>11} {'PSNR_in':>9} {'PSNR_out':>9} {'PSNR_full':>10}"
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        print(f"{s['exp']:<12} {s['split']:<6} {s['n']:>3} {s['gaussians']:>11,} "
              f"{s['psnr_in_dataset']:>9.2f} {s['psnr_out_dataset']:>9.2f} {s['psnr_full_dataset']:>10.2f}")
    print(f"\nper-image csv : {csv_path}")
    print(f"summary json  : {json_path}")
    print("\nvanilla 3DGS reference (experiments file, 1392 px): PSNR_in 26.46 dB, PSNR_full 23.26 dB")


def _save_preview(out, exp, split, name, rgb, gt, se):
    try:
        import cv2
    except Exception:
        return
    d = out / "renders" / exp / split
    d.mkdir(parents=True, exist_ok=True)

    def to_u8(t):
        return (t.detach().cpu().numpy()[..., ::-1] * 255).clip(0, 255).astype(np.uint8)

    h, w = rgb.shape[:2]
    sc = 0.5
    small = lambda a: cv2.resize(a, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    err = se.mean(-1).sqrt().detach().cpu().numpy()
    err = (np.clip(err / 0.25, 0, 1) * 255).astype(np.uint8)
    err = cv2.applyColorMap(err, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(d / f"{name}_pred.jpg"), small(to_u8(rgb)), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(d / f"{name}_gt.jpg"), small(to_u8(gt)), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(d / f"{name}_err.jpg"), small(err), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main()
