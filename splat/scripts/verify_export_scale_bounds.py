"""
verify_export_scale_bounds.py — 驗證 export_splatfacto_on_mesh()/splat_merge.py
匯出的 som.pt 裡，每個 Gaussian 的 scale 是否真的落在
[min_scale_frac x xyz_radius, upper_scale x xyz_radius] 範圍內。

不需要 GPU 渲染、不需要重新訓練，只讀存好的張量做數字檢查，幾秒鐘出結果。

用法：
    python splat/scripts/verify_export_scale_bounds.py \\
        --som-path outputs/armadillo1/.../som.pt \\
        --min-scale-frac 0.3 --upper-scale 1.0
（--min-scale-frac / --upper-scale 要跟你訓練指令裡實際用的值一致）
"""
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--som-path", type=Path, required=True, help="export_splatfacto_on_mesh() 存出來的 som.pt 路徑")
    parser.add_argument("--min-scale-frac", type=float, required=True)
    parser.add_argument("--upper-scale", type=float, required=True)
    parser.add_argument("--tol", type=float, default=1e-4, help="數值誤差容忍範圍")
    args = parser.parse_args()

    d = torch.load(args.som_path, map_location="cpu")

    scales_log = d["scales"]                      # (N, 3), log-space
    gaussians_to_mesh_indices = d["gaussians_to_mesh_indices"]  # (N,)
    xyz_radius = d["xyz_radius"]                   # (F, 3)

    real_scales = torch.exp(scales_log)            # (N, 3)
    xyz_radius_per_gauss = xyz_radius[gaussians_to_mesh_indices]  # (N, 3)

    scale_floor = args.min_scale_frac * xyz_radius_per_gauss
    scale_ceiling = args.upper_scale * xyz_radius_per_gauss

    below_floor = real_scales < (scale_floor - args.tol)
    above_ceiling = real_scales > (scale_ceiling + args.tol)

    n_gaussians = real_scales.shape[0]
    n_below = int(below_floor.any(dim=-1).sum().item())
    n_above = int(above_ceiling.any(dim=-1).sum().item())

    print(f"總 Gaussian 數: {n_gaussians}")
    print(f"低於下限 (min_scale_frac x xyz_radius) 的 Gaussian 數: {n_below} "
          f"({n_below / n_gaussians * 100:.3f}%)")
    print(f"高於上限 (upper_scale x xyz_radius) 的 Gaussian 數: {n_above} "
          f"({n_above / n_gaussians * 100:.3f}%)")

    if n_below == 0 and n_above == 0:
        print("\n通過：every Gaussian 的 scale 都落在允許範圍內。")
    else:
        print("\n沒通過：仍有 Gaussian 超出範圍，下限/上限機制沒有正確生效。")
        if n_below > 0:
            worst = (scale_floor - real_scales).max().item()
            print(f"  最嚴重的一個，比下限還小了 {worst:.6f}（世界座標單位）")

    # 額外參考：絕對尺寸分佈，方便判斷「看起來還是小」是不是純粹因為
    # min_scale_frac x 面尺寸本身在小面上換算出來就是小
    print("\n=== 絕對尺寸分佈（僅供參考，不是通過/失敗判定）===")
    for i, name in enumerate(["x", "y", "z"]):
        a = real_scales[:, i]
        print(f"  {name}: min={a.min():.5f}  p5={a.quantile(0.05):.5f}  "
              f"p50={a.median():.5f}  p95={a.quantile(0.95):.5f}")


if __name__ == "__main__":
    main()
