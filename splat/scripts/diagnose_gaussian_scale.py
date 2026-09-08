"""
diagnose_gaussian_scale.py — 診斷訓練後 Gaussian 的 scale 分佈

用來回答：
  1. 有多少比例的 Gaussian 貼著 min_scale_frac 下限（代表被系統性壓到最小）？
  2. x,y（面內）跟 z（法向厚度）各自相對於自己所屬面局部尺寸(xyz_radius)的
     比例分佈長怎樣？
  3. 「絕對尺寸過小」是不是單純因為那些 Gaussian 本來就在很小的面上
     （mesh_area_to_subdivide 切得細的地方)？

用 nerfstudio 的 eval_setup() 載入完整 pipeline（會照 config.yml 重新跑一次
mesh 讀取/細分，是確定性的，結果會跟訓練當下完全一致），再讀訓練好的權重，
這樣算出來的 xyz_radius / gaussians_to_mesh_indices 才會跟訓練時
`scales` property 用的完全一樣，不需要另外重寫一份 mesh 細分邏輯。

用法（跟訓練指令一樣的 PYTHONPATH 慣例）：
    PYTHONPATH=/workspace/DRAWER/splat python splat/scripts/diagnose_gaussian_scale.py \\
        --config-path outputs/armadillo1/armadillo1_mesh_gauss_splat/config.yml
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from nerfstudio.utils.eval_utils import eval_setup


def report(name: str, ratio: np.ndarray, floor_frac: float):
    near_floor = (ratio <= floor_frac * 1.05).mean()  # 在下限 5% 容忍範圍內算「貼著下限」
    print(f"\n=== {name}：相對 xyz_radius 的比例 ===")
    print(
        f"  min={ratio.min():.4f}  p5={np.percentile(ratio, 5):.4f}  "
        f"p50={np.percentile(ratio, 50):.4f}  p95={np.percentile(ratio, 95):.4f}  "
        f"max={ratio.max():.4f}"
    )
    print(f"  貼著下限 min_scale_frac={floor_frac} 的比例：{near_floor * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=True, help="checkpoint 旁的 config.yml 路徑")
    args = parser.parse_args()

    _, pipeline, _, step = eval_setup(args.config_path, test_mode="inference")
    model = pipeline.model
    print(f"載入 checkpoint，step={step}，num_points={model.num_points}")

    with torch.no_grad():
        xyz_radius = model.xyz_radius[model.gaussians_to_mesh_indices]  # (N, 3)：每個 Gaussian 所屬面的局部尺寸

        # 完全比照 `scales` property 的 clamp 邏輯（populate_modules / scales
        # property 見 splatfacto_on_mesh_uc.py），算出「實際渲染用」的 scale，
        # 不是只看未夾住的 raw 參數
        real_scales = torch.exp(model.gauss_params["scales"])
        scale_limit = model.config.upper_scale * xyz_radius
        real_scales = torch.minimum(real_scales, scale_limit)
        scale_floor = model.config.min_scale_frac * xyz_radius
        real_scales = torch.maximum(real_scales, scale_floor)

        ratio = (real_scales / xyz_radius).cpu().numpy()  # (N, 3)

    floor_frac = model.config.min_scale_frac
    report("x（面內）", ratio[:, 0], floor_frac)
    report("y（面內）", ratio[:, 1], floor_frac)
    report("z（法向厚度）", ratio[:, 2], floor_frac)

    # 絕對尺寸分佈：確認「過小」是不是單純因為所屬面本身就小
    abs_scales = real_scales.cpu().numpy()
    face_radius = model.xyz_radius[:, 0].cpu().numpy()  # 每個原始面的局部尺寸（x/y 共用同一個值）

    print("\n=== 絕對尺寸分佈（不是比例） ===")
    for name, idx in [("x", 0), ("y", 1), ("z", 2)]:
        a = abs_scales[:, idx]
        print(
            f"  {name}: min={a.min():.5f}  p5={np.percentile(a, 5):.5f}  "
            f"p50={np.percentile(a, 50):.5f}  p95={np.percentile(a, 95):.5f}"
        )

    print("\n=== 面局部尺寸 xyz_radius 本身的分佈（判斷小 Gaussian 是否只是因為面小）===")
    print(
        f"  min={face_radius.min():.5f}  p5={np.percentile(face_radius, 5):.5f}  "
        f"p50={np.percentile(face_radius, 50):.5f}  p95={np.percentile(face_radius, 95):.5f}"
    )

    print(
        "\n判讀方式：如果 x/y「相對 xyz_radius 的比例」大量貼著 min_scale_frac 下限，"
        "代表 min_scale_frac 這個相對值本身可能設太保守，可以考慮調高；"
        "如果比例分佈其實還好、但『絕對尺寸』分佈裡有很多很小的值，且這些值剛好"
        "對應到 xyz_radius 分佈裡偏小的那一群，代表過小只是因為那些 Gaussian 所屬"
        "的面本身就切得很細（mesh_area_to_subdivide 造成），不是 scale 機制的問題。"
    )


if __name__ == "__main__":
    main()
