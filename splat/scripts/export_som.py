"""
export_som.py — 對單一（沒有合併多個可動部件）的 splatfacto_on_mesh_uc
模型，載入訓練好的 checkpoint，呼叫 export_splatfacto_on_mesh()，
存成 som.pt。

跟 splat_merge.py 不同：splat_merge.py 是給抽屜/門這種要合併多個可動
部件的複雜流程用的；這個腳本就是單純「載入一個訓練好的模型 → 匯出」，
對應你目前的 armadillo1 單一 mesh 訓練。

用法：
    PYTHONPATH=/workspace/DRAWER/splat python scripts/export_som.py \\
        --config-path outputs/armadillo1/armadillo1_mesh_gauss_splat/config.yml

預設會存到跟 config.yml 同一層目錄下的 som.pt，也可以用 --out-path
指定別的路徑。
"""
import argparse
from pathlib import Path

import torch

from nerfstudio.utils.eval_utils import eval_setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=True, help="checkpoint 旁的 config.yml 路徑")
    parser.add_argument("--out-path", type=Path, default=None, help="輸出的 som.pt 路徑，預設跟 config.yml 同目錄")
    args = parser.parse_args()

    out_path = args.out_path or (args.config_path.parent / "som.pt")

    _, pipeline, _, step = eval_setup(args.config_path, test_mode="inference")
    model = pipeline.model
    print(f"載入 checkpoint，step={step}，num_points={model.num_points}")

    with torch.no_grad():
        save_dict = model.export_splatfacto_on_mesh()

    torch.save(save_dict, out_path)
    print(f"已存到: {out_path}")


if __name__ == "__main__":
    main()
