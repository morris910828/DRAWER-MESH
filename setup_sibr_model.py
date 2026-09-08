"""
setup_sibr_model.py
將 splat.ply + transforms.json 轉換為 SIBR Gaussian Viewer 可讀的目錄結構。

用法：
    python setup_sibr_model.py --model <splat.ply 所在目錄> --data <transforms.json 所在目錄>

範例：
    python setup_sibr_model.py \
        --model D:/SGGaussians/SGGaussians/output/armadillo1 \
        --data  D:/SGGaussians/SGGaussians/data/armadillo1

執行後用 SIBR viewer 開啟：
    SIBR_gaussianViewer_app.exe -m <model 目錄>
"""

import argparse
import json
import os
import shutil
import sys
import numpy as np


def load_transforms(data_dir):
    path = os.path.join(data_dir, "transforms.json")
    if not os.path.exists(path):
        sys.exit(f"[Error] 找不到 transforms.json: {path}")
    with open(path) as f:
        return json.load(f)


def convert_cameras(tf, data_dir):
    fl_x = tf["fl_x"]
    fl_y = tf["fl_y"]
    W = tf["w"]
    H = tf["h"]
    cameras = []
    for idx, frame in enumerate(tf["frames"]):
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        # OpenGL/Blender (Y up, Z back) → COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        pos = c2w[:3, 3].tolist()
        rot = c2w[:3, :3].tolist()
        img_name = os.path.splitext(os.path.basename(frame["file_path"]))[0]
        cameras.append({
            "id": idx,
            "img_name": img_name,
            "width": W,
            "height": H,
            "position": pos,
            "rotation": rot,
            "fy": fl_y,
            "fx": fl_x,
        })
    return cameras


def detect_sh_degree(ply_path):
    """從 PLY header 的 f_rest 數量推算 SH degree。"""
    f_rest_count = 0
    with open(ply_path, "rb") as f:
        for line in f:
            line = line.decode("ascii", errors="replace").strip()
            if "f_rest" in line:
                f_rest_count += 1
            if line == "end_header":
                break
    # f_rest 數量對應 SH degree:
    #   0 → degree 0,  9 → degree 1,  24 → degree 2,  45 → degree 3
    mapping = {0: 0, 9: 1, 24: 2, 45: 3}
    return mapping.get(f_rest_count, 1)


def detect_images_folder(data_dir, transforms):
    """從 transforms.json 的第一個 frame 推算圖片資料夾名稱。"""
    first = transforms["frames"][0]["file_path"]  # e.g. "./renders/0000.png"
    parts = first.replace("\\", "/").split("/")
    # 去掉 './' 和檔名，取資料夾名稱
    folders = [p for p in parts[:-1] if p and p != "."]
    return folders[0] if folders else "images"


def write_cfg_args(model_dir, data_dir, sh_degree, images_folder, white_background):
    # 轉為 Windows 反斜線格式（cfg_args 是 Python repr 字串）
    src = data_dir.replace("/", "\\")
    mdl = model_dir.replace("/", "\\")
    wb  = "True" if white_background else "False"
    line = (
        f"Namespace(sh_degree={sh_degree}, "
        f"source_path='{src}', "
        f"model_path='{mdl}', "
        f"images='{images_folder}', "
        f"depths='', resolution=-1, white_background={wb}, "
        f"train_test_exp=False, data_device='cuda', eval=True)"
    )
    with open(os.path.join(model_dir, "cfg_args"), "w") as f:
        f.write(line)


def setup(model_dir, data_dir, white_background, force):
    model_dir = os.path.abspath(model_dir)
    data_dir  = os.path.abspath(data_dir)

    splat_ply = os.path.join(model_dir, "splat.ply")
    if not os.path.exists(splat_ply):
        sys.exit(f"[Error] 找不到 splat.ply: {splat_ply}")

    tf = load_transforms(data_dir)

    # --- cameras.json ---
    cam_path = os.path.join(model_dir, "cameras.json")
    if force or not os.path.exists(cam_path):
        cameras = convert_cameras(tf, data_dir)
        with open(cam_path, "w") as f:
            json.dump(cameras, f)
        print(f"[OK] cameras.json  ({len(cameras)} cameras)")
    else:
        print(f"[Skip] cameras.json 已存在（用 --force 覆蓋）")

    # --- point_cloud/iteration_30000/point_cloud.ply ---
    pc_dir = os.path.join(model_dir, "point_cloud", "iteration_30000")
    os.makedirs(pc_dir, exist_ok=True)
    dst_ply = os.path.join(pc_dir, "point_cloud.ply")
    if force or not os.path.exists(dst_ply):
        shutil.copy2(splat_ply, dst_ply)
        print(f"[OK] point_cloud.ply（複製自 splat.ply）")
    else:
        print(f"[Skip] point_cloud.ply 已存在（用 --force 覆蓋）")

    # --- cfg_args ---
    cfg_path = os.path.join(model_dir, "cfg_args")
    if force or not os.path.exists(cfg_path):
        sh = detect_sh_degree(splat_ply)
        imgs = detect_images_folder(data_dir, tf)
        write_cfg_args(model_dir, data_dir, sh, imgs, white_background)
        print(f"[OK] cfg_args  (sh_degree={sh}, images='{imgs}')")
    else:
        print(f"[Skip] cfg_args 已存在（用 --force 覆蓋）")

    print()
    print("完成！用以下指令開啟 SIBR viewer：")
    viewer = os.path.join(
        os.path.dirname(__file__),
        "SIBR_viewers", "install", "bin", "SIBR_gaussianViewer_app.exe"
    )
    print(f'  "{viewer}" -m "{model_dir}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="設定 SIBR Gaussian Viewer 目錄結構")
    parser.add_argument("--model", required=True, help="splat.ply 所在目錄")
    parser.add_argument("--data",  required=True, help="transforms.json 所在目錄")
    parser.add_argument("--white-background", action="store_true", default=True,
                        help="白色背景（預設啟用）")
    parser.add_argument("--black-background", dest="white_background",
                        action="store_false", help="黑色背景")
    parser.add_argument("--force", action="store_true",
                        help="強制覆蓋已存在的檔案")
    args = parser.parse_args()
    setup(args.model, args.data, args.white_background, args.force)
