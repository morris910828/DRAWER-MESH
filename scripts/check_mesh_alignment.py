#!/usr/bin/env python
"""mesh 是否對齊照片？用多視角顏色一致性量，不需要任何人工標註。

    python scripts/check_mesh_alignment.py --data data/table \
        --mesh data/table/table_sdf_ds1/texture_mesh/mesh_table-only-400k_world.obj \
        --mesh-train data/table/table_sdf_ds1/texture_mesh/mesh_table-only-400k.obj

原理：mesh 表面上的一個點，如果幾何是對的，它在每個看得到它的視角裡都落在照片
中的同一個實體上，顏色應該一致（漫反射假設）。幾何偏了，同一個點會落到不同的東
西上，顏色就散開。所以每個取樣點的跨視角顏色離散度，就是那一處的幾何誤差指標。

這個量測在 GS 訓練之前就能做，而 GS 訓練本身分辨不出「幾何偏了」和「材質難學」。

可見性用背面剔除處理（法線朝向相機），遮擋則靠 robust 統計吸收：離群的視角用
中位數絕對離差（MAD）而不是標準差，所以少數被擋住的視角不會主導結果。

--mesh-train 給訓練座標的同一顆 mesh（頂點順序相同），用來把取樣點依高度分區
（桌腳 / 桌面 / 桌上物件），這樣可以看出誤差集中在哪裡。不給就只報整體。
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    sys.exit("需要 pillow")


def read_obj(path):
    """回傳 (verts, faces)。只讀 v 與 f，忽略 vt/vn。"""
    vs, fs = [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                vs.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    fs.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(vs, np.float64), np.asarray(fs, np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="含 transforms.json 的資料夾")
    ap.add_argument("--mesh", required=True, help="world 座標的 OBJ（GS 訓練用的那顆）")
    ap.add_argument("--mesh-train", default=None,
                    help="同一顆 mesh 的訓練座標 OBJ，用來依高度分區")
    ap.add_argument("--images", default="images_4", help="取色用的影像資料夾")
    ap.add_argument("--samples", type=int, default=6000, help="取樣的面數")
    ap.add_argument("--min-views", type=int, default=5, help="至少要被幾個視角看到才計入")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift-mm", type=float, default=0.0,
                    help="把取樣點沿法線推開這麼多毫米，用來校準色散對幾何誤差的靈敏度。"
                         "0 = 量 mesh 本身")
    ap.add_argument("--unit-mm", type=float, default=221.0,
                    help="1 個 world 單位等於多少毫米（COLMAP 尺度，預設 221）")
    args = ap.parse_args()

    data = args.data
    meta = json.load(open(os.path.join(data, "transforms.json")))
    frames = meta["frames"]

    verts, faces = read_obj(args.mesh)
    print(f"mesh: {len(verts):,} verts, {len(faces):,} tris")

    tri = verts[faces]
    centroid = tri.mean(1)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nlen = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.maximum(nlen, 1e-30)

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(faces), size=min(args.samples, len(faces)), replace=False)
    C, N = centroid[sel], normal[sel]
    if args.shift_mm:
        C = C + N * (args.shift_mm / args.unit_mm)
        print(f"取樣點沿法線推開 {args.shift_mm:.1f} mm（校準用）")

    # 分區（可選）
    zone = np.zeros(len(sel), np.int8)
    zone_names = ["整體"]
    if args.mesh_train:
        vt, _ = read_obj(args.mesh_train)
        if len(vt) != len(verts):
            print("警告：--mesh-train 頂點數不符，略過分區")
        else:
            zc = vt[faces].mean(1)[sel][:, 2]
            zone = np.digitize(zc, [-0.245, -0.16])  # 桌腳 / 桌板 / 桌上
            zone_names = ["桌腳 (z<-0.245)", "桌板 (-0.245..-0.16)", "桌上物件 (z>-0.16)"]

    # 每個取樣點在每個視角的顏色
    cols = np.full((len(sel), len(frames), 3), np.nan, np.float32)
    fx, fy = meta["fl_x"], meta["fl_y"]
    cx, cy = meta["cx"], meta["cy"]
    W0, H0 = meta["w"], meta["h"]

    for fi, fr in enumerate(frames):
        stem = os.path.splitext(os.path.basename(fr["file_path"]))[0]
        img_path = None
        for ext in (".JPG", ".jpg", ".png", ".PNG"):
            p = os.path.join(data, args.images, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        im = np.asarray(Image.open(img_path).convert("RGB"), np.float32)
        Hh, Ww = im.shape[:2]
        s = Ww / W0

        c2w = np.array(fr["transform_matrix"], np.float64)
        w2c = np.linalg.inv(c2w)
        P = (w2c[:3, :3] @ C.T).T + w2c[:3, 3]
        z = -P[:, 2]
        # 背面剔除：面法線要朝向相機
        cam_dir = c2w[:3, 3] - C
        facing = (N * cam_dir).sum(1) > 0
        u = (fx * (P[:, 0] / np.maximum(z, 1e-9)) + cx) * s
        v = (cy - fy * (P[:, 1] / np.maximum(z, 1e-9))) * s
        ok = facing & (z > 1e-6) & (u >= 0) & (u < Ww - 1) & (v >= 0) & (v < Hh - 1)
        ui, vi = u[ok].astype(int), v[ok].astype(int)
        cols[np.flatnonzero(ok), fi] = im[vi, ui]
        print(f"  [{fi+1}/{len(frames)}] {stem}  可見取樣點 {ok.sum():,}", end="\r")

    print(" " * 70, end="\r")
    nview = (~np.isnan(cols[:, :, 0])).sum(1)
    keep = nview >= args.min_views
    print(f"取樣 {len(sel):,} 個面，其中 {keep.sum():,} 個被 >= {args.min_views} 個視角看到"
          f"（中位 {int(np.median(nview))} 個視角）\n")

    # robust 離散度：每通道的 MAD，再取三通道平均。0-255 尺度。
    def mad(a):
        med = np.nanmedian(a, axis=1, keepdims=True)
        return np.nanmedian(np.abs(a - med), axis=1)

    disp = mad(cols).mean(1)  # (S,)

    print(f"{'區域':<24} {'取樣':>7} {'色散 MAD (0-255)':>20}")
    print(f"{'':<24} {'':>7} {'p25':>6} {'中位':>6} {'p75':>6}")
    groups = [(0, np.ones(len(sel), bool))] if len(zone_names) == 1 else \
             [(i, zone == i) for i in range(len(zone_names))]
    for i, m in groups:
        mm = m & keep
        if mm.sum() < 20:
            continue
        d = disp[mm]
        print(f"{zone_names[i]:<24} {mm.sum():>7,} {np.percentile(d,25):>6.1f}"
              f" {np.median(d):>6.1f} {np.percentile(d,75):>6.1f}")

    print("\n對照：同一個表面點在各視角看起來應該一樣。MAD 在 0-255 尺度上，")
    print("  < 5   幾何準（差異只來自曝光與高光）")
    print("  5-15  可疑")
    print("  > 15  該處的 mesh 與照片對不上")


if __name__ == "__main__":
    main()
