"""畫面實際上被誰塗滿？

sigma 的中位數會誤導：它把塗滿三角形的大盤子和幾乎看不見的小點等權重看待。
真正決定畫面長相的是 opacity 加權的螢幕塗抹面積 pi * sx_px * sy_px * alpha。
分成 base（被 floor 綁住、每面一顆）/ detail（面內自由）/ vertex 三層來看。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_splat_ply import read_ply

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("ply")
_ap.add_argument("--transforms", required=True)
_ap.add_argument("--frame", default=None, help="frame stem, e.g. GOPR0236")
_a = _ap.parse_args()
PLY, TF = _a.ply, _a.transforms

G, verts, faces = read_ply(PLY)
n = len(G["xyz"])
face_based = G["face"] >= 0
print(f"{PLY}\n  {n:,} gaussians  ({face_based.sum():,} face-based + {(~face_based).sum():,} vertex)")

tri = verts[faces]
centroid = tri.mean(1)
cv = np.linalg.norm(tri - centroid[:, None], axis=2)      # (F,3) 重心->角
cvm = cv.mean(1)

sx = G["scales"][:, 0]
sy = G["scales"][:, 1]
smax = np.maximum(sx, sy)

at_floor = np.zeros(n, bool)
fi = G["face"]
ok = face_based
at_floor[ok] = smax[ok] / np.maximum(cvm[fi[ok]], 1e-30) > 0.70

layer = np.full(n, 2, np.int8)          # 2 = vertex
layer[face_based & at_floor] = 0        # 0 = base
layer[face_based & ~at_floor] = 1       # 1 = detail

meta = json.load(open(TF))
frames = {os.path.splitext(os.path.basename(f["file_path"]))[0]: f for f in meta["frames"]}
STEM = _a.frame or sorted(frames)[0]
c2w = np.array(frames[STEM]["transform_matrix"], float)
R = c2w[:3, :3].copy()
R[:, 1] *= -1
R[:, 2] *= -1
cam = (G["xyz"] - c2w[:3, 3]) @ R
z = cam[:, 2]
fl, W, H = meta["fl_x"], meta["w"], meta["h"]
u = fl * cam[:, 0] / np.where(z != 0, z, 1e-9) + meta["cx"]
v = fl * cam[:, 1] / np.where(z != 0, z, 1e-9) + meta["cy"]
vis = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

sxp = sx * fl / np.maximum(z, 1e-9)
syp = sy * fl / np.maximum(z, 1e-9)
sig = np.maximum(sxp, syp)
alpha = G["alpha"]
smear = np.pi * sxp * syp * alpha        # opacity 加權的螢幕塗抹面積

tot = smear[vis].sum()
print(f"\n可見 {vis.sum():,} 顆   總塗抹 {tot:.3e} px^2   (畫面 {W*H:.3e} px^2, 重疊 {tot/(W*H):.2f}x)")

print(f"\n{'層':<8} {'顆數':>10} {'數量%':>7} {'sigma中位':>10} {'alpha中位':>10} {'塗抹%':>8}")
for i, name in ((0, "base"), (1, "detail"), (2, "vertex")):
    m = vis & (layer == i)
    if not m.any():
        continue
    print(f"{name:<8} {m.sum():>10,} {100*m.sum()/vis.sum():>6.1f}% {np.median(sig[m]):>10.2f}"
          f" {np.median(alpha[m]):>10.3f} {100*smear[m].sum()/tot:>7.1f}%")

print(f"\n按 on-screen sigma 分桶（決定字能不能被解析的是 sigma <= 1 px）：")
edges = [0, 0.5, 1, 2, 3, 5, 10, 1e9]
print(f"  {'sigma_px':>14} {'顆數':>10} {'數量%':>7} {'塗抹%':>8}")
for a, b in zip(edges[:-1], edges[1:]):
    m = vis & (sig >= a) & (sig < b)
    if not m.any():
        continue
    lab = f"{a:g}–{b:g}" if b < 1e8 else f">{a:g}"
    print(f"  {lab:>14} {m.sum():>10,} {100*m.sum()/vis.sum():>6.1f}% {100*smear[m].sum()/tot:>7.1f}%")

# 三角形本身多大？base 的尺寸下限就是由它決定的
tri_px = np.zeros(len(faces))
cc = (centroid - c2w[:3, 3]) @ R
zc = np.maximum(cc[:, 2], 1e-9)
tri_px = cvm * fl / zc
front = cc[:, 2] > 0.05
print(f"\nmesh 三角形的螢幕尺寸（重心到角，可見面 {front.sum():,}）：")
print(f"  中位 {np.median(tri_px[front]):.2f} px   q10 {np.quantile(tri_px[front],0.1):.2f}"
      f"   q90 {np.quantile(tri_px[front],0.9):.2f} px")
print(f"  -> base 盤子被綁在這個尺度上（floor = 面半徑的 0.7 倍以上）")
