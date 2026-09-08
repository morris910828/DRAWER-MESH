"""
Convert SDF mesh vertices from nerfstudio local space back to COLMAP world space,
then write as COLMAP points3D.txt (and .ply) for use as GS initialization.

The mesh.ply is in nerfstudio normalized space. The dataparser_transforms.json
defines:  X_ns = scale * (R @ X_colmap + t)
Inverse:  X_colmap = (1/scale) * R^T @ X_ns - R^T @ t

Usage:
    python scripts/mesh_to_colmap_points.py \
        --mesh_path /opt/disk/studio/studio_h200ver/studio_h200ver_sdf_recon/mesh.ply \
        --transform_json /opt/disk/studio/studio_h200ver/studio_h200ver_mesh_gauss_splat/dataparser_transforms.json \
        --output_dir /opt/disk/studio/studio_h200ver/colmap_from_mesh \
        --num_points 500000
"""

import argparse
import json
import os
import struct

import numpy as np
import trimesh
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--mesh_path", type=str, required=True)
parser.add_argument("--transform_json", type=str, required=True,
                    help="dataparser_transforms.json from nerfstudio output")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_points", type=int, default=500000,
                    help="number of vertices to sample (0 = use all)")
parser.add_argument("--color", type=int, nargs=3, default=[180, 180, 180],
                    help="RGB color for points (mesh has no vertex colors)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── load dataparser transform ─────────────────────────────────────────────────
with open(args.transform_json) as f:
    dp = json.load(f)

T34 = np.array(dp["transform"], dtype=np.float64)   # (3, 4)
scale = float(dp["scale"])

R = T34[:3, :3]   # (3, 3)  rotation part
t = T34[:3, 3]    # (3,)    translation part

# inverse:  X_colmap = (1/scale) * R^T @ X_ns - R^T @ t
R_inv = R.T
t_inv = -R_inv @ t

print(f"Transform scale: {scale:.6f}")
print(f"R:\n{R}")
print(f"t: {t}")

# ── load mesh ─────────────────────────────────────────────────────────────────
print(f"\nLoading mesh: {args.mesh_path}")
mesh = trimesh.load(args.mesh_path, process=False)
verts_ns = np.array(mesh.vertices, dtype=np.float64)   # nerfstudio space
print(f"  vertices: {len(verts_ns):,}  faces: {len(mesh.faces):,}")
print(f"  bounds (ns): {verts_ns.min(0).round(3)} → {verts_ns.max(0).round(3)}")

# ── inverse transform → COLMAP world ─────────────────────────────────────────
# X_ns = scale * (R @ X_colmap + t)
# X_colmap = R^T @ (X_ns / scale) - R^T @ t
verts_colmap = (verts_ns / scale) @ R_inv.T + t_inv  # (N, 3)
print(f"  bounds (colmap): {verts_colmap.min(0).round(3)} → {verts_colmap.max(0).round(3)}")

# ── sample vertices ───────────────────────────────────────────────────────────
N = len(verts_colmap)
n_out = args.num_points if args.num_points > 0 else N
n_out = min(n_out, N)

if n_out < N:
    # surface-area weighted sampling: sample faces proportional to area,
    # then take a vertex from each sampled face
    areas = mesh.area_faces                    # (F,)
    probs = areas / areas.sum()
    face_idx = np.random.choice(len(mesh.faces), size=n_out, replace=False, p=probs)
    vert_idx = mesh.faces[face_idx, np.random.randint(0, 3, size=n_out)]
    pts = verts_colmap[vert_idx]
else:
    pts = verts_colmap
    n_out = N

print(f"\nSampled {n_out:,} / {N:,} points")

# ── colors ────────────────────────────────────────────────────────────────────
R_col, G_col, B_col = args.color

# ── write points3D.txt ────────────────────────────────────────────────────────
txt_path = os.path.join(args.output_dir, "points3D.txt")
print(f"Writing {txt_path} ...")
with open(txt_path, "w") as f:
    f.write("# 3D point list with one line of data per point:\n")
    f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    f.write(f"# Number of points: {n_out}\n")
    for i, (x, y, z) in enumerate(tqdm(pts, desc="writing txt")):
        # empty TRACK — valid for initialization use
        f.write(f"{i+1} {x:.6f} {y:.6f} {z:.6f} {R_col} {G_col} {B_col} 0\n")

# ── write points3D.bin ────────────────────────────────────────────────────────
bin_path = os.path.join(args.output_dir, "points3D.bin")
print(f"Writing {bin_path} ...")
with open(bin_path, "wb") as f:
    # header: number of points (uint64)
    f.write(struct.pack("<Q", n_out))
    for i, (x, y, z) in enumerate(tqdm(pts, desc="writing bin")):
        # POINT3D_ID (uint64), X Y Z (double), R G B (uint8), error (double),
        # track_length (uint64), track elements...
        f.write(struct.pack("<Q", i + 1))          # point3D_id
        f.write(struct.pack("<ddd", x, y, z))       # xyz
        f.write(struct.pack("<BBB", R_col, G_col, B_col))  # rgb
        f.write(struct.pack("<d", 0.0))            # error
        f.write(struct.pack("<Q", 0))              # track_length = 0

# ── write .ply for inspection ─────────────────────────────────────────────────
ply_path = os.path.join(args.output_dir, "points3D.ply")
print(f"Writing {ply_path} ...")
pc = trimesh.PointCloud(pts, colors=np.full((len(pts), 3), args.color, dtype=np.uint8))
pc.export(ply_path)

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  mesh_to_colmap_points Summary")
print("=" * 55)
print(f"  Input mesh      : {args.mesh_path}")
print(f"  Vertices total  : {N:,}")
print(f"  Points sampled  : {n_out:,}")
print(f"  Scale factor    : {scale:.6f}")
print(f"  COLMAP bounds   :")
print(f"    min  {pts.min(0).round(4)}")
print(f"    max  {pts.max(0).round(4)}")
print(f"  Output dir      : {args.output_dir}")
print(f"    points3D.txt  : {os.path.getsize(txt_path)/1e6:.1f} MB")
print(f"    points3D.bin  : {os.path.getsize(bin_path)/1e6:.1f} MB")
print(f"    points3D.ply  : {os.path.getsize(ply_path)/1e6:.1f} MB")
print("=" * 55)
print("\nNext step: place points3D.bin (or .txt) in your COLMAP sparse/0/ directory")
print("alongside cameras.bin and images.bin, then train splatfacto.")
