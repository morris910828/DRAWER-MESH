"""
輸出 GS 訓練時實際使用的 mesh_depth mask（供視覺確認）。

使用：
  - mesh：GS 訓練時傳入的 mesh（training space OBJ，通常 mesh-clean-simplify.obj）
  - camera：GS dataparser_transforms.json 還原出的 GS-normalized camera poses
  - Y-flip：不需要（no flip）

輸出：
  <out_dir>/masks/         ← 二值 PNG mask（前景=255）
  <out_dir>/overlay/       ← 疊加在原圖（紅色=前景）

Usage:
    conda run -n drawer_splat python scripts/save_mesh_depth_masks.py \
        --data_dir      /opt/disk/drawer_dataset/studio/home \
        --mesh_path     /opt/disk/drawer_dataset/studio/home/home_sdf_recon/texture_mesh/mesh-clean-simplify.obj \
        --dataparser_tf /opt/disk/drawer_dataset/studio/home/home_gs_masked/dataparser_transforms.json \
        --out_dir       /opt/disk/drawer_dataset/studio/home/gs_masks \
        [--max_frames N]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.timer import StepTimer


def load_dataparser_transform(path):
    """Returns (T3x4 float32 tensor [3,4], scale float)."""
    with open(path) as f:
        m = json.load(f)
    T = torch.tensor(m["transform"], dtype=torch.float32)
    if T.shape[0] == 4:
        T = T[:3]
    return T, float(m["scale"])


def colmap_to_gs_cameras(transforms_json, T3x4, scale, max_frames=None):
    """
    Returns (poses_gs_3x4, frames, meta):
      poses_gs_3x4: (N,3,4) float32  GS-normalized c2w, translation scaled
    """
    with open(transforms_json) as f:
        meta = json.load(f)
    frames = meta["frames"]
    if max_frames:
        frames = frames[:max_frames]

    poses_colmap = []
    for fr in frames:
        p = torch.tensor(fr["transform_matrix"], dtype=torch.float32)
        if p.shape[0] == 3:
            p = torch.cat([p, torch.tensor([[0., 0., 0., 1.]])], dim=0)
        poses_colmap.append(p)
    poses_colmap = torch.stack(poses_colmap)  # (N,4,4)

    # Apply GS normalization: oriented = T3x4 @ pose (per frame)
    T4 = torch.eye(4, dtype=torch.float32)
    T4[:3] = T3x4
    oriented = (T4 @ poses_colmap)[:, :3, :]  # (N,3,4)
    oriented[:, :, 3] *= scale                # scale translation only
    return oriented, frames, meta


def build_projection(meta):
    fx = float(meta["fl_x"]); fy = float(meta["fl_y"])
    cx = float(meta["cx"]);   cy = float(meta["cy"])
    H  = int(meta["h"]);      W  = int(meta["w"])
    n, f = 0.001, 1e6
    n00 = 2.*fx/W; n11 = 2.*fy/H
    n02 = 2.*cx/W-1.; n12 = 2.*cy/H-1.
    n22 = (f+n)/(f-n); n23 = 2*f*n/(n-f)
    proj = torch.tensor([[n00,0,n02,0],[0,n11,n12,0],[0,0,n22,n23],[0,0,1.,0]],
                        dtype=torch.float32)
    return proj, H, W


def poses_to_square(poses_3x4):
    """NeRF → OpenGL convention, return (N,4,4) square_pose."""
    N = poses_3x4.shape[0]
    i_pose = poses_3x4.clone()
    i_pose[:, :3, 1:3] *= -1   # NeRF → OpenGL
    bot = torch.zeros(N, 1, 4); bot[:, 0, 3] = 1.
    return torch.cat([i_pose, bot], dim=1)


def rasterize_depth(verts_cuda, faces_cuda, mvp, w2c, H, W, glctx):
    import nvdiffrast.torch as dr
    verts_pad = F.pad(verts_cuda, (0, 1), value=1.0)
    verts_clip = (verts_pad @ mvp.T).unsqueeze(0).float()
    verts_cam  = (verts_pad @ w2c.T).float()
    verts_cam  = verts_cam[:, :3] / verts_cam[:, 3:]
    verts_depth_cam = verts_cam[:, -1:]

    rast, _ = dr.rasterize(glctx, verts_clip, faces_cuda.int(), (H, W))
    # no Y-flip (matches panoptic_dataparser behaviour)

    bary = torch.stack([rast[..., 0], rast[..., 1],
                        1-rast[..., 0]-rast[..., 1]], dim=-1).reshape(H, W, 3)
    pix_to_face = rast[..., -1].reshape(H, W)
    valid = pix_to_face > 0
    pix_to_face = (pix_to_face - 1).long()

    depth = torch.zeros(H, W, device="cuda")
    if valid.any():
        pd = verts_depth_cam[faces_cuda[pix_to_face[valid]].reshape(-1)].reshape(-1, 3)
        pb = bary[valid].reshape(-1, 3)
        inv_d = 1. / (pd + 1e-10)
        depth[valid] = 1. / ((inv_d * pb).sum(-1) + 1e-10)

    # 1-pixel dilation (same as panoptic_dataparser)
    invalid_idx = torch.nonzero(~valid).reshape(-1, 2)
    for dx, dy in [[-1,0],[1,0],[0,-1],[0,1]]:
        off = (invalid_idx + torch.tensor([dx,dy], device="cuda")).long()
        off[:,0].clamp_(0,H-1); off[:,1].clamp_(0,W-1)
        depth[~valid] += depth[off[:,0], off[:,1]] * 0.25
    return depth.abs().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      required=True)
    parser.add_argument("--mesh_path",     required=True,
                        help="GS 訓練時傳入的 mesh (SDF training space OBJ)")
    parser.add_argument("--dataparser_tf", required=True,
                        help="dataparser_transforms.json from GS training run")
    parser.add_argument("--out_dir",       required=True)
    parser.add_argument("--max_frames",    type=int, default=None)
    parser.add_argument("--timing", default=None, metavar="JSON",
                        help="Timing output JSON path (optional)")
    args = parser.parse_args()

    timer = StepTimer(args.timing)

    @timer.record("step_2e_save_masks")
    def run():
        import nvdiffrast.torch as dr
        from pytorch3d.io import load_objs_as_meshes

        data_dir = Path(args.data_dir)
        out_dir  = Path(args.out_dir)
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        (out_dir / "overlay").mkdir(parents=True, exist_ok=True)

        # ── Load transform & cameras ──────────────────────────────────────────
        T3x4, dp_scale = load_dataparser_transform(args.dataparser_tf)
        poses_gs, frames, meta = colmap_to_gs_cameras(
            data_dir / "transforms.json", T3x4, dp_scale, args.max_frames)
        proj, H, W = build_projection(meta)
        square_poses = poses_to_square(poses_gs)           # (N,4,4)
        mvps = proj.unsqueeze(0) @ torch.inverse(square_poses)  # (N,4,4)
        w2cs = torch.inverse(square_poses)

        # ── Load mesh (GS training space, same as --mesh_gauss_path) ─────────
        print(f"Loading mesh: {args.mesh_path}")
        mesh = load_objs_as_meshes([args.mesh_path], device='cpu')
        verts = mesh.verts_packed().cuda().float()
        faces = mesh.faces_packed().cuda().long()
        print(f"  {verts.shape[0]:,} verts, {faces.shape[0]:,} faces")

        glctx = dr.RasterizeCudaContext()

        # ── Process frames ────────────────────────────────────────────────────
        for i, frame in enumerate(frames):
            img_path = data_dir / frame["file_path"]
            if not img_path.exists():
                for ext in [".jpg",".png",".JPG",".PNG"]:
                    c = img_path.with_suffix(ext)
                    if c.exists(): img_path = c; break

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  [{i}] SKIP: {img_path.name}"); continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            mvp = mvps[i].cuda()
            w2c = w2cs[i].cuda()

            depth = rasterize_depth(verts, faces, mvp, w2c, H, W, glctx)
            mask  = (depth > 0).numpy().astype(np.uint8) * 255

            stem = Path(frame["file_path"]).stem
            cv2.imwrite(str(out_dir / "masks" / f"{stem}.png"), mask)

            # Overlay: foreground = red tint
            ov = img_rgb.copy().astype(np.float32)
            fg = mask > 0
            ov[fg, 0] = np.clip(ov[fg, 0]*0.5+127, 0, 255)
            ov[fg, 1] = np.clip(ov[fg, 1]*0.5,      0, 255)
            ov[fg, 2] = np.clip(ov[fg, 2]*0.5,      0, 255)
            cv2.imwrite(str(out_dir/"overlay"/f"{stem}.jpg"),
                        cv2.cvtColor(ov.astype(np.uint8), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            if (i+1) % 50 == 0 or (i+1) == len(frames):
                print(f"  {i+1}/{len(frames)} frames done")

        print(f"\nDone → {out_dir}/overlay/")

    run()


if __name__ == "__main__":
    main()
