#!/usr/bin/env python
"""
verify_depth_fusion.py

Verify whether per-frame Marigold monocular (affine-invariant) depth maps are
geometrically consistent with a COLMAP SfM reconstruction, by:

  1. Fitting a per-frame affine model (z_metric ~= a*d_rel + b) using the sparse
     SfM keypoints that have a 3D point (giving metric depth z = (R@Xw+t).z).
     Robust: one round of ~2.5-sigma residual rejection + refit.  If the affine
     fit is poor / a<=0, also tries an inverse-depth model (z ~= a/(d_rel+eps)+b)
     and keeps whichever fits better.
  2. Back-projecting the dense (metricized) Marigold depth to world space and
     fusing into a single colored point cloud.

Everything is in RAW COLMAP world coordinates (no nerfstudio transform applied).

Outputs:
  <out_dir>/depth_fusion_check.ply   fused Marigold cloud (xyz + rgb)
  <out_dir>/colmap_sparse.ply        COLMAP points3D cloud (xyz + rgb) reference

Run inside the `drawer_sdf` conda env (numpy, scipy, PIL).
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

# Repo's pure-python COLMAP readers (no pycolmap needed).
sys.path.insert(0, "/workspace/DRAWER/sdf")
from nerfstudio.process_data.colmap_utils import (  # noqa: E402
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    qvec2rotmat,
)


# --------------------------------------------------------------------------- #
# Minimal PLY writer
# --------------------------------------------------------------------------- #
def write_ply(path, xyz, rgb):
    """Write a binary little-endian colored point cloud PLY."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #
def build_K(cam):
    m = cam.model
    p = cam.params
    if m in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
               "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    else:
        raise ValueError(f"Unsupported camera model: {m}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K


# --------------------------------------------------------------------------- #
# Robust affine / inverse fitting
# --------------------------------------------------------------------------- #
def fit_linear(x, z):
    """Least squares z = a*x + b. Returns a, b, pred, r2."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b = coef
    pred = A @ coef
    ss_res = np.sum((z - pred) ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, pred, r2


def robust_fit(x, z):
    """Fit with one round of 2.5-sigma residual rejection then refit."""
    a, b, pred, r2 = fit_linear(x, z)
    resid = z - pred
    sigma = np.std(resid)
    if sigma > 0:
        keep = np.abs(resid) <= 2.5 * sigma
        if keep.sum() >= max(20, int(0.5 * len(x))):
            a, b, pred, r2 = fit_linear(x[keep], z[keep])
            # recompute r2 / residual stats over the kept inliers
            return a, b, r2, keep
    keep = np.ones_like(x, dtype=bool)
    return a, b, r2, keep


def fit_frame(d_rel, z_metric, eps=1e-3):
    """
    Try affine (z ~ a*d + b) and inverse (z ~ a/(d+eps) + b).
    Returns dict with chosen model parameters and quality.
    """
    # affine
    a1, b1, r2_1, keep1 = robust_fit(d_rel, z_metric)
    # inverse
    inv = 1.0 / (d_rel + eps)
    a2, b2, r2_2, keep2 = robust_fit(inv, z_metric)

    use_inverse = False
    # Prefer affine if it is valid (a>0) and decent; otherwise compare.
    affine_valid = a1 > 0 and r2_1 >= 0.5
    if not affine_valid:
        # consider inverse; pick better r2 among the two
        if r2_2 > r2_1:
            use_inverse = True

    if use_inverse:
        a, b, r2, keep = a2, b2, r2_2, keep2
        def to_metric(dd):
            return a / (dd + eps) + b
        model = "inverse"
    else:
        a, b, r2, keep = a1, b1, r2_1, keep1
        def to_metric(dd):
            return a * dd + b
        model = "affine"

    # median relative residual over inliers (vs the chosen model)
    pred_all = to_metric(d_rel)
    resid = np.abs(z_metric - pred_all)
    med_z = np.median(z_metric)
    med_rel_resid = np.median(resid) / med_z if med_z > 0 else np.nan

    return {
        "a": a, "b": b, "r2": r2, "model": model,
        "to_metric": to_metric, "med_rel_resid": med_rel_resid,
        "n_inlier": int(keep.sum()),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="/opt/disk/drawer_dataset/robot_hand/20260628_robot_hand")
    ap.add_argument("--out_dir", default=None,
                    help="defaults to data_dir")
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=14,
                    help="pixel subsample stride for dense fusion "
                         "(14 -> ~1.5M fused pts over 60 full-res frames)")
    ap.add_argument("--min_sparse", type=int, default=20,
                    help="skip frames with fewer valid sparse points")
    args = ap.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)

    sparse = os.path.join(data_dir, "sparse", "0")
    cams = read_cameras_binary(os.path.join(sparse, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse, "images.bin"))
    print(f"[load] reading points3D.bin ...", flush=True)
    points3d = read_points3d_binary(os.path.join(sparse, "points3D.bin"))
    print(f"[load] {len(cams)} cameras, {len(images)} images, "
          f"{len(points3d)} points3D", flush=True)

    # --- write COLMAP sparse cloud reference --------------------------------
    sp_xyz = np.array([p.xyz for p in points3d.values()], dtype=np.float32)
    sp_rgb = np.array([p.rgb for p in points3d.values()], dtype=np.uint8)
    sparse_ply = os.path.join(out_dir, "colmap_sparse.ply")
    write_ply(sparse_ply, sp_xyz, sp_rgb)
    print(f"[ply ] wrote {sparse_ply}  ({sp_xyz.shape[0]} pts)", flush=True)

    # --- choose frames evenly spaced ----------------------------------------
    # order images by name for deterministic, even sampling
    img_items = sorted(images.values(), key=lambda im: im.name)
    n_total = len(img_items)
    if args.max_frames >= n_total:
        sel = list(range(n_total))
    else:
        sel = np.linspace(0, n_total - 1, args.max_frames).round().astype(int)
        sel = sorted(set(int(i) for i in sel))
    print(f"[plan] using {len(sel)} of {n_total} frames, stride={args.stride}",
          flush=True)

    depth_dir = os.path.join(data_dir, "depth")
    image_dir = os.path.join(data_dir, "images")

    all_xyz = []
    all_rgb = []
    r2_list = []
    relresid_list = []
    n_inverse = 0
    n_used = 0
    n_skipped = 0

    for idx in sel:
        im = img_items[idx]
        stem = os.path.splitext(im.name)[0]
        dpath = os.path.join(depth_dir, stem + ".npy")
        ipath = os.path.join(image_dir, im.name)
        if not os.path.exists(dpath) or not os.path.exists(ipath):
            n_skipped += 1
            continue

        cam = cams[im.camera_id]
        K = build_K(cam)
        R = qvec2rotmat(im.qvec)
        t = im.tvec.reshape(3)

        # --- gather sparse correspondences ---
        ids = im.point3d_ids
        xys = im.xys
        valid = ids >= 0
        if valid.sum() < args.min_sparse:
            n_skipped += 1
            continue

        d_rel_map = np.load(dpath)
        Hd, Wd = d_rel_map.shape
        Wimg, Himg = cam.width, cam.height
        sx = Wd / float(Wimg)
        sy = Hd / float(Himg)

        us = xys[valid, 0]
        vs = xys[valid, 1]
        ids_v = ids[valid]

        z_metric = []
        d_rel = []
        for u, v, pid in zip(us, vs, ids_v):
            p = points3d.get(int(pid))
            if p is None:
                continue
            Xc = R @ p.xyz + t
            z = Xc[2]
            if z <= 0:
                continue
            # depth-map pixel coords
            du = int(round(u * sx))
            dv = int(round(v * sy))
            if du < 0 or du >= Wd or dv < 0 or dv >= Hd:
                continue
            z_metric.append(z)
            d_rel.append(d_rel_map[dv, du])

        if len(z_metric) < args.min_sparse:
            n_skipped += 1
            continue

        z_metric = np.asarray(z_metric, dtype=np.float64)
        d_rel = np.asarray(d_rel, dtype=np.float64)

        fit = fit_frame(d_rel, z_metric)
        r2_list.append(fit["r2"])
        relresid_list.append(fit["med_rel_resid"])
        if fit["model"] == "inverse":
            n_inverse += 1

        # --- dense back-projection ---
        d_full = d_rel_map  # HdxWd
        # subsample
        vv, uu = np.mgrid[0:Hd:args.stride, 0:Wd:args.stride]
        uu = uu.ravel()
        vv = vv.ravel()
        dz = fit["to_metric"](d_full[vv, uu].astype(np.float64))
        good = dz > 0
        uu, vv, dz = uu[good], vv[good], dz[good]

        # pixel coords in image (full-res) space for unprojection with K
        u_img = uu / sx
        v_img = vv / sy
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_cam = (u_img - cx) / fx * dz
        y_cam = (v_img - cy) / fy * dz
        z_cam = dz
        Xc = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N,3)
        # cam -> world : Xw = R^T (Xc - t)
        Xw = (Xc - t) @ R  # since R^T applied: (Xc-t)@R == R^T@(Xc-t) per row

        # color
        rgb_img = np.asarray(Image.open(ipath).convert("RGB"))
        Hi, Wi = rgb_img.shape[:2]
        ci = np.clip(np.round(v_img).astype(int), 0, Hi - 1)
        cj = np.clip(np.round(u_img).astype(int), 0, Wi - 1)
        cols = rgb_img[ci, cj]

        all_xyz.append(Xw.astype(np.float32))
        all_rgb.append(cols.astype(np.uint8))
        n_used += 1

    if n_used == 0:
        print("[err ] no usable frames!", flush=True)
        return

    fused_xyz = np.concatenate(all_xyz, axis=0)
    fused_rgb = np.concatenate(all_rgb, axis=0)
    fused_ply = os.path.join(out_dir, "depth_fusion_check.ply")
    write_ply(fused_ply, fused_xyz, fused_rgb)
    print(f"[ply ] wrote {fused_ply}  ({fused_xyz.shape[0]} pts)", flush=True)

    # --- summary ---
    r2_arr = np.asarray(r2_list)
    rr_arr = np.asarray(relresid_list)
    print("\n========== SUMMARY ==========")
    print(f"frames used                : {n_used}")
    print(f"frames skipped             : {n_skipped}")
    print(f"total fused points         : {fused_xyz.shape[0]}")
    print(f"per-frame R^2  min/med/max : "
          f"{r2_arr.min():.4f} / {np.median(r2_arr):.4f} / {r2_arr.max():.4f}")
    print(f"median rel residual        : {np.median(rr_arr):.4f} "
          f"(across-frame median of per-frame median|resid|/median z)")
    print(f"frames preferring inverse  : {n_inverse} / {n_used}")
    print(f"colmap sparse points       : {sp_xyz.shape[0]}")

    # verdict
    med_r2 = float(np.median(r2_arr))
    med_rr = float(np.median(rr_arr))
    good = med_r2 >= 0.8 and med_rr <= 0.10
    moderate = med_r2 >= 0.6 and med_rr <= 0.20
    print("\n========== VERDICT ==========")
    if good:
        print("GOOD: Marigold depths are geometrically consistent with COLMAP "
              f"SfM (median R^2={med_r2:.3f}, median rel residual={med_rr:.3f}); "
              "the fused cloud should coincide with the sparse cloud.")
    elif moderate:
        print("MODERATE: Marigold depths are roughly consistent with COLMAP SfM "
              f"(median R^2={med_r2:.3f}, median rel residual={med_rr:.3f}); "
              "usable after affine alignment but with noticeable per-frame scatter.")
    else:
        print("POOR: Marigold depths do NOT fit COLMAP SfM well "
              f"(median R^2={med_r2:.3f}, median rel residual={med_rr:.3f}); "
              "the fused cloud will not align cleanly with the sparse cloud.")


if __name__ == "__main__":
    main()
