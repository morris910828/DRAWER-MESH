#!/usr/bin/env python
"""Batch-transform PLY mesh files from NeRFStudio training space → COLMAP world coordinates.

Supports two input formats (auto-detected by JSON key presence):

  Format A — dataparser_transforms.json (from GS / splat training):
    { "transform": [[3x4 or 4x4 matrix]], "scale": float }

  Format B — transforms.json (raw COLMAP/DRAWER camera file):
    { "frames": [{"transform_matrix": [[4x4]]},...] }
    Replicates panoptic_dataparser's auto_orient_and_center_poses(method="up")
    + scale logic — no nerfstudio import required.

Transform (training space → COLMAP world):
    p_world = R_inv @ (p_training / scale) + t_inv
    implemented as a 4×4 affine matrix applied via trimesh.apply_transform().

Usage:
    # Format A (GS dataparser_transforms.json):
    python scripts/ply_to_world.py \\
        --transform <gs_exp>/dataparser_transforms.json \\
        --inputs    mesh.ply mesh-simplify.ply mesh-clean.ply

    # Format B (raw transforms.json from COLMAP):
    python scripts/ply_to_world.py \\
        --transform <data_dir>/transforms.json \\
        --inputs    mesh.ply mesh-simplify.ply mesh-clean.ply \\
        [--suffix _world] [--outdir /path/to/output]
"""
from __future__ import annotations
import argparse
import json
import sys
import numpy as np
import trimesh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.timer import StepTimer


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodrigues rotation: rotate unit vector a to unit vector b. (3×3)"""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-8:
        eps = (np.random.rand(3) - 0.5) * 0.01
        return _rotation_matrix_np(a + eps, b)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2],  0, -v[0]],
                  [-v[1], v[0],  0]], dtype=np.float64)
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2 + 1e-8))


def _auto_orient_and_center_up(poses: np.ndarray):
    """
    Replicate nerfstudio camera_utils.auto_orient_and_center_poses(method='up').
    poses: (N, 4, 4) float64 — camera-to-world matrices
    Returns: (transform_3x4, scale_factor)
    """
    translations = poses[:, :3, 3]           # (N, 3)
    mean_t = translations.mean(axis=0)       # (3,)

    # Average camera "up" (Y column of c2w)
    up = poses[:, :3, 1].mean(axis=0)
    up = up / (np.linalg.norm(up) + 1e-12)

    # Rotation that aligns 'up' → [0,0,1]
    R = _rotation_matrix_np(up, np.array([0., 0., 1.]))

    # 3×4 transform that centers and rotates
    transform_3x4 = np.concatenate([R, R @ (-mean_t)[:, None]], axis=1)  # (3, 4)

    # Build 4×4 and apply to poses to find scale
    T4 = np.eye(4)
    T4[:3] = transform_3x4
    oriented = (T4 @ poses)[:, :3, 3]       # (N, 3) translations in oriented space

    scale_factor = 1.0 / float(np.abs(oriented).max())
    return transform_3x4, scale_factor


def load_inverse_transform(transform_json: Path, extra_scale: float = 1.0) -> np.ndarray:
    """
    Parse the transform JSON (either format A or B) and return the 4×4 affine
    matrix that maps training-space vertices → COLMAP world coordinates.
    """
    data = json.loads(transform_json.read_text())

    if "transform" in data and "scale" in data:
        # ── Format A: dataparser_transforms.json ──────────────────────────────
        T34 = np.array(data["transform"], dtype=np.float64)
        if T34.ndim == 2 and T34.shape[0] == 4:
            T34 = T34[:3]
        scale = float(data["scale"])
        print(f"  Format A (dataparser_transforms): scale={scale:.6f}")

    elif "frames" in data:
        # ── Format B: transforms.json (COLMAP/DRAWER) ─────────────────────────
        poses_raw = np.array([fr["transform_matrix"] for fr in data["frames"]],
                             dtype=np.float64)
        if poses_raw.shape[1] == 3:
            bot = np.tile([0., 0., 0., 1.], (len(poses_raw), 1, 1))
            poses_raw = np.concatenate([poses_raw, bot], axis=1)  # (N,4,4)

        transform_3x4, scale = _auto_orient_and_center_up(poses_raw)
        T34 = transform_3x4
        print(f"  Format B (transforms.json): scale={scale:.6f}")

    else:
        raise SystemExit(f"Unrecognised JSON format in {transform_json}")

    # Fold in the dataparser's --scale_factor (default 1.0 = no-op). The panoptic
    # dataparser uses training_scale = auto_scale * scale_factor; this replicates it.
    scale = scale * extra_scale

    # Forward 4×4: p_training = scale * (T34 @ [p_world; 1])
    # → M_fwd[:3,:3] = scale*R, M_fwd[:3,3] = scale*t
    M_fwd = np.eye(4)
    M_fwd[:3, :3] = scale * T34[:3, :3]
    M_fwd[:3,  3] = scale * T34[:3,  3]

    M_inv = np.linalg.inv(M_fwd)
    print(f"  |det(M_inv[:3,:3])| = {abs(np.linalg.det(M_inv[:3,:3])):.6f} "
          f"  (expect ~{1/scale**3:.6f})")
    return M_inv


def transform_ply(input_path: Path, output_path: Path, M_inv: np.ndarray) -> None:
    """Load PLY, apply 4×4 affine M_inv to vertices, recompute normals, save."""
    mesh = trimesh.load(str(input_path), force="mesh", process=False)
    vmin = mesh.vertices.min(axis=0)
    vmax = mesh.vertices.max(axis=0)
    print(f"  {input_path.name}: {len(mesh.faces):,} faces")
    print(f"    bbox training: [{vmin[0]:.3f},{vmin[1]:.3f},{vmin[2]:.3f}] "
          f"→ [{vmax[0]:.3f},{vmax[1]:.3f},{vmax[2]:.3f}]")

    mesh.apply_transform(M_inv)

    vmin = mesh.vertices.min(axis=0)
    vmax = mesh.vertices.max(axis=0)
    print(f"    bbox world:    [{vmin[0]:.3f},{vmin[1]:.3f},{vmin[2]:.3f}] "
          f"→ [{vmax[0]:.3f},{vmax[1]:.3f},{vmax[2]:.3f}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
    print(f"    saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Transform PLY files from NeRFStudio training space to COLMAP world coords"
    )
    ap.add_argument("--transform", required=True, type=Path,
                    help="dataparser_transforms.json (GS) or transforms.json (COLMAP/DRAWER)")
    ap.add_argument("--inputs",  required=True, nargs="+", type=Path,
                    help="Input PLY file(s) in NeRFStudio training space")
    ap.add_argument("--suffix", default="_world",
                    help="Suffix added before .ply extension (default: _world)")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output directory (default: same dir as each input)")
    ap.add_argument("--timing", default=None, metavar="JSON",
                    help="Timing output JSON path (optional)")
    ap.add_argument("--extra-scale", type=float, default=1.0,
                    help="Multiply training→world scale by this (= dataparser --scale_factor; default 1.0)")
    args = ap.parse_args()

    timer = StepTimer(args.timing)

    @timer.record("step_1f_ply_to_world")
    def run():
        transform_path = args.transform.resolve()
        if not transform_path.exists():
            raise SystemExit(f"Transform file not found: {transform_path}")

        print(f"Loading transform: {transform_path.name}")
        M_inv = load_inverse_transform(transform_path, args.extra_scale)

        for inp in args.inputs:
            inp = inp.resolve()
            if not inp.exists():
                print(f"  SKIP (not found): {inp}")
                continue

            stem = inp.stem + args.suffix
            out = (args.outdir.resolve() if args.outdir else inp.parent) / (stem + inp.suffix)
            transform_ply(inp, out, M_inv)

        print("All done.")

    run()


if __name__ == "__main__":
    main()
