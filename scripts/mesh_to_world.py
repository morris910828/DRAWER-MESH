"""
Convert SDF mesh OBJ from NeRFStudio training space back to COLMAP world coordinates.

Supports two transform formats (auto-detected):
  dataparser_transforms.json  { "transform": [[3x4]], "scale": float }  (GS training output)
  transforms.json             { "frames": [...] }                        (COLMAP/DRAWER raw)

Usage:
    python mesh_to_world.py --input <mesh.obj> --transform <transforms.json> --output <mesh_world.obj>

What gets transformed:
    v  (vertices) : p_world = R_inv @ (p_training / scale) + t_inv
    vn (normals)  : n_world = R_inv @ n_gs  (renormalized; no translation/scale for directions)
    vt (uv)       : unchanged
    f  / mtllib   : unchanged  (mtllib still points to the original .mtl / .png)
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.timer import StepTimer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.timer import StepTimer


def _rotation_matrix_np(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1 + 1e-8:
        return _rotation_matrix_np(a + (np.random.rand(3) - 0.5) * 0.01, b)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2 + 1e-8))


def load_transform(path: str, extra_scale: float = 1.0):
    """Auto-detect format and return (R_inv, t_inv, scale)."""
    with open(path) as f:
        meta = json.load(f)

    if "transform" in meta and "scale" in meta:
        # Format A: dataparser_transforms.json
        T3x4 = np.array(meta["transform"], dtype=np.float64)
        if T3x4.shape[0] == 4:
            T3x4 = T3x4[:3]
        scale = float(meta["scale"])
    elif "frames" in meta:
        # Format B: transforms.json — replicate panoptic_dataparser auto_orient_and_center_poses
        poses = np.array([f["transform_matrix"] for f in meta["frames"]], dtype=np.float64)
        if poses.shape[1] == 3:
            bot = np.tile([0., 0., 0., 1.], (len(poses), 1, 1))
            poses = np.concatenate([poses, bot], axis=1)
        mean_t = poses[:, :3, 3].mean(0)
        up = poses[:, :3, 1].mean(0)
        up /= np.linalg.norm(up) + 1e-12
        R = _rotation_matrix_np(up, np.array([0., 0., 1.]))
        T3x4 = np.concatenate([R, (R @ -mean_t)[:, None]], axis=1)
        T4 = np.eye(4); T4[:3] = T3x4
        oriented_t = (T4 @ poses)[:, :3, 3]
        scale = 1.0 / float(np.abs(oriented_t).max())
    else:
        raise SystemExit(f"Unrecognised transform format: {path}")

    # Fold in the dataparser's --scale_factor (default 1.0 = no-op).
    scale = scale * extra_scale
    M_fwd = np.eye(4)
    M_fwd[:3, :3] = scale * T3x4[:3, :3]
    M_fwd[:3,  3] = scale * T3x4[:3,  3]
    M_inv = np.linalg.inv(M_fwd)
    R_inv = M_inv[:3, :3]
    t_inv = M_inv[:3,  3]
    return R_inv, t_inv, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",     required=True, help="Input .obj (training space)")
    parser.add_argument("--transform", required=True,
                        help="dataparser_transforms.json or transforms.json")
    parser.add_argument("--output",    required=True, help="Output .obj (COLMAP world space)")
    parser.add_argument("--timing", default=None, metavar="JSON",
                        help="Timing output JSON path (optional)")
    parser.add_argument("--extra-scale", type=float, default=1.0,
                        help="Multiply training→world scale by this (= dataparser --scale_factor; default 1.0)")
    args = parser.parse_args()

    timer = StepTimer(args.timing)

    @timer.record("step_1h_mesh_to_world")
    def run():
        R_inv, t_inv, scale = load_transform(args.transform, args.extra_scale)
        print(f"scale={scale:.6f}  R_inv det={np.linalg.det(R_inv):.4f}")

        # ── pass 1: collect all v and vn lines ───────────────────────────────
        v_lines, vn_lines = [], []
        line_tags = []   # ('v'|'vn'|'other', index_or_content)

        print("Reading OBJ ...", end=" ", flush=True)
        with open(args.input, "r") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if stripped.startswith("v "):
                    parts = stripped.split()
                    v_lines.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    line_tags.append(("v", len(v_lines) - 1))
                elif stripped.startswith("vn "):
                    parts = stripped.split()
                    vn_lines.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    line_tags.append(("vn", len(vn_lines) - 1))
                else:
                    line_tags.append(("other", stripped))

        print(f"{len(v_lines):,} vertices, {len(vn_lines):,} normals, "
              f"{sum(1 for t,_ in line_tags if t=='other'):,} other lines")

        # ── batch transform ───────────────────────────────────────────────────
        print("Transforming ...", end=" ", flush=True)

        v_arr = np.array(v_lines, dtype=np.float64)        # (N, 3)
        # R_inv/t_inv come from inverting M_fwd, which already carries the scale
        # (R_inv == R.T / scale), so dividing by scale here applied it a second
        # time and produced a mesh 1/scale too large -- 4.85x for this dataset,
        # which is why the mesh swamped the camera rig and every rendered mask
        # came out almost fully foreground.
        v_world = v_arr @ R_inv.T + t_inv                  # (N, 3)

        if vn_lines:
            vn_arr = np.array(vn_lines, dtype=np.float64)  # (M, 3)
            vn_world = vn_arr @ R_inv.T
            norms = np.linalg.norm(vn_world, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vn_world /= norms

        print("done")

        # ── write output ──────────────────────────────────────────────────────
        print(f"Writing {args.output} ...", end=" ", flush=True)
        with open(args.output, "w") as out:
            for tag, payload in line_tags:
                if tag == "v":
                    x, y, z = v_world[payload]
                    out.write(f"v {x:.10f} {y:.10f} {z:.10f}\n")
                elif tag == "vn":
                    x, y, z = vn_world[payload]
                    out.write(f"vn {x:.10f} {y:.10f} {z:.10f}\n")
                else:
                    out.write(payload + "\n")

        print("done")

        # ── sanity check ──────────────────────────────────────────────────────
        print("\n=== vertex range (input, training space) ===")
        for i, ax in enumerate("xyz"):
            print(f"  {ax}: [{v_arr[:, i].min():.3f}, {v_arr[:, i].max():.3f}]")
        print("=== vertex range (output, COLMAP world space) ===")
        for i, ax in enumerate("xyz"):
            print(f"  {ax}: [{v_world[:, i].min():.3f}, {v_world[:, i].max():.3f}]")

    run()


if __name__ == "__main__":
    main()
