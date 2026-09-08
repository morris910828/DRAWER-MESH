#!/usr/bin/env python
"""Filter interior (nested) meshes from a PLY file produced by marching cubes.

Strategy A: Connected-component + winding-number (ray parity) test
  - Split mesh into connected components (sorted by face count)
  - For each non-largest component, sample points from its vertices
  - Cast rays (+X axis) against the largest component; odd hits = inside
  - Discard components where >inside_thresh fraction of points are inside

Strategy B (--use_ray_filter): per-face back-face ray test (second pass)
  - Cast ray from each face center along its outward normal
  - If first hit face has dot(ray_dir, hit_normal) > 0 (same direction = back face) → interior
  - Remove those faces, then take largest connected component

Note: Does NOT require rtree — uses trimesh's pure-numpy RayMeshIntersector.

Usage:
    python scripts/filter_interior_mesh.py \\
        --input  <sdf_recon>/mesh.ply \\
        --output <sdf_recon>/mesh-clean.ply \\
        [--use_ray_filter] \\
        [--n_samples 20] \\
        [--inside_thresh 0.5]
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import trimesh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.timer import StepTimer


def points_inside_mesh(ref_mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """
    Ray-parity test with 3-axis majority vote.
    Cast rays along +X, +Y, +Z from each point; odd intersection count = inside.
    Majority vote across axes reduces edge/vertex degeneracy artifacts.
    Requires rtree (pip install rtree).
    """
    from trimesh.ray.ray_triangle import RayMeshIntersector
    intersector = RayMeshIntersector(ref_mesh)
    origins = np.asarray(points, dtype=np.float64)
    n = len(origins)

    votes = np.zeros(n, dtype=np.int32)
    for axis_dir in ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]):
        directions = np.tile(np.array(axis_dir), (n, 1))
        _tri_idx, ray_idx = intersector.intersects_id(
            origins, directions, multiple_hits=True, return_locations=False,
        )
        counts = np.bincount(ray_idx, minlength=n)
        votes += (counts % 2).astype(np.int32)

    return votes >= 2  # majority: ≥2 of 3 axes say inside


def filter_interior_components(
    mesh: trimesh.Trimesh,
    n_samples: int = 20,
    inside_thresh: float = 0.5,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """Remove connected components enclosed inside the largest component."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        if verbose:
            print("  Only 1 component — nothing to filter.")
        return mesh

    components = sorted(components, key=lambda c: len(c.faces), reverse=True)
    if verbose:
        print(f"  Found {len(components)} connected components:")
        for i, c in enumerate(components):
            print(f"    [{i}] {len(c.faces):>8,} faces, {len(c.vertices):>8,} verts")

    main_mesh = components[0]
    kept = [main_mesh]

    for i, comp in enumerate(components[1:], 1):
        # Use face centers (less prone to degeneracy than vertices) + tiny jitter
        n = min(n_samples, len(comp.faces))
        idx = np.random.choice(len(comp.faces), n, replace=False)
        sample_pts = comp.triangles_center[idx]
        # Small random jitter to avoid ray-edge/ray-vertex degeneracy
        jitter_scale = float(comp.scale) * 1e-5 if hasattr(comp, "scale") else 1e-5
        sample_pts = sample_pts + np.random.randn(*sample_pts.shape) * jitter_scale

        inside = points_inside_mesh(main_mesh, sample_pts)
        frac = float(inside.mean())

        if frac > inside_thresh:
            if verbose:
                print(f"  Component [{i}]: {frac:.0%} inside → DISCARD (interior)")
        else:
            if verbose:
                print(f"  Component [{i}]: {frac:.0%} inside → KEEP")
            kept.append(comp)

    result = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
    if verbose:
        print(f"  After component filter: {len(result.faces):,} faces "
              f"(was {len(mesh.faces):,})")
    return result


def filter_interior_faces_by_ray(
    mesh: trimesh.Trimesh,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    Per-face back-face ray test (Strategy B).
    Cast ray from face center along face normal; if first hit is a back-face
    (dot > 0), that source face is interior → remove it.
    """
    from trimesh.ray.ray_triangle import RayMeshIntersector
    intersector = RayMeshIntersector(mesh)

    face_centers = mesh.triangles_center   # (F, 3)
    face_normals = mesh.face_normals       # (F, 3)
    ray_origins = face_centers + face_normals * 1e-4

    if verbose:
        print(f"  Ray filter: casting {len(face_centers):,} rays ...")

    _tri_idx, ray_idx = intersector.intersects_id(
        ray_origins, face_normals,
        multiple_hits=False, return_locations=False,
    )
    face_idx = _tri_idx  # the hit triangle index (one per ray at most)

    if len(ray_idx) == 0:
        if verbose:
            print("  No ray hits — nothing to remove.")
        return mesh

    hit_normals = mesh.face_normals[face_idx]
    src_normals = face_normals[ray_idx]
    dot = (hit_normals * src_normals).sum(axis=1)

    bad_faces = np.zeros(len(mesh.faces), dtype=bool)
    bad_faces[ray_idx[dot > 0]] = True

    if verbose:
        n_bad = int(bad_faces.sum())
        print(f"  Ray filter: {n_bad:,} interior faces "
              f"({100 * n_bad / len(mesh.faces):.1f}%)")

    if not bad_faces.any():
        return mesh

    result = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[~bad_faces],
        process=False,
    )
    components = result.split(only_watertight=False)
    result = max(components, key=lambda c: len(c.faces))
    if verbose:
        print(f"  After ray filter: {len(result.faces):,} faces")
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Remove interior (mesh-in-mesh) components from MC PLY output"
    )
    ap.add_argument("--input",  required=True, type=Path,
                    help="Input PLY in training space (e.g. mesh.ply)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output cleaned PLY (e.g. mesh-clean.ply)")
    ap.add_argument("--use_ray_filter", action="store_true",
                    help="Also apply per-face back-face ray test (slower)")
    ap.add_argument("--n_samples",    type=int,   default=20,
                    help="Sample points per component for inside test (default 20)")
    ap.add_argument("--inside_thresh", type=float, default=0.5,
                    help="Inside fraction threshold to classify as interior (default 0.5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timing", default=None, metavar="JSON",
                    help="Timing output JSON path (optional)")
    args = ap.parse_args()

    timer = StepTimer(args.timing)

    @timer.record("step_1d_filter_interior")
    def run():
        np.random.seed(args.seed)
        input_path  = args.input.resolve()
        output_path = args.output.resolve()

        if not input_path.exists():
            raise SystemExit(f"Input not found: {input_path}")

        print(f"[1/3] Loading {input_path} ...")
        mesh = trimesh.load(str(input_path), force="mesh", process=False)
        print(f"      {len(mesh.faces):,} faces, {len(mesh.vertices):,} vertices")

        print("[2/3] Filtering interior components ...")
        mesh_clean = filter_interior_components(
            mesh,
            n_samples=args.n_samples,
            inside_thresh=args.inside_thresh,
        )

        if args.use_ray_filter:
            print("[2b/3] Per-face ray filter ...")
            mesh_clean = filter_interior_faces_by_ray(mesh_clean)

        print(f"[3/3] Saving → {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_clean.export(str(output_path))
        print(f"      Done. {len(mesh_clean.faces):,} faces saved.")

    run()


if __name__ == "__main__":
    main()
