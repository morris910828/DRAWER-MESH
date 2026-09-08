#!/usr/bin/env python
"""Simplify a PLY mesh using quadric edge collapse + isotropic remeshing (pymeshlab).

Replicates the same simplification logic used in
sdf/nerfstudio/utils/marching_cubes.py::get_surface_sliding_with_contraction().

Usage:
    python scripts/simplify_mesh.py \\
        --input  <sdf_recon>/mesh-clean.ply \\
        --output <sdf_recon>/mesh-clean-simplify.ply \\
        [--target_faces 1000000] \\
        [--min_component_faces 10000] \\
        [--no_remesh]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pymeshlab

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.timer import StepTimer


def remesh(verts: np.ndarray, faces: np.ndarray):
    triangles = verts[faces.reshape(-1)].reshape(-1, 3, 3)
    edge_01 = triangles[:, 1] - triangles[:, 0]
    edge_12 = triangles[:, 2] - triangles[:, 1]
    edge_20 = triangles[:, 0] - triangles[:, 2]
    edge_len = (
        np.sqrt(np.sum(edge_01 ** 2, axis=1))
        + np.sqrt(np.sum(edge_12 ** 2, axis=1))
        + np.sqrt(np.sum(edge_20 ** 2, axis=1))
    )
    mean_edge_len = np.mean(edge_len / 3)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(verts, faces), "mesh")
    ms.apply_filter("meshing_isotropic_explicit_remeshing",
                    targetlen=pymeshlab.PureValue(mean_edge_len))
    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  required=True, help="Input PLY (training space)")
    parser.add_argument("--output", required=True, help="Output simplified PLY")
    parser.add_argument("--target_faces", type=int, default=1_000_000,
                        help="Target face count after quadric edge collapse (default: 1000000)")
    parser.add_argument("--min_component_faces", type=int, default=10_000,
                        help="Remove connected components with fewer faces (default: 10000)")
    parser.add_argument("--no_remesh", action="store_true",
                        help="Skip isotropic remeshing after decimation")
    parser.add_argument("--timing", default=None, metavar="JSON",
                        help="Timing output JSON path (optional)")
    args = parser.parse_args()

    timer = StepTimer(args.timing)

    @timer.record("step_1e_simplify")
    def run():
        input_path  = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Loading: {input_path}")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(input_path))
        n_before = ms.current_mesh().face_number()
        print(f"  Faces before: {n_before:,}")

        print(f"Quadric edge collapse → target {args.target_faces:,} faces …")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=args.target_faces)

        if args.min_component_faces > 0:
            print(f"Removing components < {args.min_component_faces:,} faces …")
            ms.meshing_remove_connected_component_by_face_number(
                mincomponentsize=args.min_component_faces)

        if args.no_remesh:
            ms.save_current_mesh(str(output_path), save_face_color=False)
        else:
            print("Isotropic remeshing …")
            m = ms.current_mesh()
            verts, faces = remesh(m.vertex_matrix(), m.face_matrix())
            ms2 = pymeshlab.MeshSet()
            ms2.add_mesh(pymeshlab.Mesh(verts, faces), "mesh")
            ms2.save_current_mesh(str(output_path), save_face_color=False)

        ms_check = pymeshlab.MeshSet()
        ms_check.load_new_mesh(str(output_path))
        n_after = ms_check.current_mesh().face_number()
        print(f"  Faces after:  {n_after:,}")
        print(f"Saved → {output_path}")

    run()


if __name__ == "__main__":
    main()
