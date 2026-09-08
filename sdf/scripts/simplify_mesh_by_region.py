"""
Region-aware mesh simplification using pymeshlab.
Applies different face targets to the inner SDF core region and the outer scene region.

Usage:
    python scripts/simplify_mesh_by_region.py \
        --input_mesh  <sdf_dir>/mesh_full20.ply \
        --output_mesh <sdf_dir>/mesh_full20-simplify.ply \
        --inner_box_min "-0.9 -0.9 -0.9" \
        --inner_box_max "0.9 0.9 0.9" \
        --inner_target 1000000 \
        --outer_target 200000 \
        --min_component_faces 500
"""

import argparse
import numpy as np
import pymeshlab
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_mesh",  required=True)
parser.add_argument("--output_mesh", required=True)
parser.add_argument("--inner_box_min", default="-0.9 -0.9 -0.9",
                    help="Space-separated XYZ min of inner region (world coords)")
parser.add_argument("--inner_box_max", default="0.9 0.9 0.9",
                    help="Space-separated XYZ max of inner region (world coords)")
parser.add_argument("--inner_target",  type=int, default=1_000_000,
                    help="Target face count for inner (high-detail) region")
parser.add_argument("--outer_target",  type=int, default=200_000,
                    help="Target face count for outer (scene background) region")
parser.add_argument("--min_component_faces", type=int, default=500,
                    help="Remove connected components smaller than this")
# Decimation quality params
parser.add_argument("--quality_thr",       type=float, default=0.3)
parser.add_argument("--preserve_boundary", action="store_true", default=False)
parser.add_argument("--preserve_normal",   action="store_true", default=False)
parser.add_argument("--preserve_topology", action="store_true", default=False)
parser.add_argument("--planar_quadric",    action="store_true", default=False)
args = parser.parse_args()

bmin = [float(v) for v in args.inner_box_min.split()]
bmax = [float(v) for v in args.inner_box_max.split()]

print(f"Loading: {args.input_mesh}")
ms = pymeshlab.MeshSet()
ms.load_new_mesh(args.input_mesh)
m = ms.current_mesh()
print(f"  vertices: {m.vertex_number():,}  faces: {m.face_number():,}")

# ── Step 1: remove small disconnected components ──────────────────────────────
if args.min_component_faces > 0:
    before = ms.current_mesh().face_number()
    ms.meshing_remove_connected_component_by_face_number(
        mincomponentsize=args.min_component_faces, removeunref=True
    )
    after = ms.current_mesh().face_number()
    print(f"Removed small components: {before - after:,} faces removed")

# ── Step 2: select inner region faces ─────────────────────────────────────────
# Select vertices inside the bounding box, then propagate to faces
cond = (
    f"(x >= {bmin[0]}) && (x <= {bmax[0]}) && "
    f"(y >= {bmin[1]}) && (y <= {bmax[1]}) && "
    f"(z >= {bmin[2]}) && (z <= {bmax[2]})"
)
print(f"Selecting inner region: {cond}")
ms.compute_selection_by_condition_per_vertex(condselect=cond)
ms.compute_selection_transfer_vertex_to_face()

inner_count = ms.current_mesh().selected_face_number()
total_count = ms.current_mesh().face_number()
outer_count = total_count - inner_count
print(f"  Inner faces: {inner_count:,}  Outer faces: {outer_count:,}")

# ── Step 3: simplify inner region (selected=True) ─────────────────────────────
eff_inner = min(args.inner_target, inner_count)
print(f"\nSimplifying inner region → {eff_inner:,} faces ...")
ms.meshing_decimation_quadric_edge_collapse(
    targetfacenum=eff_inner,
    qualitythr=args.quality_thr,
    preserveboundary=args.preserve_boundary,
    preservenormal=args.preserve_normal,
    preservetopology=args.preserve_topology,
    planarquadric=args.planar_quadric,
    optimalplacement=True,
    autoclean=True,
    selected=True,
)
after_inner = ms.current_mesh().face_number()
print(f"  After: {after_inner:,} total faces")

# ── Step 4: invert selection → simplify outer region ─────────────────────────
ms.apply_selection_inverse(invfaces=True, invverts=False)
outer_now = ms.current_mesh().selected_face_number()
eff_outer = min(args.outer_target, outer_now)
print(f"\nSimplifying outer region → {eff_outer:,} faces ...")
ms.meshing_decimation_quadric_edge_collapse(
    targetfacenum=eff_outer,
    qualitythr=args.quality_thr,
    preserveboundary=args.preserve_boundary,
    preservenormal=args.preserve_normal,
    preservetopology=args.preserve_topology,
    planarquadric=True,   # planar for floors/walls
    optimalplacement=True,
    autoclean=True,
    selected=True,
)

final = ms.current_mesh().face_number()
print(f"\nFinal: {final:,} faces")

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path(args.output_mesh)
out.parent.mkdir(parents=True, exist_ok=True)
ms.save_current_mesh(str(out), save_face_color=False)
size_mb = out.stat().st_size / 1e6
print(f"Saved → {out}  ({size_mb:.1f} MB)")
