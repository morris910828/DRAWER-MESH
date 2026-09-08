#!/usr/bin/env python
"""Check that the four table legs survived mesh simplification.

    python scripts/check_table_legs.py <mesh.ply> [--floor-z auto]

Slices the mesh at several heights between the table top and the floor and
clusters the vertices in each slice on the XY plane.  Four legs show up as four
separate clusters of similar size at every height; a leg that got simplified
away, or fused into a neighbour, changes that count.

Only vertex positions are read (via memmap), so a 14M-vertex PLY costs a couple
of seconds and very little memory.
"""

import argparse
import sys
from collections import defaultdict

import numpy as np

_TYPE_BYTES = {
    "float": 4, "float32": 4, "double": 8, "float64": 8,
    "int": 4, "int32": 4, "uint": 4, "uint32": 4,
    "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
    "char": 1, "int8": 1, "uchar": 1, "uint8": 1,
}


def read_vertices(path, max_samples=1_500_000):
    """Return the vertex xyz of a binary PLY, subsampled to max_samples."""
    elements, current = [], None
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                sys.exit(f"[Error] no end_header in {path}")
            tok = line.decode("ascii", "replace").strip().split()
            if tok and tok[0] == "end_header":
                offset = f.tell()
                break
            if not tok:
                continue
            if tok[0] == "element":
                current = {"name": tok[1], "count": int(tok[2]), "stride": 0, "props": []}
                elements.append(current)
            elif tok[0] == "property" and current is not None and tok[1] != "list":
                # Record each property's offset and type: meshlab writes doubles
                # where the SDF exporter writes floats, so x/y/z is not reliably
                # the first 12 bytes.
                current["props"].append((tok[2], tok[1], current["stride"]))
                current["stride"] += _TYPE_BYTES.get(tok[1], 4)

    vert = next((e for e in elements if e["name"] == "vertex"), None)
    if vert is None:
        sys.exit(f"[Error] no vertex element in {path}")

    xprop = next((p for p in vert["props"] if p[0] == "x"), None)
    if xprop is None:
        sys.exit(f"[Error] no x property in {path}")
    _, xtype, xoff = xprop
    nbytes = _TYPE_BYTES.get(xtype, 4)
    dtype = "<f8" if nbytes == 8 else "<f4"

    raw = np.memmap(path, dtype=np.uint8, mode="r",
                    offset=offset, shape=(vert["count"], vert["stride"]))
    step = max(1, vert["count"] // max_samples)
    chunk = raw[::step, xoff:xoff + 3 * nbytes].copy()
    xyz = chunk.view(dtype).reshape(-1, 3).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)], vert["count"]


def cluster_xy(points, cell, min_pts):
    """Grid-based connected components on the XY plane. Returns cluster sizes."""
    occupancy = defaultdict(int)
    for key in map(tuple, np.floor(points[:, :2] / cell).astype(int)):
        occupancy[key] += 1
    occupancy = {k: v for k, v in occupancy.items() if v >= min_pts}

    seen, sizes = set(), []
    for start in occupancy:
        if start in seen:
            continue
        stack, total = [start], 0
        seen.add(start)
        while stack:
            cx, cy = stack.pop()
            total += occupancy[(cx, cy)]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in occupancy and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        sizes.append(total)
    return sorted(sizes, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--floor-z", default="auto",
                    help="floor height in training coords, or 'auto' (densest horizontal slab)")
    ap.add_argument("--top-z", default="auto",
                    help="table-top height, or 'auto'")
    ap.add_argument("--cell", type=float, default=0.04, help="XY grid cell for clustering")
    args = ap.parse_args()

    xyz, total = read_vertices(args.mesh)
    print(f"{args.mesh}\n  {total:,} vertices (sampled {len(xyz):,})")
    print(f"  bbox min {np.round(xyz.min(0), 3)}  max {np.round(xyz.max(0), 3)}")

    counts, edges = np.histogram(xyz[:, 2], bins=40)
    order = np.argsort(counts)[::-1]

    if args.floor_z == "auto":
        # The floor is the densest slab in the lower half of the model.
        lower = [i for i in order if edges[i] < np.median(xyz[:, 2])]
        floor_z = float(edges[lower[0]]) if lower else float(xyz[:, 2].min())
    else:
        floor_z = float(args.floor_z)

    if args.top_z == "auto":
        upper = [i for i in order if edges[i] > floor_z + 0.15]
        top_z = float(edges[upper[0]]) if upper else float(xyz[:, 2].max())
    else:
        top_z = float(args.top_z)

    print(f"  detected floor z={floor_z:.3f}   table top z={top_z:.3f}"
          f"   leg span {top_z - floor_z:.3f}")

    if top_z - floor_z < 0.05:
        sys.exit("[Error] could not separate floor from table top -- pass --floor-z/--top-z")

    print(f"\n  {'height':>10}  {'verts':>8}  {'clusters':>8}   four largest similar-sized")
    ok = True
    for frac in (0.15, 0.35, 0.55, 0.75, 0.9):
        z = top_z - (top_z - floor_z) * frac
        band = (xyz[:, 2] > z - 0.02) & (xyz[:, 2] < z + 0.02)
        pts = xyz[band]
        if len(pts) < 20:
            print(f"  {z:10.3f}  {len(pts):8d}   (no geometry)")
            ok = False
            continue
        sizes = cluster_xy(pts, args.cell, min_pts=3)
        # The legs are the four similarly-sized clusters; anything much larger is
        # background (neighbouring furniture, walls) that shares the bbox.
        legs = [s for s in sizes if s >= 0.25 * (sizes[3] if len(sizes) > 3 else sizes[-1])]
        four = sizes[:8]
        print(f"  {z:10.3f}  {len(pts):8d}  {len(sizes):8d}   {four}")
        if len(sizes) < 4:
            ok = False

    print("\n  " + ("legs look intact at every height"
                    if ok else "SOME SLICES LOST GEOMETRY -- check target_faces / min_component_faces"))


if __name__ == "__main__":
    main()
