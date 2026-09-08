#!/usr/bin/env python
"""Crop the middle table (and everything standing on it) out of the SDF mesh.

    python scripts/crop_table.py --input mesh_table.ply --output mesh_table-only.ply

The default box was measured from the reconstructed mesh itself, in training
coordinates:

  * long axis  a in [-0.67, +0.53]  -- the tabletop reaches a = +0.489 at its
    rounded end and the neighbouring table starts at +0.508; a = -0.673 is a
    clean gap (7 vertices, vs ~500 either side). Span 1.16 units = 1.24 m, the
    real length of the table.
  * short axis b in [-0.42, +0.45]  -- the tabletop runs -0.378..+0.398 and the
    histogram is empty either side out to +-0.55.

The box is deliberately a little looser than the table: what it over-collects is
disconnected from it and the component filter below removes it, whereas a box
cut too tight shaves the rounded corner off and no later step can put it back.
  * z > -0.625 -- the floor slab sits at z = -0.655, so this keeps the legs to
    within ~3 cm of the ground without dragging the floor along.

Nothing bounds z from above: the screen, the ONYX magnifier with its arm (which
reaches ~55 cm above the table top) and the standing signs all come with it.

Faces are kept when all three of their vertices survive, and vertex indices are
renumbered, so the result is a valid standalone PLY.
"""

import argparse
import sys

import numpy as np

_TYPE_BYTES = {
    "float": 4, "float32": 4, "double": 8, "float64": 8,
    "int": 4, "int32": 4, "uint": 4, "uint32": 4,
    "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
    "char": 1, "int8": 1, "uchar": 1, "uint8": 1,
}

# Oriented box of the table, in training coordinates.
CENTRE = np.array([0.0442, -0.2175, -0.1828])
AXIS_A = np.array([0.7318, -0.6814, -0.0120])   # along the table's length
AXIS_B = np.array([0.6815, 0.7317, 0.0138])     # across its width


def read_ply(path):
    """Parse a binary-little-endian PLY into (vertices, faces, header info)."""
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
                current = {"name": tok[1], "count": int(tok[2]),
                           "stride": 0, "list": None, "props": []}
                elements.append(current)
            elif tok[0] == "property" and current is not None:
                if tok[1] == "list":
                    current["list"] = (_TYPE_BYTES.get(tok[2], 1),
                                       _TYPE_BYTES.get(tok[3], 4))
                else:
                    current["props"].append((tok[2], tok[1], current["stride"]))
                    current["stride"] += _TYPE_BYTES.get(tok[1], 4)

    vert = next(e for e in elements if e["name"] == "vertex")
    face = next((e for e in elements if e["name"] == "face"), None)
    if face is None:
        sys.exit(f"[Error] no face element in {path}")

    xprop = next(p for p in vert["props"] if p[0] == "x")
    _, xtype, xoff = xprop
    nb = _TYPE_BYTES.get(xtype, 4)
    xyz_dtype = "<f8" if nb == 8 else "<f4"

    vraw = np.memmap(path, dtype=np.uint8, mode="r",
                     offset=offset, shape=(vert["count"], vert["stride"]))
    xyz = vraw[:, xoff:xoff + 3 * nb].copy().view(xyz_dtype).reshape(-1, 3).astype(np.float64)

    cbytes, ibytes = face["list"]
    frec = cbytes + 3 * ibytes
    foff = offset + vert["count"] * vert["stride"]
    fraw = np.memmap(path, dtype=np.uint8, mode="r",
                     offset=foff, shape=(face["count"], frec))
    counts = fraw[:, 0]
    if not np.all(counts == 3):
        sys.exit("[Error] mesh is not purely triangular")
    tris = fraw[:, cbytes:cbytes + 3 * ibytes].copy().view("<i4").reshape(-1, 3)
    return xyz, tris


def write_ply(path, xyz, tris):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(tris)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(xyz.astype("<f4").tobytes())
        rec = np.empty((len(tris), 13), dtype=np.uint8)
        rec[:, 0] = 3
        rec[:, 1:] = tris.astype("<i4").view(np.uint8).reshape(-1, 12)
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--a-min", type=float, default=-0.67)
    ap.add_argument("--a-max", type=float, default=0.53)
    ap.add_argument("--b-min", type=float, default=-0.42)
    ap.add_argument("--b-max", type=float, default=0.45)
    ap.add_argument("--z-min", type=float, default=-0.625)
    ap.add_argument("--keep-components", type=int, default=1,
                    help="keep only the N largest connected components (0 = keep all). "
                         "The table and everything resting on it form one component "
                         "holding 99.4%% of the mesh; the rest is floating SDF noise "
                         "and slivers of neighbouring furniture clipped by the box.")
    args = ap.parse_args()

    xyz, tris = read_ply(args.input)
    print(f"in : {len(xyz):,} vertices, {len(tris):,} faces")

    rel = xyz - CENTRE
    a = rel @ AXIS_A
    b = rel @ AXIS_B
    keep = ((a >= args.a_min) & (a <= args.a_max)
            & (b >= args.b_min) & (b <= args.b_max)
            & (xyz[:, 2] >= args.z_min))
    print(f"     {keep.sum():,} vertices inside the box ({100 * keep.mean():.1f}%)")

    # Keep a face only when the whole triangle survives, then renumber.
    face_keep = keep[tris].all(axis=1)
    remap = np.full(len(xyz), -1, dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    out_tris = remap[tris[face_keep]].astype(np.int32)
    out_xyz = xyz[keep]

    # Vertices orphaned by face removal would confuse downstream tools.
    used = np.zeros(len(out_xyz), dtype=bool)
    used[out_tris.ravel()] = True
    if not used.all():
        remap2 = np.full(len(out_xyz), -1, dtype=np.int64)
        remap2[used] = np.arange(used.sum())
        out_tris = remap2[out_tris].astype(np.int32)
        out_xyz = out_xyz[used]
        print(f"     dropped {(~used).sum():,} vertices left with no face")

    # A box cut leaves whatever else happened to fall inside it: floaters the SDF
    # invented in mid-air, and slivers of the neighbouring table sheared off by
    # the box faces. They are all disconnected from the table, so keeping the
    # largest component(s) removes them without touching the object itself.
    if args.keep_components > 0:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        edges = np.concatenate([out_tris[:, [0, 1]], out_tris[:, [1, 2]], out_tris[:, [2, 0]]])
        graph = coo_matrix(
            (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
            shape=(len(out_xyz), len(out_xyz)),
        )
        ncomp, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels)
        biggest = np.argsort(sizes)[::-1][:args.keep_components]
        comp_keep = np.isin(labels, biggest)
        if not comp_keep.all():
            print(f"     {ncomp} components; keeping {args.keep_components} largest "
                  f"({100 * comp_keep.mean():.2f}% of vertices), dropping "
                  f"{(~comp_keep).sum():,} vertices of debris")
            face_ok = comp_keep[out_tris].all(axis=1)
            remap3 = np.full(len(out_xyz), -1, dtype=np.int64)
            remap3[comp_keep] = np.arange(comp_keep.sum())
            out_tris = remap3[out_tris[face_ok]].astype(np.int32)
            out_xyz = out_xyz[comp_keep]

    print(f"out: {len(out_xyz):,} vertices, {len(out_tris):,} faces")
    print(f"     bbox min {np.round(out_xyz.min(0), 3)}  max {np.round(out_xyz.max(0), 3)}")
    write_ply(args.output, out_xyz, out_tris)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
