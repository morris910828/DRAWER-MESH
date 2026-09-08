#!/usr/bin/env python
"""
Measure GAP coverage on an exported splatfacto_on_mesh_uc .ply: what fraction of the mesh
surface has no Gaussian actually reaching it.

    python scripts/analyze_gap_coverage.py data/table/table_gs13/export_ply/splat.ply
    python scripts/analyze_gap_coverage.py <ply> --samples 16 --report-faces gaps.txt

WHY THIS EXISTS, AND HOW IT DIFFERS FROM analyze_splat_ply.py's [1]

report_coverage() in the model -- and [1] in analyze_splat_ply.py, which reproduces it --
computes an AREA BUDGET: sum of pi*sx*sy*alpha over a face's Gaussians, capped at that
face's own triangle area. Two properties make it the wrong number for "is every part of
the surface covered":

  1. It is blind to PLACEMENT. A face whose Gaussians all pile up at its centroid scores
     the same as one whose Gaussians are spread evenly. On table_gs11 the median face
     carries 2.418x its own area in budget while 9.03% of faces sit under half of it --
     surplus and starvation side by side, and the number cannot tell where either is.
  2. It rewards SIZE x OPACITY, which is the definition of a blurry Gaussian. Pushing it
     up pushes the model toward exactly the large opaque plates that cannot represent a
     2 px text stroke. Across gs10 -> gs13 the metric and on-screen sharpness moved in
     opposite directions every single time.

This tool measures the union instead: sample points across every triangle and ask, for
each one, whether ANY Gaussian actually reaches it. That is the "no gaps" question, and
it is satisfiable by many small Gaussians just as well as by one large one -- so unlike
the area budget, it does not have a preferred answer that happens to be blurry.

WHAT COUNTS AS COVERED

A Gaussian's contribution at a point is alpha * exp(-0.5 * M^2), where M is the in-plane
Mahalanobis distance from its centre to that point. In-plane only, using the Gaussian's
own local x/y axes: the same simplification _compute_coverage_density() makes, and for the
same reason -- these Gaussians are flattened onto the surface (face_flat_coef 0.05) and
both the sample and the Gaussian lie on it, so the out-of-plane term carries no
information and its tiny sigma would make the Mahalanobis distance numerically explosive.

A point is covered at threshold t if the STRONGEST single Gaussian reaching it is >= t.
Max, not sum: "covered by at least one Gaussian" is the stated requirement, and summing
would let several far-away tails add up to a phantom pass. Results are reported at three
thresholds because the honest answer depends on what you mean by reaching:

    0.5   solidly covered -- inside roughly the 1-sigma ellipse of an opaque Gaussian
    0.1   visibly covered
    0.01  the rasterizer still writes something here (its own cutoff is near 1/255)

WHICH GAUSSIANS ARE TESTED AGAINST A FACE

Its own (mesh_face_idx == that face), those of its edge-adjacent faces, and the three
vertex-layer Gaussians anchored at its corners. Gaussians two or more faces away are
ignored, which makes every number here CONSERVATIVE: real coverage is at least this good,
never worse. On this mesh a Gaussian would have to be several times its face's size to
reach that far, so the omission is small -- but it is an omission, not an approximation
that averages out.

Sample points are a barycentric grid, area-weighted, so a large triangle counts for more
of the surface than a small one -- the reported percentage is of mesh AREA, not of faces.
"""
import argparse
import os
import sys

import numpy as np

HEADER_LIMIT = 1 << 20


def read_ply(path):
    """Same layout analyze_splat_ply.py reads: a vertex element of float properties (plus
    an int mesh_face_idx), then mesh_vertex and mesh_face elements."""
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            chunk = f.readline()
            if not chunk:
                raise ValueError("no end_header found -- is this the exporter's ply?")
            header += chunk
            if len(header) > HEADER_LIMIT:
                raise ValueError("header too large")
        offset = len(header)

        els = []
        for line in header.decode("ascii", "replace").splitlines():
            if line.startswith("element"):
                els.append([line.split()[1], int(line.split()[2]), []])
            elif line.startswith("property") and els:
                els[-1][2].append(line)
        if len(els) < 3:
            raise ValueError("expected vertex + mesh_vertex + mesh_face elements")

        n_g, n_mv, n_mf = els[0][1], els[1][1], els[2][1]
        names = [l.split()[-1] for l in els[0][2] if "list" not in l]
        dt = np.dtype([(nm, "<i4" if nm == "mesh_face_idx" else "<f4") for nm in names])
        G = np.memmap(path, dtype=dt, mode="r", offset=offset, shape=(n_g,))

        off2 = offset + dt.itemsize * n_g
        MV = np.memmap(
            path, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]),
            mode="r", offset=off2, shape=(n_mv,),
        )
        f.seek(off2 + 12 * n_mv)
        raw = np.frombuffer(f.read(n_mf * 13), dtype=np.uint8).reshape(n_mf, 13)
        if not (raw[:, 0] == 3).all():
            raise ValueError("non-triangle face in ply")
        F = raw[:, 1:].copy().view("<i4").reshape(n_mf, 3).astype(np.int64)
        V = np.stack([MV["x"], MV["y"], MV["z"]], 1).astype(np.float64)
    return G, V, F


def quat_axes(q):
    """Local x and y axes of each Gaussian, from its (w, x, y, z) quaternion.

    Only two of the three axes are built: the third is the flattened normal direction,
    which this tool deliberately does not use (see the module docstring)."""
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ax = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1)
    ay = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], 1)
    return ax, ay


def bary_grid(k):
    """k*(k+1)/2 barycentric sample points, one per cell of a k-row subdivision, taken at
    each cell's own centroid so no sample lands exactly on an edge or vertex (where two
    faces would both claim it)."""
    pts = []
    for i in range(k):
        for j in range(k - i):
            a = (i + 1.0 / 3.0) / k
            b = (j + 1.0 / 3.0) / k
            pts.append((a, b, 1.0 - a - b))
            if i + j < k - 1:
                a2 = (i + 2.0 / 3.0) / k
                b2 = (j + 2.0 / 3.0) / k
                pts.append((a2, b2, 1.0 - a2 - b2))
    return np.array(pts, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--samples", type=int, default=4,
                    help="barycentric subdivision level k; yields k^2 points per face (default 4 -> 16)")
    ap.add_argument("--report-faces", default=None,
                    help="write the worst faces (index, gap fraction) to this file")
    ap.add_argument("--no-vertex-layer", action="store_true",
                    help="exclude the vertex filler layer, to measure what it is actually holding up")
    ap.add_argument("--chunk", type=int, default=20000, help="faces per batch")
    args = ap.parse_args()

    G, V, F = read_ply(args.ply)
    n_mf = F.shape[0]
    fidx = np.asarray(G["mesh_face_idx"])
    n_v = int((fidx < 0).sum())
    print(os.path.basename(args.ply))
    print("  %d gaussians (%d face-based + %d vertex-layer), %d faces, %d mesh verts"
          % (len(fidx), len(fidx) - n_v, n_v, n_mf, V.shape[0]))

    g_xyz = np.stack([G["x"], G["y"], G["z"]], 1).astype(np.float64)
    g_sx = np.exp(G["scale_0"].astype(np.float64))
    g_sy = np.exp(G["scale_1"].astype(np.float64))
    g_al = 1.0 / (1.0 + np.exp(-np.clip(G["opacity"].astype(np.float64), -60, 60)))
    g_ax, g_ay = quat_axes(np.stack([G["rot_0"], G["rot_1"], G["rot_2"], G["rot_3"]], 1).astype(np.float64))

    # --- per-face Gaussian lists, as a CSR-style (order, start) pair -------------------
    fb = np.where(fidx >= 0)[0]
    order = fb[np.argsort(fidx[fb], kind="stable")]
    start = np.searchsorted(fidx[order], np.arange(n_mf + 1))
    per_face = np.diff(start)
    print("  face-based gaussians per face: median %.1f  q10 %.1f  q90 %.1f  (%d faces with none)"
          % (np.median(per_face), np.percentile(per_face, 10), np.percentile(per_face, 90),
             int((per_face == 0).sum())))

    # --- vertex-layer gaussian id for each mesh vertex --------------------------------
    # The exporter appends this layer last, one row per mesh vertex, in vertex order.
    vg = np.arange(len(fidx) - n_v, len(fidx)) if n_v else None
    if args.no_vertex_layer and vg is not None:
        print("  EXCLUDING the vertex filler layer (%d rows) from the test" % n_v)
        vg = None
    if n_v and n_v != V.shape[0]:
        print("  note: vertex-layer count != mesh vertex count; skipping that layer")
        vg = None

    # --- edge-adjacent faces ----------------------------------------------------------
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    e = np.sort(e, axis=1)
    key = e[:, 0] * (V.shape[0] + 1) + e[:, 1]
    o = np.argsort(key, kind="stable")
    ks, fs = key[o], np.tile(np.arange(n_mf), 3)[o]
    same = np.where(ks[1:] == ks[:-1])[0]
    nbr = np.full((n_mf, 3), -1, np.int64)
    fill = np.zeros(n_mf, np.int64)
    for a, b in ((fs[same], fs[same + 1]), (fs[same + 1], fs[same])):
        for i in range(len(a)):
            if fill[a[i]] < 3:
                nbr[a[i], fill[a[i]]] = b[i]
                fill[a[i]] += 1
    print("  edge-adjacent faces found for %.1f%% of faces (3 each on a closed manifold)"
          % (100.0 * (nbr >= 0).sum() / (3.0 * n_mf)))

    bary = bary_grid(args.samples)
    K = len(bary)
    thresholds = (0.5, 0.1, 0.01)
    best_sum = np.zeros(len(thresholds))
    gap_area = np.zeros(len(thresholds))
    tot_area = 0.0
    worst = []

    fv_all = V[F]
    area_all = 0.5 * np.linalg.norm(
        np.cross(fv_all[:, 1] - fv_all[:, 0], fv_all[:, 2] - fv_all[:, 0]), axis=1)

    # Faces are visited in order of how many Gaussians they own, so each batch is
    # homogeneous. Batching them in face order instead makes every face in a batch pay for
    # the single busiest one -- the slice loop below runs to the batch maximum -- which on
    # this mesh is a large constant factor for nothing.
    face_order = np.argsort(per_face, kind="stable")
    for lo in range(0, n_mf, args.chunk):
        hi = min(lo + args.chunk, n_mf)
        idx = face_order[lo:hi]
        fv = fv_all[idx]                                     # (B,3,3)
        pts = np.einsum("kb,fbc->fkc", bary, fv)             # (B,K,3)
        best = np.zeros((hi - lo, K))

        def apply(gid, mask):
            """gid: (B,S) gaussian ids, -1 where absent. Update `best` in place."""
            if gid.size == 0:
                return
            valid = mask & (gid >= 0)
            if not valid.any():
                return
            gi = np.where(valid, gid, 0)
            d = pts[:, None, :, :] - g_xyz[gi][:, :, None, :]          # (B,S,K,3)
            u = np.einsum("bskc,bsc->bsk", d, g_ax[gi])
            w = np.einsum("bskc,bsc->bsk", d, g_ay[gi])
            m2 = (u / np.maximum(g_sx[gi], 1e-30)[:, :, None]) ** 2 \
               + (w / np.maximum(g_sy[gi], 1e-30)[:, :, None]) ** 2
            contrib = g_al[gi][:, :, None] * np.exp(-0.5 * np.minimum(m2, 200.0))
            contrib = np.where(valid[:, :, None], contrib, 0.0)
            np.maximum(best, contrib.max(axis=1), out=best)

        # own face, in slices so the (B,S,K,3) temporary stays bounded. Gathered with
        # arithmetic on the CSR (order, start) pair rather than a per-face Python loop --
        # at 1.4M faces the loop version dominates the runtime completely.
        n_ord = len(order)
        mx = int(per_face[idx].max()) if len(idx) else 0
        SL = 16
        for s0 in range(0, mx, SL):
            s1 = min(s0 + SL, mx)
            s = np.arange(s0, s1)[None, :]                       # (1,S)
            valid = s < per_face[idx][:, None]                   # (B,S)
            pos = np.minimum(start[idx][:, None] + s, n_ord - 1)
            gid = np.where(valid, order[pos], -1)
            apply(gid, valid)

        # edge-adjacent faces: their nearest few Gaussians are the ones that can reach in
        s = np.arange(SL)[None, :]
        for nb in range(3):
            nf = nbr[idx, nb]
            ok = nf >= 0
            nf_s = np.where(ok, nf, 0)
            valid = ok[:, None] & (s < per_face[nf_s][:, None])
            pos = np.minimum(start[nf_s][:, None] + s, n_ord - 1)
            gid = np.where(valid, order[pos], -1)
            apply(gid, valid)

        # the three vertex-layer gaussians at this face's corners
        if vg is not None:
            apply(vg[F[idx]], np.ones((hi - lo, 3), bool))

        a = area_all[idx]
        tot_area += a.sum()
        for t, th in enumerate(thresholds):
            frac_gap = (best < th).mean(axis=1)
            gap_area[t] += (frac_gap * a).sum()
        if args.report_faces:
            fg = (best < thresholds[1]).mean(axis=1)
            bad = np.where(fg > 0)[0]
            if len(bad):
                worst.append(np.stack([idx[bad], fg[bad]], 1))
        print("\r  scanning faces %d/%d" % (hi, n_mf), end="", file=sys.stderr)
    print("", file=sys.stderr)

    print()
    print("GAP COVERAGE -- fraction of mesh AREA with no Gaussian reaching it")
    print("  (conservative: only own-face, edge-adjacent and corner-vertex Gaussians tested)")
    print()
    print("    %-34s %12s %12s" % ("threshold", "covered", "GAP"))
    for t, th in enumerate(thresholds):
        cov = 1.0 - gap_area[t] / tot_area
        label = {0.5: "0.50  solidly covered",
                 0.1: "0.10  visibly covered",
                 0.01: "0.01  rasterizer writes something"}[th]
        print("    %-34s %11.3f%% %11.3f%%" % (label, cov * 100, (1 - cov) * 100))
    print()
    print("  %d sample points per face, area-weighted; total mesh area %.3f" % (K, tot_area))

    if args.report_faces and worst:
        w = np.concatenate(worst)
        w = w[np.argsort(-w[:, 1])]
        with open(args.report_faces, "w") as fh:
            fh.write("# face_index\tgap_fraction (at threshold 0.1)\n")
            for fi, g in w[:200000]:
                fh.write("%d\t%.4f\n" % (int(fi), g))
        print("  wrote %d faces with a gap to %s" % (len(w), args.report_faces))


if __name__ == "__main__":
    sys.exit(main())
