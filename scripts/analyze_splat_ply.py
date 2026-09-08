#!/usr/bin/env python
"""
Measure an exported splatfacto_on_mesh_uc .ply: coverage, Gaussian shape, and on-screen
size. Reads the ply directly (numpy only) -- no torch, no GPU, no nerfstudio import -- so
it runs anywhere, including on a laptop against a scp'd export.

    python scripts/analyze_splat_ply.py data/table/table_gs9/export_ply/splat.ply \
        [--transforms data/table/transforms.json] [--frame GOPR0245]

Reports four things:

  1. AREA COVERAGE -- reproduces the model's own "Gaussian area coverage: X%" log line
     (report_coverage()): sum of pi*sx*sy*alpha per face, capped at that face's area.
     Optimistic by construction: it is an area budget and is blind to WHERE on the face
     the area sits, so a face whose Gaussians all pile up at its centre still scores 100%.

  2. CORNER COVERAGE -- the measure that matches "every triangle corner must be covered":
     the same density _compute_coverage_density() uses, alpha * exp(-d^2 / 2*sigma_min^2),
     accumulated at each face centroid and each mesh vertex. sigma_min = min(sx, sy), the
     conservative isotropic radius, so an elongated Gaussian cannot fake coverage in its
     narrow direction. Distances are plain 3D here rather than the model's face-local 2D;
     both points lie on the surface, so the two agree closely.

  3. SHAPE -- in-plane aspect ratio max(sx,sy)/min(sx,sy), split by whether a row sits at
     the base layer's scale floor. The hard cap is upper_scale / min_scale_frac, because
     both bounds are per-axis and isotropic; rows piled up exactly at that value are
     clamped, not converged.

  3b. STEINER ELLIPSE -- each face's minimum-area enclosing ellipse (what
     config.anisotropic_base_floor floors the base layer to), the mesh's own triangle
     aspect distribution, how closely the floored rows currently match it, and the
     footprint-area / corner-reach tradeoff against both isotropic floors.

  4. ON-SCREEN SIGMA -- sigma projected to pixels for one training camera, which is what
     actually decides whether text is resolvable. Needs --transforms.
"""
import argparse
import json
import os
import sys

import numpy as np

HEADER_LIMIT = 1 << 20


def read_ply(path):
    """Parse the exporter's ply: a vertex element of float properties (+ one int
    mesh_face_idx), then mesh_vertex and mesh_face elements."""
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

    text = header.decode("ascii", "replace")
    if "binary_little_endian" not in text:
        raise ValueError("only binary_little_endian ply is supported")

    elements, props = [], None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            props = []
            elements.append({"name": parts[1], "count": int(parts[2]), "props": props})
        elif parts[0] == "property" and props is not None:
            props.append(parts[1:])

    by_name = {e["name"]: e for e in elements}
    for need in ("vertex", "mesh_vertex", "mesh_face"):
        if need not in by_name:
            raise ValueError("ply has no '%s' element (not a splatfacto_on_mesh export?)" % need)

    gauss = by_name["vertex"]
    names = [p[-1] for p in gauss["props"]]
    if any(p[0] == "list" for p in gauss["props"]):
        raise ValueError("unexpected list property in the gaussian element")
    stride = 4 * len(names)  # every gaussian property is a 4-byte float or int
    n = gauss["count"]

    raw = np.memmap(path, dtype=np.uint8, mode="r")  # whole file; mesh elements follow
    table = raw[offset : offset + n * stride].reshape(n, stride)

    def col(name, dtype=np.float32):
        i = names.index(name)
        buf = np.ascontiguousarray(table[:, i * 4 : (i + 1) * 4]).tobytes()
        return np.frombuffer(buf, dtype=dtype)

    G = {
        "xyz": np.stack([col("x"), col("y"), col("z")], 1),
        "scales": np.exp(np.stack([col("scale_0"), col("scale_1"), col("scale_2")], 1)),
        "alpha": 1.0 / (1.0 + np.exp(-col("opacity"))),
        "face": col("mesh_face_idx", np.int32) if "mesh_face_idx" in names else np.full(n, -1, np.int32),
        # World-space orientation, needed to evaluate a Gaussian anisotropically rather
        # than through the isotropic sigma_min stand-in (see section 2).
        "quats": np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], 1),
    }

    pos = offset + n * stride
    nv = by_name["mesh_vertex"]["count"]
    verts = np.frombuffer(np.ascontiguousarray(raw[pos : pos + nv * 12]).tobytes(),
                          dtype=np.float32).reshape(nv, 3)
    pos += nv * 12
    nf = by_name["mesh_face"]["count"]
    # each face row is 1 uchar count + 3 int32 indices
    fb = np.frombuffer(np.ascontiguousarray(raw[pos : pos + nf * 13]).tobytes(),
                       dtype=np.uint8).reshape(nf, 13)
    faces = np.frombuffer(np.ascontiguousarray(fb[:, 1:]).tobytes(), dtype=np.int32).reshape(nf, 3)
    return G, verts, faces


def gaussian_inplane_axes(quats):
    """Each Gaussian's local x and y axes in world space, from the exporter's (w,x,y,z).

    Only the two in-plane axes are returned, and the caller projects offsets into them
    after removing the face-normal component. That is deliberate, and it is what makes
    this comparable to the model: _compute_coverage_density() works entirely in the
    face's own 2D plane and never sees a normal-direction offset at all. Measuring the
    full 3D covariance instead would divide any normal offset by the Gaussian's
    thickness, which is face_flat_coef (0.05) times its in-plane size -- so mesh
    curvature between a face's plane and its corners would dominate the result and swamp
    the anisotropy this is meant to isolate.
    """
    q = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-30)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)
    return R[:, :, 0], R[:, :, 1]  # columns = local x, y axes in world space


def inplane_maha2(d, e1, e2, fn, sx, sy):
    """Squared Mahalanobis distance of offset `d`, in the face plane.

    d: (N,3) world offset. fn: (N,3) unit face normal -- its component is dropped first,
    mirroring the model working in face-local 2D. e1/e2: Gaussian in-plane axes.
    """
    d = d - (d * fn).sum(1, keepdims=True) * fn
    u = (d * e1).sum(1)
    v = (d * e2).sum(1)
    return (u / np.maximum(sx, 1e-12)) ** 2 + (v / np.maximum(sy, 1e-12)) ** 2


def iso_maha2(d, fn, sigma_min):
    """The same offset under the model's isotropic sigma_min, also in the face plane, so
    the two numbers differ only by the anisotropy and not by which distance is used."""
    d = d - (d * fn).sum(1, keepdims=True) * fn
    return (d ** 2).sum(1) / sigma_min ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--transforms", default=None, help="transforms.json, for the pixel-size report")
    ap.add_argument("--frame", default=None, help="frame stem, e.g. GOPR0245 (default: first)")
    ap.add_argument("--coverage-target", type=float, default=1.0)
    args = ap.parse_args()

    G, verts, faces = read_ply(args.ply)
    face_based = G["face"] >= 0
    fi = G["face"][face_based].astype(np.int64)
    sx, sy = G["scales"][face_based, 0], G["scales"][face_based, 1]
    alpha = G["alpha"][face_based]
    pos = G["xyz"][face_based]

    tri = verts[faces]
    centroid = tri.mean(1)
    face_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    cv = np.linalg.norm(tri - centroid[:, None], axis=2)  # (F,3) centroid->corner

    print("%s" % os.path.basename(args.ply))
    print("  %d gaussians (%d face-based + %d vertex-layer), %d faces, %d mesh verts"
          % (len(G["alpha"]), face_based.sum(), (~face_based).sum(), len(faces), len(verts)))

    # ---- 1. area coverage (report_coverage) -------------------------------------
    contrib = np.pi * sx * sy * alpha
    covered = np.zeros(len(faces))
    np.add.at(covered, fi, contrib)
    capped = np.minimum(covered, face_area)
    ratio = covered / np.maximum(face_area, 1e-30)
    print("\n[1] AREA COVERAGE (matches the model's own log line)")
    print("    %.2f%%  (%.6f / %.6f)" % (100 * capped.sum() / face_area.sum(),
                                         capped.sum(), face_area.sum()))
    print("    redundancy (uncapped/total)   : %.2fx" % (covered.sum() / face_area.sum()))
    print("    faces with zero gaussian area : %d" % int((covered == 0).sum()))
    print("    faces under 50%% of own area   : %d (%.2f%%)"
          % (int((ratio < 0.5).sum()), 100 * (ratio < 0.5).mean()))
    print("    per-face ratio q1/q5/q25/q50  : %s"
          % "/".join("%.3f" % np.quantile(ratio, q) for q in (0.01, 0.05, 0.25, 0.50)))

    # ---- 2. corner coverage (_compute_coverage_density) --------------------------
    sigma_min = np.maximum(np.minimum(sx, sy), 1e-12)
    d2_cent = ((pos - centroid[fi]) ** 2).sum(1)
    cent_cov = np.zeros(len(faces))
    np.add.at(cent_cov, fi, alpha * np.exp(-0.5 * d2_cent / sigma_min ** 2))

    vert_cov = np.zeros(len(verts))
    for k in range(3):
        d2 = ((pos - verts[faces[fi, k]]) ** 2).sum(1)
        np.add.at(vert_cov, faces[fi, k], alpha * np.exp(-0.5 * d2 / sigma_min ** 2))

    tgt = args.coverage_target
    print("\n[2] CORNER COVERAGE (the measure that matches 'every corner covered')")
    print("    target %.2f" % tgt)
    for label, cov in (("face centroids", cent_cov), ("mesh vertices ", vert_cov)):
        print("    %s : >=target %6.2f%%   short(<0.9x) %6.2f%%   q1/q5/q25/q50 = %s"
              % (label, 100 * (cov >= tgt).mean(), 100 * (cov < 0.9 * tgt).mean(),
                 "/".join("%.3f" % np.quantile(cov, q) for q in (0.01, 0.05, 0.25, 0.50))))
    print("    Read '>=target' together with 'short': when the distribution piles up")
    print("    exactly ON the target (a single base plate contributing alpha*exp(0)=alpha")
    print("    and nothing else reaching the centroid), float noise splits it either side")
    print("    of the line and '>=target' reads low while almost nothing is really short.")
    print("    That pile-up is itself the finding: no redundancy, the plate carries it alone.")
    print("    NOTE: excludes the vertex filler layer, which adds its own opacity at each")
    print("          vertex, so real vertex coverage is somewhat higher than shown.")

    # ---- 2b. the same points, measured anisotropically ---------------------------
    # sigma_min = min(sx, sy) throws away everything the wide axis reaches. This repeats
    # section 2 with the real elliptical Gaussian, both measured in the face plane so the
    # only difference between them IS the anisotropy. The gap is what
    # config.anisotropic_coverage_density recovers, and therefore how much of the base
    # layer adaptive_coverage_floor could release once it can see the detail layer's
    # actual reach.
    fnorm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fnorm = fnorm / np.maximum(np.linalg.norm(fnorm, axis=1, keepdims=True), 1e-30)
    e1, e2 = gaussian_inplane_axes(G["quats"][face_based])
    fn = fnorm[fi]
    sig = sigma_min

    iso_c2 = np.zeros(len(faces)); ani_c2 = np.zeros(len(faces))
    d = centroid[fi] - pos
    np.add.at(iso_c2, fi, alpha * np.exp(-0.5 * iso_maha2(d, fn, sig)))
    np.add.at(ani_c2, fi, alpha * np.exp(-0.5 * inplane_maha2(d, e1, e2, fn, sx, sy)))

    iso_v2 = np.zeros(len(verts)); ani_v2 = np.zeros(len(verts))
    for k in range(3):
        vid = faces[fi, k]
        d = verts[vid] - pos
        np.add.at(iso_v2, vid, alpha * np.exp(-0.5 * iso_maha2(d, fn, sig)))
        np.add.at(ani_v2, vid, alpha * np.exp(-0.5 * inplane_maha2(d, e1, e2, fn, sx, sy)))

    print()
    print("[2b] THE SAME POINTS, ANISOTROPIC (what anisotropic_coverage_density measures)")
    print("     Both rows are face-plane distances, so they differ only in whether the")
    print("     Gaussian is treated as an ellipse or as a circle of its narrow axis.")
    for label, iso, ani in (("face centroids", iso_c2, ani_c2),
                            ("mesh vertices ", iso_v2, ani_v2)):
        print("    %s : isotropic median %.3f -> anisotropic %.3f  (%+.1f%%)"
              % (label, np.median(iso), np.median(ani),
                 100 * (np.median(ani) / max(np.median(iso), 1e-9) - 1)))
    print("    faces clearing a floor-release target (centroid AND all 3 corners):")
    for t in (0.9, 1.0, 1.2, 1.5):
        def frac(c, v):
            return 100.0 * ((c >= t) & (v[faces] >= t).all(axis=1)).mean()
        print("      target %.1f : isotropic %6.2f%%  ->  anisotropic %6.2f%%"
              % (t, frac(iso_c2, iso_v2), frac(ani_c2, ani_v2)))
    print("    (measured with floors ENGAGED, so both columns understate the release rate")
    print("     the model computes -- it re-measures with every floor dropped. The gap")
    print("     between the columns is the part the metric change unlocks.)")

    # ---- 3. shape ----------------------------------------------------------------
    aspect = np.maximum(sx, sy) / np.maximum(np.minimum(sx, sy), 1e-30)
    at_floor = np.maximum(sx, sy) / (cv.mean(1)[fi]) > 0.70
    print("\n[3] SHAPE (in-plane aspect ratio)")
    for label, m in (("at base floor", at_floor), ("free / detail", ~at_floor)):
        if not m.any():
            continue
        a = aspect[m]
        print("    %-14s n=%-9d median %.3f   circles(<1.001) %.1f%%   q90 %.2f  q99 %.2f"
              % (label, m.sum(), np.median(a), 100 * (a < 1.001).mean(),
                 np.quantile(a, 0.90), np.quantile(a, 0.99)))
    top = np.quantile(aspect, 0.99)
    print("    q99 = %.3f -- if this equals upper_scale/min_scale_frac exactly, the" % top)
    print("    aspect ratio is being CLAMPED by the isotropic bounds, not chosen.")

    # ---- 3b. Steiner-ellipse fit / headroom --------------------------------------
    # Per-face Steiner circumellipse semi-axes, computed straight from the mesh in the
    # ply (no 2D frame needed: M = (1/3) sum d_i d_i^T over the 3 centroid offsets is
    # rank 2, and its two non-zero eigenvalues give semi-axis = sqrt(2*lambda)).
    d3 = tri - centroid[:, None]                        # (F, 3, 3)
    M = np.einsum("fki,fkj->fij", d3, d3) / 3.0         # (F, 3, 3), rank 2
    ev = np.linalg.eigvalsh(M)                          # ascending, ev[:, 0] ~ 0
    semi_major = np.sqrt(np.maximum(2.0 * ev[:, 2], 0.0))
    semi_minor = np.sqrt(np.maximum(2.0 * ev[:, 1], 0.0))
    st_aspect = semi_major / np.maximum(semi_minor, 1e-30)

    hi, lo = np.maximum(sx, sy), np.minimum(sx, sy)
    print()
    print("[3b] STEINER ELLIPSE (what anisotropic_base_floor gives each face)")
    print("    mesh triangle shape : aspect median %.2f  q90 %.2f  q99 %.2f"
          % (np.median(st_aspect), np.quantile(st_aspect, 0.90), np.quantile(st_aspect, 0.99)))
    # How well the floored rows currently match their face's ellipse. Before the patch
    # this is ~1.0 vs ~st_aspect (round plates on non-round triangles); after it, the
    # two should track.
    m = at_floor
    if m.any():
        want = st_aspect[fi][m]
        got = aspect[m]
        print("    floored rows        : their aspect median %.3f vs their faces' %.3f"
              % (np.median(got), np.median(want)))
        print("                          (these two tracking each other is the patch working;")
        print("                           got ~1.0 while want > 1 is the isotropic floor)")
    # Footprint area the base layer would claim under each floor rule. This is the
    # overlap/blur budget: how far each plate spills past its own triangle.
    a_ell = np.pi * semi_major * semi_minor
    a_mean = np.pi * cv.mean(1) ** 2
    a_max = np.pi * cv.max(1) ** 2
    print("    base footprint area : ellipse %.3fx the cv_radius circle, %.3fx the"
          % (np.median(a_ell / a_mean), np.median(a_ell / a_max)))
    print("                          corner-reaching circle (median over faces)")
    # Corner reach under each rule, in sigma. The ellipse and the max circle both put
    # every vertex at exactly 1.0; the mean circle -- today's default -- does not.
    reach_mean = (cv / cv.mean(1)[:, None]).max(1)
    reach_max = (cv / np.maximum(cv.max(1)[:, None], 1e-30)).max(1)
    print("    farthest corner at  : %.3f sigma with cv_radius (%.3f of peak), %.3f with"
          % (np.median(reach_mean), np.exp(-0.5 * np.median(reach_mean) ** 2), np.median(reach_max)))
    print("                          the corner-reaching circle, 1.000 with the ellipse")

    # ---- 4. on-screen sigma ------------------------------------------------------
    if args.transforms:
        meta = json.load(open(args.transforms))
        frames = {os.path.splitext(os.path.basename(f["file_path"]))[0]: f for f in meta["frames"]}
        stem = args.frame or sorted(frames)[0]
        if stem not in frames:
            print("\n[4] frame %r not in transforms.json (have %d)" % (stem, len(frames)))
            return
        c2w = np.array(frames[stem]["transform_matrix"], float)
        R = c2w[:3, :3].copy()
        R[:, 1] *= -1
        R[:, 2] *= -1  # OpenGL -> OpenCV
        cam = (G["xyz"] - c2w[:3, 3]) @ R
        z = cam[:, 2]
        fl, W, H = meta["fl_x"], meta["w"], meta["h"]
        u = fl * cam[:, 0] / np.where(z != 0, z, 1e-9) + meta["cx"]
        v = fl * cam[:, 1] / np.where(z != 0, z, 1e-9) + meta["cy"]
        vis = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        s_all = np.maximum(G["scales"][:, 0], G["scales"][:, 1])
        px = s_all * fl / np.maximum(z, 1e-9)
        print("\n[4] ON-SCREEN SIGMA -- camera %s (%dx%d, fl=%.0f)" % (stem, W, H, fl))
        for label, m in (("face-based", vis & face_based), ("vertex layer", vis & ~face_based)):
            if not m.any():
                continue
            print("    %-12s n=%-9d depth median %.3f   sigma_px q10/50/90 = %.2f/%.2f/%.2f"
                  % (label, m.sum(), np.median(z[m]),
                     np.quantile(px[m], 0.10), np.median(px[m]), np.quantile(px[m], 0.90)))
        print("    Text strokes in these photos are roughly 2 px wide: a sigma above ~1 px")
        print("    cannot resolve them no matter how well its colour is trained.")


if __name__ == "__main__":
    sys.exit(main())
