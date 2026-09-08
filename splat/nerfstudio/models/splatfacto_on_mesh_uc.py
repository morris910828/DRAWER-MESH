# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import trimesh
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
import cv2

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from gsplat.cuda_legacy._wrapper import num_sh_bases

from pytorch_msssim import SSIM
from torch.nn import Parameter
# from typing_extensions import Literal
from typing import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.misc import torch_compile

from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_quaternion,
    quaternion_invert,
    quaternion_multiply,
    euler_angles_to_matrix,
)
from pytorch3d.io import load_objs_as_meshes, save_obj
from collections import namedtuple
from pytorch3d.renderer import TexturesUV, TexturesVertex, RasterizationSettings, MeshRenderer, MeshRendererWithFragments, MeshRasterizer, HardPhongShader, PointLights
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as F_V

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def area(triangles):
    # Extract the vertices of the triangles
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Compute the lengths of the sides of the triangles
    a = torch.norm(B - C, dim=1)
    b = torch.norm(C - A, dim=1)
    c = torch.norm(A - B, dim=1)

    # Compute the semi-perimeter of each triangle
    s = (a + b + c) / 2

    # Compute the area of each triangle using Heron's formula
    area = torch.sqrt(s * (s - a) * (s - b) * (s - c))

    return area

def circumcircle_radius(triangles):
    # Extract the vertices of the triangles
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Compute the lengths of the sides of the triangles
    a = torch.norm(B - C, dim=1)
    b = torch.norm(C - A, dim=1)
    c = torch.norm(A - B, dim=1)

    # Compute the semi-perimeter of each triangle
    s = (a + b + c) / 2

    # Compute the area of each triangle using Heron's formula
    area = torch.sqrt(s * (s - a) * (s - b) * (s - c))

    # Compute the circumcircle radius
    R = (a * b * c) / (4 * area)

    return R


def face_ring_radius(mesh_faces, mesh_faces_verts):
    """
    Local scale estimate per face based on its 1-ring (edge-adjacent) neighbor faces,
    instead of the face's own circumcircle radius.

    A single triangle's circumradius is unstable: it blows up as the triangle
    degenerates (area -> 0 => circumradius -> inf), and it only looks at that one
    triangle, ignoring how big its neighbors actually are. Two adjacent faces of very
    different circumradius therefore produce a sharp jump in the Gaussian scale bound
    right at their shared edge, which shows up as uneven coverage/gaps along that
    edge. Averaging the distance from a face's own center to each of its edge-adjacent
    neighbors' centers gives a size estimate that varies smoothly across the mesh and
    stays well-behaved even when the face itself is a degenerate sliver, as long as
    its neighbors aren't.

    Args:
        mesh_faces: (F, 3) long tensor of vertex indices per face.
        mesh_faces_verts: (F, 3, 3) vertex positions per face.

    Returns:
        (F,) tensor, one radius per face.
    """
    F = mesh_faces.shape[0]
    device = mesh_faces.device

    edges = torch.stack([
        mesh_faces[:, [0, 1]],
        mesh_faces[:, [1, 2]],
        mesh_faces[:, [2, 0]],
    ], dim=1).reshape(-1, 2)  # (3F, 2)
    edges_sorted, _ = torch.sort(edges, dim=-1)
    face_ids = torch.arange(F, device=device).repeat_interleave(3)  # (3F,)

    _, edge_inverse = torch.unique(edges_sorted, dim=0, return_inverse=True)

    order = torch.argsort(edge_inverse)
    sorted_face_ids = face_ids[order]
    sorted_edge_ids = edge_inverse[order]

    # A manifold edge is shared by exactly 2 faces, so after sorting by edge id the
    # two faces sharing an edge land in consecutive positions.
    same_next = sorted_edge_ids[:-1] == sorted_edge_ids[1:]
    idx = torch.where(same_next)[0]
    neighbor_a = sorted_face_ids[idx]
    neighbor_b = sorted_face_ids[idx + 1]

    pair_self = torch.cat([neighbor_a, neighbor_b], dim=0)
    pair_other = torch.cat([neighbor_b, neighbor_a], dim=0)

    face_centers = mesh_faces_verts.mean(dim=1)  # (F, 3)
    dists = torch.norm(face_centers[pair_self] - face_centers[pair_other], dim=-1)

    ring_sum = torch.zeros(F, device=device, dtype=dists.dtype).scatter_add_(0, pair_self, dists)
    ring_cnt = torch.zeros(F, device=device, dtype=dists.dtype).scatter_add_(0, pair_self, torch.ones_like(dists))

    has_neighbors = ring_cnt > 0
    ring_radius = torch.where(
        has_neighbors,
        ring_sum / ring_cnt.clamp(min=1),
        circumcircle_radius(mesh_faces_verts),  # fallback: face has no edge-adjacent neighbor
    )
    return ring_radius


def face_adjacency_pairs(mesh_faces, share="vertex"):
    """
    Adjacent face pairs, as two parallel (E,) index tensors, canonicalized to a<b and
    deduped so each unordered pair appears exactly once (a symmetric penalty over these
    already covers both directions; emitting both would just double the weight).

    share="vertex" (default) -- faces sharing at least one VERTEX. This is what
    color_consistency_lambda wants, and the default changed from "edge" to "vertex" on
    2026-08-06 after measuring the exported armadillo. Classifying every spatially
    overlapping Gaussian pair by mesh topology showed the edge-only version had a large
    blind spot:

        same face      52.2% of overlapping pairs, median |dC| 0.0011  [was covered]
        edge-adjacent  42.2%                       median |dC| 0.0019  [was covered]
        vertex-only     5.6%                       median |dC| 0.0056  [NOT covered]
        truly distant   0.0%                       median |dC| 0.0059  [NOT covered]

    i.e. the loss flattened everything it could reach to ~0.001-0.002 and left the
    corner-neighbor population sitting 5.3x higher (6.2x at p90), carrying 17.2% of the
    total flicker potential. A triangle has 3 edge neighbors but ~12 vertex neighbors,
    and base Gaussians alone have a footprint ~2.6-2.8x their own triangle's area, so
    those corner neighbors overlap just as physically as the edge ones do -- they were
    simply invisible to an edge-hash adjacency. Note the "truly distant" (mesh fold)
    class turned out to be negligible in bulk here, which is why this stays a cheap
    topological query instead of a spatial search.

    share="edge" -- the original behavior (faces sharing an edge), kept because it is
    the textbook 1-ring and callers may want the tighter relation.

    Assumes a manifold mesh, same assumption face_ring_radius() already documents. A
    boundary edge (only 1 face) contributes no edge pair; a non-manifold edge (3+ faces)
    under share="edge" contributes only the consecutive pairs that fall out of the sort,
    a silent under-count rather than a crash. share="vertex" has no such limitation --
    it enumerates the incident-face table directly.

    Args:
        mesh_faces: (F, 3) long tensor of vertex indices per face.
        share: "vertex" or "edge" -- what two faces must have in common to be a pair.

    Returns:
        (face_a, face_b), each (E,) long tensors, with face_a < face_b elementwise.
    """
    F = mesh_faces.shape[0]
    device = mesh_faces.device

    if share == "edge":
        edges = torch.stack([
            mesh_faces[:, [0, 1]],
            mesh_faces[:, [1, 2]],
            mesh_faces[:, [2, 0]],
        ], dim=1).reshape(-1, 2)  # (3F, 2)
        edges_sorted, _ = torch.sort(edges, dim=-1)
        face_ids = torch.arange(F, device=device).repeat_interleave(3)  # (3F,)

        _, edge_inverse = torch.unique(edges_sorted, dim=0, return_inverse=True)
        order = torch.argsort(edge_inverse)
        sorted_face_ids = face_ids[order]
        sorted_edge_ids = edge_inverse[order]

        same_next = sorted_edge_ids[:-1] == sorted_edge_ids[1:]
        idx = torch.where(same_next)[0]
        a, b = sorted_face_ids[idx], sorted_face_ids[idx + 1]
    elif share == "vertex":
        # Vertex -> incident faces, as a dense (V, K) table. K is the max valence, ~6-12
        # on a normal mesh, so this stays small; -1 marks unused slots.
        vids = mesh_faces.reshape(-1)  # (3F,)
        fids = torch.arange(F, device=device).repeat_interleave(3)  # (3F,)
        order = torch.argsort(vids)
        v_s, f_s = vids[order], fids[order]
        _, counts = torch.unique_consecutive(v_s, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(v_s.shape[0], device=device) - torch.repeat_interleave(starts, counts)
        V = int(vids.max().item()) + 1
        table = torch.full((V, int(counts.max().item())), -1, dtype=torch.long, device=device)
        table[v_s, slot] = f_s

        # For each face, every face incident to any of its 3 vertices.
        nb = table[mesh_faces].reshape(F, -1)  # (F, 3K)
        self_id = torch.arange(F, device=device).unsqueeze(1).expand_as(nb)
        valid = (nb >= 0) & (nb != self_id)
        a, b = self_id[valid], nb[valid]
    else:
        raise ValueError(f"share must be 'vertex' or 'edge', got {share!r}")

    pairs = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], dim=-1)
    pairs = torch.unique(pairs, dim=0) if pairs.numel() > 0 else pairs.reshape(0, 2)
    return pairs[:, 0].contiguous(), pairs[:, 1].contiguous()


def face_centroid_vertex_radius(mesh_faces_verts):
    """
    Average distance from each face's own centroid to its 3 vertices -- i.e. how far
    a Gaussian centered at the centroid (means_2d starts there) actually has to reach
    to cover this face's own corners.

    Briefly changed to dists.max(dim=-1) (2026-07-24) to guarantee reaching the single
    farthest corner on irregular/obtuse triangles -- reverted the same day: the base
    layer's in-plane floor (self.cv_radius, see the scales property) keys off this
    value, and the larger max-based radius pushed every base Gaussian further into its
    neighbors' territory, re-introducing the base-vs-base overlap (and resulting
    blur/softness) that using cv_radius instead of the full 1-ring xyz_radius was
    originally meant to reduce (see the severe flicker reported 2026-07-18, same root
    cause). Coverage-vs-overlap is a real tradeoff here, not a bug either way; mean is
    the reverted-to default until a better middle ground is found -- some
    irregular-triangle corners may stay under-covered as a result.

    Deliberately NOT circumcircle_radius here: circumradius is the radius of the
    circle through all 3 vertices, and for a thin/obtuse triangle the circumcenter
    can sit far outside the triangle, making circumradius blow up well past the
    triangle's actual size (the same instability face_ring_radius() was written to
    avoid in the first place). Complex/high-curvature areas of a mesh tend to have
    more irregularly-shaped triangles, so using circumradius as a size floor there
    would inflate the bound most exactly where a tighter, more curvature-conforming
    Gaussian is wanted -- Gaussians would visibly poke out past the true surface.
    Centroid-to-vertex distance stays proportional to the triangle's actual extent
    for any triangle shape, so it can't do that.

    Args:
        mesh_faces_verts: (F, 3, 3) vertex positions per face.

    Returns:
        (F,) tensor, one radius per face.
    """
    centroid = mesh_faces_verts.mean(dim=1, keepdim=True)  # (F, 1, 3)
    dists = torch.norm(mesh_faces_verts - centroid, dim=-1)  # (F, 3)
    return dists.mean(dim=-1)  # (F,)


def steiner_ellipse_axes(a2d, b2d, c2d, max_aspect=8.0):
    """
    Per-face Steiner circumellipse: the unique ellipse centered on a triangle's centroid
    that passes through all three of its vertices, which is also the minimum-area ellipse
    enclosing that triangle.

    This is the "better middle ground" face_centroid_vertex_radius()'s docstring asks for.
    That function has to pick between the MEAN centroid-to-vertex distance (tight, but
    leaves the farthest corner of an irregular triangle under-covered) and the MAX (reaches
    every corner, but inflates every base plate deep into its neighbours' territory -- the
    2026-07-18 flicker). The choice is only forced because a circle has a single degree of
    freedom: to reach the farthest corner it must overshoot equally in every other
    direction. An ellipse aligned to the triangle's own principal axes has no such
    conflict, so it does not have to trade one failure for the other.

    The property that makes this work is that the Steiner circumellipse's area is
    (4*pi)/(3*sqrt(3)) ~= 2.418 times the triangle's area for EVERY triangle, whatever its
    shape, while a circumscribed circle's area-to-triangle-area ratio diverges as the
    triangle gets slivery. So this reaches all three corners (strictly better coverage than
    the mean-based radius) while still covering only a small constant multiple of each
    face's own area (far less overlap than the max-based radius). On an equilateral face
    the ellipse degenerates to the circumcircle and nothing changes; the entire gain is on
    irregular faces, which is exactly where coverage was short.

    Derivation: with d_i the three vertex offsets from the centroid, M = (1/3) sum_i
    d_i d_i^T satisfies x^T M^-1 x = 2 at every d_i, so the ellipse {x : x^T M^-1 x <= 2}
    passes through all three vertices; its semi-axes are sqrt(2 * eigenvalue) of M and its
    major axis is M's leading eigenvector. (Check: an equilateral triangle with circumradius
    R gives M = (R^2 / 2) I, hence semi-axes R and R -- the circumcircle, as expected.)

    Args:
        a2d, b2d, c2d: (F, 2) triangle vertices in each face's own 2D frame -- the frame
            mesh_triangles_2d_coords_* live in, whose axes are faces_axis_x / faces_axis_y.
            Those are the same two axes a Gaussian's local x/y scales are measured along
            (see faces_quats in populate_modules), which is what lets the returned semi-axes
            be used directly as an (x, y) scale floor.
        max_aspect: cap on semi_major / semi_minor. Degenerate sliver faces would otherwise
            yield an arbitrarily elongated ellipse -- numerically fragile, and it would
            render as a needle. The cap WIDENS the minor axis instead of shrinking the major
            one, so the reaches-every-corner guarantee survives it (a wider ellipse still
            contains the vertices). Pass None or 0 to disable.

    Returns:
        axes: (F, 2) semi-axis lengths, (major, minor), with the major axis along `theta`.
        theta: (F,) in-plane angle of the major axis in the face's own 2D frame -- directly
            comparable to (and assignable into) gauss_params["quats"][:, 0].
    """
    verts = torch.stack([a2d, b2d, c2d], dim=1)  # (F, 3, 2)
    d = verts - verts.mean(dim=1, keepdim=True)  # (F, 3, 2)
    # M = (1/3) sum_i d_i d_i^T -- symmetric 2x2, written out per-component rather than as
    # a batched matmul so this stays a handful of elementwise ops over F.
    m11 = (d[..., 0] * d[..., 0]).mean(dim=1)
    m22 = (d[..., 1] * d[..., 1]).mean(dim=1)
    m12 = (d[..., 0] * d[..., 1]).mean(dim=1)
    half_tr = 0.5 * (m11 + m22)
    disc = torch.sqrt(torch.clamp((0.5 * (m11 - m22)) ** 2 + m12 ** 2, min=0.0))
    lam_max = torch.clamp(half_tr + disc, min=0.0)
    lam_min = torch.clamp(half_tr - disc, min=0.0)
    semi_major = torch.sqrt(2.0 * lam_max)
    semi_minor = torch.sqrt(2.0 * lam_min)
    if max_aspect is not None and max_aspect > 0:
        semi_minor = torch.maximum(semi_minor, semi_major / max_aspect)
    # Leading eigenvector angle of a symmetric 2x2. m12 == 0 and m11 > m22 gives 0 (major
    # along x); m12 == 0 and m11 < m22 gives pi/2 (major along y), both as they should be.
    theta = 0.5 * torch.atan2(2.0 * m12, m11 - m22)
    return torch.stack([semi_major, semi_minor], dim=-1), theta


def one_ring_ellipse_axes(edge_src, offsets_2d, num_verts, max_aspect=8.0):
    """
    Per-vertex 1-ring ellipse: the vertex layer's counterpart to steiner_ellipse_axes().

    Same reason for existing, one layer over. The vertex layer's size is a single scalar
    per vertex -- the MAX 1-ring edge length, see populate_modules -- copied into both
    in-plane axes, so a vertex Gaussian sitting at its ceiling or its floor has sx == sy
    exactly. That is a circle by construction, not by the optimizer's choice, and it is
    the same failure anisotropic_base_floor fixed for the base layer. The fix has to be
    made separately here because the two layers derive their size from different geometry
    and share no code path.

    Two things differ from the triangle case and neither is cosmetic:

    (1) The ellipse is centred on the VERTEX, not on the centroid of its neighbours. A
        vertex Gaussian is anchored exactly at its vertex (vertex_positions ==
        mesh_verts, fixed), so the offsets that decide how far it must reach are measured
        from there. Centring on the 1-ring's centroid would describe an ellipse the
        Gaussian is not sitting in.

    (2) A triangle's Steiner ellipse passes through all three vertices for free -- with
        three offsets summing to zero, d_i^T M^-1 d_i == 2 identically. That identity is
        specific to three points and does NOT generalise to an n-gon 1-ring, so an
        ellipse built from M alone would reach some neighbours and miss others, silently
        losing the reach guarantee the current isotropic radius provides. So M is used
        only for the ellipse's SHAPE and ORIENTATION (its principal axes), and the whole
        ellipse is then scaled up by the single factor that brings the farthest neighbour
        exactly onto its boundary. That keeps the existing guarantee intact and exact --
        "reaches every 1-ring neighbour", the same thing scatter_reduce(amax) on edge
        length provides today -- while letting the shape follow the ring's own anisotropy.
        On an isotropic 1-ring M is a multiple of the identity and the result degenerates
        to precisely today's circumscribing circle, so nothing changes there; the entire
        gain is on stretched/irregular rings.

    Args:
        edge_src: (E,) source vertex id of each DIRECTED 1-ring edge. Every neighbour
            relation must appear once per direction, i.e. the same both-directions edge
            list the isotropic vertex_radius is built from.
        offsets_2d: (E, 2) the neighbour's offset from its source vertex, already
            projected onto that vertex's own tangent frame (vertex_axis_x, vertex_axis_y).
            Those are the axes a vertex Gaussian's local x/y scales are measured along
            (see vertex_quats in populate_modules), which is what lets the returned
            semi-axes be used directly as an (x, y) scale bound.
        num_verts: V, so vertices with no incident edge still get a row.
        max_aspect: cap on semi_major / semi_minor, applied to the SHAPE before the
            enclosing scale is solved for, so the scale is computed against the ellipse
            that will actually be used and the reach guarantee survives the cap. As in
            steiner_ellipse_axes() the cap WIDENS the minor axis rather than shrinking the
            major one. Pass None or 0 to disable.

    Returns:
        axes: (V, 2) semi-axis lengths, (major, minor), with the major axis along `theta`.
        theta: (V,) in-plane angle of the major axis in the vertex's own tangent frame --
            directly comparable to (and assignable into) vertex_gauss_params["quats"][:, 0].
    """
    dev, dtp = offsets_2d.device, offsets_2d.dtype
    u, t = offsets_2d[:, 0], offsets_2d[:, 1]
    # dtype taken from the offsets, not left at the torch.zeros default: scatter requires
    # self and src to match exactly, so a float64 caller would otherwise fail here.
    zeros = lambda: torch.zeros(num_verts, device=dev, dtype=dtp)
    cnt = zeros().scatter_add_(0, edge_src, torch.ones_like(u))
    cnt_safe = cnt.clamp(min=1.0)
    # M = (1/n) sum_i d_i d_i^T over the 1-ring, symmetric 2x2, accumulated per component
    # because the ring is ragged (valence varies per vertex) -- a dense (V, n, 2) tensor
    # would have to be padded to the maximum valence and the padding would bias M.
    m11 = zeros().scatter_add_(0, edge_src, u * u) / cnt_safe
    m22 = zeros().scatter_add_(0, edge_src, t * t) / cnt_safe
    m12 = zeros().scatter_add_(0, edge_src, u * t) / cnt_safe
    # Same closed-form symmetric-2x2 eigendecomposition as steiner_ellipse_axes().
    half_tr = 0.5 * (m11 + m22)
    disc = torch.sqrt(torch.clamp((0.5 * (m11 - m22)) ** 2 + m12 ** 2, min=0.0))
    lam_max = torch.clamp(half_tr + disc, min=0.0)
    lam_min = torch.clamp(half_tr - disc, min=0.0)
    theta = 0.5 * torch.atan2(2.0 * m12, m11 - m22)

    # Unit-scale shape only -- the enclosing factor is solved for below.
    shape_major = torch.sqrt(lam_max).clamp(min=1e-12)
    shape_minor = torch.sqrt(lam_min)
    if max_aspect is not None and max_aspect > 0:
        shape_minor = torch.maximum(shape_minor, shape_major / max_aspect)
    shape_minor = shape_minor.clamp(min=1e-12)

    # Smallest uniform scale s with every neighbour inside {(u/(s*a))^2 + (t/(s*b))^2 <= 1},
    # i.e. s = max_i sqrt((u_i/a)^2 + (t_i/b)^2) in the ellipse's own frame. amax, exactly
    # as the isotropic radius uses amax over edge length, so the farthest neighbour lands
    # on the boundary and every other one is strictly inside.
    cos_t, sin_t = torch.cos(theta)[edge_src], torch.sin(theta)[edge_src]
    u_rot = u * cos_t + t * sin_t
    t_rot = -u * sin_t + t * cos_t
    rho2 = (u_rot / shape_major[edge_src]) ** 2 + (t_rot / shape_minor[edge_src]) ** 2
    s2 = zeros().scatter_reduce(
        0, edge_src, rho2, reduce="amax", include_self=False
    )
    scale = torch.sqrt(s2.clamp(min=0.0))
    # A vertex with no incident edge (isolated, or a mesh the caller trimmed) would get
    # scale 0 and collapse to a point; leave it at its shape so the caller's own clamp on
    # the isotropic radius still governs it.
    scale = torch.where(cnt > 0, scale, torch.ones_like(scale))
    return torch.stack([shape_major * scale, shape_minor * scale], dim=-1), theta


def detect_fold_safe_radius(mesh_faces, mesh_faces_verts, own_ring_radius, safety_frac=0.5, fold_ratio=1.5):
    """
    2026-08-02, redesigned same day after the first version's fixed-hop-count exclusion
    (2-ring) turned out not to be density-invariant: every existing per-face size bound
    (face_ring_radius, face_centroid_vertex_radius) only ever looks at TOPOLOGICAL
    neighbors (faces reachable by walking shared edges). None of them can "see" a mesh
    fold -- e.g. an armpit, where the arm's triangles and the torso's triangles are many
    edges apart along the surface but nearly touching in 3D space. A face's size bound
    there gets computed as if its nearest surface were still its own topological
    neighbors, with nothing stopping it from reaching straight across the fold into the
    facing, UNRELATED surface. That produces a forced, permanent, never-resolvable
    overlap between two near-coplanar semi-transparent Gaussian populations whose blend
    order flips with viewing angle -- the sort-flip flicker reported at every limb/body
    junction. (Root-caused by direct code comparison against DRAWER-main, which had no
    scale floor at all and so could let its optimizer freely shrink away redundant
    overlap; this project's coverage-guarantee machinery -- min_scale_frac, the base
    layer, the vertex/centroid filler layers -- structurally cannot do that, so the
    overlap this creates at folds has nowhere to go.)

    FIRST VERSION (reverted): excluded a face's 2-ring (self + edge-adjacent + their
    edge-adjacent) neighborhood from the "foreign" search, on the theory that anything
    closer than that must be a fold. Broke badly on this mesh: triangle density varies a
    lot (small, dense triangles in high-curvature regions like fingers/ears), and a FIXED
    hop count covers very little real 3D distance in dense regions -- so ordinary,
    non-folded curved surface there got misclassified as folded and had its radius
    wrongly shrunk, producing many new coverage holes without fixing the flicker (the
    genuine folds weren't even the dominant source of what got flagged).

    THIS VERSION: only excludes the direct 1-ring (edge-adjacent) neighbors from the
    "foreign" search -- those are unconditionally expected to be close, no threshold
    needed. Whether the nearest remaining ("foreign") face actually counts as a fold is
    then decided by comparing its distance against `fold_ratio * own_ring_radius`
    (own_ring_radius = face_ring_radius()'s 1-ring-average output, the same
    density-adaptive local-scale estimate already used everywhere else in this file) --
    small triangles get a small threshold, large triangles get a large one, so the same
    physical ratio applies mesh-wide regardless of local density. Only when a foreign
    face is closer than that scaled threshold does it actually count as a fold and cap
    the radius; otherwise the returned cap is +inf, a guaranteed no-op through
    torch.minimum.

    Args:
        mesh_faces: (F, 3) long tensor of vertex indices per face.
        mesh_faces_verts: (F, 3, 3) vertex positions per face.
        own_ring_radius: (F,) or (F, 1) tensor, this face's own 1-ring-average radius
            (face_ring_radius()'s output) -- the density-adaptive "what counts as
            normal neighbor spacing here" reference this whole redesign keys off.
        safety_frac: fraction of the raw nearest-foreign-face distance to allow when a
            fold IS detected -- 0.5 (default) splits the gap evenly, matching that both
            sides of a fold apply the same cap. Never tuned.
        fold_ratio: how many multiples of own_ring_radius a foreign face must be closer
            than to count as a fold -- 1.5 (default, never tuned) means "closer than
            1.5x how far my own 1-ring neighbors are", deliberately conservative (a
            regular 2-ring neighbor in a uniform mesh sits at roughly 2x 1-ring
            distance) so ordinary curved-but-unfolded surface stays clear of the
            threshold even where triangulation is a little irregular.

    Returns:
        (fold_cap, is_fold, nearest_foreign):
        fold_cap: (F, 1) tensor, meant to be combined with an existing per-face radius
            via torch.minimum (never used to widen a bound, only to tighten one):
            safety_frac * nearest-non-1-ring-face-distance where that distance is below
            the scaled threshold, +inf (no-op) everywhere else.
        is_fold: (F,) bool tensor, the same detection this function already computes to
            build fold_cap, exposed separately (2026-08-03) so callers can target OTHER
            per-face relaxations at exactly these faces too -- e.g. the base layer's
            opacity floor (see config.fold_base_opacity_floor): globally lowering that
            floor to resolve fold flicker made the WHOLE model visibly more transparent,
            since it relaxed every face's base, not just the folded ones. Reusing this
            same mask keeps the relaxation surgical instead of global.
        nearest_foreign: (F,) long tensor, the index of the nearest non-1-ring face
            (-1 if none) -- i.e. WHICH face each fold faces across the gap, exposed
            2026-08-06 for color_consistency_lambda's fold term. Valid for every face,
            but only meaningful where is_fold is True (elsewhere it is just "the nearest
            face that happens not to share an edge", an ordinary non-fold neighbor).
    """
    F = mesh_faces.shape[0]
    device = mesh_faces.device
    centers = mesh_faces_verts.mean(dim=1)  # (F, 3)
    own_ring_radius = own_ring_radius.reshape(-1)  # (F,)

    # ---- exclusion table: self + every face sharing a VERTEX. Widened from
    # self + edge-adjacent on 2026-08-06, because the edge-only version made this
    # function fire almost everywhere: measured on the armadillo mesh it flagged
    # 86.72% of ALL faces as folds, and the "foreign" face it found was overwhelmingly
    # just a corner neighbor on the same smooth sheet (median normal angle between the
    # two faces: 14.5 degrees; only 0.1% of them were more than 90 degrees apart, i.e.
    # actually facing each other). The premise stated above -- "a regular 2-ring
    # neighbor in a uniform mesh sits at roughly 2x 1-ring distance" -- simply does not
    # hold: a triangle has 3 edge neighbors but ~12 vertex neighbors, and those corner
    # neighbors sit at essentially the SAME centroid distance as the edge ones, so any
    # fold_ratio above ~1.0 was guaranteed to fire on ordinary curved surface. This is
    # the same "ordinary surface misclassified as folded" failure the FIRST VERSION note
    # above describes; making the threshold density-adaptive fixed how it scaled, not
    # the fact that the excluded neighborhood was too small. Excluding the full vertex
    # neighborhood drops the flag rate to 6.96% on the same mesh.
    vids = mesh_faces.reshape(-1)  # (3F,)
    fids = torch.arange(F, device=device).repeat_interleave(3)  # (3F,)
    v_order = torch.argsort(vids)
    v_s, f_s = vids[v_order], fids[v_order]
    _, v_counts = torch.unique_consecutive(v_s, return_counts=True)
    v_starts = torch.cumsum(v_counts, 0) - v_counts
    v_slot = torch.arange(v_s.shape[0], device=device) - torch.repeat_interleave(v_starts, v_counts)
    V = int(vids.max().item()) + 1
    v_table = torch.full((V, int(v_counts.max().item())), -1, dtype=torch.long, device=device)
    v_table[v_s, v_slot] = f_s

    self_ids = torch.arange(F, device=device).reshape(-1, 1)
    # (F, 1 + 3K): self + every face incident to any of this face's 3 vertices.
    allowed = torch.cat([self_ids, v_table[mesh_faces].reshape(F, -1)], dim=1)

    # ---- chunked nearest-non-1-ring-face search (same double-chunk pattern as
    # _find_nearest_faces(), just centroid-to-centroid Euclidean distance -- a coarse
    # proxy is enough here, this only needs to detect "something foreign is close",
    # not the exact closest point on that face).
    _G_CHUNK = 1_000
    _F_CHUNK = 20_000
    best_dist = torch.full((F,), float("inf"), device=device)
    # WHICH face won, not just how far it was (2026-08-06). The search already visits
    # every candidate; keeping the argmin alongside the min costs one extra (F,) tensor
    # and lets color_consistency_lambda reuse this exact same "topologically distant but
    # geometrically close" relation as its third penalty term, instead of running a
    # second, near-identical spatial search of its own. -1 where nothing foreign exists.
    best_idx = torch.full((F,), -1, dtype=torch.long, device=device)
    for g_start in range(0, F, _G_CHUNK):
        g_end = min(g_start + _G_CHUNK, F)
        g = g_end - g_start
        q = centers[g_start:g_end]  # (g, 3)
        q_allowed = allowed[g_start:g_end]  # (g, 4)
        chunk_best = torch.full((g,), float("inf"), device=device)
        chunk_idx = torch.full((g,), -1, dtype=torch.long, device=device)
        for f_start in range(0, F, _F_CHUNK):
            f_end = min(f_start + _F_CHUNK, F)
            cand = centers[f_start:f_end]  # (f, 3)
            d = torch.cdist(q, cand)  # (g, f)
            local = q_allowed - f_start  # (g, 4)
            valid = (q_allowed >= 0) & (local >= 0) & (local < (f_end - f_start))
            if valid.any():
                rows = torch.arange(g, device=device).unsqueeze(1).expand(-1, q_allowed.shape[1])[valid]
                cols = local[valid]
                d[rows, cols] = float("inf")
            f_min, f_arg = d.min(dim=1)
            # Strict < so the earliest chunk wins ties, making the result independent of
            # _F_CHUNK (a >= would let a later chunk overwrite an equally-close face).
            improved = f_min < chunk_best
            chunk_best = torch.where(improved, f_min, chunk_best)
            chunk_idx = torch.where(improved, f_arg + f_start, chunk_idx)
        best_dist[g_start:g_end] = chunk_best
        best_idx[g_start:g_end] = chunk_idx

    # Density-adaptive gate: only a nearest-foreign-face distance that's suspiciously
    # small RELATIVE TO this face's own local scale counts as a fold. A dense/small-
    # triangle region and a coarse/large-triangle region can have wildly different
    # absolute best_dist values for entirely ordinary (unfolded) geometry -- comparing
    # against a fixed number would reintroduce the same density-invariance bug this
    # redesign exists to fix.
    threshold = fold_ratio * own_ring_radius
    is_fold = best_dist < threshold
    fold_cap = torch.where(is_fold, safety_frac * best_dist, torch.full_like(best_dist, float("inf")))
    return fold_cap.reshape(-1, 1), is_fold, best_idx


def compute_min_distance(v, v1, v2, v3):
    """
    Computes the minimum distance from point v to the edges of the triangle formed by v1, v2, and v3.
    Supports batch inference.

    Args:
    v (Tensor): Tensor of shape (batch_size, 3), the point from which distances are computed.
    v1, v2, v3 (Tensor): Tensors of shape (batch_size, 3), representing the vertices of the triangle.

    Returns:
    Tensor: Minimum distance from v to the triangle edges for each batch element.
    """

    def distance_to_edge(v, v_start, v_end):
        # Compute edge vector e and vector from start to v (w)
        e = v_end - v_start
        w = v - v_start

        # Projection scalar t
        e_dot_e = torch.sum(e * e, dim=-1, keepdim=True)
        t = torch.sum(w * e, dim=-1, keepdim=True) / e_dot_e

        # Clamp t to [0, 1] to handle closest point on the edge segment
        t_clamped = torch.clamp(t, 0, 1)

        # Closest point on the edge
        closest_point = v_start + t_clamped * e

        # Compute distance from v to closest point
        distance = torch.norm(v - closest_point, dim=-1)

        return distance

    # Compute distances to each edge
    d1 = distance_to_edge(v, v1, v2)  # Edge from v1 to v2
    d2 = distance_to_edge(v, v2, v3)  # Edge from v2 to v3
    d3 = distance_to_edge(v, v3, v1)  # Edge from v3 to v1

    # Return the minimum distance across the edges
    min_distance = torch.min(torch.stack([d1, d2, d3], dim=-1), dim=-1).values

    return min_distance


def compute_triangle_vertices(a, b, c):
    """
    Compute the coordinates of triangle vertices A, B, and C
    given side lengths a, b, and c in a batched manner.

    Args:
        a: Tensor of side lengths |BC|.
        b: Tensor of side lengths |CA|.
        c: Tensor of side lengths |AB|.

    Returns:
        A (0, 0), B (c, 0), C (x_C, y_C): Coordinates of the triangle vertices.
    """
    # Vertices A and B
    A_x, A_y = torch.zeros_like(a), torch.zeros_like(a)  # A is at (0, 0)
    B_x, B_y = c, torch.zeros_like(c)  # B is at (c, 0)

    # Compute C coordinates
    x_C = (c ** 2 - a ** 2 + b ** 2) / (2 * c)  # x-coordinate of C
    y_C = torch.sqrt(b ** 2 - x_C ** 2)  # y-coordinate of C

    A = torch.stack((A_x, A_y), dim=-1)
    B = torch.stack((B_x, B_y), dim=-1)
    C = torch.stack((x_C, y_C), dim=-1)

    return A, B, C


def barycentric_coordinates(P, A, B, C):
    """
    Compute the barycentric coordinates for a point P relative to triangle ABC.

    Args:
        P_x: Tensor of x-coordinates of point P.
        P_y: Tensor of y-coordinates of point P.
        A, B, C: Tuples of tensors representing the coordinates of points A, B, and C.

    Returns:
        alpha, beta, gamma: Barycentric coordinates of the point P.
    """
    A_x, A_y = A[:, 0], A[:, 1]
    B_x, B_y = B[:, 0], B[:, 1]
    C_x, C_y = C[:, 0], C[:, 1]
    P_x, P_y = P[:, 0], P[:, 1]

    # Compute the denominator (area of the triangle ABC)
    By_Cy = B_y - C_y
    Ax_Cx = A_x - C_x
    Cx_Bx = C_x - B_x
    Ay_Cy = A_y - C_y
    Px_Cx = P_x - C_x
    Py_Cy = P_y - C_y

    denominator = By_Cy * Ax_Cx + Cx_Bx * Ay_Cy

    # Compute the barycentric coordinates
    alpha = (By_Cy * Px_Cx + Cx_Bx * Py_Cy) / denominator
    beta = (-Ay_Cy * Px_Cx + Ax_Cx * Py_Cy) / denominator
    gamma = 1 - alpha - beta

    return torch.stack([alpha, beta, gamma], dim=-1)


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    height, width = image.shape[:2]
    # print("height: ", height)
    # print("width: ", width)

    scaling_factor = 1.0 / d
    # weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    # return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)
    height_new = int(torch.floor(0.5 + (torch.tensor([float(height)]) * scaling_factor)).to(torch.int64))
    width_new = int(torch.floor(0.5 + (torch.tensor([float(width)]) * scaling_factor)).to(torch.int64))

    # print("height_new: ", height_new)
    # print("width_new: ", width_new)

    image_resized = F_V.resize(image.permute(2, 0, 1), [height_new, width_new])
    image_resized = image_resized.permute(1, 2, 0)

    return image_resized

@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class SplatfactoOnMeshUCModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: SplatfactoOnMeshUCModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.05
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    cull_at_floor_gaussians: bool = False
    """Whether to cull non-base Gaussians whose rendered in-plane (x, y) scale is pinned
    at the min_scale_frac floor (see the scales property / already_at_floor in
    split_gaussians/dup_gaussians for the exact same check). Off by default so existing
    runs are unaffected. Historically these floor-pinned Gaussians were left alone
    because removing them could reopen coverage gaps -- but with use_base_layer (plus
    the vertex-/centroid-anchored filler layers) now guaranteeing every face/vertex has
    permanent coverage independent of this population, a detail-layer Gaussian stuck at
    the floor is no longer load-bearing for coverage, just visual clutter of
    barely-visible dots. See cull_gaussians()."""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    opacity_reset_value: float = 0.8
    """Upper bound the periodic opacity reset clamps every raw opacity logit to."""
    opacity_reset_min_value: float = 0.5
    """Lower bound the periodic opacity reset clamps every raw opacity logit to.

    NOTE these two defaults reproduce this file's historical hardcoded [0.5, 0.8] and are
    the OPPOSITE of vanilla splatfacto, which uses [cull_alpha_thresh * 1.5,
    cull_alpha_thresh * 2.0] -- i.e. [0.075, 0.10] at cull_alpha_thresh=0.05. The point of
    3DGS's opacity reset is to push opacity DOWN so redundant Gaussians fall below the cull
    threshold and the survivors have to re-earn their opacity; clamping to a 0.5 FLOOR
    instead does the reverse, forcing every Gaussian in the model back to at least half
    opaque every reset_alpha_every * refine_every steps.

    Measured consequence on table_gs6 (2026-08-24): 98.1% of face-based Gaussians ended
    training at opacity > 0.9. With everything opaque there is no alpha blending left, so
    whichever Gaussian sorts in front takes the pixel outright -- which is both why fine
    detail cannot be anti-aliased and why the render speckles (at ~1 Gaussian per 1.7 px
    with sigma ~2.5 px, the front-most one is close to arbitrary).

    The floor was originally raised so occluded Gaussians could not die from the reset.
    With adaptive_coverage_floor that job belongs to the coverage machinery (base rows are
    exempt from culling outright, and the rescue re-spawns genuine gaps), so lowering these
    toward vanilla is now safe. Left at the historical values so nothing changes unless you
    ask for it."""
    densify_grad_thresh: float = 0.0004
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh_frac: float = 0.5
    """Below this fraction of a Gaussian's own face's xyz_radius (in-plane, x/y), a
    high-gradient Gaussian is *duplicated* (dup_gaussians, shrinks both copies by 1.6x);
    at or above it, it's *split* (split_gaussians, shrinks by 1.6x once into 3 samples).
    Replaces the old densify_size_thresh, a fixed absolute value (0.01) inherited from
    vanilla splatfacto that measured against real-world scene units, not this mesh's
    normalized coordinate scale. On this mesh essentially every face's natural
    ceiling-init scale already sits below 0.01 (measured: 100% of faces at
    init_copies_per_face=4, 95.9% even at 1 -- see pipeline_overview.md), so with the old
    threshold virtually all densification silently took the dup-and-shrink path and the
    split path was practically unreachable, causing progressive size erosion each time a
    Gaussian re-triggered densification -- worst in high-curvature regions, since
    curvature_densify_scale boosts their effective gradient and so how often they
    re-trigger. 0.5 means a freshly-seeded Gaussian (starts at upper_scale x xyz_radius)
    stays in the split bucket for its first one or two densification events (1/1.6 ~=
    0.625, so after one split/dup it's still >= 0.5x) and only falls into the dup bucket
    once it's already meaningfully smaller than its own face -- unverified against an
    actual training run, may need retuning."""
    n_split_samples: int = 3
    """number of samples to split gaussians into"""
    dup_size_shrink_factor: float = 1.2
    """Divisor applied to a Gaussian's scale (both original and its duplicate) each time
    dup_gaussians fires on it. Separate from split_gaussians' shrink (hardcoded 1.6,
    reasonable for "large Gaussian -> 3 smaller ones covering the same area" -- that
    branch only fires above densify_size_thresh_frac x xyz_radius, comfortably clear of
    min_scale_frac, so it doesn't feed the floor-convergence problem this addresses).

    Added 2026-07-17, lowered from 1.6 (dup_gaussians used to share split_gaussians'
    factor): dup_gaussians is the branch that fires on already-small Gaussians, often
    repeatedly (curvature_densify_scale / coverage_densify_scale both boost how often
    densification triggers, and there is no mechanism anywhere that grows an existing
    Gaussian's scale back up) -- 1.6 applied every trigger drove most of the model down to
    min_scale_frac regardless of where that floor was set (confirmed: median x/y ratio sat
    exactly on min_scale_frac at both 0.3 and 0.5). A gentler factor slows that descent and
    leaves more overlap between newly-duplicated Gaussians, matching the user's stated
    "overlap over gaps" priority. Combined with skipping the shrink entirely once a
    Gaussian's *rendered* size is already at the floor (see dup_gaussians) -- this value
    only matters for Gaussians still above the floor. Never tuned/validated; 1.2 is a
    guess gentler than 1.6, not a measured optimum."""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    detail_elevate_min_frac: float = -1.0
    """Lower bound on a non-base Gaussian's offset along its face normal, as a fraction of
    the face radius. -1.0 reproduces the symmetric [-elevate_coef, +elevate_coef] range
    every run so far has used. 0.0 confines the detail layer to the OUTWARD side of its
    face, i.e. in front of the base plate.

    Why. Measured on table_gs20's export, offsets along the face normal in units of face
    radius (the mesh is closed and consistently wound -- signed volume +3.90, and 100.00%
    of 2,063,187 shared edges have same-facing normals -- so "outward" is "towards any
    camera that can see this face", independent of viewpoint):

        base    1,136,576 rows   p10 -0.000  p50 -0.000  p90 +0.000   (pinned)
        detail  3,679,132 rows   p10 -0.500  p50 -0.000  p90 +0.500   (saturating both caps)

        50.9% of detail sits BEHIND the base plate.

    Those rows are trained, occupy memory, and carry a median alpha of 0.996, but an
    opacity-0.950 plate spanning the whole face sits in front of them, so they contribute
    almost nothing. Front and back halves have identical median size (0.3299 of face
    radius), so this is not a size-based division of labour -- it is half the detail layer
    being wasted.

    It is also self-sustaining: a Gaussian hidden behind an opaque plate barely affects any
    rendered pixel, so it receives almost no gradient, so nothing ever pushes it out. The
    clean 50/50 split is the initial random distribution frozen in place, not a decision the
    optimiser made.

    Setting this to 0.0 roughly doubles the effective detail population at zero memory cost.
    Coverage is unaffected: _compute_coverage_density() measures distances in the face
    plane, which the normal offset does not change.

    Base rows are unaffected either way -- they are position-pinned with zero offset."""
    cull_screen_size_min: float = 0.0
    """Cull non-base Gaussians whose on-screen size stays BELOW this, as a fraction of the
    larger image dimension -- the mirror image of cull_screen_size, which culls the ones
    that get too big. 0 disables it, so runs that do not set it are bit-identical.

    Why this exists. Measured on table_gs20's export, by on-screen sigma:

        sigma <= 0.5 px : 581,116 rows (10.6% of the model) painting 0.0% of the frame
        sigma 0.5-1 px  : 712,768 rows (13.0%)              painting 1.2%
        sigma 3-10 px   : 1.40M rows   (25.6%)              painting 80.9%

    A tenth of the population contributes nothing visible. It is not harmless: it is a
    tenth of the memory budget, and table_gs21 died at 69% precisely because the budget ran
    out -- 205,888 faces were flagged for coverage repair and coverage_rescue was allowed
    to spawn zero Gaussians. Deleting rows that paint nothing buys back exactly the room
    that the rescue needed.

    Why not min_scale_frac instead. Raising that floor to bite at ~0.5 px needs
    min_scale_frac ~ 0.16, which drops the reachable in-plane aspect ratio
    (upper_scale / min_scale_frac) to about 3.1 while the needle cull fires at
    max_gauss_ratio * 4 = 20. That is the exact configuration gs11 had to fix: a cull
    threshold above the reachable maximum never fires at all. A separate screen-space
    bound has no such coupling.

    Choosing a value. The units are the same as cull_screen_size (radius / max(H, W)), and
    the mapping from radius to sigma is the rasteriser's, not ours -- so read it off the
    distribution this class logs at every cull rather than deriving it. The log line prints
    the p1/p10/p50 of max_2Dsize over rows that were actually seen; a threshold at or just
    below the p10 removes the invisible tail. Start conservative and raise it only after
    seeing the culled count and the run's own coverage figure.

    Safety. Only rows with max_2Dsize > 0 are eligible: a zero means the row was not seen
    by any view this refine cycle (or was just split/duplicated and its entry zero-filled),
    which is not the same as being small. Beyond that the two existing guarantees still
    apply -- base rows are never culled, and a row that would leave its face empty is kept
    and has its opacity raised instead. And max_2Dsize stops being accumulated at
    stop_split_at, after which it is None and this check is skipped entirely."""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 25000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = True
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 2.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    opacity_reg_lambda: float = 0.001
    """weight of opacity binarization loss to push opacities towards 0 or 1"""
    init_opacity: float = 0.9
    """Initial opacity for newly created face-based Gaussians (seed, split/dup children,
    append_from_mesh, and coverage-rescue spawns) -- NOT the base layer's rendered floor
    (base_opacity_floor, applied separately as a property-level clamp regardless of this
    value) and NOT the vertex-/centroid-anchored filler layers (they start at a fixed,
    unrelated neutral 0.5, see populate_modules). Original DRAWER-main value was 0.1;
    raised to 0.9 during this project's development so freshly seeded/split/dup Gaussians
    start comfortably above cull_alpha_thresh instead of near it."""
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    mesh_area_to_subdivide: float = 1e-5
    upper_scale: float = 0.5
    unconstrained_scale: bool = True
    unconstrained_elevate: bool = True
    face_flat_coef: float = 0.005
    elevate_coef: float = 0.15
    cone_coef: float = 10.0 * np.pi / 180.0
    acm_lambda: float = 20.0
    mesh_depth_lambda: float = 0.1
    gaussian_save_extra_info_path: str = None
    curvature_densify_scale: float = 5.0
    """Boost densification gradients in high-curvature regions (nose, fingers, sharp edges).
    0 disables curvature weighting; higher values concentrate more Gaussians at curved areas."""
    coverage_densify_scale: float = 0.0
    """Boost densification gradients wherever coverage_lambda's own geometric check
    (mesh vertices / face centroids, see that config's docstring) finds a coverage
    deficit -- same mechanism as curvature_densify_scale, just weighted by "how
    undercovered is this face" instead of "how curved is this face". 0 disables (default,
    matches original behavior).

    Added 2026-07-17 because coverage_lambda and densify_grad_thresh are otherwise
    decoupled: coverage_lambda can tell a face is undercovered and nudge nearby
    Gaussians' position/scale/opacity toward it, but it cannot spawn new Gaussians --
    only densify_grad_thresh (a *photometric*-gradient trigger) can, via
    split_gaussians/dup_gaussians. In occluded, low-texture, or otherwise weakly-observed
    regions, photometric gradient can stay below densify_grad_thresh indefinitely even
    while coverage_lambda keeps reporting a real geometric gap there -- densification
    never fires and the gap never gets more Gaussians, no matter how long training runs.
    This lets a geometric deficit lower the bar for triggering densification directly,
    the same way curvature already does, so a persistently undercovered face can still
    get new Gaussians even with weak photometric signal.

    This is a *number*, not *size*, lever: it grows coverage by adding more Gaussians in
    genuinely gapped areas rather than inflating existing ones (unlike min_scale_frac),
    so it shouldn't by itself increase overlap/flicker risk the way a bigger scale floor
    does. Trade-off: more Gaussians get spawned overall wherever a deficit persists, so
    training time/GPU memory both go up, similar to lowering densify_grad_thresh directly
    but targeted at coverage gaps specifically rather than blanket-lowering the trigger
    everywhere. Brand new, never run."""
    min_scale_frac: float = 0.05
    """Minimum scale as a fraction of each face's bounding radius. Prevents Gaussians from
    shrinking so small they no longer cover their assigned triangle face. 0 disables."""
    base_floor_reaches_corners: bool = False
    """Size the base layer's in-plane scale floor off its face's OWN corner reach
    (cv_radius_base_floor) instead of off the copies_scale_shrink_power-divided
    xyz_radius, so a base Gaussian actually reaches its own triangle's corners.

    Fixes a measured discrepancy between use_base_layer's docstring ("its in-plane (x,y)
    rendered size is floored at the FULL face xyz_radius, so it always reaches its own
    corners") and what the code does. The floor is
    min(cv_radius_base_floor, xyz_radius), and xyz_radius has already been divided by
    init_copies_per_face ** copies_scale_shrink_power a few lines earlier in
    populate_modules -- so the min() always selects the shrunk value and the promise never
    held. Measured on the exported table_gs6 (init_copies_per_face=2,
    copies_scale_shrink_power=0.3, divisor 2**0.3 = 1.231): all 1,497,524 base rows sat at
    exactly 0.812 x cv_radius, none at or above 1.0. A base plate therefore only reached
    its own mean corner at 1.23 sigma and its farthest corner at 1.30 sigma, where the
    Gaussian is down to 0.47 / 0.43 of peak -- i.e. every triangle corner on the mesh was
    being covered at under half strength by the very layer whose job is to guarantee it.

    The shrink divisor is correct for ordinary detail copies (N of them partition a face,
    so each should claim 1/sqrt(N) of it) and wrong for the base plate, which is one per
    face and must span the face alone. This separates the two rules.

    Cost: floored base plates get 1.23x larger in-plane at the current settings, so faces
    that keep their floor get blurrier. That is why this pairs with
    adaptive_coverage_floor -- with floors released wherever detail already covers, the
    bigger plate only lands where there is no detail to lose. Turning this on WITHOUT
    adaptive_coverage_floor makes the whole model blurrier, uniformly.

    Off by default: it changes rendered geometry for every base row on every existing
    run."""
    anisotropic_base_floor: bool = False
    """Give the base layer an ELLIPTICAL scale floor, shaped and oriented to its own
    triangle (its Steiner circumellipse, see steiner_ellipse_axes()), instead of a single
    isotropic radius applied to both in-plane axes.

    Fixes the reason base plates render as circles rather than ellipses. The floor is
    self.base_floor_xy, built as cv_radius_base_floor.expand(-1, 2) in populate_modules --
    one scalar per face written into BOTH the x and y bound. Any base row the floor
    actually binds therefore has sx == sy exactly, i.e. a perfect circle, by definition and
    not by the optimizer's choice. Measured on the exported table_gs8: of 1,282,316 rows
    sitting at the base floor, median in-plane aspect ratio 1.000 and 96.8% below 1.001.
    Note that min_scale_frac and max_gauss_ratio cannot touch this -- floored base rows
    take the _base_floor_xy branch of the scales property, which neither value feeds into,
    so tuning them only ever reshapes the detail layer.

    This also resolves the coverage-vs-overlap tradeoff face_centroid_vertex_radius()
    documents as unresolved, rather than picking a side of it. The Steiner ellipse passes
    through all three vertices (so corner coverage is strictly better than today's
    mean-based radius, which reaches the mean corner distance and undershoots the farthest
    one) while enclosing a fixed 2.418x the face's own area for any triangle shape (so
    overlap is far below the max-based radius, whose circumscribed circle blows up on
    slivers). On near-equilateral faces the two coincide and nothing changes; the gain is
    concentrated on irregular faces, which is where coverage was actually short.

    Implies pinning each base row's in-plane rotation to its face's Steiner major axis (see
    _inplane_theta()) -- the two are inseparable: an elliptical floor written into the (x,
    y) scale bounds only describes the intended ellipse while the Gaussian's local x axis
    points along that ellipse's major axis. Out-of-plane tilt stays trainable, so the
    2026-08-06 fix for base plates crossing at sharp dihedral angles is unaffected.

    Pair with anisotropic_coverage_density: leaving that off measures every ellipse by its
    NARROW axis, which reports a correctly-covering elliptical plate as under-covering and
    pulls it straight back to circular via the floor mask. Off by default: it changes
    rendered geometry for every base row on every existing run."""
    base_floor_corner_sigma: float = 1.0
    """How many sigma out the triangle's vertices land on a base plate whose anisotropic
    floor is engaged. The floor is the Steiner semi-axes divided by this, so 1.0 puts every
    vertex exactly on the 1-sigma ellipse (exp(-0.5) = 0.607 of peak) and smaller values
    grow the plate to cover the corners harder. For reference, base_floor_reaches_corners
    lands the farthest corner at about 1.06 sigma today, so 1.0 here is very slightly
    stronger than that and markedly stronger than the 1.30 sigma the pre-fix floor gave.
    Only read when anisotropic_base_floor is on."""
    base_floor_max_aspect: float = 8.0
    """Cap on a base plate's floor aspect ratio (semi_major / semi_minor), applied inside
    steiner_ellipse_axes() by WIDENING the minor axis, never shrinking the major one, so
    the corner guarantee survives the cap. Degenerate sliver faces -- which marching-cubes
    meshes produce in quantity -- would otherwise get arbitrarily needle-like floors. Note
    the needle cull at max_gauss_ratio * 4 cannot delete a base row (cull_gaussians() ends
    with culls &= ~is_base), so this is about numerical sanity and rendering, not survival.
    Only read when anisotropic_base_floor is on."""
    anisotropic_vertex_floor: bool = False
    """Give the vertex filler layer an ELLIPTICAL size bound, shaped and oriented to its
    own 1-ring (see one_ring_ellipse_axes()), instead of one isotropic radius applied to
    both in-plane axes.

    The vertex-layer counterpart of anisotropic_base_floor, and needed separately because
    that flag does not reach this layer: it only rewrites the base layer's floor
    (base_floor_xy_aniso, consumed by the scales property). The vertex layer's size comes
    from self.vertex_log_scales, built in populate_modules as
    vertex_radius.repeat(1, 3) -- ONE scalar per vertex written into both the x and y
    bound -- and _vertex_scales() derives both its ceiling and its floor
    (min_vertex_scale_frac x ceiling) from that. So any vertex row sitting at either bound
    has sx == sy exactly, a circle by definition rather than by the optimizer's choice.

    A WARNING ABOUT THE EVIDENCE, because this flag was originally justified by a number
    that turned out to be false. The exported ply appeared to show 59.1% of vertex rows at
    in-plane aspect below 1.001, which read as this defect biting hard. It was not: the
    exporter was capping every population at a FACE-derived bound (see its own note), which
    clipped both in-plane axes of 56.9% of the vertex layer to the same value and wrote
    them out as exact circles. Read back from the checkpoint instead, that same model had
    aspect median 1.2055 with 0.2% circles -- the defect is real in principle but was
    nowhere near binding in practice.

    What it actually buys, measured rather than assumed: on this mesh the 1-ring ellipse
    covers the identical neighbours at 0.952x the isotropic circle's footprint, i.e. ~5%.
    The 1-rings here are close to isotropic (aspect median 1.11, q99 1.52). Expect
    materially more only on a mesh with genuinely stretched triangulation.

    Unlike the base layer there is no corner_sigma here. The isotropic radius this
    replaces is used directly as sigma (vertex_radius is the max 1-ring edge length, with
    vertex_radius_scale at its 1.0 default), so the ellipse is used directly too and a vertex
    on an isotropic 1-ring keeps exactly the size it has today.

    Implies pinning each vertex row's in-plane rotation to its 1-ring ellipse's major axis
    (see _vertex_inplane_theta()), for the same inseparable reason the base layer's pin is
    not optional: bounding local x and y separately only describes the intended ellipse
    while local x points along that ellipse's major axis. Out-of-plane tilt stays
    trainable within cone_coef, exactly as before.

    Pair with anisotropic_coverage_density: this layer's contribution to a face centroid
    is measured through v_sigma_min in _compute_coverage_density(), i.e. by its NARROW
    axis, which would report a correctly-covering elliptical filler as under-covering and
    pull it back toward circular via vertex_needs_floor. That function's anisotropic branch
    covers the vertex term too. Off by default: it changes rendered geometry for every
    vertex row on every existing run."""
    vertex_floor_max_aspect: float = 8.0
    """Cap on a vertex ellipse's major/minor ratio (see anisotropic_vertex_floor). Same
    role and same default as base_floor_max_aspect, and applied the same way inside
    one_ring_ellipse_axes() -- by WIDENING the minor axis, never shrinking the major one,
    and before the enclosing scale is solved for, so the reaches-every-neighbour guarantee
    survives it. Matters more here than on a triangle: a vertex on a mesh boundary or on a
    crease sees a 1-ring that is genuinely close to collinear, which would otherwise yield
    an arbitrarily elongated ellipse and render as a needle.

    Only read when anisotropic_vertex_floor is on."""
    vertex_radius_scale: float = 1.0
    """Scale on the vertex layer's size CEILING, i.e. on its 1-ring reach. 1.0 is the
    historical behaviour (reach the farthest 1-ring neighbour exactly); below 1.0 shrinks
    every vertex Gaussian proportionally, ellipse and all.

    Was the hardcoded local _VERTEX_RADIUS_SAFETY_MARGIN, which has only ever been 1.0 or
    1.3 (the 1.3 was reverted 2026-07-29 for crowding out the face-based populations).
    Promoted to a config because the ceiling turned out to be the only thing holding this
    layer's size up, and there was no way to lower it.

    What made that visible: on table_gs11 the vertex layer's TRUE rendered size (read back
    from the checkpoint, since the export misreports it -- see the exporter's own note)
    has a minor-axis median of 0.00663 against the face-based population's 0.00111, i.e.
    ~6x. Training had already pushed it from the 0.01025 ceiling down to 0.00875 and
    stopped; the floor was measurably NOT binding (releasing every floor changes the
    rendered aspect from 1.2051 to 1.2055 and nothing else), so min_vertex_scale_frac is
    not the lever here and never was. After gs11 took the face-based on-screen sigma from
    2.50 px to 1.55 px, this layer is what is left: 324,432 rows still visible at ~6x the
    size of everything around them.

    Why shrinking it is close to free, from the definition rather than from a guess: a
    vertex Gaussian's contribution to vert_cov AT ITS OWN ANCHOR is opacity * exp(0) --
    see _compute_coverage_density() -- which does not depend on its size at all. So
    lowering this cannot reduce coverage at the vertex the Gaussian is anchored to. It
    only reduces the cross-contribution to nearby face CENTROIDS, and that contribution is
    a wide, low-opacity smear: coverage bought with blur, which is the same trade gs8
    rejected when it set coverage_lambda to 0. Vertex coverage measured 95.46% at target
    with a median of 3.107 against a target of 1.0 on gs11, so there is room.

    Not a floor and not a guarantee: at values below 1.0 the layer no longer reaches every
    1-ring neighbour, which is the property the isotropic radius (and the ellipse built on
    it) was constructed to provide. Watch the mesh-vertices row of the corner-coverage
    report, not just the appearance."""
    anisotropic_coverage_density: bool = False
    """Measure coverage with a true anisotropic Mahalanobis distance in
    _compute_coverage_density(), instead of collapsing every Gaussian to the isotropic
    sigma_min = min(sx, sy).

    sigma_min was chosen so "anisotropy can't fake coverage", and for a freely-oriented
    detail Gaussian that is a reasonable guard. But it makes the metric structurally unable
    to score an ellipse that genuinely does cover its target: stretch a base plate along
    its triangle's long axis and sigma_min falls, its face drops below
    coverage_floor_target / coverage_rescue_thresh, and the floor re-engages and squeezes
    it back to round. That is a closed loop that returns the base layer to circles no
    matter how the scale bounds are relaxed, so anisotropic_base_floor is not fully
    effective without this. Measuring (u/sx)^2 + (v/sy)^2 in the Gaussian's own frame is
    simply the correct evaluation of the same Gaussian this metric already claims to be
    integrating -- an ellipse still scores low toward a corner it does not reach.

    Requires knowing each Gaussian's in-plane angle, which comes from _inplane_theta() --
    the same value the quats property renders with. Off by default: it changes what
    adaptive_coverage_floor / coverage_rescue_thresh decide on every existing run."""
    scale_reg_exclude_base: bool = False
    """Exclude currently-floored base rows from use_scale_regularization's aspect-ratio
    penalty.

    With anisotropic_base_floor on, a base plate's elongation is deliberate and dictated by
    its triangle's geometry, while scale_reg penalises every in-plane aspect ratio above
    max_gauss_ratio with weight 1.0 across all rows indiscriminately. Leaving it on means
    the loss is actively pushing the base layer back toward the circles this whole change
    exists to eliminate -- and on a mesh this size the base layer dominates the mean, so it
    also drowns out the signal on the detail rows the regulariser is meant for. Scoped to
    _base_floor_mask() rather than is_base: a base row adaptive_coverage_floor has released
    renders at its raw size like any other Gaussian and should be regularised like one."""
    adaptive_coverage_floor: bool = False
    """Apply the base layer's full-face scale/opacity floor (and the vertex layer's scale
    floor) ONLY to faces/vertices that are actually under-covered right now, instead of
    unconditionally to every face on the mesh.

    Added 2026-08-24 after measuring the exported table_gs6 model: 50.9% of all face-based
    Gaussians sat exactly at the base floor and ~32% exactly at the upper_scale ceiling --
    i.e. ~83% of the model was pinned at a hard bound rather than at a size the optimizer
    chose, and 98.1% rendered at opacity > 0.9. On the sign in that scene the base plates
    measured sigma = 2.5 px against 2 px text strokes, so a full-face, single-colour,
    opaque plate was structurally averaging every letter into its background: a Gaussian
    has ONE colour, so a plate spanning both black stroke and white paper can only ever
    converge to the grey mean. Training the base layer's colour harder cannot fix that --
    only letting it SHRINK can, which the unconditional floor forbids.

    The observation that makes the conditional form safe: the floor is only load-bearing
    on faces whose OTHER Gaussians do not already cover them. On a face carrying real
    texture detail, densification has already put enough small Gaussians there to cover it
    outright, so the plate is pure blur with no coverage value. On a smooth or occluded
    face the plate is the only thing holding coverage up -- and there is no detail there to
    lose. So keying the floor off a live coverage measurement gives up nothing: the
    guarantee still holds everywhere it is needed, and the blur disappears everywhere it is
    not.

    Mechanically the guarantee gets STRONGER, not weaker: update_coverage_floor_mask()
    re-measures every refine_every steps (100 by default) instead of the previous
    coverage_rescue_thresh cadence of reassign_face_every (3000), and it measures with
    every floor RELEASED, so a face only loses its floor if it is genuinely covered
    without it. Faces are re-floored the moment that stops being true.

    Per-FACE and per-VERTEX state (self.face_needs_floor / self.vertex_needs_floor), NOT
    per-Gaussian: the mesh is fixed for the whole run, so unlike is_base/is_rescue these
    need no lifecycle maintenance through split/dup/cull/reassign. They are recomputed
    from scratch in load_state_dict() too, so an exported/resumed model reproduces the same
    rendered sizes rather than falling back to all-floors-on.

    Off by default -- existing runs are bit-identical with this False."""
    coverage_floor_target: float = 0.9
    """Coverage density (same measure as coverage_target, see _compute_coverage_density())
    a face's centroid and all 3 of its corners must reach, measured with every floor
    released, before that face's base floor is dropped for the next refine cycle. Only
    meaningful when adaptive_coverage_floor is True.

    Deliberately a HIGH bar, unlike coverage_rescue_thresh's deliberately low one: dropping
    a floor is only safe if the face is comfortably covered without it, so this should sit
    near coverage_target, not near the rescue threshold. Lower it toward ~0.7 to let more
    faces go floor-free (sharper, more gap risk); raise it toward 1.0 to keep more floors
    (safer, blurrier). Never tuned."""
    enable_fold_detection: bool = False
    """Whether to cap each face's size bound (self.radius, see populate_modules) using
    detect_fold_safe_radius() -- catches mesh folds (e.g. an armpit) where topologically
    distant faces are geometrically close, which none of the existing 1-ring/own-corner
    size bounds can see. See detect_fold_safe_radius()'s docstring for the full
    reasoning, including the first (2-ring, fixed-hop-count) version's failure and the
    2026-08-02 same-day redesign to a density-adaptive ratio against each face's own
    R_1ring instead. Still defaulted OFF: the redesign fixes the specific known failure
    mode (dense/small-triangle regions getting misclassified as folds) but has not yet
    been validated on the real mesh -- no local GPU access to test it directly. Turn on
    explicitly via the training command to try it, and check for new coverage holes the
    same way the first version's regression was caught."""
    fold_safety_frac: float = 0.5
    """Fraction of the raw nearest-non-1-ring-neighbor distance a face is allowed to
    reach toward when a fold is detected (see detect_fold_safe_radius()). 0.5 splits the
    gap evenly between both sides of the fold. Never tuned."""
    fold_ratio: float = 1.5
    """How many multiples of a face's own R_1ring a non-1-ring-neighbor face must be
    closer than to count as a fold (see detect_fold_safe_radius()) -- the density-
    adaptive replacement for the first version's fixed 2-ring exclusion. 1.5 is
    deliberately conservative (a regular 2-ring neighbor sits at roughly 2x 1-ring
    distance in a uniform mesh) so ordinary curved-but-unfolded surface stays clear of
    the threshold even where triangulation is a little irregular. Never tuned."""
    min_vertex_scale_frac: float = 0.5
    """Minimum scale as a fraction of each vertex Gaussian's fixed 1-ring-based ceiling (see
    populate_modules' vertex layer). Only meaningful once the vertex layer's scale/rotation
    are made trainable (2026-07-31) -- higher than min_scale_frac's default on purpose: this
    layer exists specifically as coverage insurance, so its floor should protect that role
    rather than allow it to shrink as aggressively as an ordinary detail Gaussian can."""
    use_vertex_layer: bool = True
    """Keep the vertex-anchored filler layer: one Gaussian per mesh vertex, sized to that
    vertex's 1-ring reach, never split/culled/reassigned. True is the historical behaviour
    -- the layer had no switch at all before 2026-08-30 and was always built.

    It exists as coverage insurance (see populate_modules), and it is not free. Measured on
    table_gs11 against one training camera, weighting each Gaussian by the screen area it
    paints (pi * sigma_x_px * sigma_y_px * alpha):

        face-based    4,715,865 rows   69.6% of the painted area   sigma median 1.55 px
        vertex layer    702,813 rows   30.4% of the painted area   sigma median 6.29 px

    13% of the count paints 30% of the image, at four times the size of everything else. On
    a scene whose text strokes are ~2 px wide, that population cannot represent detail and
    covers a third of the frame with what it can represent instead.

    What it buys, also measured (scripts/analyze_gap_coverage.py, the union "no gaps" test,
    on table_gs11 with the layer excluded from the test):

        threshold 0.50    99.507% -> 95.725% covered
        threshold 0.10    99.989% -> 99.153%
        threshold 0.01    99.998% -> 99.745%

    So it is genuinely holding up about a quarter of a percent of the surface at the
    rasterizer threshold -- real, but small, and worth weighing against the third of the
    frame it blurs. Turn it off and MEASURE both sides; do not assume either.

    Off means the parameter dict is never created, which the exporter already keys on
    (hasattr), so the layer disappears from training, rendering, coverage and export
    together. The per-vertex geometry tensors are still built -- they are a few MB and
    other code reads them."""
    use_base_layer: bool = False
    """Give every face one permanent 'base' Gaussian that structurally guarantees the face
    stays covered, instead of relying on gradient-driven mechanisms or periodic rescue
    patches to keep it covered. A base Gaussian: is never culled (any reason), never has
    its face reassigned, is position-pinned to its face's centroid (means/elevate gradients
    have no effect on it), has its in-plane (x,y) rendered size floored at the FULL face
    xyz_radius (so it always reaches its own corners; with upper_scale=1.0 that pins x,y
    exactly), and has its rendered opacity floored at base_opacity_floor. Its color/SH,
    z-thickness, and rotation (2026-08-06: within the same cone_coef tilt limit as
    detail/vertex, previously locked to its face's exact orientation) all train normally,
    and it can still act as a split/dup *source* (children are ordinary non-base
    Gaussians) so detail still densifies on top of it.

    Added 2026-07-17 after every probabilistic/patch mechanism (coverage_lambda,
    coverage_densify_scale, coverage_rescue_thresh) still failed to reach the user's
    requirement of 100% visually gap-free mesh coverage: densification triggers are
    ultimately photometric-gradient-gated, rescue is a periodic trigger with a low bar
    (0.2) whose patches later training can re-shrink/fade, and training's net refinement
    forces (split/dup shrink, cull, opacity binarization) all push coverage down between
    patches. The base layer converts "every face is covered" from an optimization target
    into an invariant that holds by construction at every step. Cost: more overlap
    (explicitly matches the user's overlap-over-gaps priority), base Gaussians in
    high-curvature areas are full-face-sized so fine detail there depends on the detail
    layer on top, and num_faces Gaussians are permanently resident. Membership is tracked
    in self.is_base, which is maintained exactly like gaussians_to_mesh_indices everywhere
    the Gaussian population changes (split/dup/cull/rescue/append/save/load)."""
    base_opacity_floor: float = 0.9
    """Minimum rendered opacity for base-layer Gaussians (see use_base_layer); applied in
    the opacities property with a straight-through clamp, so photometric training can never
    fade a base Gaussian into an effective hole (cull_alpha_thresh can also never catch one
    -- doubly so, since culling explicitly skips the base layer).

    Raised 0.7 -> 0.9 (2026-07-18) as an anti-flicker measure: with alpha compositing,
    whatever shows through a front splat is (1 - its alpha) of the one behind it, so when
    the depth-sort order of two overlapping near-coplanar plates flips with viewing angle,
    the visible color jump is proportional to that leakage -- 30% at 0.7, 10% at 0.9. The
    periodic opacity reset still clamps the RAW logit into [0.5, 0.8]; the rendered floor
    here simply keeps base at 0.9 regardless, which is fine -- base is exempt from the
    cull that reset exists to enable anyway."""
    fold_base_opacity_floor: float = 0.02
    """Rendered opacity floor for base-layer Gaussians on faces detect_fold_safe_radius()
    flagged as folded (see is_fold_face, config.enable_fold_detection), used INSTEAD of
    base_opacity_floor for just those rows. Added 2026-08-03: globally lowering
    base_opacity_floor to let the optimizer fade away redundant fold overlap made the
    WHOLE model visibly more transparent, since every face's base relaxed, not just the
    folded ones -- most of the mesh gained nothing from that (no fold there to resolve)
    and just lost its coverage guarantee for free. Keying the relaxation off is_fold_face
    keeps base_opacity_floor at its normal, coverage-safe value everywhere else. Only
    takes effect when enable_fold_detection is True; is_fold_face is all-False otherwise
    so this is unreachable. Never tuned -- same starting guess as fold_safety_frac's
    counterpart on the size side."""
    init_copies_per_face: int = 4
    """Number of Gaussians to place per face at initialization. 1 = original behavior (centroid
    only). 4 = centroid + near each vertex. More copies give better initial coverage for faces
    that receive little gradient during training (occluded areas, back of model)."""
    copies_scale_shrink_power: float = 0.5
    """When init_copies_per_face > 1, each copy's in-plane (x,y) scale bound is divided by
    init_copies_per_face ** copies_scale_shrink_power, so the N copies on a face partition it
    instead of each one being bounded as if it alone covered the whole face (which causes heavy
    mutual overlap -> redundant, slightly-mismatched colors competing for depth/blend order ->
    flicker and speckle). 0.5 = divide by sqrt(N), area-preserving for uniform circle packing.
    Lower values shrink less, letting each copy reach further into its corner of the triangle
    at the cost of some overlap; raise if coverage still looks fine but speckle persists, lower
    if some triangle corners are left under-covered. 0 disables shrinking entirely."""
    grazing_weight_enabled: bool = True
    """Down-weight the photometric loss for pixels only observed at a grazing angle (surface
    normal nearly perpendicular to the viewing direction). Grazing observations are few in number
    and noisy (foreshortened, stronger specular/Fresnel response), so letting them fully drive
    color fitting makes those Gaussians flicker as the view angle changes."""
    grazing_weight_power: float = 2.0
    """Exponent applied to the clamped normal-vs-view-direction cosine to get the loss weight.
    Higher values fall off faster for oblique views."""
    grazing_weight_floor: float = 0.1
    """Minimum loss weight even at a fully grazing angle, so those pixels still get some gradient."""
    stop_acm_after_split: bool = True
    """Zero out acm_lambda once step >= stop_split_at. Keeping it active during the post-split
    cull-only phase fights the natural pruning of now-redundant Gaussians (it keeps their
    opacity propped up so total accumulation stays near 1), preventing the point count from
    settling down the way it does when acm_lambda is 0."""
    color_consistency_lambda: float = 0.0
    """Weight of the color-consistency loss: penalizes DC-color disagreement between
    Gaussians that spatially overlap, so that a depth-sort flip between two of them
    doesn't visibly change the pixel. 0 disables.

    Added 2026-08-06 after measuring the exported armadillo model to find what actually
    drives the overlap-edge flicker. Two findings drove this:

      * 97.2% of the total flicker potential (sum of a1*a2*|dC| over all Gaussian pairs
        within 2x their in-plane radius) comes from pairs on the SAME surface sheet, not
        from two different body parts overlapping (2.8%). The "arm over torso" framing
        was wrong.
      * Overlapping neighbors disagree badly on color: median |dC| 0.048, p90 0.173 on a
        0-1 scale, while their opacities sit at a median of 1.0. Two near-coincident,
        near-opaque Gaussians with different colors mean whichever sorts in front wins
        the pixel outright, so a sort flip is a hard color jump -- and at grazing angles
        (silhouettes, i.e. exactly the "overlap edges" reported) neighboring Gaussians
        along the surface are at nearly equal depth, so flips happen readily.

    Why the disagreement exists at all: the photometric loss only ever sees the final
    composited pixel, so it constrains whichever Gaussian is in FRONT and leaves the one
    hidden behind free to drift to any color. That drift is invisible until the view
    rotates enough to swap them.

    Since flicker magnitude goes as a1*a2*|dC|, this attacks the |dC| factor directly
    without touching Gaussian count, size, or position -- so it costs no coverage. The
    tradeoff is that pushed too high it also smooths away legitimate high-frequency
    surface detail; start small.

    Covers the base+detail populations (everything indexed by gaussians_to_mesh_indices).
    The vertex layer is excluded -- it isn't face-indexed, and its opacity trains down to
    ~0.007 in practice, so it contributes almost nothing to a1*a2. Only the DC term is
    constrained, not features_rest: DC dominates the color, and leaving the view-dependent
    part free avoids over-constraining genuine specular/angular variation."""
    coverage_lambda: float = 0.0
    """Weight of the point-sampled coverage loss. Evaluates the actual Gaussian density
    (opacity * exp(-d^2 / 2*sigma_min^2)) received at every mesh vertex and every face
    centroid, and penalizes sample points below coverage_target. Unlike a per-face footprint
    *area* budget, this is position-aware: Gaussians piling up at the face center cannot
    satisfy it while the triangle corners stay bare — the gradient pulls Gaussians toward
    (and grows them over) the uncovered spots. Geometric and camera-free, so it also closes
    gaps in occluded regions that photometric loss never sees. 0 disables.

    Note: this loss can only grow/reposition *existing* Gaussians toward a gap, it cannot
    spawn new ones -- see coverage_densify_scale to also let a coverage deficit found here
    lower the bar for triggering actual densification (split_gaussians/dup_gaussians)."""
    coverage_target: float = 0.7
    """Density each sample point (mesh vertex / face centroid) must reach before its
    coverage-loss term is satisfied. 1.0 demands fully opaque coverage everywhere; lower
    values tolerate thinner overlap at corners."""
    coverage_rescue_thresh: float = 0.2
    """Below this coverage density (same measure as coverage_target, see
    _compute_coverage_density()) at ANY of a face's 4 check points -- its centroid or any
    of its 3 corners -- that face gets a fresh Gaussian spawned near whichever point was
    worst, unconditionally -- called from reassign_gaussians_to_nearest_face() (gated by
    reassign_face_every), not gated on photometric gradient at all. Added 2026-07-17: even
    coverage_densify_scale-boosted densify_grad_thresh is still ultimately a
    photometric-gradient trigger, which can in principle stay below threshold forever for a
    genuinely under-observed face (occluded, low-texture) no matter how badly it's covered
    -- this is the deterministic backstop that doesn't depend on gradient at all.

    First version of this (same day) only checked centroid coverage, spawning only at the
    centroid -- corners kept coming up empty regardless, because a face's own Gaussians
    typically sit nearer its interior, so a comfortably-covered centroid routinely masked
    still-empty corners. Checking all 4 points and spawning near the worst one (not always
    the centroid) directly targets whichever part of the triangle is actually gapped.

    Deliberately much lower than coverage_target (0.2 vs typically 1.0): coverage_target is
    a smooth loss target the optimizer nudges every Gaussian toward asymptotically, and
    it's normal/expected for many faces to sit just under it most of the time even when
    genuinely well covered. Reusing that same bar here would fire on nearly every face on
    nearly every reassign_face_every cycle -- a hard, unconditional population-changing
    action, not a soft gradient nudge -- causing runaway Gaussian growth. This threshold is
    meant to catch only faces that are still badly, persistently undercovered, not merely
    short of the aspirational target. A face that's still below this threshold after being
    rescued once gets rescued again next cycle (one Gaussian added per cycle, not
    front-loaded), so persistent gaps close gradually across cycles rather than in one
    shot. Watch the "spawning N gaussian(s)" log line the first time this runs -- if N is
    large relative to num_points, this threshold may need to be lowered further. 0 disables
    (only the old exactly-zero-Gaussians case would ever get rescued, matching pre-2026-07-17
    behavior)."""
    coverage_rescue_every: int = 0
    """Run the deterministic coverage rescue (see coverage_rescue_thresh) every this many
    steps from refinement_after(), instead of only piggybacking on
    reassign_gaussians_to_nearest_face()'s reassign_face_every cadence. 0 keeps the
    original behaviour (rescue only fires when a reassignment happens).

    Added 2026-08-24 alongside adaptive_coverage_floor, which needs it. Once the base
    layer's floor is conditional, the rescue stops being a rarely-reached backstop and
    becomes the mechanism that actually maintains the coverage invariant on faces whose
    floor has been dropped -- and a 3000-step check interval is far too coarse for that
    job: split/dup shrink, cull and opacity binarisation can all open a gap within a
    single refine cycle, so a gap could persist for up to 3000 steps before anything
    noticed. Set this to refine_every (100) so the check runs at the same cadence as
    every other population change.

    Cheap: the measurement is _compute_coverage_density(), which refinement_after()
    already computes every cycle anyway whenever coverage_densify_scale > 0."""
    max_gaussians: int = 0
    """Hard ceiling on the FACE-BASED Gaussian count. 0 disables it, which is the historical
    behaviour. Once the population reaches this, densification and the coverage rescue both
    stop adding; everything else -- culling, reassignment, floor bookkeeping -- keeps
    running, so the run continues improving the Gaussians it already has.

    Counted in the same units as self.num_points, i.e. means.shape[0], which is the
    face-based population ALONE. The vertex filler layer is a separate parameter dict of
    fixed size (one row per mesh vertex, 707,579 on this mesh) and is not included. A value
    set from a total-population budget is therefore too generous by exactly that count --
    size it against the face-based figure the export reports, not the total.

    This exists because the memory budget is empirical and the failure is expensive. On a
    32 GB card at full resolution, table_gs12 completed with 5.48M rows; two later runs
    died of CUDA OOM hours in, and a run here costs about 4.5 hours. Nothing in the model
    previously bounded the population at all -- densification is its only unbounded
    producer -- so any change that reduced culling pressure could and did walk straight
    off that cliff.

    The second of those OOMs is worth recording, because the cause was a change working as
    intended rather than a bug. init_copies_per_face 4 seeds 6.98M rows (1.4M base + 4 per
    face), which is already ABOVE what gs12 finished with: gs12 only fit because opaque
    base plates (base_opacity_floor 0.95) left the detail beneath them unable to earn
    opacity, so ~1.5M rows were culled away early. Lowering base_opacity_floor to let that
    detail survive -- the entire point of the change -- removed the culling pressure the
    seed size had been silently relying on.

    Set it from the measured budget, not from the target: a value the run reaches is a
    signal to reconsider the seed (init_copies_per_face) or the split rate
    (densify_grad_thresh), not something to raise until it stops printing."""
    coverage_rescue_max_count: int = 0
    """Absolute ceiling on how many Gaussians one coverage-rescue cycle may spawn. 0 leaves
    coverage_rescue_max_frac as the only cap, which is the historical behaviour.

    coverage_rescue_max_frac is a fraction of the CURRENT population, so whenever it binds
    on every cycle the population compounds instead of growing: 5% per cycle over the ~26
    cycles between warmup and step 3100 is 1.05^26 = 3.6x. That is latent rather than
    theoretical -- it put a run over 32 GB at step 3101, at HALF resolution, on a config
    whose predecessor fit at FULL resolution with 5.48M rows. It stays harmless only while
    few faces qualify and culling keeps pace, which is the regime coverage_rescue_thresh's
    0.2 default sits in; raise the threshold toward coverage_target and the cap starts
    binding every cycle, at which point the compounding term takes over.

    An absolute count removes that: total addition becomes linear and knowable before the
    run, with max_count * (max_num_iterations / coverage_rescue_every) as the worst case.

    Both caps apply when both are set; the tighter one wins."""
    coverage_rescue_max_frac: float = 0.05
    """Safety cap on the deterministic coverage rescue: at most this fraction of the
    current Gaussian count may be spawned in any single rescue call. When more faces
    qualify than the cap allows, the worst-deficit faces are served first and the rest
    wait for the next cycle (they stay flagged, so nothing is silently dropped).

    Exists because raising coverage_rescue_thresh toward coverage_target -- which is
    exactly what adaptive_coverage_floor asks you to do -- removes the property that made
    the unbounded spawn safe. See coverage_rescue_thresh's docstring: its 0.2 default is
    deliberately far below coverage_target precisely so it fires on almost nothing; at a
    high threshold it can instead fire on nearly every face, every cycle, which is a hard
    population-changing action rather than a soft gradient nudge. This bounds the worst
    case to steady growth instead of a single-cycle explosion. 0 disables the cap."""
    reassign_face_every: int = 3000
    """Every this many steps, re-bind each Gaussian to whichever face of the whole mesh its
    current 3D position is actually closest to (exact nearest-triangle query, see
    _find_nearest_faces() -- no longer restricted to a local neighborhood as of
    2026-07-26), instead of leaving it permanently bound to whichever face it was created
    on. A Gaussian
    can drift away from its birth face's true footprint (elevate offset, or scale growing
    past that face's own extent), especially near tight self-occlusion gaps between two
    different parts of the mesh -- when that happens, coverage_lambda and curvature-weighted
    densification keep crediting the (now wrong) birth face while the face the Gaussian
    actually ended up covering never gets credit, which makes that signal misleading.
    0 disables (original behavior: face binding is permanent from creation).

    Enabled 2026-07-17 (was 0/disabled -- user had chosen the export-only "option A" over
    this "option B" earlier, but coverage measured after a full training run still fell
    short of coverage_target=1.0, and by the end of that run 46.6% of all Gaussians had
    drifted from their birth face -- strong evidence that coverage_lambda/curvature
    weighting were working off substantially stale groupings for a large fraction of
    training, not just a rare edge case). 3000 matches the existing
    reset_alpha_every(30) x refine_every(100) = 3000-step opacity-reset rhythm already in
    refinement_after() -- reusing an interval already proven not to destabilize training,
    rather than introducing an unrelated new one. This is far less frequent than every
    refine_every(100) cycle, so the scale-clamp discontinuity described in
    reassign_gaussians_to_nearest_face()'s docstring (moving to a smaller-xyz_radius face
    reclamps a Gaussian's rendered size, with no change to its raw trained parameter) fires
    only ~8 times total before stop_split_at(25000), not up to 245 times -- and unlike the
    export-only case, training continues afterward so the optimizer has room to adapt each
    time, the same way it already recovers from the discontinuity a fresh split/dup
    creates. Traced dup_in_all_optim()/the empty-face-rescue spawning path in
    reassign_gaussians_to_nearest_face() end-to-end before enabling this -- both look
    consistent with the same patterns split_gaussians/dup_gaussians already exercise -- but
    this combination (periodic reassignment + this mesh + this training config) has never
    actually been run. Watch the first training run closely."""
    freeze_geometry_at_step: int = -1
    """Once step reaches this value, stop training every geometry-related raw parameter
    (means_2d, normal_elevates, scales, quats) by turning off its requires_grad -- only
    features_dc/features_rest/opacities keep training for the remainder of the run.
    -1 disables (geometry trains for the whole run, original behavior).

    Added 2026-07-25 for a two-stage "freeze geometry, then fit appearance alone"
    workflow (same spirit as SGGaussians' separate geometry/appearance training
    stages): position/scale/coverage and color are otherwise trained simultaneously
    the whole time, so color's gradient has to compete with a target (Gaussian
    position/size/count) that's still actively moving, especially during the
    split/dup/cull/reassign churn earlier in training -- letting geometry settle first
    and fitting color against a now-fixed arrangement should give color a cleaner,
    non-moving target to converge against. Intended to be set well after
    stop_split_at, once coverage/size have been tuned to a satisfactory state on their
    own (see pipeline_overview.md).

    Implementation note: cull_gaussians() (and any other path that rebuilds
    self.gauss_params[name] as a fresh nn.Parameter, e.g. split/dup/rescue) resets
    requires_grad to its default (True) on the new tensor, silently undoing a one-time
    freeze. To stay robust across those replacements, refinement_after()
    re-asserts the freeze every refine_every cycle once step >= this value, rather
    than toggling it once -- idempotent and cheap (four requires_grad_ calls), and
    means the window where a rebuilt parameter could sneak in a gradient step is at
    most one refine_every cycle, not the rest of training. Adam simply skips a
    parameter whose .grad stays None (requires_grad=False means autograd never
    populates it), so no optimizer/momentum-state surgery is needed on top of this.
    Brand new, never run."""


class SplatfactoOnMeshUCModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SplatfactoOnMeshUCModelConfig

    def __init__(
        self,
        *args,
        seed_mesh: Optional[Dict] = None,
        **kwargs,
    ):
        self.seed_mesh = seed_mesh
        assert self.seed_mesh is not None, "splatfacto on mesh needs a mesh to run"
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        means = self.seed_mesh["means"]
        num_points = means.shape[0]

        normal_elevates = torch.nn.Parameter(torch.zeros(num_points).float())

        self.xys_grad_norm = None
        self.max_2Dsize = None

        self.mesh_verts = mesh_verts = self.seed_mesh["mesh_verts"].clone().cuda()
        self.mesh_faces = mesh_faces = self.seed_mesh["mesh_faces"].clone().cuda()
        self.mesh_faces_verts = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3).cuda()

        # Per-face local scale: the LARGER of (a) the 1-ring neighbor-average distance
        # (face_ring_radius() -- smooth across the mesh, doesn't blow up on this face's
        # own skinny/degenerate shape, see its docstring) and (b) this face's own
        # centroid-to-vertex average distance (face_centroid_vertex_radius() -- how far
        # a Gaussian centered on this face actually has to reach to cover its own
        # corners). (a) alone reflects how big the *neighbors* are, not this face
        # itself -- a face noticeably bigger or more elongated than its neighborhood
        # could get a size bound too small to cover its own vertices, leaving a gap no
        # matter how init_copies_per_face/upper_scale are set. Taking the max with (b)
        # guarantees the bound is always at least big enough to reach this face's own
        # corners -- deliberately NOT circumcircle_radius for (b), see
        # face_centroid_vertex_radius() docstring: circumradius blows up on thin/
        # irregular triangles (more common in complex/high-curvature mesh regions),
        # which would inflate the bound exactly where a tighter, curvature-conforming
        # Gaussian is wanted and make it visibly poke out past the true surface.
        # User's stated priority is "overlap is fine, gaps are not", so biasing toward
        # the larger of (a)/(b) is the right tradeoff, just with a self-size measure
        # that can't wildly overshoot the way circumradius can. The percentile clamp a
        # few lines below still catches any remaining extreme outlier.
        _cv_radius = face_centroid_vertex_radius(self.mesh_faces_verts)
        _ring_radius = face_ring_radius(self.mesh_faces, self.mesh_faces_verts)
        self.radius = torch.maximum(
            _ring_radius,
            _cv_radius,
        ).reshape(-1, 1).clone()

        # Edge-adjacent face pairs, for color_consistency_lambda (see its docstring).
        # Precomputed once here rather than per-step: the mesh topology is fixed for the
        # whole run, and only which Gaussians sit on which face changes. Kept even when
        # the loss is disabled -- it's two (E,) index tensors, negligible next to the
        # per-Gaussian state, and this keeps append_from_mesh/load bookkeeping uniform.
        self.face_adj_a, self.face_adj_b = face_adjacency_pairs(self.mesh_faces)
        # Cap self.radius at mesh folds (2026-08-02, redesigned same day -- see
        # detect_fold_safe_radius()'s docstring for the fixed-hop-count version that
        # preceded this and why it had to be replaced): the max() above only ever
        # reasons about topological neighbors, so it has no way to notice when a face's
        # true nearest surface in 3D space is a topologically distant, unrelated part of
        # the mesh (e.g. an armpit). Root-caused by comparing against DRAWER-main, which
        # had no scale floor at all and so could let its optimizer freely shrink away
        # this kind of forced overlap -- this project's coverage-guarantee machinery
        # (min_scale_frac, base layer, vertex/centroid filler layers) structurally
        # cannot, so capping the bound itself is the only remaining lever. torch.minimum
        # only ever tightens self.radius, never widens it, so this is a no-op everywhere
        # the mesh doesn't fold (and, with the ratio-based redesign, everywhere a nearby
        # non-1-ring face is merely a normal consequence of dense triangulation rather
        # than a genuine fold).
        # The fold search feeds TWO independent consumers now, so run it if EITHER wants
        # it and let each take only what it asked for (it's a chunked all-pairs centroid
        # search over every face -- doing it twice would be pure waste):
        #   * enable_fold_detection -> radius cap + is_fold_face (opacity floor)
        #   * color_consistency_lambda -> the fold face PAIRS (see _set_face_overlap_pairs)
        _need_fold_search = self.config.enable_fold_detection or self.config.color_consistency_lambda > 0
        if _need_fold_search:
            _fold_cap, _is_fold, _nearest_foreign = detect_fold_safe_radius(
                self.mesh_faces, self.mesh_faces_verts, _ring_radius,
                self.config.fold_safety_frac, self.config.fold_ratio,
            )
        if self.config.enable_fold_detection:
            self.is_fold_face = _is_fold
            self.radius = torch.minimum(self.radius, _fold_cap)
        else:
            # All-False so the opacities property's per-face floor lookup (see
            # config.fold_base_opacity_floor) doesn't need to re-check the config flag
            # itself -- degrades to "every face uses base_opacity_floor" for free.
            # NOTE: deliberately stays all-False even when the search above DID run for
            # color_consistency_lambda's sake. is_fold_face's only job is gating the
            # per-face opacity relaxation, and that must keep keying off
            # enable_fold_detection alone -- otherwise turning on the color loss would
            # silently drop folded faces' base opacity floor from base_opacity_floor to
            # fold_base_opacity_floor (0.7 -> 0.02 with current settings), a large and
            # completely unrelated behavior change.
            self.is_fold_face = torch.zeros(self.mesh_faces.shape[0], dtype=torch.bool, device=self.mesh_faces.device)
        # Fold pairs use the raw geometric detection regardless of enable_fold_detection:
        # whether two surfaces face each other across a gap is a fact about the mesh, not
        # about whether we chose to cap radii because of it.
        self._set_face_overlap_pairs(
            _is_fold if _need_fold_search else None,
            _nearest_foreign if _need_fold_search else None,
        )
        # Each face's OWN corner reach (centroid-to-vertex average), kept separately from
        # self.radius/xyz_radius: the base layer's in-plane scale floor keys off this
        # (just enough to guarantee reaching this face's own corners) rather than the
        # 1-ring-max radius above (usually much larger than the face itself, which made
        # every base Gaussian reach deep into its neighbors' territory -- heavy
        # base-vs-base overlap between near-coplanar semi-transparent plates whose blend
        # order flips with viewing angle, i.e. the severe flicker reported 2026-07-18).
        self.cv_radius = _cv_radius.reshape(-1, 1).clone()
        # Per-face MAX corner distance (not mean) -- used only to widen the base
        # layer's floor on the minority of faces where the mean underestimates the
        # reach actually needed for the single farthest corner (irregular/obtuse
        # triangles). Applying max globally (replacing cv_radius entirely) does fix
        # that under-coverage, but reintroduces base-vs-base overlap/blur across the
        # WHOLE mesh (tried 2026-07-24, reverted 2026-07-25 -- see
        # pipeline_overview.md), because every face's base floor grows, not just the
        # ones that actually need it. Restricting the wider floor to just the
        # irregular faces keeps that fix local instead of global.
        _centroid = self.mesh_faces_verts.mean(dim=1, keepdim=True)
        _cv_radius_max = torch.norm(self.mesh_faces_verts - _centroid, dim=-1).max(dim=-1).values
        # Threshold: a face counts as "irregular enough" once its farthest corner is
        # more than 30% beyond the mean of all three -- an equilateral triangle scores
        # exactly 1.0 here, so 1.3 catches meaningfully lopsided triangles without
        # widening the (large majority, roughly-regular) rest of the mesh. Not tuned/
        # validated against an actual coverage measurement; a starting guess.
        _irregular = (_cv_radius_max / _cv_radius.clamp(min=1e-12)) > 1.3
        self.cv_radius_base_floor = torch.where(_irregular, _cv_radius_max, _cv_radius).reshape(-1, 1).clone()
        self.xyz_radius = self.radius.clone().repeat(1, 3)
        self.xyz_radius[:, 2] *= self.config.face_flat_coef
        # All init_copies_per_face Gaussians on a face share the same face radius, so without
        # this, every copy's in-plane scale bound is sized as if it alone covered the whole
        # face. With N>1 that means each copy can grow to the full face size, so they end up
        # heavily overlapping (redundant, slightly-mismatched colors competing for depth order
        # -> flicker/speckle across the surface, not just at silhouettes). Shrink the x,y bound
        # per copy so N of them partition the face instead of each claiming all of it.
        if self.config.init_copies_per_face > 1 and self.config.copies_scale_shrink_power > 0:
            self.xyz_radius[:, :2] /= self.config.init_copies_per_face ** self.config.copies_scale_shrink_power
        # Clamp xyz_radius: even the 1-ring radius can still be large at isolated
        # irregular spots (e.g. a small cluster of degenerate faces neighboring each
        # other). Cap at 99th-pct × 5 as a safety net.
        _r99 = torch.quantile(self.radius.squeeze(), 0.99).item()
        self.xyz_radius = torch.clamp(self.xyz_radius, max=_r99 * 5.0)

        # Un-shrunk in-plane reference for the base layer's coverage floor (see
        # config.base_floor_reaches_corners). Same 99th-pct x 5 safety clamp as
        # xyz_radius above, but WITHOUT the copies_scale_shrink_power division: that
        # division exists so N ordinary copies partition a face between them, which is
        # the right rule for a detail Gaussian and the wrong one for the single plate
        # whose entire job is to span the face by itself.
        self.base_floor_xy = torch.clamp(
            self.cv_radius_base_floor, max=_r99 * 5.0
        ).expand(-1, 2).contiguous()

        v_a = self.mesh_faces_verts[:, 0]
        v_b = self.mesh_faces_verts[:, 1]
        v_c = self.mesh_faces_verts[:, 2]

        self.mesh_triangles_edge_ab = v_b - v_a
        self.mesh_triangles_edge_bc = v_c - v_b
        self.mesh_triangles_edge_ca = v_a - v_c

        self.mesh_triangles_edge_len_a = torch.linalg.norm(self.mesh_triangles_edge_bc, ord=2, dim=-1)
        self.mesh_triangles_edge_len_b = torch.linalg.norm(self.mesh_triangles_edge_ca, ord=2, dim=-1)
        self.mesh_triangles_edge_len_c = torch.linalg.norm(self.mesh_triangles_edge_ab, ord=2, dim=-1)

        self.mesh_triangles_2d_coords_a, self.mesh_triangles_2d_coords_b, self.mesh_triangles_2d_coords_c = compute_triangle_vertices(self.mesh_triangles_edge_len_a, self.mesh_triangles_edge_len_b, self.mesh_triangles_edge_len_c)
        means_2d_coords = (self.mesh_triangles_2d_coords_a + self.mesh_triangles_2d_coords_b + self.mesh_triangles_2d_coords_c) / 3
        means_2d = torch.nn.Parameter(means_2d_coords)

        # Elliptical base-layer floor (see config.anisotropic_base_floor). Precomputed here
        # rather than in the scales property because it depends only on the mesh, which is
        # fixed for the whole run -- the same treatment cv_radius / base_floor_xy get above.
        # It has to live down here rather than beside them because it needs the face-local
        # 2D triangle coords, which are only built a few lines up.
        #
        # Stored already divided by base_floor_corner_sigma, so the tensor IS the floor:
        # semi_axis / k places each vertex exactly k sigma out along that axis.
        _steiner_axes, _steiner_theta = steiner_ellipse_axes(
            self.mesh_triangles_2d_coords_a,
            self.mesh_triangles_2d_coords_b,
            self.mesh_triangles_2d_coords_c,
            max_aspect=self.config.base_floor_max_aspect,
        )
        # base_ellipse_theta is the face's Steiner major-axis angle in its own 2D frame,
        # which is the frame gauss_params["quats"][:, 0] is measured in -- see
        # _inplane_theta(), which pins base rows to it so their local x axis (the axis
        # base_floor_xy_aniso[:, 0] bounds) actually points along the ellipse's major axis.
        self.base_ellipse_theta = _steiner_theta.reshape(-1).contiguous()
        # Same 99th-pct x 5 safety clamp as xyz_radius / base_floor_xy above, and for the
        # same reason (isolated clusters of degenerate faces), just applied per-axis here.
        self.base_floor_xy_aniso = torch.clamp(
            _steiner_axes / max(self.config.base_floor_corner_sigma, 1e-6), max=_r99 * 5.0
        ).contiguous()
        if self.config.anisotropic_base_floor:
            # Report what the new floor actually costs on THIS mesh, at startup, so the
            # tradeoff is visible before 30k iterations rather than inferred from the
            # export afterwards. The two ratios are against the isotropic floors this
            # replaces: cv_radius (today's default, which undershoots the far corner) and
            # cv_radius_base_floor (base_floor_reaches_corners, which reaches it by
            # inflating every plate). Both are footprint AREA, i.e. how much of a
            # neighbour's territory each base plate claims -- the quantity that turns into
            # overlap, depth-order flicker and blur.
            _asp = _steiner_axes[:, 0] / _steiner_axes[:, 1].clamp(min=1e-12)
            _area_ell = _steiner_axes[:, 0] * _steiner_axes[:, 1]
            _area_mean = self.cv_radius.squeeze(-1) ** 2
            _area_max = self.cv_radius_base_floor.squeeze(-1) ** 2
            _capped = float((_asp >= self.config.base_floor_max_aspect - 1e-6).float().mean())
            CONSOLE.log(
                f"anisotropic base floor: aspect median {_asp.median().item():.2f} "
                f"q90 {_asp.quantile(0.90).item():.2f} q99 {_asp.quantile(0.99).item():.2f}, "
                f"{100.0 * _capped:.2f}% at the {self.config.base_floor_max_aspect:.0f}:1 cap; "
                f"footprint area vs cv_radius circle {(_area_ell / _area_mean.clamp(min=1e-12)).median().item():.3f}x, "
                f"vs corner-reaching circle {(_area_ell / _area_max.clamp(min=1e-12)).median().item():.3f}x "
                f"(corners reached at {self.config.base_floor_corner_sigma:.2f} sigma either way)"
            )


        if self.config.unconstrained_scale:
            # Start every Gaussian at its maximum allowed size (upper_scale x
            # xyz_radius) instead of deriving an initial guess from k-NN neighbor
            # spacing. A Gaussian only grows beyond a small initial guess via
            # photometric gradient during training, and that gradient is weak or
            # absent in occluded/low-texture/rarely-observed regions -- if it starts
            # small there, it never grows, leaving that area under-covered no matter
            # how long training runs (this is the same root cause as the persistent
            # "coverage isn't complete" symptom discussed throughout this project).
            # Starting at the ceiling, combined with init_copies_per_face for density,
            # gets full coverage right from initialization instead of depending on
            # training to discover it. Training can still shrink individual copies
            # down (as low as min_scale_frac) where a smaller footprint fits better;
            # this only changes the starting point, not the trainable range.
            scales = torch.nn.Parameter(torch.log(self.config.upper_scale * self.xyz_radius.clone() + 1e-20))
        else:
            scales = torch.nn.Parameter(torch.zeros(num_points, 3).float())


        self.normals = self.seed_mesh["normals"].clone().cuda()
        self.normals = torch.nn.functional.normalize(self.normals, dim=-1, p=2)
        self.faces_axis_x = torch.nn.functional.normalize(self.mesh_triangles_edge_ab, dim=-1, p=2)
        self.faces_axis_y = torch.cross(self.normals, self.faces_axis_x, dim=-1).reshape(-1, 3)
        self.faces_axis_y = torch.nn.functional.normalize(self.faces_axis_y, dim=-1, p=2)

        # axis_x_norm_value = torch.sum(self.faces_axis_x * self.faces_axis_x, dim=-1)
        # print("self.faces_axis_x: ", axis_x_norm_value.mean(), axis_x_norm_value.max(), axis_x_norm_value.min())
        # axis_y_norm_value = torch.sum(self.faces_axis_y * self.faces_axis_y, dim=-1)
        # print("self.faces_axis_y: ", axis_y_norm_value.mean(), axis_y_norm_value.max(), axis_y_norm_value.min())
        # assert False

        rot_mat = torch.stack([self.faces_axis_x, self.faces_axis_y, self.normals], dim=2).cuda()
        self.faces_quats = matrix_to_quaternion(rot_mat)



        # quats = torch.nn.Parameter(self.faces_quats.clone())
        quats = torch.nn.Parameter(torch.zeros(num_points, 3).float())
        self.dim_sh = dim_sh = num_sh_bases(self.config.sh_degree)

        shs = torch.zeros((self.seed_mesh["features_dc"].shape[0], dim_sh, 3)).float().cuda()
        if self.config.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(self.seed_mesh["features_dc"].clone())
            shs[:, 1:, 3:] = 0.0
        else:
            CONSOLE.log("use color only optimization with sigmoid activation")
            shs[:, 0, :3] = torch.logit(self.seed_mesh["features_dc"].clone(), eps=1e-10)
        features_dc = torch.nn.Parameter(shs[:, 0, :])
        self.features_dc_dim = features_dc.shape[-1]
        features_rest = torch.nn.Parameter(shs[:, 1:, :])
        self.features_rest_dims = features_rest.shape[-2:]


        opacities = torch.nn.Parameter(torch.logit(self.config.init_opacity * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means_2d": means_2d,
                "normal_elevates": normal_elevates,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.gaussians_to_mesh_indices = torch.arange(num_points, device="cuda")

        # Place N Gaussians per face at different positions for better initial coverage.
        # Faces with low image gradient (occluded areas, back of model) never receive
        # densification, so a single centroid Gaussian leaves most of the face uncovered.
        # With N=4: centroid + one position near each vertex guarantees full coverage.
        _N = self.config.init_copies_per_face
        if _N > 1:
            _nf = self.mesh_faces.shape[0]
            _a = self.mesh_triangles_2d_coords_a   # (nf, 2)
            _b = self.mesh_triangles_2d_coords_b
            _c = self.mesh_triangles_2d_coords_c
            _fixed_w = torch.tensor([
                [1/3, 1/3, 1/3],   # centroid
                [0.6, 0.2, 0.2],   # near vertex 0
                [0.2, 0.6, 0.2],   # near vertex 1
                [0.2, 0.2, 0.6],   # near vertex 2
            ], dtype=torch.float32, device=_a.device)[:min(_N, 4)]  # (min(N,4), 3)
            _bary_w = _fixed_w.unsqueeze(1).expand(-1, _nf, -1)  # (min(N,4), nf, 3)
            if _N > 4:
                # Copies beyond the 4 fixed positions: per-face uniform random points
                # inside the triangle (parallelogram-fold trick).
                _uv = torch.rand(_N - 4, _nf, 2, device=_a.device)
                _flip = _uv.sum(dim=-1) > 1
                _uv[_flip] = 1 - _uv[_flip]
                _rand_w = torch.cat([_uv, 1 - _uv.sum(dim=-1, keepdim=True)], dim=-1)
                _bary_w = torch.cat([_bary_w, _rand_w], dim=0)  # (N, nf, 3)
            # Build (nf*N, 2) means_2d in face-major order: face0_copy0, face0_copy1, ...
            _pos = (
                _bary_w[..., 0:1] * _a.unsqueeze(0) +
                _bary_w[..., 1:2] * _b.unsqueeze(0) +
                _bary_w[..., 2:3] * _c.unsqueeze(0)
            )  # (N, nf, 2)
            _pos = _pos.permute(1, 0, 2).reshape(-1, 2)  # (nf*N, 2)
            # Replicate all params (repeat_interleave gives face-major order: [f0c0,f0c1,...,f1c0,...])
            for _pname in list(self.gauss_params.keys()):
                self.gauss_params[_pname] = torch.nn.Parameter(
                    self.gauss_params[_pname].data.repeat_interleave(_N, dim=0)
                )
            self.gauss_params["means_2d"] = torch.nn.Parameter(_pos)
            self.gaussians_to_mesh_indices = torch.arange(_nf, device="cuda").repeat_interleave(_N)

        # Base-layer membership (see config.use_base_layer): copy 0 of each face's seeds
        # -- the centroid copy (_fixed_w row 0); with init_copies_per_face=1 that's every
        # seed. Deliberately a plain tensor (not a Parameter/buffer): persisted through
        # gaussian_on_mesh_extra_info.pt alongside gaussians_to_mesh_indices (see
        # get_gaussian_param_groups / load_state_dict) and kept in sync with every
        # operation that grows/shrinks the Gaussian population, exactly like
        # gaussians_to_mesh_indices is. All-False when the feature is disabled, so every
        # is_base-aware code path degrades to the original behavior without re-checking
        # the config flag.
        _n_total = self.gaussians_to_mesh_indices.shape[0]
        if self.config.use_base_layer:
            self.is_base = (torch.arange(_n_total, device="cuda") % max(_N, 1)) == 0
        else:
            self.is_base = torch.zeros(_n_total, dtype=torch.bool, device="cuda")
        # Per-Gaussian coverage-rescue membership (see config.coverage_rescue_thresh
        # and the scales property): none of the initial seed Gaussians are rescue
        # spawns, so this starts all-False, same lifecycle discipline as is_base --
        # maintained through cull/split/dup/append/save/load.
        self.is_rescue = torch.zeros(_n_total, dtype=torch.bool, device="cuda")
        # One-cull-cycle grace period for cull_at_floor_gaussians (2026-07-29): a row
        # whose face just changed (reassign_gaussians_to_nearest_face(), or a fresh
        # coverage-rescue spawn) can have its RENDERED scale reclamped purely by the
        # relabel -- moving to a face with a larger xyz_radius makes the same absolute
        # scale look relatively tiny against the new (larger) min_scale_frac floor, with
        # no actual shrinking having happened (see reassign_gaussians_to_nearest_face's
        # "SCALE-CLAMP CAVEAT" docstring, which assumed only "training has room to adapt
        # over subsequent steps" -- true before cull_at_floor_gaussians existed, but that
        # assumption breaks if the very next cull_gaussians() call (as little as
        # refine_every steps later) can permanently delete the row before any such
        # adaptation happens). Transient, not persisted: set True only in
        # _reproject_to_new_faces() and the coverage-rescue spawn below, read once by
        # cull_gaussians() to exempt those rows from the at-floor check, then reset to
        # all-False at the end of that same cull_gaussians() call. Lifecycle otherwise
        # maintained exactly like is_base/is_rescue (grown/filtered at every population
        # change) so indices never desync.
        self.skip_floor_cull = torch.zeros(_n_total, dtype=torch.bool, device="cuda")

        # Adaptive coverage floor state (see config.adaptive_coverage_floor). Indexed by
        # FACE / VERTEX, not by Gaussian: the mesh is fixed for the whole run, so unlike
        # is_base/is_rescue/skip_floor_cull these two need no maintenance in
        # split/dup/cull/reassign/append/save/load -- nothing that changes the Gaussian
        # population can change their size or meaning.
        #
        # Start all-True (every floor engaged), which is exactly the pre-2026-08-24
        # behaviour: a floor is only ever dropped by update_coverage_floor_mask() actually
        # measuring that face as covered without it. So every window where the mask is
        # stale -- before warmup_length, between refine cycles, immediately after a resume
        # -- fails safe toward coverage rather than toward sharpness.
        self.face_needs_floor = torch.ones(
            self.mesh_faces.shape[0], dtype=torch.bool, device="cuda"
        )
        self.vertex_needs_floor = torch.ones(
            self.mesh_verts.shape[0], dtype=torch.bool, device="cuda"
        )

        # Precompute per-face curvature: deviation of each face normal from the
        # smoothed vertex normal (average of adjacent face normals).
        # High value = sharp/curved region (nose, fingers, edges).
        _nf = self.mesh_faces.shape[0]
        _nv = self.mesh_verts.shape[0]
        _dev = self.mesh_verts.device
        _vn_sum = torch.zeros(_nv, 3, device=_dev)
        _vn_cnt = torch.zeros(_nv, device=_dev)
        _face_normals = self.normals[:_nf]
        for _i in range(3):
            _vn_sum.scatter_add_(0, self.mesh_faces[:, _i].unsqueeze(1).expand(-1, 3), _face_normals)
            _vn_cnt.scatter_add_(0, self.mesh_faces[:, _i],
                                 torch.ones(_nf, device=_dev))
        _vn_avg = torch.nn.functional.normalize(
            _vn_sum / _vn_cnt.unsqueeze(1).clamp(min=1), dim=1)
        _face_vn_avg = _vn_avg[self.mesh_faces]              # [F, 3, 3]
        _face_cos = (_face_normals.unsqueeze(1) * _face_vn_avg).sum(dim=-1).mean(dim=1)  # [F]
        self.face_curvature = (1.0 - _face_cos).clamp(min=0).detach()  # 0=flat, large=curved

        # Vertex-anchored coverage-filler Gaussians (2026-07-26): one permanent
        # Gaussian per mesh vertex, entirely independent of the face-based
        # gaussians_to_mesh_indices system (never culled, never split/duplicated,
        # never reassigned -- population size is always exactly _nv). Purpose: a
        # per-face-independent architecture structurally cannot guarantee a shared
        # mesh vertex is covered without either (a) every incident face's own
        # Gaussian independently reaching that vertex -- which is exactly the
        # "several independent Gaussians converging at the same point" geometry
        # already identified as this project's hardest, mathematically-unavoidable
        # flicker source (see pipeline_overview.md's cross-face-overlap analysis), or
        # (b) a single shared entity at that vertex instead (this). Same idea as
        # SGGaussians' vertex-anchored geometry Gaussians (see pipeline_overview.md's
        # SGGaussians comparison), applied narrowly here as a coverage-only filler
        # layer on top of the existing base+detail system, not a replacement for it.
        #
        # Deliberately much simpler than the face-based Gaussians: position, scale,
        # and rotation are FIXED (plain tensors, not nn.Parameter) -- pinned exactly
        # at the vertex, sized off the vertex's own 1-ring neighbor distance (same
        # idea as SGGaussians' compute_vertex_radii_approx), oriented to the
        # averaged vertex normal (_vn_avg, already computed above for curvature).
        # Only color and opacity train, so the optimizer can fade a vertex Gaussian
        # toward invisible wherever the existing per-face system already covers that
        # spot well, and keep it visible only where it's actually filling a real gap
        # -- this is the "overshoot is fine, appearance must not suffer" tradeoff the
        # user asked for: geometric overlap with neighboring face-based Gaussians is
        # allowed by construction, but nothing forces these to stay opaque where
        # they're not earning their keep.
        self.vertex_positions = self.mesh_verts.clone()  # (V, 3), fixed world position

        # 1-ring neighbor distance per vertex (max, not mean/average -- see
        # SGGaussians comparison: max guarantees reaching the single farthest
        # neighbor, which is what "cover this vertex's local neighborhood" needs).
        _v_edges = torch.cat([
            self.mesh_faces[:, [0, 1]],
            self.mesh_faces[:, [1, 2]],
            self.mesh_faces[:, [2, 0]],
        ], dim=0)  # (3F, 2)
        _v_edges = torch.cat([_v_edges, _v_edges.flip(1)], dim=0)  # undirected, both directions
        _edge_len = torch.norm(
            self.mesh_verts[_v_edges[:, 1]] - self.mesh_verts[_v_edges[:, 0]], dim=-1
        )
        _vertex_radius = torch.zeros(_nv, device=_dev)
        _vertex_radius = _vertex_radius.scatter_reduce(
            0, _v_edges[:, 0], _edge_len, reduce="amax", include_self=False
        )
        # Safety margin (2026-07-27, set to 1.3; reverted to 1.0 on 2026-07-29): the
        # 1.3x overshoot was added because the exact 1-ring max distance left no slack
        # for gaps sitting between vertices or in a small/dense triangle's interior
        # (reported via a coverage-visualization tool showing ~0.3% remaining gaps,
        # concentrated in high-curvature areas). But the user then observed the 1.3x
        # vertex layer alone already covers most of each face's area, crowding out any
        # reason for the face-based (base/detail) and centroid-anchored (see section 37
        # of pipeline_overview.md) populations to exist -- exactly the failure mode this
        # margin was risking. Reverted to 1.0 (exact 1-ring reach, no overshoot) so the
        # vertex layer goes back to covering only the immediate area around each vertex,
        # leaving the triangle interior to the face-based/centroid populations. The
        # centroid-anchored filler layer (added since the 1.3x margin, also section 37)
        # now independently covers the "hole in a triangle's interior, away from any
        # vertex" case that 1.3x was patched in to handle, so this reach is no longer
        # this layer's job alone.
        # Was the hardcoded _VERTEX_RADIUS_SAFETY_MARGIN; now config.vertex_radius_scale,
        # whose docstring carries the history above plus why lowering it below 1.0 is
        # cheap in coverage and expensive in nothing else. 1.0 reproduces this line's
        # previous behaviour exactly.
        self.vertex_radius = (
            _vertex_radius.clamp(min=1e-6) * self.config.vertex_radius_scale
        ).reshape(-1, 1)  # (V, 1)

        # self.vertex_log_scales is built a little further down, below the tangent frame
        # rather than here beside the radius it comes from: under
        # config.anisotropic_vertex_floor the in-plane bound is an ellipse expressed in
        # that frame's axes, so the frame has to exist first. Same treatment, and for the
        # same reason, as base_floor_xy_aniso sitting below the face-local 2D coords.

        # Arbitrary-but-consistent tangent frame from the averaged vertex normal
        # (Gram-Schmidt against a reference axis, same spirit as faces_quats' x/y
        # axes but a face has a natural edge to use for axis_x; a vertex doesn't, so
        # any consistent in-plane frame works). This is the FIXED base frame that
        # get_outputs()'s vertex quats computation rotates on top of (2026-07-31) --
        # same role as faces_quats for the face-based populations.
        self.vertex_normals = _vn_avg  # (V, 3)
        _ref = torch.tensor([0.0, 1.0, 0.0], device=_dev).expand(_nv, 3)
        _nearly_parallel = torch.abs((_vn_avg * _ref).sum(dim=-1)) > 0.99
        _ref = torch.where(
            _nearly_parallel.unsqueeze(-1),
            torch.tensor([1.0, 0.0, 0.0], device=_dev).expand(_nv, 3),
            _ref,
        )
        _vaxis_x = torch.nn.functional.normalize(torch.cross(_ref, _vn_avg, dim=-1), dim=-1)
        _vaxis_y = torch.nn.functional.normalize(torch.cross(_vn_avg, _vaxis_x, dim=-1), dim=-1)
        self.vertex_quats = matrix_to_quaternion(
            torch.stack([_vaxis_x, _vaxis_y, _vn_avg], dim=2)
        )  # (V, 4), fixed, already unit-norm
        # Kept as tensors, not just folded into the quaternion above: the elliptical
        # in-plane bound below is expressed in these two axes, and
        # _compute_coverage_density() needs them again to measure how far a vertex
        # Gaussian reaches toward a face centroid along the axis it is actually wide on.
        self.vertex_axis_x = _vaxis_x  # (V, 3), fixed
        self.vertex_axis_y = _vaxis_y  # (V, 3), fixed

        # In-plane size bound for the vertex layer (see config.anisotropic_vertex_floor).
        # _vertex_scales() reads self.vertex_log_scales as its CEILING and
        # min_vertex_scale_frac x that as its FLOOR, so making this anisotropic makes both
        # anisotropic -- which is the whole point: a row pinned at either bound is what
        # renders as a circle today.
        _vertex_xyz_radius = self.vertex_radius.repeat(1, 3)
        if self.config.anisotropic_vertex_floor:
            # Offsets to every 1-ring neighbour, in each source vertex's own tangent
            # frame. _v_edges is already both-directions (built for the isotropic radius
            # above), so each vertex sees its full ring.
            _e_src, _e_dst = _v_edges[:, 0], _v_edges[:, 1]
            _e_off = self.mesh_verts[_e_dst] - self.mesh_verts[_e_src]  # (E, 3)
            _e_off_2d = torch.stack([
                (_e_off * _vaxis_x[_e_src]).sum(dim=-1),
                (_e_off * _vaxis_y[_e_src]).sum(dim=-1),
            ], dim=-1)  # (E, 2)
            _vtx_axes, _vtx_theta = one_ring_ellipse_axes(
                _e_src, _e_off_2d, _nv, max_aspect=self.config.vertex_floor_max_aspect
            )
            # config.vertex_radius_scale has to be applied HERE as well, not only to
            # self.vertex_radius above. one_ring_ellipse_axes() measures the raw mesh
            # 1-ring, so it returns full reach regardless of the scale, and the line below
            # overwrites the in-plane columns of _vertex_xyz_radius wholesale -- without
            # this multiply, vertex_radius_scale silently governs only the z column
            # whenever this config is on, i.e. it does nothing to the footprint it exists
            # to shrink. (Exactly that happened on table_gs12, which trained 30k steps at
            # scale 0.6 with the in-plane size unchanged. The startup log below is the
            # tell: its area ratio is measured against vertex_radius, which IS scaled, so
            # the mismatch showed up as 2.644x where gs11 read 0.952x.)
            _vtx_axes = _vtx_axes * self.config.vertex_radius_scale
            # Same 99th-pct x 5 safety clamp the face-based radii get, per axis, against
            # isolated clusters of degenerate 1-rings. The min matches the clamp
            # self.vertex_radius already carries, so a vertex with no incident edge at all
            # lands on the same lower bound here as it does on the isotropic path instead
            # of collapsing to a point.
            _vr99 = torch.quantile(self.vertex_radius.reshape(-1).float(), 0.99)
            _vtx_axes = torch.clamp(_vtx_axes, min=1e-6, max=(_vr99 * 5.0).item())
            self.vertex_ellipse_theta = _vtx_theta.reshape(-1).contiguous()
            _vertex_xyz_radius[:, :2] = _vtx_axes
            # z is deliberately left on the isotropic vertex_radius: this change is about
            # the in-plane footprint, and plate thickness was never part of the reach
            # guarantee (same reasoning as the base layer leaving z on min_scale_frac).
            _asp = _vtx_axes[:, 0] / _vtx_axes[:, 1].clamp(min=1e-12)
            _area_ell = _vtx_axes[:, 0] * _vtx_axes[:, 1]
            _area_circ = self.vertex_radius.reshape(-1) ** 2
            _capped = float((_asp >= self.config.vertex_floor_max_aspect - 1e-6).float().mean())
            CONSOLE.log(
                f"anisotropic vertex floor: aspect median {_asp.median().item():.2f} "
                f"q90 {_asp.quantile(0.90).item():.2f} q99 {_asp.quantile(0.99).item():.2f}, "
                f"{100.0 * _capped:.2f}% at the {self.config.vertex_floor_max_aspect:.0f}:1 cap; "
                f"radius scale {self.config.vertex_radius_scale:.2f}; "
                f"footprint area vs the 1-ring circle {(_area_ell / _area_circ.clamp(min=1e-12)).median().item():.3f}x "
                f"(both carry that scale; a ratio near 1/scale^2 instead of near 1 means it "
                f"reached only one of the two)"
            )
        else:
            # Unchanged isotropic bound: one radius in both in-plane axes, which is what
            # forces sx == sy on every floored or ceilinged row.
            self.vertex_ellipse_theta = torch.zeros(_nv, device=_dev)
        _vertex_xyz_radius[:, 2] = self.vertex_radius.reshape(-1) * self.config.face_flat_coef
        self.vertex_log_scales = torch.log(_vertex_xyz_radius + 1e-20)  # (V, 3), fixed

        # Seed color: average of the ORIGINAL per-face seed texture color (see
        # populate_modules' features_dc seeding above) across each vertex's
        # incident faces -- same scatter pattern as _vn_sum/_vn_cnt just above,
        # reusing _vn_cnt as the per-vertex incident-face count.
        _vc_sum = torch.zeros(_nv, 3, device=_dev)
        _seed_rgb = self.seed_mesh["features_dc"].to(_dev)  # (F, 3), still per-ORIGINAL-face here
        for _i in range(3):
            _vc_sum.scatter_add_(0, self.mesh_faces[:, _i].unsqueeze(1).expand(-1, 3), _seed_rgb)
        _vertex_seed_rgb = (_vc_sum / _vn_cnt.unsqueeze(1).clamp(min=1)).clamp(1e-6, 1 - 1e-6)

        _vertex_shs = torch.zeros(_nv, self.dim_sh, 3, device=_dev)
        if self.config.sh_degree > 0:
            _vertex_shs[:, 0, :3] = RGB2SH(_vertex_seed_rgb)
        else:
            _vertex_shs[:, 0, :3] = torch.logit(_vertex_seed_rgb, eps=1e-10)

        # Only built when config.use_vertex_layer is on. Everything downstream keys on
        # the attribute existing -- the exporter already did (hasattr), and the render,
        # optimizer-group and coverage paths below now do too -- so leaving it uncreated
        # removes the layer from every one of them at once rather than in five places
        # that could drift apart.
        self.vertex_gauss_params = None if not self.config.use_vertex_layer else torch.nn.ParameterDict({
            "features_dc": torch.nn.Parameter(_vertex_shs[:, 0, :]),
            "features_rest": torch.nn.Parameter(_vertex_shs[:, 1:, :]),
            # Start at a moderate 0.5 (not the usual init_opacity face-based init): unlike a
            # freshly seeded face-based Gaussian, this one starts already overlapping
            # a fully-formed base+detail layer, so there's no "empty face" period
            # where it alone needs to carry coverage -- let photometric gradient
            # decide whether it should become more or less visible from a neutral
            # starting point, rather than assuming it's needed at full strength.
            "opacities": torch.nn.Parameter(torch.logit(0.5 * torch.ones(_nv, 1, device=_dev))),
            # Trainable scale/rotation (2026-07-31): self.vertex_log_scales/self.vertex_quats
            # above stay as the FIXED reference (ceiling for scale, base frame for rotation)
            # -- these raw params start exactly AT that reference (scales) or at zero offset
            # (quats), so initial rendered behavior is unchanged from the fully-fixed version;
            # training can only move away from there. See get_outputs() for how they're
            # rendered (clamped scale, cone-limited tilt -- same pattern as the face-based
            # scales/quats properties, just keyed off vertex_radius/vertex_quats instead of
            # xyz_radius/faces_quats).
            "scales": torch.nn.Parameter(self.vertex_log_scales.clone()),
            "quats": torch.nn.Parameter(torch.zeros(_nv, 3, device=_dev)),
        })

        # Face-centroid coverage-filler Gaussians -- REMOVED 2026-08-06 at the user's
        # request, to test whether base+detail+vertex alone can reach 100% coverage
        # without this fourth layer. This layer existed (2026-07-27) because a coverage
        # tool showed the vertex layer + a 1.3x radius margin still left ~0.3% of gaps in
        # a triangle's INTERIOR (away from any vertex, which the vertex layer structurally
        # cannot reach). Two things have changed since that measurement, though, which is
        # why this is worth re-testing rather than assumed to reopen the same gap: (1)
        # the vertex layer's scale/rotation became trainable (2026-07-31), so it can grow
        # within its 1-ring ceiling to reach farther into a face's interior than it could
        # when fully fixed; (2) _compute_coverage_density() was just made vertex-aware
        # (2026-08-05), so coverage_rescue_thresh can now detect and deterministically
        # patch any interior gap that does reopen, instead of being structurally blind to
        # it. If gaps reappear, the fix is a face-centroid-anchored population identical
        # in spirit to the vertex layer above: fixed position at
        # self.mesh_faces_verts.mean(dim=1), fixed size from self.cv_radius (with a
        # safety margin), trainable color/opacity only, never split/duplicated/culled.

        if self.config.gaussian_save_extra_info_path is None:
            self.save_extra_info_path = os.path.join(self.seed_mesh["mesh_dir"], "gaussian_on_mesh_extra_info.pt")
        else:
            self.save_extra_info_path = self.config.gaussian_save_extra_info_path
            os.makedirs(os.path.dirname(self.config.gaussian_save_extra_info_path), exist_ok=True)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity('alex', normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        self.articulate_transform = None
        self.visible_gs_indices = None

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    def _clip_means_2d_to_face(self, means_2d, face_idx):
        """
        Clip a set of face-local 2D points into their own face's triangle, using the
        exact same barycentric clip-and-renormalize the means property applies before
        ever turning means_2d into a world-space position. means_2d itself is an
        unconstrained per-Gaussian nn.Parameter -- nothing stops the raw value from
        drifting outside its assigned face's triangle (training gradient, or a stale
        reassignment -- see _reproject_to_new_faces()) -- so any code that wants "the
        position this Gaussian actually renders at" must go through this, not read
        gauss_params["means_2d"] directly. Straight-through: the forward value is the
        clamped point, but gradient still flows to the raw means_2d as if unclamped, so
        callers that need gradient (e.g. get_loss_dict()'s coverage_lambda) still work.

        Args:
            means_2d: (N, 2) raw face-local coordinates (NOT necessarily gathered by
                gaussians_to_mesh_indices -- caller decides, see face_idx).
            face_idx: (N,) which row of mesh_triangles_2d_coords_* each point belongs
                to (normally self.gaussians_to_mesh_indices, but kept as a parameter
                since callers may already have it gathered for other reasons).
        """
        triangles_2d_coords_a = self.mesh_triangles_2d_coords_a[face_idx]
        triangles_2d_coords_b = self.mesh_triangles_2d_coords_b[face_idx]
        triangles_2d_coords_c = self.mesh_triangles_2d_coords_c[face_idx]

        means_2d_bary_coords = barycentric_coordinates(
            means_2d,
            triangles_2d_coords_a,
            triangles_2d_coords_b,
            triangles_2d_coords_c
        )
        means_2d_bary_coords = torch.clip(means_2d_bary_coords, 0, 1)
        means_2d_bary_coords = means_2d_bary_coords / torch.sum(means_2d_bary_coords, dim=-1).reshape(-1, 1)

        means_2d_limited = torch.sum(means_2d_bary_coords.reshape(-1, 3, 1) * torch.stack([triangles_2d_coords_a, triangles_2d_coords_b, triangles_2d_coords_c], dim=1), dim=1)
        means_2d_limited = means_2d_limited.reshape(-1, 2)

        return means_2d + means_2d_limited.detach() - means_2d.detach()

    @property
    def means(self):

        # if self.articulate_means:
        #     return self.external_means.to(self.gauss_params["bary_coords"].device)

        means_2d = self._clip_means_2d_to_face(
            self.gauss_params["means_2d"], self.gaussians_to_mesh_indices
        )

        means_2d_transformed = torch.sum(
            means_2d.unsqueeze(-1) * torch.stack([self.faces_axis_x[self.gaussians_to_mesh_indices], self.faces_axis_y[self.gaussians_to_mesh_indices]], dim=1), dim=1).reshape(-1, 3)
        means = means_2d_transformed + self.mesh_faces_verts[self.gaussians_to_mesh_indices][:, 0]

        normals = self.normals[self.gaussians_to_mesh_indices]
        radius = self.radius[self.gaussians_to_mesh_indices]

        if self.config.unconstrained_elevate:
            elevate = self.gauss_params["normal_elevates"].reshape(-1, 1)
            maximum_elevate = radius.reshape(-1, 1) * self.config.elevate_coef
            # Non-base rows can be confined to the outward side of the face so they are not
            # buried behind the base plate -- see config.detail_elevate_min_frac. The
            # default (-1.0) is the historical symmetric bound. is_base rows keep the
            # symmetric range because they are position-pinned at zero offset anyway.
            if self.config.detail_elevate_min_frac > -1.0:
                _lo = torch.full_like(radius.reshape(-1, 1), -self.config.elevate_coef)
                _lo[~self.is_base] = self.config.detail_elevate_min_frac
                minimum_elevate = radius.reshape(-1, 1) * _lo
            else:
                minimum_elevate = -radius.reshape(-1, 1) * self.config.elevate_coef
            elevate_limited = torch.where(elevate < maximum_elevate, elevate, maximum_elevate)
            elevate_limited = torch.where(elevate_limited > minimum_elevate, elevate_limited, minimum_elevate)
            elevate = elevate_limited.detach() + elevate - elevate.detach()
            elevate = normals * elevate

        else:
            normal_elevates = torch.sigmoid(self.gauss_params["normal_elevates"]) - 0.5
            elevate = normals * normal_elevates.reshape(-1, 1) * radius.reshape(-1, 1)

        means += elevate

        if self.config.use_base_layer and bool(self.is_base.any()):
            # Base-layer Gaussians are position-pinned to their face's 3D centroid with
            # zero normal offset (see config.use_base_layer): torch.where routes gradient
            # only through the non-base branch, so a base Gaussian's means_2d /
            # normal_elevates parameters receive zero gradient and simply stay at their
            # seeded values -- position is guaranteed by construction, not by hoping the
            # optimizer leaves it alone.
            base_centroids = self.mesh_faces_verts[self.gaussians_to_mesh_indices].mean(dim=1)
            means = torch.where(self.is_base.unsqueeze(-1), base_centroids, means)

        if self.articulate_transform is not None:
            transform_indices_list = self.articulate_transform['transform_indices_list']
            transform_matrix_list = self.articulate_transform['transform_matrix_means_list']

            for transform_indices, transform_matrix in zip(transform_indices_list, transform_matrix_list):
                transform_indices = transform_indices.to(means.device)
                transform_matrix = transform_matrix.to(means.device)

                means_to_transform = means[transform_indices]
                means_to_transform = torch.nn.functional.pad(means_to_transform, (0, 1), "constant", 1.0)
                means_to_transform = means_to_transform @ transform_matrix.T
                means_to_transform = means_to_transform[:, :3] / means_to_transform[:, 3:]

                means = means.reshape(-1).masked_scatter(transform_indices.reshape(-1, 1).repeat(1, 3).reshape(-1), means_to_transform.reshape(-1)).reshape(-1, 3)

        return means

    def _rescue_widened_xyz_radius(self):
        """
        self.xyz_radius[self.gaussians_to_mesh_indices], with rows marked self.is_rescue
        given the full, undivided per-face radius (self.radius) as their in-plane (x, y)
        reference instead of the copies_scale_shrink_power-divided value every ordinary
        N-way partitioning copy on a face shares.

        Coverage-rescue Gaussians (see config.coverage_rescue_thresh) are spawned
        specifically to plug a face that's still under-covered after everything else;
        sizing them off the same shrunk ceiling/floor as an ordinary copy can leave them
        too small to actually close that gap. Unlike the base layer (which widens the
        floor for EVERY Gaussian on its face), this only touches the specific rows
        marked is_rescue -- the other copies on the same face keep their normal shrunk
        ceiling, so this can't reintroduce the widespread overlap/blur a global ceiling
        change caused (2026-07-24/25, see pipeline_overview.md).

        Shared by the scales property and the already_at_floor checks in
        split_gaussians()/dup_gaussians(): both need the SAME effective radius rescue
        rows are actually rendered against, or the already-at-floor check would compare
        a rescue row's (correctly widened) rendered size against the wrong (narrow,
        un-widened) floor reference and never recognize it as already at its own floor.
        Added 2026-07-26, never run.
        """
        xyz_radius = self.xyz_radius[self.gaussians_to_mesh_indices]
        if bool(self.is_rescue.any()):
            _rescue_radius_xy = self.radius[self.gaussians_to_mesh_indices].expand(-1, 2)
            xyz_radius = xyz_radius.clone()
            xyz_radius[:, :2] = torch.where(
                self.is_rescue.unsqueeze(-1), _rescue_radius_xy, xyz_radius[:, :2]
            )
        return xyz_radius

    def _base_floor_mask(self):
        """
        Which rows actually get the base layer's full-face scale/opacity floor applied
        THIS cycle.

        Without adaptive_coverage_floor this is just self.is_base -- every base row, every
        step, exactly as before. With it on, a base row only keeps its floor while its own
        face is still measured as under-covered (see update_coverage_floor_mask()); on a
        face that is already covered by its ordinary detail Gaussians the base row is
        released and behaves like any other Gaussian: free to shrink toward min_scale_frac
        and free to fade. It stays exempt from culling and stays position-pinned either
        way, so the face never loses its permanent occupant -- only the size/opacity
        clamps are conditional, never membership.

        Single source of truth on purpose: the scales property, the opacities property and
        split_gaussians()'s shrink guard must all agree on which rows are floored, or a row
        could be shrunk in the raw parameter while still rendering at a floor it can never
        get back under (or vice versa).
        """
        if not self.config.adaptive_coverage_floor:
            return self.is_base
        return self.is_base & self.face_needs_floor[self.gaussians_to_mesh_indices]

    def _inplane_theta(self):
        """
        The in-plane rotation angle each face-based Gaussian actually renders with,
        expressed in its face's own 2D frame.

        Single source of truth, in the same spirit as _base_floor_mask(): the quats
        property turns this into a rotation, and _compute_coverage_density() needs the very
        same angle to work out how far an anisotropic Gaussian reaches toward one specific
        corner. If those two ever disagreed, coverage would be measured for an ellipse
        pointing somewhere other than where the Gaussian is drawn -- a discrepancy that
        would be invisible in the logs and would corrupt every floor/rescue decision.

        With anisotropic_base_floor on, base rows are pinned to their face's Steiner major
        axis. This is exactly parallel to the means property pinning base rows to the face
        centroid, and rests on the same reasoning: the base plate's entire job is to span
        its own triangle, so its orientation is a property of that triangle rather than
        something the optimizer should have to rediscover per face. torch.where routes
        gradient only through the non-base branch, so a base row's raw quats[:, 0] receives
        no gradient and simply stays at its seeded value -- the alignment is guaranteed by
        construction, not by hoping the optimizer leaves it alone.

        The pin is not optional under anisotropic_base_floor and is deliberately not a
        separate config: base_floor_xy_aniso bounds the local x and y axes separately, so
        it only describes the intended ellipse while local x points along that ellipse's
        major axis. An unpinned angle would leave the two silently inconsistent -- a floor
        shaped for one orientation applied to a Gaussian sitting at another, which is
        strictly worse than the isotropic floor it replaces.

        Only the IN-PLANE angle is pinned. Out-of-plane tilt (quats[:, 1:], bounded by
        cone_coef) stays fully trainable, so the 2026-08-06 change that freed base rotation
        to resolve base plates crossing at sharp dihedral angles keeps working: that fix
        needs tilt, which an in-plane spin could never have provided anyway.
        """
        thetas = self.gauss_params["quats"][:, 0]
        if (
            self.config.anisotropic_base_floor
            and self.config.use_base_layer
            and bool(self.is_base.any())
        ):
            pinned = self.base_ellipse_theta[self.gaussians_to_mesh_indices]
            thetas = torch.where(self.is_base, pinned, thetas)
        return thetas

    @property
    def scales(self):
        if self.config.unconstrained_scale:
            real_scales = torch.exp(self.gauss_params["scales"])
            xyz_radius = self._rescue_widened_xyz_radius()
            scale_limit = self.config.upper_scale * xyz_radius
            real_scales_limited = torch.where(real_scales <= scale_limit, real_scales, scale_limit)
            real_scales = real_scales_limited.detach() + real_scales - real_scales.detach()
            if self.config.min_scale_frac > 0 or self.config.use_base_layer:
                scale_floor = self.config.min_scale_frac * xyz_radius
                if self.config.use_base_layer:
                    # Base-layer Gaussians' in-plane floor is their face's OWN
                    # centroid-to-vertex reach -- the minimum that still guarantees the
                    # base covers its own face's corners. Originally this floor was the
                    # full xyz_radius (the max-with-1-ring-average radius), which is
                    # usually much larger than the face itself: every base reached deep
                    # into neighboring faces, and that heavy overlap between
                    # near-coplanar semi-transparent plates (whose depth-sort order flips
                    # with viewing angle) was the dominant cause of the severe flicker
                    # reported 2026-07-18. Floor-at-own-reach keeps the coverage
                    # guarantee while letting photometric training shrink base overlap
                    # back out; the ceiling (upper_scale x xyz_radius) is unchanged, so a
                    # base can still GROW into its neighbors where that genuinely helps.
                    # min() with the ceiling guards the pathological case of the
                    # 99th-pct clamp in populate_modules pushing xyz_radius below
                    # cv_radius. z keeps the ordinary min_scale_frac floor -- the base
                    # should stay a thin plate; thickness is not part of the guarantee.
                    #
                    # self.cv_radius_base_floor (not self.cv_radius): mean-based cv_radius
                    # everywhere except the minority of irregular/obtuse faces where mean
                    # underestimates the reach needed for the single farthest corner --
                    # those faces use the max-based distance instead, see its definition
                    # in populate_modules for why this is scoped to just those faces
                    # rather than applied to the whole mesh (2026-07-26).
                    if self.config.anisotropic_base_floor:
                        # Elliptical floor: [:, 0] bounds the Steiner major axis and
                        # [:, 1] the minor one, which lines up with the Gaussian's local
                        # x / y axes precisely because _inplane_theta() pins this row's
                        # in-plane angle to that same major axis. Takes precedence over
                        # base_floor_reaches_corners -- both answer "how far must a base
                        # plate reach to cover its own face", and this one answers it per
                        # direction instead of with a single radius. See
                        # config.anisotropic_base_floor.
                        _base_floor_xy = self.base_floor_xy_aniso[self.gaussians_to_mesh_indices]
                    elif self.config.base_floor_reaches_corners:
                        _base_floor_xy = self.base_floor_xy[self.gaussians_to_mesh_indices]
                    else:
                        _base_floor_xy = torch.minimum(
                            self.cv_radius_base_floor[self.gaussians_to_mesh_indices], xyz_radius[:, :2]
                        )
                    # _base_floor_mask(), not is_base: with adaptive_coverage_floor on,
                    # base rows whose face is already covered without them keep only the
                    # ordinary min_scale_frac floor, so they can shrink to sub-pixel like
                    # any detail Gaussian. See config.adaptive_coverage_floor.
                    _floored_rows = self._base_floor_mask()
                    scale_floor[:, :2] = torch.where(
                        _floored_rows.unsqueeze(-1), _base_floor_xy, scale_floor[:, :2]
                    )
                real_scales_floored = torch.maximum(real_scales, scale_floor)
                real_scales = real_scales_floored.detach() + real_scales - real_scales.detach()
            return torch.log(real_scales + 1e-20)


        else:
            upper_scale = self.config.upper_scale
            local_scales = torch.sigmoid(self.gauss_params["scales"])
            xyz_radius = self.xyz_radius[self.gaussians_to_mesh_indices]
            scales = torch.log(local_scales * xyz_radius * upper_scale + 1e-20)

            return scales

        # return self.gauss_params["scales"]

    @property
    def quats(self):

        # _inplane_theta(), not the raw parameter: base rows may be pinned to their
        # face's Steiner major axis (see anisotropic_base_floor). Identical to the raw
        # value whenever that is off.
        thetas = self._inplane_theta().reshape(-1, 1)
        xy_rotation_quats = axis_angle_to_quaternion(torch.nn.functional.pad(thetas, (2, 0), "constant", 0.0))

        alphas = self.gauss_params["quats"][:, 1]
        phis = self.gauss_params["quats"][:, 2]

        phis_limited = torch.clip(phis, 0.0, self.config.cone_coef)
        phis = phis + phis_limited.detach() - phis.detach()

        z_rotation_axis = torch.nn.functional.pad(torch.stack([torch.cos(alphas), torch.sin(alphas)], dim=-1), (0, 1), "constant", 0.0)

        z_rotation_quats = axis_angle_to_quaternion(z_rotation_axis * phis.reshape(-1, 1))

        faces_quats = self.faces_quats[self.gaussians_to_mesh_indices]
        quats = quaternion_multiply(faces_quats, quaternion_multiply(z_rotation_quats, xy_rotation_quats))

        # Base-layer rotation made trainable 2026-08-06 (within the same cone_coef tilt
        # limit as detail/vertex, via the composition above -- no separate override here
        # anymore). Was previously locked to the face's exact orientation via
        # torch.where(is_base, faces_quats, quats), which also zeroed gradient into the
        # base rows' raw quats offsets. That lock is what forced two adjacent base plates
        # at a sharp mesh dihedral angle to physically cross -- each was rigidly pinned to
        # its own face's plane with no room to tilt away from the other, regardless of how
        # steep the fold was. Letting the optimizer tilt within the existing cone limit
        # gives it a way to resolve that crossing directly, instead of only being solvable
        # by remeshing (e.g. Loop subdivision) to reduce the dihedral angle beforehand.
        # self.gauss_params["quats"] already has a row for every face-based Gaussian
        # (base and detail share the same tensor, split only by the is_base mask) and
        # starts at all-zeros for every row, so this doesn't change the rendered value at
        # step 0 -- only what training is allowed to do with it afterward. Base's
        # POSITION stays pinned (see the means property above): only orientation is
        # freed, so the "every face permanently has one Gaussian at its centroid"
        # coverage guarantee is untouched.

        if self.articulate_transform is not None:
            transform_indices_list = self.articulate_transform['transform_indices_list']
            transform_matrix_list = self.articulate_transform['transform_matrix_quats_list']

            for transform_indices, transform_matrix in zip(transform_indices_list, transform_matrix_list):
                transform_indices = transform_indices.to(quats.device)
                transform_matrix = transform_matrix.to(quats.device)

                quats_to_transform = quats[transform_indices]
                quats_to_transform_mat = quaternion_to_matrix(quats_to_transform)
                quats_to_transform_mat = transform_matrix @ quats_to_transform_mat
                quats_to_transform = matrix_to_quaternion(quats_to_transform_mat)

                quats = quats.reshape(-1).masked_scatter(transform_indices.reshape(-1, 1).repeat(1, 4).reshape(-1), quats_to_transform.reshape(-1)).reshape(-1, 4)

        return quats

    def _vertex_scales(self):
        """
        Trainable-within-bounds rendered scale for the vertex-anchored filler layer
        (see populate_modules) -- 2026-07-31, was fully fixed before. Same two-step
        straight-through clamp as the scales property above, just keyed off
        vertex_radius instead of xyz_radius. Ceiling is the ORIGINAL fixed value
        (self.vertex_log_scales, computed once in populate_modules from the 1-ring
        distance) -- training can only shrink from there, never grow past it, so
        this can't reintroduce the over-large-vertex-layer overlap/flicker problem
        the 1.3x-margin revert (2026-07-29) fixed. Floor is
        config.min_vertex_scale_frac of that same ceiling.

        Both bounds inherit whatever shape vertex_log_scales has, so this function needs
        no anisotropy branch of its own: with config.anisotropic_vertex_floor on, that
        tensor holds the 1-ring ellipse's two semi-axes instead of one radius twice, and
        the ceiling and the floor become elliptical together. That is what stops a row
        pinned at either bound from being a circle by construction -- see that config's
        docstring, and _vertex_inplane_theta() for the rotation pin the elliptical bound
        is inseparable from.
        """
        raw = torch.exp(self.vertex_gauss_params["scales"])
        ceiling = torch.exp(self.vertex_log_scales)
        limited = torch.where(raw <= ceiling, raw, ceiling)
        raw = limited.detach() + raw - raw.detach()
        floor = self.config.min_vertex_scale_frac * ceiling
        if self.config.adaptive_coverage_floor:
            # Same conditional treatment as the base layer (see _base_floor_mask): this
            # layer is coverage insurance, so its floor is only load-bearing at vertices
            # that are still short WITHOUT it. Measured on table_gs6 it was the single
            # blurriest population in the model -- 707,579 Gaussians at a median sigma of
            # 5.3 px (0.71x its own ceiling, i.e. nowhere near this floor, it simply never
            # shrank) with 40% at opacity > 0.9 -- because coverage_lambda's gradient
            # pushes sigma and opacity up on it and, unlike the face-based population, it
            # can never split, so one Gaussian has to span its whole 1-ring alone.
            # Released rows fall back to 0 (only the ceiling still binds), so photometric
            # loss can finally shrink them where the face-based population already covers
            # that vertex.
            floor = torch.where(
                self.vertex_needs_floor.unsqueeze(-1), floor, torch.zeros_like(floor)
            )
        floored = torch.maximum(raw, floor)
        raw = floored.detach() + raw - raw.detach()
        return torch.log(raw + 1e-20)

    def _vertex_inplane_theta(self):
        """
        The in-plane rotation angle each vertex Gaussian actually renders with, expressed
        in its own tangent frame. The vertex-layer counterpart of _inplane_theta(), and
        the single source of truth for the same reason: _vertex_quats() turns this into a
        rotation and _compute_coverage_density() needs the identical angle to work out how
        far an elliptical vertex Gaussian reaches toward a face centroid. If the two ever
        disagreed, coverage would be measured for an ellipse pointing somewhere other than
        where the Gaussian is drawn, and every vertex_needs_floor decision downstream would
        be made on that wrong number.

        With anisotropic_vertex_floor on, every vertex row is pinned to its 1-ring
        ellipse's major axis. Unlike _inplane_theta() there is no mask here: that function
        pins only the base rows out of a tensor it shares with the detail layer, whereas
        this layer is entirely vertex rows, so the pin is unconditional when the config is
        on -- so it returns the pinned angle outright rather than selecting per row. That
        tensor carries no grad, so vertex_gauss_params["quats"][:, 0] simply receives none
        and stays at its seeded value; the alignment holds by construction rather than by
        the optimizer choosing to leave it alone, same as the base pin.

        Only the IN-PLANE angle is pinned; out-of-plane tilt stays trainable within
        cone_coef, so this layer keeps the freedom it was given in 2026-07-31.
        """
        if self.config.anisotropic_vertex_floor:
            return self.vertex_ellipse_theta
        return self.vertex_gauss_params["quats"][:, 0]

    def _vertex_quats(self):
        """
        Trainable rotation for the vertex-anchored filler layer -- 2026-07-31, was
        fully fixed before. Exact same in-plane-spin + cone-limited-tilt composition
        as the quats property above, just rotating on top of self.vertex_quats (the
        fixed Gram-Schmidt tangent frame from populate_modules) instead of
        faces_quats. No is_base-style override: every row here is eligible.

        The in-plane angle comes from _vertex_inplane_theta(), not the raw parameter:
        under anisotropic_vertex_floor it is pinned to the 1-ring ellipse's major axis.
        Identical to the raw value whenever that config is off.
        """
        raw_q = self.vertex_gauss_params["quats"]
        thetas = self._vertex_inplane_theta().reshape(-1, 1)
        xy_rotation_quats = axis_angle_to_quaternion(torch.nn.functional.pad(thetas, (2, 0), "constant", 0.0))

        alphas = raw_q[:, 1]
        phis = raw_q[:, 2]
        phis_limited = torch.clip(phis, 0.0, self.config.cone_coef)
        phis = phis + phis_limited.detach() - phis.detach()

        z_rotation_axis = torch.nn.functional.pad(
            torch.stack([torch.cos(alphas), torch.sin(alphas)], dim=-1), (0, 1), "constant", 0.0
        )
        z_rotation_quats = axis_angle_to_quaternion(z_rotation_axis * phis.reshape(-1, 1))

        return quaternion_multiply(self.vertex_quats, quaternion_multiply(z_rotation_quats, xy_rotation_quats))

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        if self.visible_gs_indices is not None:
            # Articulation/masked-rendering path (not used during training): deliberately
            # NOT base-floored -- this mask exists to make non-visible Gaussians fully
            # transparent, and flooring base rows back up would defeat it.
            opacities = self.gauss_params["opacities"]
            opacities_mask = torch.zeros((opacities.shape[0], 1), device="cuda", dtype=torch.float32)
            opacities_mask[self.visible_gs_indices] = 1.0
            opacities = opacities * opacities_mask + torch.logit(torch.tensor([1e-6]).reshape(1, 1)).to("cuda").float() * (1 - opacities_mask)
            return opacities
        else:
            opacities = self.gauss_params["opacities"]
            if self.config.use_base_layer and bool(self.is_base.any()):
                # Straight-through floor (same pattern as the scales property): base-layer
                # rows render at >= base_opacity_floor no matter where photometric
                # training pushes the raw logit, so a base Gaussian can never fade into an
                # effective hole. Gradient still flows to the raw parameter.
                # NOTE for writers: this property no longer always returns the Parameter
                # itself -- code that mutates opacity data must go through
                # self.gauss_params["opacities"].data, not self.opacities.data (the
                # opacity-reset block in refinement_after was updated accordingly).
                # Per-Gaussian floor (2026-08-03): faces detect_fold_safe_radius() flagged
                # as folded use fold_base_opacity_floor instead of base_opacity_floor, so
                # the optimizer has room to fade away redundant fold overlap ONLY there --
                # globally lowering base_opacity_floor made the whole model transparent,
                # since every face's base relaxed, not just the folded ones. is_fold_face
                # is all-False when enable_fold_detection is off, so this degrades to the
                # original single-floor behavior for free.
                is_fold_gauss = self.is_fold_face[self.gaussians_to_mesh_indices]
                floor_value = torch.where(
                    is_fold_gauss,
                    torch.full_like(is_fold_gauss, self.config.fold_base_opacity_floor, dtype=opacities.dtype),
                    torch.full_like(is_fold_gauss, self.config.base_opacity_floor, dtype=opacities.dtype),
                )
                floor_logit = torch.logit(floor_value).unsqueeze(-1)
                # _base_floor_mask(), not is_base -- see config.adaptive_coverage_floor.
                # The opacity floor has to be released on exactly the same rows the scale
                # floor is: a released base row that shrank to sub-pixel but stayed pinned
                # at base_opacity_floor would be a hard opaque dot with nothing behind it
                # to justify the opacity.
                floored = torch.where(
                    self._base_floor_mask().unsqueeze(-1),
                    torch.maximum(opacities, floor_logit),
                    opacities,
                )
                opacities = floored.detach() + opacities - opacities.detach()
            return opacities

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000

        newp = dict["gauss_params.scales"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))

        if os.path.exists(self.save_extra_info_path):
            print("extra info is loaded from: ", self.save_extra_info_path)
            extra_info = torch.load(self.save_extra_info_path)
            self.gaussians_to_mesh_indices = extra_info["gaussians_to_mesh_indices"].cuda()
            if "is_base" in extra_info:
                self.is_base = extra_info["is_base"].cuda()
            else:
                # Extra-info file from before the base layer existed: no membership info,
                # treat everything as non-base (matches pre-base-layer behavior).
                self.is_base = torch.zeros(newp, dtype=torch.bool, device="cuda")
            if "is_rescue" in extra_info:
                self.is_rescue = extra_info["is_rescue"].cuda()
            else:
                # Extra-info file from before coverage-rescue ceiling tracking existed.
                self.is_rescue = torch.zeros(newp, dtype=torch.bool, device="cuda")
            # skip_floor_cull is purely transient (reset every cull_gaussians() call,
            # see its docstring in populate_modules), never persisted to extra_info --
            # a resumed run hasn't just reassigned anything, so all-False is the
            # correct starting state, not something to recover from a saved file.
            self.skip_floor_cull = torch.zeros(newp, dtype=torch.bool, device="cuda")
        super().load_state_dict(dict, **kwargs)
        # Rebuild the adaptive coverage floor mask from the state we just restored, rather
        # than persisting it: it is a pure function of the Gaussians + mesh, so recomputing
        # is both simpler and immune to the extra_info file drifting out of sync with the
        # checkpoint. Without this, an exported or resumed model would keep populate_modules'
        # conservative all-floors-on mask and silently reinstate a full-face opaque plate on
        # every face that training had released -- i.e. the export would be visibly blurrier
        # than the model that produced it.
        if self.config.adaptive_coverage_floor:
            if self.gaussians_to_mesh_indices.shape[0] != self.gauss_params["scales"].shape[0]:
                # Mismatched extra_info (see get_gaussian_param_groups' skip_save note):
                # leave the mask conservative rather than crashing here -- this is not the
                # right place to surface that error, and all-floors-on is always safe.
                CONSOLE.log(
                    "[yellow]adaptive coverage floor: gaussians_to_mesh_indices "
                    f"({self.gaussians_to_mesh_indices.shape[0]}) does not match the loaded "
                    f"checkpoint ({self.gauss_params['scales'].shape[0]}); keeping every floor "
                    "engaged[/yellow]"
                )
            else:
                self.update_coverage_floor_mask()

    def append_from_mesh(self, texture_mesh_path):

        mesh = load_objs_as_meshes([texture_mesh_path], device='cpu')
        mesh_verts = mesh.verts_packed().clone().reshape(-1, 3)
        mesh_faces = mesh.faces_packed().clone().reshape(-1, 3)

        normals = mesh.faces_normals_packed().clone().reshape(-1, 3)

        N_Gaussians = mesh_faces.shape[0]
        triangles = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3)
        means = torch.mean(triangles, dim=1)
        radius = circumcircle_radius(triangles)

        pix_to_face = torch.arange(N_Gaussians)
        bary_coords = torch.ones(N_Gaussians, 3) / 3

        Mesh_Fragments = namedtuple("Mesh_Fragments", ['pix_to_face', 'bary_coords'])
        mesh_fragments = Mesh_Fragments(
            pix_to_face=pix_to_face.reshape(1, 1, N_Gaussians, 1),
            bary_coords=bary_coords.reshape(1, 1, N_Gaussians, 1, 3),
        )

        area_to_subdivide = self.config.mesh_area_to_subdivide
        n_subdivision_max_iter = 4
        subdivision_iter = 0
        while True:
            areas = area(triangles)
            if torch.all(areas <= area_to_subdivide) or subdivision_iter >= n_subdivision_max_iter:
                break
            face_to_subdivide = (areas > area_to_subdivide)

            mesh_faces_subdivided = mesh_faces[face_to_subdivide]

            triangles_subdivided = mesh_verts[mesh_faces_subdivided.reshape(-1)].reshape(-1, 3, 3)

            mesh_verts_added = torch.cat([
                (triangles_subdivided[:, 0] + triangles_subdivided[:, 1]) / 2,
                (triangles_subdivided[:, 0] + triangles_subdivided[:, 2]) / 2,
                (triangles_subdivided[:, 1] + triangles_subdivided[:, 2]) / 2,
            ], dim=0)

            num_verts_before = mesh_verts.shape[0]
            num_subdivided_faces = triangles_subdivided.shape[0]

            verts_a_idxs = num_verts_before + torch.arange(num_subdivided_faces)
            verts_b_idxs = num_verts_before + num_subdivided_faces + torch.arange(num_subdivided_faces)
            verts_c_idxs = num_verts_before + num_subdivided_faces * 2 + torch.arange(num_subdivided_faces)

            verts_0_idxs = mesh_faces_subdivided[:, 0]
            verts_1_idxs = mesh_faces_subdivided[:, 1]
            verts_2_idxs = mesh_faces_subdivided[:, 2]

            faces_0ab = torch.stack([verts_0_idxs, verts_a_idxs, verts_b_idxs], dim=-1)
            faces_1ca = torch.stack([verts_1_idxs, verts_c_idxs, verts_a_idxs], dim=-1)
            faces_2bc = torch.stack([verts_2_idxs, verts_b_idxs, verts_c_idxs], dim=-1)
            faces_acb = torch.stack([verts_a_idxs, verts_c_idxs, verts_b_idxs], dim=-1)

            bary_coords_to_subdivide = bary_coords[face_to_subdivide]
            weight_0 = bary_coords_to_subdivide[:, 0]
            weight_1 = bary_coords_to_subdivide[:, 1]
            weight_2 = bary_coords_to_subdivide[:, 2]
            bary_coords_0ab = torch.stack([weight_0 + 0.5 * (weight_1 + weight_2), 0.5 * weight_1, 0.5 * weight_2], dim=-1)
            bary_coords_1ca = torch.stack([0.5 * weight_0, weight_1 + 0.5 * (weight_0 + weight_2), 0.5 * weight_2], dim=-1)
            bary_coords_2bc = torch.stack([0.5 * weight_0, 0.5 * weight_1, weight_2 + 0.5 * (weight_0 + weight_1)], dim=-1)

            mesh_faces[face_to_subdivide] = faces_acb
            mesh_faces = torch.cat([
                mesh_faces,
                faces_0ab, faces_1ca, faces_2bc
            ], dim=0)

            pix_to_face = torch.cat([pix_to_face] + [pix_to_face[face_to_subdivide]] * 3, dim=0)
            bary_coords = torch.cat([
                bary_coords,
                bary_coords_0ab, bary_coords_1ca, bary_coords_2bc
            ], dim=0)

            mesh_verts = torch.cat([
                mesh_verts,
                mesh_verts_added
            ], dim=0)

            triangles = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3)

            radius = circumcircle_radius(triangles)
            N_Gaussians = mesh_faces.shape[0]
            means = torch.mean(triangles, dim=1)
            normals = torch.cat([normals] + [normals[face_to_subdivide]] * 3, dim=0)
            # features_dc = torch.cat([features_dc] + [features_dc[face_to_subdivide]] * 3, dim=0)

            subdivision_iter += 1

        Mesh_Fragments = namedtuple("Mesh_Fragments", ['pix_to_face', 'bary_coords'])
        mesh_fragments = Mesh_Fragments(
            pix_to_face=pix_to_face.reshape(1, 1, N_Gaussians, 1),
            bary_coords=bary_coords.reshape(1, 1, N_Gaussians, 1, 3),
        )
        features_dc = mesh.textures.sample_textures(mesh_fragments).reshape(N_Gaussians, 3)

        append_num_gaussians = mesh_faces.shape[0]
        num_existing_gaussians = self.gauss_params["means_2d"].shape[0]

        original_face_cnt = self.mesh_faces.shape[0]
        append_indices = torch.arange(append_num_gaussians) + original_face_cnt
        append_gaussians_indices = torch.arange(append_num_gaussians) + num_existing_gaussians

        radius = torch.abs(radius).reshape(-1, 1).clone().cuda()
        mesh_verts = mesh_verts.cuda()
        first_edge = mesh_verts[mesh_faces[:, :2].reshape(-1)].reshape(-1, 2, 3)
        first_edge_vec = (first_edge[:, 0] - first_edge[:, 1]).reshape(-1, 3)
        bottom_length = torch.sum(first_edge_vec ** 2, dim=-1) ** 0.5
        cross_height = (area(triangles).cuda() * 2) / bottom_length
        # xyz_radius = torch.stack(
        #     [bottom_length.reshape(-1) * 3.0, cross_height.reshape(-1) * 3.0, radius.reshape(-1) * 0.05], dim=-1)
        # radius *= 50
        xyz_radius = radius.clone().repeat(1, 3)
        xyz_radius[:, 2] *= self.config.face_flat_coef

        self.radius = torch.cat([self.radius, radius.clone().reshape(-1, 1).to(self.radius.device)], dim=0)
        self.xyz_radius = torch.cat([self.xyz_radius, xyz_radius.clone().to(self.xyz_radius.device)], dim=0)

        original_num_verts = self.mesh_verts.shape[0]

        self.mesh_verts = torch.cat([self.mesh_verts, mesh_verts.to(self.mesh_verts.device)], dim=0)
        self.mesh_faces = torch.cat([self.mesh_faces, mesh_faces.to(self.mesh_faces.device) + original_num_verts],
                                    dim=0)
        self.mesh_faces_verts = self.mesh_verts[self.mesh_faces.reshape(-1)].reshape(-1, 3, 3).cuda()
        self.normals = torch.cat([self.normals, normals.to(self.normals.device)])

        # Recompute (not extend) the color_consistency_lambda adjacency: the appended
        # part's vertex indices were just offset into the combined buffer above, so
        # running the edge hash over the merged mesh_faces is both simpler than splicing
        # two index lists and correct if the two parts happen to share vertices.
        self.face_adj_a, self.face_adj_b = face_adjacency_pairs(self.mesh_faces)
        # Same reasoning for the fold pairs, and here recomputing is not merely simpler
        # but REQUIRED: joining two parts can put one part's surface right up against the
        # other's, creating folds that neither part had on its own. Runs the full search
        # again (no is_fold/nearest_foreign to pass in at this point), which is why it is
        # skipped outright when the color loss is off.
        self._set_face_overlap_pairs()

        mesh_faces_verts = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3).cuda()

        # Extend cv_radius for the appended faces: the scales property gathers
        # self.cv_radius_base_floor[gaussians_to_mesh_indices] for ALL rows whenever the
        # base layer is on, so leaving it un-extended would index out of bounds even
        # though the appended Gaussians themselves are never base.
        _appended_cv_radius = face_centroid_vertex_radius(mesh_faces_verts).reshape(-1, 1).to(self.cv_radius.device)
        self.cv_radius = torch.cat([self.cv_radius, _appended_cv_radius], dim=0)
        # No irregular-face widening for appended faces (see cv_radius_base_floor in
        # populate_modules) -- base membership never applies to this splat_merge path
        # anyway, so the plain mean-based value is never actually read as a base floor;
        # it only needs to exist so the gather above doesn't go out of bounds.
        self.cv_radius_base_floor = torch.cat([self.cv_radius_base_floor, _appended_cv_radius], dim=0)
        # Same reasoning for is_fold_face (2026-08-03, see the opacities property): fold
        # detection was never run against this splat_merge path either, so appended faces
        # just need a same-length False extension to keep the opacities property's
        # is_fold_face[gaussians_to_mesh_indices] gather in bounds.
        self.is_fold_face = torch.cat([
            self.is_fold_face,
            torch.zeros(mesh_faces.shape[0], dtype=torch.bool, device=self.is_fold_face.device),
        ], dim=0)

        v_a = mesh_faces_verts[:, 0]
        v_b = mesh_faces_verts[:, 1]
        v_c = mesh_faces_verts[:, 2]

        mesh_triangles_edge_ab = v_b - v_a
        mesh_triangles_edge_bc = v_c - v_b
        mesh_triangles_edge_ca = v_a - v_c

        mesh_triangles_edge_len_a = torch.linalg.norm(mesh_triangles_edge_bc, ord=2, dim=-1)
        mesh_triangles_edge_len_b = torch.linalg.norm(mesh_triangles_edge_ca, ord=2, dim=-1)
        mesh_triangles_edge_len_c = torch.linalg.norm(mesh_triangles_edge_ab, ord=2, dim=-1)

        mesh_triangles_2d_coords_a, mesh_triangles_2d_coords_b, mesh_triangles_2d_coords_c = compute_triangle_vertices(
            mesh_triangles_edge_len_a, mesh_triangles_edge_len_b, mesh_triangles_edge_len_c)
        means_2d_coords = (mesh_triangles_2d_coords_a + mesh_triangles_2d_coords_b + mesh_triangles_2d_coords_c) / 3
        means_2d = means_2d_coords

        self.mesh_triangles_edge_ab = torch.cat([self.mesh_triangles_edge_ab, mesh_triangles_edge_ab.to(self.mesh_triangles_edge_ab.device)], dim=0)
        self.mesh_triangles_edge_bc = torch.cat([self.mesh_triangles_edge_bc, mesh_triangles_edge_bc.to(self.mesh_triangles_edge_bc.device)], dim=0)
        self.mesh_triangles_edge_ca = torch.cat([self.mesh_triangles_edge_ca, mesh_triangles_edge_ca.to(self.mesh_triangles_edge_ca.device)], dim=0)

        self.mesh_triangles_edge_len_a = torch.cat([self.mesh_triangles_edge_len_a, mesh_triangles_edge_len_a.to(self.mesh_triangles_edge_len_a.device)], dim=0)
        self.mesh_triangles_edge_len_b = torch.cat([self.mesh_triangles_edge_len_b, mesh_triangles_edge_len_b.to(self.mesh_triangles_edge_len_b.device)], dim=0)
        self.mesh_triangles_edge_len_c = torch.cat([self.mesh_triangles_edge_len_c, mesh_triangles_edge_len_c.to(self.mesh_triangles_edge_len_c.device)], dim=0)

        self.mesh_triangles_2d_coords_a = torch.cat([self.mesh_triangles_2d_coords_a, mesh_triangles_2d_coords_a.to(self.mesh_triangles_2d_coords_a.device)], dim=0)
        self.mesh_triangles_2d_coords_b = torch.cat([self.mesh_triangles_2d_coords_b, mesh_triangles_2d_coords_b.to(self.mesh_triangles_2d_coords_b.device)], dim=0)
        self.mesh_triangles_2d_coords_c = torch.cat([self.mesh_triangles_2d_coords_c, mesh_triangles_2d_coords_c.to(self.mesh_triangles_2d_coords_c.device)], dim=0)

        # Extend the base layer's elliptical floor over the appended faces (see
        # config.anisotropic_base_floor). Both tensors are indexed by face id everywhere
        # they are read, so they have to grow with the mesh or every gather past the old
        # face count would be out of bounds. Clamped against the existing floor's own
        # maximum rather than a freshly computed 99th percentile: the appended faces are a
        # small addition to an established mesh, so the established scale is the right
        # reference, and this keeps a handful of degenerate new faces from setting it.
        _new_axes, _new_theta = steiner_ellipse_axes(
            mesh_triangles_2d_coords_a,
            mesh_triangles_2d_coords_b,
            mesh_triangles_2d_coords_c,
            max_aspect=self.config.base_floor_max_aspect,
        )
        _new_axes = _new_axes.to(self.base_floor_xy_aniso.device) / max(
            self.config.base_floor_corner_sigma, 1e-6
        )
        if self.base_floor_xy_aniso.numel() > 0:
            _new_axes = torch.clamp(_new_axes, max=self.base_floor_xy_aniso.max().item())
        self.base_floor_xy_aniso = torch.cat([self.base_floor_xy_aniso, _new_axes], dim=0)
        self.base_ellipse_theta = torch.cat(
            [self.base_ellipse_theta, _new_theta.reshape(-1).to(self.base_ellipse_theta.device)], dim=0
        )


        if self.config.unconstrained_scale:
            means_data = torch.mean(mesh_faces_verts, dim=1).reshape(-1, 3)
            distances, _ = self.k_nearest_sklearn(means_data, 3)
            distances = torch.from_numpy(distances)
            # find the average of the three nearest neighbors for each point and use that as the scale
            avg_dist = distances.mean(dim=-1, keepdim=True)
            scales = torch.log(avg_dist.repeat(1, 3))
        else:
            scales = torch.zeros(append_num_gaussians, 3).float()

        normals = torch.nn.functional.normalize(normals, dim=-1, p=2).cuda()
        faces_axis_x = torch.nn.functional.normalize(mesh_triangles_edge_ab, dim=-1, p=2).cuda()
        faces_axis_y = torch.cross(normals, faces_axis_x, dim=-1).reshape(-1, 3)
        faces_axis_y = torch.nn.functional.normalize(faces_axis_y, dim=-1, p=2)

        self.faces_axis_x = torch.cat([self.faces_axis_x, faces_axis_x.to(self.faces_axis_x.device)], dim=0)
        self.faces_axis_y = torch.cat([self.faces_axis_y, faces_axis_y.to(self.faces_axis_y.device)], dim=0)

        rot_mat = torch.stack([faces_axis_x, faces_axis_y, normals], dim=2).cuda()
        faces_quats = matrix_to_quaternion(rot_mat)
        self.faces_quats = torch.cat([self.faces_quats, faces_quats.to(self.faces_quats.device)], dim=0)

        quats = torch.nn.Parameter(torch.zeros(append_num_gaussians, 3).float())

        self.gaussians_to_mesh_indices = torch.cat([self.gaussians_to_mesh_indices, torch.arange(append_num_gaussians, device="cuda") + original_face_cnt], dim=0)
        # Appended-mesh Gaussians are not base: the whole base machinery (position pin,
        # scale/opacity floors, cull immunity) was never validated for this splat_merge
        # path. (_find_nearest_faces() now queries self.mesh_faces_verts directly, which
        # IS extended above, so face reassignment does cover appended faces as of
        # 2026-07-26 -- just the base-layer exemption above that doesn't apply to them.)
        self.is_base = torch.cat([
            self.is_base,
            torch.zeros(append_num_gaussians, dtype=torch.bool, device=self.is_base.device),
        ], dim=0)
        self.is_rescue = torch.cat([
            self.is_rescue,
            torch.zeros(append_num_gaussians, dtype=torch.bool, device=self.is_rescue.device),
        ], dim=0)
        self.skip_floor_cull = torch.cat([
            self.skip_floor_cull,
            torch.zeros(append_num_gaussians, dtype=torch.bool, device=self.skip_floor_cull.device),
        ], dim=0)

        self.gauss_params["means_2d"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["means_2d"].detach(),
            means_2d.float().cuda()
        ], dim=0))

        self.gauss_params["normal_elevates"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["normal_elevates"].detach(),
            torch.zeros(append_num_gaussians).float().cuda()
        ], dim=0))

        self.gauss_params["scales"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["scales"].detach(),
            scales.float().cuda()
        ], dim=0))

        self.gauss_params["quats"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["quats"].detach(),
            quats.float().cuda()
        ], dim=0))

        self.gauss_params["features_dc"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["features_dc"].detach(),
            RGB2SH(features_dc.reshape(-1, 3)).cuda()
        ], dim=0))

        self.gauss_params["features_rest"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["features_rest"].detach(),
            torch.zeros(append_num_gaussians, self.dim_sh - 1, 3).float().cuda()
        ], dim=0))

        self.gauss_params["opacities"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["opacities"].detach(),
            torch.logit(self.config.init_opacity * torch.ones(append_num_gaussians, 1)).float().cuda()
        ], dim=0))

        return append_indices, append_gaussians_indices


    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    # Face-based param group names only -- get_gaussian_param_groups() also returns
    # the vertex-anchored filler population's groups (vertex_features_dc/
    # vertex_features_rest/vertex_opacities), which have a completely different size
    # (one row per mesh VERTEX, never split/duplicated/culled) and would size-mismatch
    # against a mask sized for the face-based population if remove_from_all_optim/
    # dup_in_all_optim below applied it to them too.
    _FACE_PARAM_GROUPS = (
        "means_2d", "normal_elevates", "scales", "quats", "features_dc", "features_rest", "opacities",
    )

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group in self._FACE_PARAM_GROUPS:
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param_groups[group])
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group in self._FACE_PARAM_GROUPS:
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param_groups[group], n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # self.radii/self.xys now also include the vertex-anchored filler
            # population appended after the face-based one (see get_outputs) -- slice
            # down to just the face-based rows first, since that filler population is
            # never split/duplicated/culled/reassigned and must not feed these stats
            # (self.xys_grad_norm/self.vis_counts/self.max_2Dsize are all sized to
            # self.num_points, the face-based count only).
            radii_face = self.radii[: self.n_face_based_gaussians]
            xys_grad_face = self.xys.absgrad[0][: self.n_face_based_gaussians]  # type: ignore

            # keep track of a moving average of grad norms
            visible_mask = (radii_face > 0).flatten()
            grads = xys_grad_face[visible_mask].norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
                self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)

            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(radii_face, dtype=torch.float32)
            newradii = radii_face.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval > min(self.num_train_data, 1000) + self.config.refine_every
            )
            # Hard population ceiling (see config.max_gaussians). Checked here rather than
            # inside split/dup so culling, reassignment and the floor bookkeeping below all
            # still run -- the run keeps improving what it has, it just stops adding. The
            # coverage rescue enforces the same ceiling separately, in its own cap list;
            # gating only this path is what let a run reach 9.7M rows.
            hit_ceiling = False
            if self.config.max_gaussians > 0 and self.num_points >= self.config.max_gaussians:
                if do_densification:
                    CONSOLE.log(
                        f"max_gaussians reached ({self.num_points:,} >= "
                        f"{self.config.max_gaussians:,}) -- densification paused this cycle"
                    )
                    # Remember that densification was cancelled BY THE CEILING rather than
                    # by the opacity-reset schedule, so the cull below still runs. Without
                    # this the run deadlocks: cull_gaussians() is only reachable from the
                    # densification branch before stop_split_at, so hitting the ceiling
                    # stopped culling too, the population could no longer shrink, and
                    # coverage_rescue was permanently denied its budget. Observed on gs21
                    # (5.5M, step 18890) and gs22 (6.5M, step 6590): rescue reporting
                    # "spawning 0 gaussian(s)" against 200k+ faces flagged for repair,
                    # while 416k rows sat below cull_alpha_thresh waiting to be removed.
                    # This is what the comment above always claimed already happened.
                    hit_ceiling = True
                do_densification = False
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                if self.config.curvature_densify_scale > 0:
                    _curv_w = 1.0 + self.config.curvature_densify_scale * (
                        self.face_curvature[self.gaussians_to_mesh_indices] /
                        (self.face_curvature.max() + 1e-8)
                    )
                    avg_grad_norm = avg_grad_norm * _curv_w.squeeze()
                if self.config.coverage_densify_scale > 0 and self.config.coverage_lambda > 0:
                    # Same idea as curvature weighting above, but weighted by "how
                    # undercovered is this face" (coverage_lambda's own geometric check)
                    # instead of "how curved is this face" -- see coverage_densify_scale
                    # docstring for why this is needed (coverage_lambda can't spawn new
                    # Gaussians on its own, only densify_grad_thresh -- a photometric
                    # trigger -- can, so a geometrically-gapped but photometrically-quiet
                    # face would otherwise never get new Gaussians).
                    vert_cov, cent_cov = self._compute_coverage_density()
                    tgt = self.config.coverage_target
                    cent_deficit = torch.nn.functional.relu(tgt - cent_cov)  # (num_faces,)
                    vert_deficit_per_face = torch.nn.functional.relu(tgt - vert_cov)[
                        self.mesh_faces
                    ].mean(dim=1)  # (num_faces,), avg of this face's 3 vertices' deficits
                    # Worse of "this face's own centroid is under-covered" and "this
                    # face's corners are under-covered" -- either alone is a real gap.
                    face_deficit = torch.maximum(cent_deficit, vert_deficit_per_face)
                    _cov_w = 1.0 + self.config.coverage_densify_scale * (
                        face_deficit[self.gaussians_to_mesh_indices] / (tgt + 1e-8)
                    )
                    avg_grad_norm = avg_grad_norm * _cov_w.squeeze()
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                # in-plane footprint only, same reasoning as the cull_scale_thresh check above.
                # Threshold is relative to each Gaussian's own face's xyz_radius (not a fixed
                # absolute value) -- see densify_size_thresh_frac docstring for why.
                size_thresh = self.config.densify_size_thresh_frac * self.xyz_radius[
                    self.gaussians_to_mesh_indices, :2
                ].max(dim=-1).values
                splits = (self.scales[:, :2].exp().max(dim=-1).values > size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)
                splits_idxs = self.gaussians_to_mesh_indices[splits].repeat(nsamps)

                dups = (self.scales[:, :2].exp().max(dim=-1).values <= size_thresh).squeeze()
                dups &= high_grads
                dups_idxs = self.gaussians_to_mesh_indices[dups]

                dup_params = self.dup_gaussians(dups)

                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )
                    # print(name, self.gauss_params[name].shape)
                self.gaussians_to_mesh_indices = torch.cat([
                    self.gaussians_to_mesh_indices,
                    splits_idxs,
                    dups_idxs
                ], dim=0)
                # Children of splits/dups are ordinary detail Gaussians, never base --
                # even when the split/dup source was a base Gaussian (base can seed
                # detail on top of itself, but base membership itself is only ever
                # created at initialization). Same for is_rescue: even if the parent was
                # a rescue-ceiling Gaussian, its children go back to the ordinary
                # shared/shrunk ceiling -- rescue status is only ever assigned at the
                # moment of a coverage-rescue spawn, not inherited.
                self.is_base = torch.cat([
                    self.is_base,
                    torch.zeros(
                        splits_idxs.shape[0] + dups_idxs.shape[0],
                        dtype=torch.bool,
                        device=self.is_base.device,
                    ),
                ], dim=0)
                self.is_rescue = torch.cat([
                    self.is_rescue,
                    torch.zeros(
                        splits_idxs.shape[0] + dups_idxs.shape[0],
                        dtype=torch.bool,
                        device=self.is_rescue.device,
                    ),
                ], dim=0)
                self.skip_floor_cull = torch.cat([
                    self.skip_floor_cull,
                    torch.zeros(
                        splits_idxs.shape[0] + dups_idxs.shape[0],
                        dtype=torch.bool,
                        device=self.skip_floor_cull.device,
                    ),
                ], dim=0)
                # print("gaussians_to_mesh_indices: ", self.gaussians_to_mesh_indices.shape)
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif (
                self.step >= self.config.stop_split_at or hit_ceiling
            ) and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)


            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
                # Configurable as of 2026-08-24 (see config.opacity_reset_min_value for why
                # the historical hardcoded [0.5, 0.8] is the opposite of vanilla splatfacto
                # and what it measured out to). Defaults reproduce the old values exactly.
                reset_value = self.config.opacity_reset_value
                min_reset_value = self.config.opacity_reset_min_value
                # Must write through gauss_params, NOT self.opacities.data: the opacities
                # property may return a computed (non-leaf) tensor when use_base_layer is
                # on, and assigning .data on that would silently modify a temporary.
                self.gauss_params["opacities"].data = torch.clamp(
                    self.gauss_params["opacities"].data,
                    min=torch.logit(torch.tensor(min_reset_value, device=self.device)).item(),
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

            did_rescue = False
            if (
                self.config.reassign_face_every > 0
                and self.step % self.config.reassign_face_every == 0
            ):
                self.reassign_gaussians_to_nearest_face(optimizers)
                # reassign_gaussians_to_nearest_face() always ends by calling
                # coverage_rescue_spawn() itself, so don't run it twice this step.
                did_rescue = True

            if (
                self.config.coverage_rescue_every > 0
                and not did_rescue
                and self.step % self.config.coverage_rescue_every == 0
            ):
                # Per-cycle coverage repair. Required by adaptive_coverage_floor: once a
                # face's floor can be released, this is what keeps the invariant true
                # between the (much rarer) reassignment passes. See
                # config.coverage_rescue_every.
                self.coverage_rescue_spawn(optimizers)

            if (
                self.config.freeze_geometry_at_step >= 0
                and self.step >= self.config.freeze_geometry_at_step
            ):
                # See freeze_geometry_at_step docstring: re-asserted every cycle
                # (idempotent) rather than toggled once, because cull_gaussians/
                # split/dup/rescue all rebuild self.gauss_params[name] as a fresh
                # nn.Parameter (default requires_grad=True), which would otherwise
                # silently re-enable gradients on these after the first cull event
                # past this step.
                for _geom_name in ("means_2d", "normal_elevates", "scales", "quats"):
                    self.gauss_params[_geom_name].requires_grad_(False)

            # LAST in this function on purpose: every population change for this cycle
            # (split, dup, cull, reassign, rescue) has landed by now, so the mask the next
            # refine_every steps render against is measured against the Gaussians that
            # actually exist. Measuring earlier would decide which floors to drop based on
            # a population that is about to change. No-op unless adaptive_coverage_floor.
            self.update_coverage_floor_mask()

    def _find_nearest_faces(self):
        """
        For every Gaussian, find which face of the WHOLE mesh its current 3D position is
        actually closest to, searching every face (no local-neighborhood restriction).
        Shared by reassign_gaussians_to_nearest_face() (periodic, mid-training) and
        finalize_face_assignment() (one-shot, at export time).

        Replaces the 2026-07-17 version of this function (candidate list = current face
        + its 1-ring + 2-ring neighbors, up to 13 faces, see the removed
        build_face_reassign_candidates()): that approach could never find a Gaussian's
        true nearest face if it had drifted farther than 2 rings from whatever face it
        was currently labeled with -- structurally impossible to detect or correct, since
        the candidate list itself wouldn't contain the right answer.

        2026-07-26: first attempt used pytorch3d's point_face_distance (an internal API,
        unverifiable locally -- no pytorch3d install on this dev machine) and crashed on
        the very first remote run with a return-value unpacking mismatch (the function's
        actual return signature didn't match what was assumed from memory). Reverted to a
        pure-PyTorch implementation using this file's own already-proven distance
        approximation instead of trusting an unverified external API: the same
        "project onto the face's plane, clip barycentric coords to the triangle,
        renormalize" trick the old candidate-based version and the `means` property both
        already use, just evaluated against EVERY face instead of a restricted candidate
        list.

        2026-07-29: that bary-clip-renormalize distance was still only an approximation
        of "closest point on this triangle" -- independently clamping each barycentric
        coordinate to [0, 1] and renormalizing does not, in general, recover the true
        nearest point (e.g. clamping two negative coordinates to 0 can renormalize the
        third above 1, landing outside the triangle in the wrong place). On
        irregular/obtuse triangles (common in this mesh's high-curvature regions) this
        could pick the wrong face as "nearest" -- user reported Gaussians whose center
        genuinely sat on a neighboring triangle even after finalize_face_assignment()
        had already run. Replaced with an exact vertex/edge/interior case split
        (Ericson's ClosestPtPointTriangle, vectorized) inside the loop below -- this is
        the "separate case split" the old version of this docstring said was missing.

        Cost: O(N_gaussians x N_faces), no spatial acceleration structure, chunked over
        BOTH Gaussians and faces to bound memory (a single (all Gaussians, all faces)
        intermediate would be enormous -- e.g. ~4e5 x 1e5 here). Only fires every
        reassign_face_every steps (a handful of times total), so the added cost is
        per-call, not per-step, but expect this to take noticeably longer than the old
        candidate-restricted version.

        Returns:
            means, cur_face, best_face, changed as described above. Unlike the previous
            version, this no longer needs to bail out (return None) for a model that went
            through append_from_mesh(): the query works directly off self.mesh_faces_verts
            (already correctly extended there), with no separate candidate table to fall
            out of sync. Callers still handle a None return for forward compatibility, but
            it will not actually happen from this implementation.
        """
        means = self.means  # (N, 3)
        cur_face = self.gaussians_to_mesh_indices  # (N,)
        N = means.shape[0]
        device = means.device

        tri_origin_all = self.mesh_faces_verts[:, 0]  # (T, 3)
        axis_x_all = self.faces_axis_x  # (T, 3)
        axis_y_all = self.faces_axis_y  # (T, 3)
        normal_all = self.normals  # (T, 3)
        a2_all = self.mesh_triangles_2d_coords_a  # (T, 2)
        b2_all = self.mesh_triangles_2d_coords_b
        c2_all = self.mesh_triangles_2d_coords_c
        T = tri_origin_all.shape[0]

        # Candidate pre-filter. The exact test below used to run for every
        # (Gaussian, face) pair -- ~2e11 evaluations of a 30-tensor closest-point
        # routine, four hours per call, i.e. ~40 h added to a 30k-step run at
        # reassign_face_every=3000. A Gaussian's nearest face is always among its
        # nearest few by centroid, so a KD-tree over the centroids narrows each one
        # to _REASSIGN_K candidates and the identical exact math runs on those.
        # Verified against brute force on this mesh: the resulting distance is
        # bit-identical (max diff 0.000) at every offset from the surface tested,
        # 1 mm to 160 mm. Face *ids* differ on 1-40% of Gaussians depending on
        # offset, but only where two faces are exactly equidistant -- ties, which
        # every downstream use (coverage, densification, scale floor) treats alike.
        # The mesh is static, so the tree is built once and cached.
        _REASSIGN_K = min(48, T)
        if getattr(self, "_face_tree_size", None) != T:
            from scipy.spatial import cKDTree
            self._face_tree = cKDTree(self.mesh_faces_verts.mean(dim=1).detach().cpu().numpy())
            self._face_tree_size = T
        _, cand_np = self._face_tree.query(
            means.detach().cpu().numpy(), k=_REASSIGN_K, workers=-1
        )
        cand_all = torch.as_tensor(
            np.asarray(cand_np).reshape(N, _REASSIGN_K), device=device, dtype=torch.long
        )

        _G_CHUNK = 20_000
        best_face_chunks = []
        for g_start in range(0, N, _G_CHUNK):
            g_end = min(g_start + _G_CHUNK, N)
            pts = means[g_start:g_end]  # (g, 3)
            g = pts.shape[0]

            if True:
                cand = cand_all[g_start:g_end]          # (g, K) face indices
                origin = tri_origin_all[cand]           # (g, K, 3)
                axis_x = axis_x_all[cand]
                axis_y = axis_y_all[cand]
                normal = normal_all[cand]
                a2 = a2_all[cand]                       # (g, K, 2)
                b2 = b2_all[cand]
                c2 = c2_all[cand]

                rel = pts.unsqueeze(1) - origin  # (g, f, 3)
                plane_dist = (rel * normal).sum(dim=-1)  # (g, f)
                local_2d = torch.stack([
                    (rel * axis_x).sum(dim=-1),
                    (rel * axis_y).sum(dim=-1),
                ], dim=-1)  # (g, f, 2)

                # Exact closest-point-on-triangle (2026-07-29, replaces the old
                # bary-clip-renormalize approximation -- see this function's docstring
                # for why that approximation could pick the wrong "nearest face" on
                # irregular/obtuse triangles). Vectorized version of Ericson's
                # ClosestPtPointTriangle ("Real-Time Collision Detection"): explicit
                # vertex/edge/interior region case split via the six half-plane tests,
                # instead of independently clamping each barycentric coordinate to
                # [0, 1] and renormalizing (which does NOT recover the true nearest
                # point in general -- e.g. clamping two negative coordinates to 0
                # independently can renormalize the third above 1, still outside the
                # triangle in the wrong place).
                A = a2  # (g, f, 2) -- already per-Gaussian after the KD-tree gather
                B = b2
                C = c2
                P = local_2d  # (g, f, 2)

                ab = B - A
                ac = C - A
                ap = P - A
                d1 = (ab * ap).sum(-1)  # (g, f)
                d2 = (ac * ap).sum(-1)

                bp = P - B
                d3 = (ab * bp).sum(-1)
                d4 = (ac * bp).sum(-1)

                cp = P - C
                d5 = (ab * cp).sum(-1)
                d6 = (ac * cp).sum(-1)

                va = d3 * d6 - d5 * d4
                vb = d5 * d2 - d1 * d6
                vc = d1 * d4 - d3 * d2

                _EPS = 1e-12
                # Interior region (fallback/lowest priority): va+vb+vc's sign matches
                # this mesh's fixed local-2D winding (mesh_triangles_2d_coords_* is
                # built by compute_triangle_vertices(), a canonical layout independent
                # of the source mesh's own winding, so the sign is consistent across
                # every face) -- preserve sign when guarding against division by ~0,
                # unlike a plain clamp(min=eps) which would silently flip it.
                denom_in = va + vb + vc
                denom_in = torch.where(denom_in.abs() < _EPS, torch.full_like(denom_in, _EPS), denom_in)
                v_in = (vb / denom_in).unsqueeze(-1)
                w_in = (vc / denom_in).unsqueeze(-1)
                closest = A + ab * v_in + ac * w_in

                # Edge BC: reached only when va <= 0 and both region tests are >= 0, so
                # their sum is already guaranteed non-negative -- plain clamp is safe.
                case6 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
                denom6 = ((d4 - d3) + (d5 - d6)).clamp(min=_EPS)
                w6 = ((d4 - d3) / denom6).unsqueeze(-1)
                closest = torch.where(case6.unsqueeze(-1), B + w6 * (C - B), closest)

                # Edge AC: d2 >= 0 and d6 <= 0 under this case, so d2 - d6 >= 0.
                case5 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
                denom5 = (d2 - d6).clamp(min=_EPS)
                w5 = (d2 / denom5).unsqueeze(-1)
                closest = torch.where(case5.unsqueeze(-1), A + w5 * ac, closest)

                # Vertex C
                case4 = (d6 >= 0) & (d5 <= d6)
                closest = torch.where(case4.unsqueeze(-1), C.expand_as(closest), closest)

                # Edge AB: d1 >= 0 and d3 <= 0 under this case, so d1 - d3 >= 0.
                case3 = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
                denom3 = (d1 - d3).clamp(min=_EPS)
                v3 = (d1 / denom3).unsqueeze(-1)
                closest = torch.where(case3.unsqueeze(-1), A + v3 * ab, closest)

                # Vertex B
                case2 = (d3 >= 0) & (d4 <= d3)
                closest = torch.where(case2.unsqueeze(-1), B.expand_as(closest), closest)

                # Vertex A -- checked last so it wins ties, matching Ericson's
                # original early-return order (vertex regions take priority over
                # edge/interior regions when multiple conditions could apply at a
                # shared corner).
                case1 = (d1 <= 0) & (d2 <= 0)
                closest = torch.where(case1.unsqueeze(-1), A.expand_as(closest), closest)

                local_2d_clamped = closest  # (g, f, 2)
                in_plane_dist = torch.norm(local_2d - local_2d_clamped, dim=-1)  # (g, f)
                total_dist = torch.sqrt(in_plane_dist ** 2 + plane_dist ** 2 + 1e-12)

                _, chunk_argmin = total_dist.min(dim=1)  # (g,) index into the K candidates
                # Map back from candidate slot to the real face id.
                best_idx = cand.gather(1, chunk_argmin.unsqueeze(1)).squeeze(1)

            best_face_chunks.append(best_idx)

        best_face = torch.cat(best_face_chunks, dim=0)
        changed = best_face != cur_face
        # Base-layer Gaussians are pinned to their own face's centroid and must never be
        # rebound (their face binding IS the coverage guarantee). Their nearest face is
        # their own face by construction anyway; this mask just makes that explicit.
        # No-op when the base layer is disabled (is_base all False). Covers both callers
        # (finalize_face_assignment and reassign_gaussians_to_nearest_face).
        changed = changed & ~self.is_base
        return means, cur_face, best_face, changed

    def _reproject_to_new_faces(self, means, changed, best_face):
        """
        For every Gaussian whose face assignment is changing (changed[i] == True),
        recompute means_2d / normal_elevates in the NEW face's local frame so the
        *world-space* means stays continuous across the reassignment, and update
        gaussians_to_mesh_indices. scales are already absolute world lengths (not
        face-relative) so need no change; quats keeps its current raw offset -- a small,
        trainable discontinuity if the new face's normal differs noticeably from the old
        one, which further training steps will fine-tune away, same as a freshly
        split/duplicated Gaussian does (at export time, after training has stopped,
        this discontinuity -- if any -- is permanent). As of 2026-07-26 the nearest-face
        search is exact/global (see _find_nearest_faces()), not restricted to a small
        local neighborhood -- a Gaussian that drifted far from its labeled face can now
        correctly jump straight to its true nearest face in one step, which is the point,
        but means this discontinuity is no longer bounded by "how far a local
        neighborhood reaches": a badly-drifted Gaussian can move farther in one
        reassignment than the old candidate-restricted version ever could have moved it.

        Returns the number of Gaussians actually moved.
        """
        n_changed = int(changed.sum().item())
        if n_changed == 0:
            return 0

        idx_changed = torch.where(changed)[0]
        new_face = best_face[idx_changed]

        origin_new = self.mesh_faces_verts[new_face][:, 0]
        axis_x_new = self.faces_axis_x[new_face]
        axis_y_new = self.faces_axis_y[new_face]
        normal_new = self.normals[new_face]

        rel_new = means[idx_changed] - origin_new
        self.gauss_params["means_2d"].data[idx_changed] = torch.stack(
            [(rel_new * axis_x_new).sum(dim=-1), (rel_new * axis_y_new).sum(dim=-1)], dim=-1
        )
        # NOTE: this raw value gets clamped to the new face's [-radius*elevate_coef,
        # radius*elevate_coef] range by the `means` property same as any other step --
        # if the true offset exceeds that bound the Gaussian will land as close as the
        # new face's bound allows, not exactly continuous.
        self.gauss_params["normal_elevates"].data[idx_changed] = (rel_new * normal_new).sum(dim=-1)

        self.gaussians_to_mesh_indices[idx_changed] = new_face
        # See populate_modules' skip_floor_cull docstring: relabeling can reclamp the
        # rendered scale purely from the new face's (possibly larger) xyz_radius, with
        # no actual shrinking having happened -- give these rows one cull_gaussians()
        # cycle of immunity from cull_at_floor_gaussians before they're evaluated
        # against it again. No-op when called from finalize_face_assignment() (export
        # time, no more cull_gaussians() calls follow).
        self.skip_floor_cull[idx_changed] = True
        return n_changed

    def finalize_face_assignment(self):
        """
        One-shot version of reassign_gaussians_to_nearest_face(), meant to run once after
        training has fully stopped -- currently called from export_splatfacto_on_mesh()
        and report_coverage() (both AFTER_TRAIN-only call sites, see their comments/
        docstrings for why the timing matters), regardless of whether
        config.reassign_face_every was ever enabled during training. Without this,
        gaussians_to_mesh_indices would just reflect whichever face each Gaussian was
        assigned at *birth* (inherited through split/dup), never updated to match where
        it actually ended up -- exactly the "Gaussian doesn't look like it's on its
        assigned triangle" mismatch this whole feature exists to fix. This is the "option
        A" (one-shot, final-only) design the user chose over "option B" (periodic
        mid-training reassignment, config.reassign_face_every / the plumbing in
        reassign_gaussians_to_nearest_face() below, left in place but disabled by
        default and not used in the training command).

        Callers that also read self.scales must do so BEFORE calling this, not after --
        see export_splatfacto_on_mesh()'s comment for why (relabeling changes which
        face's xyz_radius bounds the scales clamp).

        Unlike reassign_gaussians_to_nearest_face(): no optimizer involved (training has
        already stopped by export time) and does NOT spawn new Gaussians on faces left
        with zero coverage -- injecting a brand new, never-trained Gaussian into an
        already-finished result would just show up as an untrained-looking patch, with no
        further training steps left to refine it away.
        """
        with torch.no_grad():
            result = self._find_nearest_faces()
            if result is None:
                return
            means, _, best_face, changed = result
            n_changed = self._reproject_to_new_faces(means, changed, best_face)
            CONSOLE.log(
                f"finalize_face_assignment: {n_changed}/{self.num_points} gaussians "
                f"relabeled to their true nearest face for export"
            )

    def reassign_gaussians_to_nearest_face(self, optimizers: Optimizers):
        """
        Re-bind each Gaussian to whichever face its current 3D position is actually
        closest to (see _find_nearest_faces()), instead of leaving it permanently bound
        to whichever face it was created on. See config.reassign_face_every docstring
        for why this matters (coverage_lambda / curvature-weighted densification
        crediting the wrong face once a Gaussian has drifted).

        SCALE-CLAMP CAVEAT (found 2026-07-20 via the export-time version of this same
        reassignment, re-assessed 2026-07-17 for periodic mid-training use):
        self.scales' clamp bounds ([min_scale_frac, upper_scale] x xyz_radius) key off
        self.gaussians_to_mesh_indices -- the exact same array this function reassigns.
        Moving a Gaussian to a face with a smaller xyz_radius than its old one reclamps
        its rendered ceiling/floor purely from being relabeled, with no change to its
        actual raw scale parameter. At export time (finalize_face_assignment(), a true
        one-shot with no more training afterward) this was a real bug, fixed by reading
        self.scales before relabeling. Here, mid-training, that fix doesn't apply --
        deliberately: training continues after this call, so a Gaussian reclamped to its
        new face's bound is a discontinuity the optimizer has room to adapt to over
        subsequent steps, the same way it already recovers from the discontinuity a fresh
        split/dup introduces. Left un-mitigated on purpose; see reassign_face_every's
        docstring for the reasoning behind the 3000-step interval chosen to keep this
        infrequent.

        Gaussians are allowed to move freely (no veto to protect a face from being
        emptied). Densification is otherwise only driven by photometric gradient
        (densify_grad_thresh, optionally boosted by coverage_densify_scale) and
        coverage_lambda only grows/repositions EXISTING Gaussians -- neither one is a
        *guarantee*: a genuinely under-observed face can keep photometric gradient below
        threshold indefinitely no matter how boosted, so without an unconditional
        backstop a face left badly undercovered (whether by reassignment emptying it out,
        or just never having enough gradient signal to densify) could stay that way
        permanently. See coverage_rescue_thresh's docstring for the deterministic,
        gradient-free rescue this does instead -- every face below that threshold at
        *any* of its 4 check points (centroid or its 3 corners) gets exactly one fresh
        Gaussian spawned near whichever of those 4 is worst, regardless of why.
        """
        with torch.no_grad():
            result = self._find_nearest_faces()
            if result is None:
                return
            means, cur_face, best_face, changed = result
            N = cur_face.shape[0]
            n_changed = int(changed.sum().item())
            CONSOLE.log(f"reassign_gaussians_to_nearest_face: {n_changed}/{N} gaussians moved to a closer face")
            self._reproject_to_new_faces(means, changed, best_face)

        # Extracted to coverage_rescue_spawn() (2026-08-24) so the same check can also run
        # every refine cycle -- see config.coverage_rescue_every. Still called here
        # unconditionally regardless of that setting: the reassignment above is explicitly
        # allowed to empty a face outright, and cull_gaussians() asserts that no face is
        # ever left with zero Gaussians, so this call is load-bearing on its own.
        self.coverage_rescue_spawn(optimizers)

    def coverage_rescue_spawn(self, optimizers: Optimizers):
        """
        Deterministic, gradient-free coverage repair: every face still below
        coverage_rescue_thresh at ANY of its 4 check points (centroid or its 3 corners)
        gets one fresh Gaussian spawned near whichever of those points is worst.

        Split out of reassign_gaussians_to_nearest_face() (2026-08-24) with no change to
        what it does, only to when it can be called. It used to be reachable only on the
        reassign_face_every cadence (3000 steps), which was fine while the base layer put
        an unconditional full-face plate on every face -- with that plate present the
        rescue almost never fired, and a 3000-step latency on a mechanism that rarely
        triggers costs nothing.

        adaptive_coverage_floor changes that: on a face whose floor has been released, this
        is now the mechanism that maintains the coverage invariant, and every force that
        can open a gap (split/dup shrink, cull, opacity binarisation) acts once per
        refine_every. Set coverage_rescue_every to refine_every so repair runs at the same
        cadence as damage.
        """
        with torch.no_grad():
            # Deterministic coverage rescue (see coverage_rescue_thresh docstring): every
            # face whose coverage density is still below threshold at ANY of its 4 check
            # points -- centroid or any of its 3 corners -- gets a fresh Gaussian, no
            # gradient condition involved. Checking only cent_cov here (2026-07-17,
            # coverage_rescue_thresh's first version) missed corner-only gaps entirely: a
            # face's own Gaussians typically sit nearer its interior, so a well-covered
            # centroid was routinely masking still-empty corners -- exactly the "many
            # triangle corners have nothing on them" symptom reported after testing this.
            # Also spawn *at* whichever of the 4 points is worst (not always the centroid)
            # so the new Gaussian actually lands near the gap instead of adding more
            # density where it wasn't needed.
            vert_cov, cent_cov = self._compute_coverage_density()
            rescue_thresh = max(self.config.coverage_rescue_thresh, 1e-8)
            # max(..., 1e-8) guarantees a face left with literally zero Gaussians
            # (cov == 0, e.g. emptied by the reassignment just above) always gets rescued
            # even if coverage_rescue_thresh is set to 0 -- cull_gaussians() asserts no
            # face is ever left at zero, so this floor isn't optional the way the rest of
            # the threshold is.
            cent_def = torch.nn.functional.relu(rescue_thresh - cent_cov)  # (num_faces,)
            corner_def = torch.nn.functional.relu(
                rescue_thresh - vert_cov[self.mesh_faces]
            )  # (num_faces, 3), this face's 3 corners' deficits
            # (num_faces, 4): [centroid, corner0, corner1, corner2]
            all_def = torch.cat([cent_def.unsqueeze(1), corner_def], dim=1)
            worst_def, worst_idx = all_def.max(dim=1)  # (num_faces,), (num_faces,)

            rescue_faces = torch.where(worst_def > 0)[0]
            n_qualified = int(rescue_faces.shape[0])
            # Safety cap (see config.coverage_rescue_max_frac). Serve the worst-deficit
            # faces first; the rest stay short and simply qualify again next cycle, so
            # nothing is dropped, growth is just rate-limited. Only matters once
            # coverage_rescue_thresh is raised near coverage_target -- at its default 0.2
            # so few faces qualify that this never binds.
            # Caps, whichever is tightest -- see config.coverage_rescue_max_count for why
            # the fraction alone compounds once it binds on every cycle.
            #
            # config.max_gaussians is one of them. It was first written to gate only
            # densification, on the stated reasoning that "the rescue is already
            # rate-limited by coverage_rescue_max_frac / coverage_rescue_max_count" -- which
            # is false whenever max_count is left at its 0 default, since the fraction is
            # the compounding cap and not a bound at all. A run then died with split/dup
            # producing 258 rows in the cycle that the rescue produced 483,463, against a
            # ceiling that never looked at it. Densification is NOT this model's only
            # unbounded producer; the rescue is the bigger one.
            _caps = []
            if self.config.max_gaussians > 0:
                _caps.append(max(self.config.max_gaussians - self.num_points, 0))
            if self.config.coverage_rescue_max_frac > 0:
                _caps.append(int(self.config.coverage_rescue_max_frac * self.num_points))
            if self.config.coverage_rescue_max_count > 0:
                _caps.append(int(self.config.coverage_rescue_max_count))
            # Faces with LITERALLY ZERO Gaussians bypass every cap. cull_gaussians()
            # asserts that no face is ever at zero, and with use_base_layer off nothing
            # else maintains that: base rows are what used to guarantee it (culls &=
            # ~is_base), and reassignment routinely empties a face, leaving this rescue as
            # the only thing that refills it. Rate-limiting that duty turns a soft budget
            # into a hard crash -- table_gs17 died at step 24999 with 44,296 empty faces
            # because a ceiling below the seed size blocked every spawn for the whole run.
            # This is the same "isn't optional" carve-out the 1e-8 floor on rescue_thresh
            # above already makes, applied to the caps as well.
            _empty = torch.bincount(
                self.gaussians_to_mesh_indices, minlength=self.mesh_faces.shape[0]
            ) == 0
            _must = torch.where(_empty)[0]
            if _caps:
                cap = max(min(_caps), 0)
                if n_qualified > cap:
                    keep = torch.topk(worst_def[rescue_faces], cap).indices
                    rescue_faces = rescue_faces[keep]
            if _must.numel():
                # Union, so an empty face is served whether or not the cap already kept it.
                rescue_faces = torch.unique(torch.cat([rescue_faces, _must]))
                CONSOLE.log(
                    f"coverage_rescue_spawn: {_must.numel()} face(s) with zero Gaussians "
                    f"served regardless of the caps"
                )
            n_rescue = int(rescue_faces.shape[0])
            CONSOLE.log(
                f"coverage_rescue_spawn: spawning {n_rescue} gaussian(s) "
                f"({n_qualified} face(s) qualified) below "
                f"coverage_rescue_thresh={self.config.coverage_rescue_thresh} "
                f"(centroid or corner)"
            )
            if n_rescue == 0:
                return

            xyz_radius_rescue = self.xyz_radius[rescue_faces]  # (n_rescue, 3)

            # Barycentric weight for where to spawn: centroid, or biased toward whichever
            # corner was worst (0.8/0.1/0.1 -- close to that vertex without literally
            # sitting on it, same spirit as the near-vertex init_copies_per_face seed
            # positions, just more aggressive since this is specifically targeting a
            # corner gap).
            _rescue_bary = torch.tensor(
                [
                    [1 / 3, 1 / 3, 1 / 3],  # worst point was the centroid
                    [0.8, 0.1, 0.1],  # worst point was corner 0
                    [0.1, 0.8, 0.1],  # worst point was corner 1
                    [0.1, 0.1, 0.8],  # worst point was corner 2
                ],
                dtype=torch.float32,
                device=self.device,
            )
            bary_w = _rescue_bary[worst_idx[rescue_faces]]  # (n_rescue, 3)

            new_means_2d = (
                bary_w[:, 0:1] * self.mesh_triangles_2d_coords_a[rescue_faces]
                + bary_w[:, 1:2] * self.mesh_triangles_2d_coords_b[rescue_faces]
                + bary_w[:, 2:3] * self.mesh_triangles_2d_coords_c[rescue_faces]
            )
            new_normal_elevates = torch.zeros(n_rescue, device=self.device)
            # Start at the midpoint of the allowed [min_scale_frac, upper_scale] range
            # -- same spirit as any other fresh Gaussian, gets clamped/refined from there.
            _mid_frac = 0.5 * (self.config.min_scale_frac + self.config.upper_scale)
            new_scales = torch.log(_mid_frac * xyz_radius_rescue + 1e-20)
            new_quats = torch.zeros(n_rescue, 3, device=self.device)

            # Reuse the original mesh texture color for that face, same source
            # populate_modules() used for the very first seeding.
            seed_features_dc = self.seed_mesh["features_dc"].to(self.device)[rescue_faces]
            new_shs = torch.zeros(n_rescue, self.dim_sh, 3, device=self.device)
            if self.config.sh_degree > 0:
                new_shs[:, 0, :3] = RGB2SH(seed_features_dc)
            else:
                new_shs[:, 0, :3] = torch.logit(seed_features_dc, eps=1e-10)
            new_opacities = torch.logit(self.config.init_opacity * torch.ones(n_rescue, 1, device=self.device))

            new_params = {
                "means_2d": new_means_2d,
                "normal_elevates": new_normal_elevates,
                "scales": new_scales,
                "quats": new_quats,
                "features_dc": new_shs[:, 0, :],
                "features_rest": new_shs[:, 1:, :],
                "opacities": new_opacities,
            }

            for name, param in self.gauss_params.items():
                self.gauss_params[name] = torch.nn.Parameter(
                    torch.cat([param.detach(), new_params[name]], dim=0)
                )
            self.gaussians_to_mesh_indices = torch.cat([self.gaussians_to_mesh_indices, rescue_faces], dim=0)
            # Rescue spawns are ordinary detail Gaussians, not base (base membership is
            # only ever created at initialization; with the base layer enabled, rescue
            # rarely fires at all since every face's base keeps its density up).
            self.is_base = torch.cat([
                self.is_base,
                torch.zeros(n_rescue, dtype=torch.bool, device=self.is_base.device),
            ], dim=0)
            # These rows ARE marked rescue (see config.coverage_rescue_thresh and the
            # scales property): a face only reaches this spawn path because it's
            # genuinely under-covered, and sizing this new Gaussian off the same
            # copies_scale_shrink_power-divided ceiling every ordinary N-way
            # partitioning copy uses can leave it too small to actually close that gap.
            # is_rescue gives just this row a wider, undivided per-face ceiling/floor
            # reference instead, without touching any other Gaussian on the same face.
            self.is_rescue = torch.cat([
                self.is_rescue,
                torch.ones(n_rescue, dtype=torch.bool, device=self.is_rescue.device),
            ], dim=0)
            # Same one-cycle grace period as a reassigned row (see populate_modules):
            # this Gaussian is brand new this cycle, sized off the midpoint of its
            # face's allowed range -- give it until the NEXT cull_gaussians() call
            # before the at-floor check can consider it, rather than possibly
            # evaluating it in the same cycle before it's had any training steps.
            self.skip_floor_cull = torch.cat([
                self.skip_floor_cull,
                torch.ones(n_rescue, dtype=torch.bool, device=self.skip_floor_cull.device),
            ], dim=0)

            # dup_in_optim only uses this mask's *count* (to size the zero-init
            # optimizer state it appends), not which specific Gaussians it points at --
            # any n_rescue valid indices into the pre-append arrays work as the template.
            template_idcs = torch.arange(n_rescue, device=self.device)
            self.dup_in_all_optim(optimizers, template_idcs, 1)

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones based on in-plane footprint only (exclude normal-direction thickness,
            # so thickening via face_flat_coef doesn't get misclassified as "too big" and culled)
            toobigs = (torch.exp(self.scales[:, :2]).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                if self.max_2Dsize is not None:
                    toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        # hard-cull needle-like Gaussians in XY plane
        scale_exp_xy = torch.exp(self.scales[:, :2])
        needle_ratio = scale_exp_xy.amax(dim=-1) / (scale_exp_xy.amin(dim=-1) + 1e-8)
        needles = (needle_ratio > self.config.max_gauss_ratio * 4).squeeze()
        culls = culls | needles

        # Cull rows that stay invisibly small on screen (see config.cull_screen_size_min).
        # max_2Dsize is only maintained before stop_split_at and is reset to None at the
        # end of every refine cycle, so this is a no-op late in training. The > 0 test is
        # load-bearing: a zero entry means "not seen this cycle" or "just created", never
        # "small". Reported quantiles are over seen rows only, for choosing the threshold.
        toosmall_count = 0
        size_report = ""
        if self.max_2Dsize is not None:
            seen = (self.max_2Dsize > 0).squeeze()
            if seen.any():
                seen_sizes = self.max_2Dsize.squeeze()[seen]
                q = torch.quantile(
                    seen_sizes.float(),
                    torch.tensor([0.01, 0.10, 0.50], device=seen_sizes.device),
                )
                size_report = (
                    f", max_2Dsize seen p1/p10/p50 = "
                    f"{q[0].item():.2e}/{q[1].item():.2e}/{q[2].item():.2e}"
                )
            if self.config.cull_screen_size_min > 0 and seen.any():
                toosmall = seen & (
                    self.max_2Dsize.squeeze() < self.config.cull_screen_size_min
                )
                culls = culls | toosmall
                toosmall_count = int(torch.sum(toosmall).item())

        at_floor_count = 0
        if self.config.cull_at_floor_gaussians:
            # Same check as already_at_floor in split_gaussians()/dup_gaussians(): use
            # the rescue-widened radius reference so a coverage-rescue Gaussian is
            # compared against ITS OWN (possibly widened) floor, not a narrower one it
            # was never sized against. 1% tolerance for floating point, same as there.
            floor_xy = self.config.min_scale_frac * self._rescue_widened_xyz_radius()[:, :2]
            at_floor = (scale_exp_xy <= floor_xy * 1.01).all(dim=-1)
            # Exempt rows whose face just changed this training run (reassignment or a
            # fresh rescue spawn) -- see populate_modules' skip_floor_cull docstring:
            # without this, a relabel alone (not any actual shrinking) can make a row
            # look "at floor" against its new face's xyz_radius and get permanently
            # deleted before it has a chance to adapt.
            at_floor = at_floor & ~self.skip_floor_cull
            culls = culls | at_floor
            at_floor_count = torch.sum(at_floor).item()

        # Base-layer Gaussians are never culled, for ANY reason -- including the
        # split-prune extra_cull_mask (a base Gaussian that acts as a split source keeps
        # existing alongside its children). This is the core of the coverage guarantee:
        # every face keeps its base at every step, so no face can ever go uncovered (and
        # the never-zero-Gaussians-per-face assert below can never fire for base faces).
        # is_base is all-False when the feature is disabled, making this a no-op.
        culls = culls & ~self.is_base

        faces_gaussian_cnt = torch.bincount(self.gaussians_to_mesh_indices)
        assert faces_gaussian_cnt.shape[0] == self.mesh_faces.shape[0]
        assert (not torch.any(faces_gaussian_cnt == 0)), torch.count_nonzero(faces_gaussian_cnt == 0)

        gaussians_to_mesh_indices_after_culls = self.gaussians_to_mesh_indices[~culls]
        faces_gaussian_cnt_after_culls = torch.bincount(gaussians_to_mesh_indices_after_culls)
        if faces_gaussian_cnt_after_culls.shape[0] < self.mesh_faces.shape[0]:
            faces_gaussian_cnt_after_culls = torch.nn.functional.pad(faces_gaussian_cnt_after_culls, (0, self.mesh_faces.shape[0] - faces_gaussian_cnt_after_culls.shape[0]), "constant", 0)

        pruned_faces_gs = torch.isin(self.gaussians_to_mesh_indices, torch.arange(self.mesh_faces.shape[0], device="cuda")[faces_gaussian_cnt_after_culls == 0])
        rescued_mask = culls & pruned_faces_gs  # would-be-culled but last on their face
        culls = torch.logical_and(culls, ~pruned_faces_gs)

        # Rescued Gaussians were transparent (below cull_alpha_thresh); boost their opacity
        # so they don't permanently occupy a face while contributing nothing to coverage.
        if rescued_mask.any():
            rescue_opacity = max(self.config.cull_alpha_thresh * 8.0, 0.4)
            rescue_logit = torch.logit(torch.tensor(rescue_opacity, device=self.device))
            self.gauss_params["opacities"].data[rescued_mask] = torch.clamp(
                self.gauss_params["opacities"].data[rescued_mask],
                min=rescue_logit.item(),
            )

        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])
            # print(name, self.gauss_params[name].shape)
        self.gaussians_to_mesh_indices = self.gaussians_to_mesh_indices[~culls]
        self.is_base = self.is_base[~culls]
        self.is_rescue = self.is_rescue[~culls]
        # skip_floor_cull's one-cycle grace period ends here: filter alongside the
        # other per-Gaussian arrays for surviving rows, then unconditionally reset to
        # all-False -- this cull_gaussians() call is exactly the "next cull cycle" the
        # grace period was deferring to, so every surviving row (reassigned or not) is
        # fully eligible for the at-floor check again starting next cycle.
        self.skip_floor_cull = torch.zeros_like(self.gaussians_to_mesh_indices, dtype=torch.bool)
        # print("self.gaussians_to_mesh_indices: ", self.gaussians_to_mesh_indices.shape)
        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, "
            f"{toosmall_count} too small on screen, "
            f"{at_floor_count} at min-scale floor, {self.num_points} remaining)"
            + size_report
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")

        # def sample_pts_in_triangle(num_samples):
        #     uv = np.random.rand(num_samples, 2)
        #     cond = np.sum(uv, axis=-1) > 1
        #     uv[cond] = 1 - uv[cond]
        #     uvw = np.concatenate([uv, 1 - np.sum(uv, axis=-1, keepdims=True)], axis=-1)
        #     return torch.from_numpy(uvw).float()
        #
        # new_bary_coords = torch.log(sample_pts_in_triangle(n_splits * samps).reshape(samps * n_splits, 3).cuda() + 1e-20)
        # new_normal_elevates = self.gauss_params["normal_elevates"][split_mask].repeat(samps)

        # start with sample new scales
        size_fac = 1.6
        if self.config.unconstrained_scale:
            real_scales = torch.exp(self.scales[split_mask])
            new_scales = torch.log(real_scales / size_fac + 1e-20).repeat(samps, 1)
            # self.scales is a computed property (non-leaf), so assigning to it is a NO-OP.
            # Write directly to gauss_params["scales"] to actually reduce the original's scale.
            # Base-layer originals are excluded: their rendered size is floored at full
            # face radius anyway (shrinking the raw value would be invisible, only digging
            # the raw parameter below a floor it can never render under), and unlike
            # ordinary splits they are not pruned afterward (cull skips base), so the
            # usual "shrink the original because it's about to be replaced by its
            # children" rationale doesn't apply. Children (new_scales above, computed from
            # the rendered/floored size) still start at 1/1.6 of the base's size.
            # Same already-at-floor guard as dup_gaussians (2026-07-17 fix, never mirrored
            # here): once a Gaussian's *rendered* size is already at/near min_scale_frac,
            # shrinking its raw parameter further has no visual effect (self.scales clamps
            # it there either way) and only pushes the raw value deeper below the floor,
            # making it that much harder for the optimizer to ever grow it back. Split
            # fires on the larger, high-gradient Gaussians most of the time, but a
            # Gaussian can still be re-triggered repeatedly over training and its size can
            # reach the floor by other means (reassignment to a smaller face,
            # coverage/curvature-boosted densification) -- this was previously unguarded
            # here even though the identical case was fixed for dup_gaussians.
            # Rescue-widened, not raw self.xyz_radius: a rescue row's *rendered* size
            # (below) is measured against its own widened ceiling/floor (see the scales
            # property / _rescue_widened_xyz_radius()), so the floor compared against
            # here must be the same widened value or already_at_floor would almost
            # never recognize a rescue row as being at its own (wider) floor.
            xyz_radius_all = self._rescue_widened_xyz_radius()  # (N, 3)
            floor_xy = self.config.min_scale_frac * xyz_radius_all[:, :2]  # (N, 2)
            rendered_xy = torch.exp(self.scales[:, :2])  # (N, 2), clamped/rendered size
            already_at_floor = (rendered_xy <= floor_xy * 1.01).all(dim=-1)  # (N,)
            # _base_floor_mask(), not is_base: the reason base originals are excluded from
            # the shrink is that their rendered size is floored anyway, so shrinking the
            # raw value is invisible and only digs it below a floor it can never render
            # under. That reasoning applies to a *floored* base row, not to one whose
            # floor adaptive_coverage_floor has released -- a released row renders at its
            # raw size like any other Gaussian and must shrink with its children, or
            # splitting it would leave the parent permanently oversized.
            shrink_mask = split_mask & (~already_at_floor) & (~self._base_floor_mask())
            self.gauss_params["scales"].data[shrink_mask] = torch.log(
                torch.exp(self.gauss_params["scales"].data[shrink_mask]) / size_fac + 1e-20
            )

        else:
            upper_scale = self.config.upper_scale
            real_scales = torch.exp(self.scales[split_mask])
            new_real_scales = real_scales / size_fac
            xyz_radius_split = self.xyz_radius[self.gaussians_to_mesh_indices][split_mask]
            new_scales = torch.logit((new_real_scales / upper_scale) / xyz_radius_split).repeat(samps, 1)

        # then sample new means
        real_scale_min_xy = torch.min(real_scales[:, :2], dim=-1)[0].reshape(-1, 1).repeat(samps, 1)
        radius_splits = torch.abs(torch.randn(n_splits * samps, 1, device="cuda", dtype=torch.float32)) * 0.5
        theta_splits = torch.rand(n_splits * samps, 1, device="cuda", dtype=torch.float32) * torch.pi * 2.0

        # bary_coords = torch.nn.functional.softmax(self.gauss_params["bary_coords"], dim=-1)[split_mask]
        mesh_faces_verts = self.mesh_faces_verts[self.gaussians_to_mesh_indices][split_mask]
        # # normals = self.normals[self.gaussians_to_mesh_indices][split_mask]
        # # radius = self.radius[self.gaussians_to_mesh_indices][split_mask]
        #
        # v_proj_splits = torch.sum(mesh_faces_verts * bary_coords.reshape(-1, 3, 1), dim=1).reshape(-1, 3)

        means_2d = self.gauss_params["means_2d"][split_mask]
        faces_axis_x = self.faces_axis_x[self.gaussians_to_mesh_indices][split_mask]
        faces_axis_y = self.faces_axis_y[self.gaussians_to_mesh_indices][split_mask]
        means_2d_transformed = torch.sum(
            means_2d.unsqueeze(-1) * torch.stack([faces_axis_x, faces_axis_y], dim=1), dim=1).reshape(-1, 3)
        v_proj_splits = means_2d_transformed + mesh_faces_verts[:, 0]

        largest_radius_splits = compute_min_distance(v_proj_splits, mesh_faces_verts[:, 0], mesh_faces_verts[:, 1], mesh_faces_verts[:, 2])
        largest_radius_splits = largest_radius_splits.unsqueeze(-1).repeat(samps, 1)
        v_proj_splits = v_proj_splits.repeat(samps, 1)

        x_axis_splits = faces_axis_x.repeat(samps, 1)
        y_axis_splits = faces_axis_y.repeat(samps, 1)

        real_radius_splits = radius_splits * real_scale_min_xy
        real_radius_splits = torch.where(real_radius_splits > largest_radius_splits * 0.99, largest_radius_splits * 0.99, real_radius_splits)

        direction_splits = torch.cos(theta_splits) * x_axis_splits + torch.sin(theta_splits) * y_axis_splits

        # norm_value = torch.sum(direction_splits * direction_splits, dim=-1)
        # print("direction_splits: ", norm_value.mean(), norm_value.max(), norm_value.min())
        # assert False

        v_sample_proj_splits = v_proj_splits + real_radius_splits * direction_splits
        v_sample_proj_splits_2d_coords = v_sample_proj_splits - mesh_faces_verts[:, 0].repeat(samps, 1)

        new_means_2d = torch.stack([
            torch.sum(v_sample_proj_splits_2d_coords * x_axis_splits, dim=-1).reshape(-1),
            torch.sum(v_sample_proj_splits_2d_coords * y_axis_splits, dim=-1).reshape(-1),
        ], dim=-1)

        new_normal_elevates = self.gauss_params["normal_elevates"][split_mask].repeat(samps)

        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 5, sample new quats
        new_quats = self.gauss_params["quats"][split_mask].repeat(samps, 1)
        out = {
            "means_2d": new_means_2d,
            "normal_elevates": new_normal_elevates,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")

        # Unlike split_gaussians (which shrinks scale on split), this used to clone the
        # original's scale unchanged into the new copy. Two identically-sized Gaussians
        # landing near each other doesn't reduce the "small scale + high gradient"
        # trigger condition, so a Gaussian that duplicated once could duplicate again
        # next refine cycle, and again: runaway duplication of tiny, heavily overlapping
        # clones (more tiny points, more near-coincident duplicates fighting for blend
        # order -> flicker). Shrink scale on duplication too, so duplicating actually
        # reduces each copy's footprint instead of just adding a same-size neighbor.
        #
        # 2026-07-17: uses dup_size_shrink_factor (gentler, configurable) instead of
        # sharing split_gaussians' factor, and skips shrinking entirely for Gaussians
        # whose *rendered* (clamped) size is already at/near their face's min_scale_frac
        # floor. Found via diagnose_gaussian_scale.py: the x/y size ratio distribution's
        # median sat exactly on min_scale_frac (both at 0.3 and again at 0.5 after raising
        # it) no matter how the split/dup threshold or coverage mechanisms were tuned --
        # every densification event, split or dup, unconditionally shrinks scale and
        # nothing ever grows an existing Gaussian back, so with curvature_densify_scale
        # and coverage_densify_scale both now boosting how *often* densification
        # triggers, most of the population drifts down to the floor over the training
        # window regardless of where that floor is set. Once a Gaussian's rendered size
        # is already floored, shrinking its raw parameter further doesn't change what
        # actually renders (self.scales clamps it there either way) -- it only pushes the
        # raw value further below the floor for no visual effect, making it that much
        # harder to recover later if the optimizer ever wants to grow it back. Skipping
        # the shrink there stops digging that hole deeper; it's still duplicated (spread
        # to a different position within the face below), so this isn't a wasted
        # coincident clone even without shrinking.
        if self.config.unconstrained_scale:
            # Rescue-widened, not raw self.xyz_radius: a rescue row's *rendered* size
            # (below) is measured against its own widened ceiling/floor (see the scales
            # property / _rescue_widened_xyz_radius()), so the floor compared against
            # here must be the same widened value or already_at_floor would almost
            # never recognize a rescue row as being at its own (wider) floor.
            xyz_radius_all = self._rescue_widened_xyz_radius()  # (N, 3)
            floor_xy = self.config.min_scale_frac * xyz_radius_all[:, :2]  # (N, 2)
            rendered_xy = torch.exp(self.scales[:, :2])  # (N, 2), clamped/rendered size
            already_at_floor = (rendered_xy <= floor_xy * 1.01).all(dim=-1)  # (N,)
            # ~_base_floor_mask(): a FLOORED base-layer Gaussian's rendered size is the
            # full face radius (not min_scale_frac x radius, so already_at_floor won't
            # catch it) and shrinking its raw value is invisible for the same reason --
            # skip it, same rationale as split_gaussians. A base row whose floor
            # adaptive_coverage_floor has released is NOT skipped: it renders at its raw
            # size like any other Gaussian, so it has to shrink with its duplicate or dup
            # would just add a second copy at the original (too large) size.
            shrink_mask = dup_mask & (~already_at_floor) & (~self._base_floor_mask())
            self.gauss_params["scales"].data[shrink_mask] = torch.log(
                torch.exp(self.gauss_params["scales"].data[shrink_mask]) / self.config.dup_size_shrink_factor + 1e-20
            )

        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]

        # In mesh-constrained splatting, duplicates at the same means_2d receive identical
        # gradients and converge to the same position, wasting the duplicate.
        # Spread duplicates randomly within their mesh face so each copy can cover
        # a different part of the triangle.
        mesh_faces_verts = self.mesh_faces_verts[self.gaussians_to_mesh_indices][dup_mask]
        faces_axis_x = self.faces_axis_x[self.gaussians_to_mesh_indices][dup_mask]
        faces_axis_y = self.faces_axis_y[self.gaussians_to_mesh_indices][dup_mask]

        means_2d = self.gauss_params["means_2d"][dup_mask]
        means_2d_transformed = torch.sum(
            means_2d.unsqueeze(-1) * torch.stack([faces_axis_x, faces_axis_y], dim=1), dim=1
        ).reshape(-1, 3)
        v_proj = means_2d_transformed + mesh_faces_verts[:, 0]

        largest_radius = compute_min_distance(
            v_proj, mesh_faces_verts[:, 0], mesh_faces_verts[:, 1], mesh_faces_verts[:, 2]
        ).unsqueeze(-1)
        radius_dups = torch.rand(n_dups, 1, device=self.device, dtype=torch.float32) * largest_radius
        theta_dups = torch.rand(n_dups, 1, device=self.device, dtype=torch.float32) * torch.pi * 2.0
        direction_dups = torch.cos(theta_dups) * faces_axis_x + torch.sin(theta_dups) * faces_axis_y

        v_sample = v_proj + radius_dups * direction_dups
        v_sample_2d = v_sample - mesh_faces_verts[:, 0]
        new_means_2d = torch.stack([
            torch.sum(v_sample_2d * faces_axis_x, dim=-1),
            torch.sum(v_sample_2d * faces_axis_y, dim=-1),
        ], dim=-1)
        new_dups["means_2d"] = new_means_2d
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN],
                self.report_coverage,
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def report_coverage(self, step: int):
        """Report how much of the mesh surface is actually reached by at least one Gaussian.

        This is a UNION test, sampled across every triangle: lay points over each face, ask
        each one whether any Gaussian reaches it, and report the area-weighted fraction that
        do. It answers the question the project actually cares about -- "is any part of the
        mesh left with nothing on it" -- and it is satisfied equally well by many small
        Gaussians or by one large one.

        WHAT THIS REPLACED, AND WHY (2026-08-31). The previous version summed
        pi * sx * sy * alpha per face and capped it at the face's own triangle area. That is
        an AREA BUDGET, and it is wrong for this question in three separate ways:

          - Blind to placement. A face whose Gaussians all pile up at its centroid scores
            the same as one covered evenly, so the number cannot distinguish "covered" from
            "has enough total footprint somewhere".
          - Weighted by opacity, which conflates how big a Gaussian is with how solid it is.
            Two Gaussians at alpha 0.5 sum to the same "area" as one at alpha 1.0 while
            covering twice the surface between them.
          - Excluded the vertex filler layer entirely -- it reads self.opacities and
            self.scales, which are the face-based population only, so 707,579 Gaussians that
            genuinely sit on the surface contributed nothing.

        Taken together those made it a proxy for size-times-opacity, i.e. for blur: on
        table_gs14 it read 79.15% while the surface was in fact 100.000% covered, and every
        attempt to push it up made the render blurrier. It is not a coverage number and it
        was never safe to read as one.

        HOW THE NUMBER IS BUILT. Sample points are a barycentric grid per face, weighted by
        face area so the result is a fraction of mesh AREA rather than of faces. A Gaussian's
        contribution at a point is alpha * exp(-0.5 * M^2) with M the in-plane Mahalanobis
        distance, computed in 3D against the Gaussian's own local x/y axes -- the same
        quantity the rasterizer accumulates, and the same one _compute_coverage_density()
        uses at its four check points.

        MAX, not sum: the requirement is "reached by at least one Gaussian", and summing
        would let several distant tails add up to a phantom pass. Three thresholds are
        reported because "reaches" has no single honest definition -- 0.50 is solidly inside
        a Gaussian, 0.01 is roughly where the rasterizer stops writing anything at all.

        Each face is tested against its own Gaussians, those of its edge-adjacent faces, and
        the vertex-layer Gaussians at its three corners. A Gaussian two or more faces away is
        not tested, which makes every number here CONSERVATIVE -- the true coverage is at
        least this good. On this mesh a Gaussian would have to be several times its face's
        size to reach that far.

        Registered as an AFTER_TRAIN callback, so like the previous version it may call
        finalize_face_assignment(). Do not add a mid-training call site: that would turn
        this into the periodic reassignment config.reassign_face_every exists to control.
        """
        # 3 -> 9 points per face. Enough to catch a bare corner or a bare middle without
        # making this pass expensive; it runs once, after training.
        K_LEVEL = 3
        THRESHOLDS = (0.5, 0.1, 0.01)

        with torch.no_grad():
            means = self.means.detach()
            scales = torch.exp(self.scales.detach())
            quats = self.quats.detach()
            opac = torch.sigmoid(self.opacities.detach()).squeeze(-1)

            self.finalize_face_assignment()
            gs_face = self.gaussians_to_mesh_indices
            dev = means.device
            nf = self.mesh_faces.shape[0]
            nv = self.mesh_verts.shape[0]

            R = quaternion_to_matrix(quats)
            g_ax, g_ay = R[:, :, 0], R[:, :, 1]
            g_sx = scales[:, 0].clamp(min=1e-12)
            g_sy = scales[:, 1].clamp(min=1e-12)

            # --- barycentric sample grid, cell centroids so nothing lands on an edge ------
            bary = []
            for a in range(K_LEVEL):
                for b in range(K_LEVEL - a):
                    u, v = (a + 1 / 3) / K_LEVEL, (b + 1 / 3) / K_LEVEL
                    bary.append((u, v, 1 - u - v))
                    if a + b < K_LEVEL - 1:
                        u2, v2 = (a + 2 / 3) / K_LEVEL, (b + 2 / 3) / K_LEVEL
                        bary.append((u2, v2, 1 - u2 - v2))
            bary = torch.tensor(bary, device=dev, dtype=means.dtype)  # (K, 3)
            K = bary.shape[0]

            fv = self.mesh_faces_verts  # (F, 3, 3)
            face_area = area(fv)

            # --- per-face Gaussian lists, CSR style --------------------------------------
            order = torch.argsort(gs_face)
            per_face = torch.bincount(gs_face, minlength=nf)
            start = torch.zeros(nf + 1, dtype=torch.long, device=dev)
            start[1:] = torch.cumsum(per_face, 0)

            # --- edge-adjacent faces, up to 3 each ----------------------------------------
            e = torch.cat([self.mesh_faces[:, [0, 1]], self.mesh_faces[:, [1, 2]],
                           self.mesh_faces[:, [2, 0]]], dim=0)
            e, _ = torch.sort(e, dim=1)
            key = e[:, 0].to(torch.int64) * (nv + 1) + e[:, 1].to(torch.int64)
            ko = torch.argsort(key)
            ks = key[ko]
            fs = torch.arange(nf, device=dev).repeat(3)[ko]
            same = torch.nonzero(ks[1:] == ks[:-1], as_tuple=False).squeeze(-1)
            src = torch.cat([fs[same], fs[same + 1]])
            dst = torch.cat([fs[same + 1], fs[same]])
            so = torch.argsort(src, stable=True)
            src, dst = src[so], dst[so]
            first = torch.ones_like(src, dtype=torch.bool)
            first[1:] = src[1:] != src[:-1]
            pos = torch.arange(src.shape[0], device=dev)
            rank = pos - torch.cummax(torch.where(first, pos, torch.zeros_like(pos)), 0)[0]
            nbr = torch.full((nf, 3), -1, dtype=torch.long, device=dev)
            keep = rank < 3
            nbr[src[keep], rank[keep]] = dst[keep]

            has_vtx = getattr(self, "vertex_gauss_params", None) is not None
            if has_vtx:
                v_pos = self.vertex_positions
                v_sc = torch.exp(self._vertex_scales().detach())
                v_R = quaternion_to_matrix(self._vertex_quats().detach())
                v_op = torch.sigmoid(self.vertex_gauss_params["opacities"].detach()).squeeze(-1)

            SLOT = 16          # Gaussians pulled from each contributing face per pass
            CHUNK = 8192
            n_ord = order.shape[0]
            covered = torch.zeros(len(THRESHOLDS), device=dev, dtype=means.dtype)
            bare_faces = 0

            # Busiest faces last, so a batch is not padded out to one outlier's count.
            face_order = torch.argsort(per_face)

            for lo in range(0, nf, CHUNK):
                idx = face_order[lo:lo + CHUNK]
                B = idx.shape[0]
                pts = torch.einsum("kb,fbc->fkc", bary, fv[idx])   # (B, K, 3)
                best = torch.zeros(B, K, device=dev, dtype=means.dtype)

                def accumulate(gid, valid):
                    if gid.numel() == 0 or not bool(valid.any()):
                        return
                    gi = torch.where(valid, gid, torch.zeros_like(gid))
                    d = pts.unsqueeze(1) - means[gi].unsqueeze(2)          # (B, S, K, 3)
                    u = torch.einsum("bskc,bsc->bsk", d, g_ax[gi])
                    w = torch.einsum("bskc,bsc->bsk", d, g_ay[gi])
                    m2 = (u / g_sx[gi].unsqueeze(-1)) ** 2 + (w / g_sy[gi].unsqueeze(-1)) ** 2
                    c = opac[gi].unsqueeze(-1) * torch.exp(-0.5 * m2.clamp(max=200.0))
                    torch.maximum(best, torch.where(valid.unsqueeze(-1), c, torch.zeros_like(c)).amax(dim=1), out=best)

                # own face, sliced so the (B, S, K, 3) temporary stays bounded
                mx = int(per_face[idx].max().item()) if B else 0
                for s0 in range(0, mx, SLOT):
                    off = torch.arange(s0, min(s0 + SLOT, mx), device=dev).unsqueeze(0)
                    valid = off < per_face[idx].unsqueeze(1)
                    gid = order[(start[idx].unsqueeze(1) + off).clamp(max=n_ord - 1)]
                    accumulate(gid, valid)

                # edge-adjacent faces
                off = torch.arange(SLOT, device=dev).unsqueeze(0)
                for nb in range(3):
                    nf_i = nbr[idx, nb]
                    ok = nf_i >= 0
                    nf_s = torch.where(ok, nf_i, torch.zeros_like(nf_i))
                    valid = ok.unsqueeze(1) & (off < per_face[nf_s].unsqueeze(1))
                    gid = order[(start[nf_s].unsqueeze(1) + off).clamp(max=n_ord - 1)]
                    accumulate(gid, valid)

                # the three vertex-layer Gaussians at this face's corners
                if has_vtx:
                    vi = self.mesh_faces[idx]                                # (B, 3)
                    d = pts.unsqueeze(1) - v_pos[vi].unsqueeze(2)
                    u = torch.einsum("bskc,bsc->bsk", d, v_R[vi][:, :, :, 0])
                    w = torch.einsum("bskc,bsc->bsk", d, v_R[vi][:, :, :, 1])
                    m2 = (u / v_sc[vi][:, :, 0].clamp(min=1e-12).unsqueeze(-1)) ** 2 \
                       + (w / v_sc[vi][:, :, 1].clamp(min=1e-12).unsqueeze(-1)) ** 2
                    c = v_op[vi].unsqueeze(-1) * torch.exp(-0.5 * m2.clamp(max=200.0))
                    torch.maximum(best, c.amax(dim=1), out=best)

                a_chunk = face_area[idx]
                for t, th in enumerate(THRESHOLDS):
                    covered[t] += ((best >= th).to(means.dtype).mean(dim=1) * a_chunk).sum()
                bare_faces += int(((best < THRESHOLDS[1]).any(dim=1)).sum().item())

            total = face_area.sum()
            pct = (covered / total.clamp(min=1e-20) * 100).tolist()
            CONSOLE.print(
                f"[bold]Gaussian coverage (union over the mesh, area-weighted, "
                f"{K} samples/face):[/bold]\n"
                f"    reached at alpha >= 0.50 (solidly covered)  : {pct[0]:7.3f}%\n"
                f"    reached at alpha >= 0.10 (visibly covered)  : {pct[1]:7.3f}%\n"
                f"    reached at alpha >= 0.01 (anything at all)  : {pct[2]:7.3f}%\n"
                f"    faces with an uncovered sample at 0.10     : {bare_faces} / {nf}\n"
                f"    mesh area {total.item():.3f}; conservative -- only own-face, "
                f"edge-adjacent and corner-vertex Gaussians are tested"
            )

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        skip_save = False
        if self.step == 0 and os.path.exists(self.save_extra_info_path):
            try:
                existing_len = torch.load(self.save_extra_info_path)["gaussians_to_mesh_indices"].shape[0]
                # At step 0 (before Trainer._load_checkpoint runs), a fresh mesh-only init
                # can be much smaller than a previously-saved run that had already split
                # many times. Resuming via --load-dir resizes gauss_params to match that
                # larger checkpoint and re-reads this same file for gaussians_to_mesh_indices,
                # so don't clobber it with the smaller fresh array here -- otherwise the two
                # end up mismatched sizes and get_outputs crashes right after resuming.
                # Any mismatch at step 0 means a resume is in progress, in EITHER
                # direction. The original test only covered "checkpoint is bigger",
                # assuming densification only ever grows the population -- but culling
                # shrinks it, and table_gs24 reached step 20000 with 2,404,821 rows
                # against a 5,580,916 seed. The fresh array then overwrote the saved one
                # and step 20000 became unrecoverable: gauss_params live in the
                # checkpoint, but which face each row belongs to and which rows are base
                # live only here. Gated on step 0 so that normal training saves, where
                # the size legitimately changes every cycle, are unaffected.
                skip_save = (
                    self.step == 0
                    and existing_len > 0
                    and existing_len != self.gaussians_to_mesh_indices.shape[0]
                )
            except Exception:
                skip_save = False
        if not skip_save:
            torch.save({
                "gaussians_to_mesh_indices": self.gaussians_to_mesh_indices.cpu(),
                # Same lifecycle as gaussians_to_mesh_indices (plain tensor, not in the
                # state dict): must be persisted here or a resumed run would lose base
                # membership and silently drop every base-layer guarantee.
                "is_base": self.is_base.cpu(),
                "is_rescue": self.is_rescue.cpu(),
            }, self.save_extra_info_path)
            print("extra info is saved to: ", self.save_extra_info_path)
        gps = {
            name: [self.gauss_params[name]]
            for name in ["means_2d", "normal_elevates", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }
        # Vertex-anchored coverage-filler Gaussians (see populate_modules/get_outputs):
        # own param group names, not merged into the face-based ones above --
        # dup_in_optim()/remove_from_optim() assume exactly one tensor per group when
        # the face-based population is split/duplicated/culled, so mixing the two
        # populations into the same group would break that surgery. Registered in
        # method_configs.py's "splatfacto_on_mesh_uc"/"splatfacto_on_mesh_uc_longer"
        # optimizers dicts (own LR, same as their face-based counterparts).
        if self.vertex_gauss_params is not None:
            gps["vertex_features_dc"] = [self.vertex_gauss_params["features_dc"]]
            gps["vertex_features_rest"] = [self.vertex_gauss_params["features_rest"]]
            gps["vertex_opacities"] = [self.vertex_gauss_params["opacities"]]
        # Scale/rotation made trainable-within-bounds 2026-07-31 (see _vertex_scales()/
        # _vertex_quats()) -- same own-group-per-tensor reasoning as the three above.
            gps["vertex_scales"] = [self.vertex_gauss_params["scales"]]
            gps["vertex_quats"] = [self.vertex_gauss_params["quats"]]
        return gps

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        # Vertex-anchored coverage-filler Gaussians (see populate_modules): appended
        # after the face-based population, not cropped (self.crop_box only matters for
        # interactive non-training visualization; skipping crop for this small, fixed
        # population is an accepted simplification, not a correctness issue during
        # training/export). n_face_based is captured so after_train() can slice back
        # down to just the face-based rows for its densification bookkeeping -- this
        # filler population is never split/duplicated/culled/reassigned, so it must
        # never feed those stats.
        n_face_based = means_crop.shape[0]
        if self.vertex_gauss_params is not None:
            vertex_colors = torch.cat(
                (
                    self.vertex_gauss_params["features_dc"][:, None, :],
                    self.vertex_gauss_params["features_rest"],
                ),
                dim=1,
            )
            means_crop = torch.cat([means_crop, self.vertex_positions], dim=0)
            # vertex scale/rotation are now trainable-within-bounds (2026-07-31) -- see
            # _vertex_scales()/_vertex_quats().
            scales_crop = torch.cat([scales_crop, self._vertex_scales()], dim=0)
            quats_crop = torch.cat([quats_crop, self._vertex_quats()], dim=0)
            colors_crop = torch.cat([colors_crop, vertex_colors], dim=0)
            opacities_crop = torch.cat([opacities_crop, self.vertex_gauss_params["opacities"]], dim=0)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac, "round")
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        camera.rescale_output_resolution(camera_scale_fac, "round")  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N + V, 2] (N face-based + V vertex-anchored)
        self.radii = info["radii"][0]  # [N + V]
        # after_train()'s densification bookkeeping (xys_grad_norm/vis_counts) must
        # only ever see the face-based rows -- the vertex-anchored filler population is
        # never split/duplicated/culled/reassigned, so it has no business feeding those
        # stats. Recorded here (not recomputed in after_train) since n_face_based is only
        # available where means_crop is actually assembled.
        self.n_face_based_gaussians = n_face_based
        alpha = alpha[:, ...]

        grazing_weight = None
        if self.config.grazing_weight_enabled and self.training:
            with torch.no_grad():
                # Extended with vertex_normals to match means_crop's now-larger count
                # (face-based + vertex-anchored) -- see the concatenation above.
                normals_crop = self.normals[self.gaussians_to_mesh_indices]
                if self.vertex_gauss_params is not None:
                    normals_crop = torch.cat([normals_crop, self.vertex_normals], dim=0)
                if crop_ids is not None:
                    normals_crop = normals_crop[crop_ids]
                cam_pos = optimized_camera_to_world[0, :3, 3]
                view_dirs = torch.nn.functional.normalize(cam_pos - means_crop, dim=-1)
                # cos of the angle between the face normal and the viewing direction; clamp
                # negative (back-facing) values to 0 since those shouldn't be visible anyway.
                cos_angle = torch.sum(normals_crop * view_dirs, dim=-1, keepdim=True).clamp(0.0, 1.0)
                weight_render, weight_alpha, _ = rasterization(
                    means=means_crop,
                    quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                    scales=torch.exp(scales_crop),
                    opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                    colors=cos_angle.expand(-1, 3),
                    viewmats=viewmat,
                    Ks=K,
                    width=W,
                    height=H,
                    tile_size=BLOCK_WIDTH,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode="RGB",
                    sh_degree=None,
                    sparse_grad=False,
                    absgrad=False,
                )
                # normalize by accumulation to get the average reliability of whatever surface
                # is visible at each pixel, independent of how opaque that pixel ended up being.
                grazing_weight = (weight_render[..., :1] / (weight_alpha + 1e-6)).clamp(0.0, 1.0)

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "grazing_weight": grazing_weight.squeeze(0) if grazing_weight is not None else None,  # type: ignore
        }  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        # print("predicted_rgb: ", predicted_rgb.shape)
        # print("gt_rgb: ", gt_rgb.shape)
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)

        return metrics_dict

    def _set_face_overlap_pairs(self, is_fold=None, nearest_foreign=None):
        """
        Build (and store) the face pairs that are geometrically overlapping but
        topologically DISTANT -- the mesh folds. Third input to color_consistency_lambda,
        added 2026-08-06.

        Why this exists: face_adjacency_pairs() is purely topological, so no matter how
        wide its neighborhood is it can never relate two sheets that touch in 3D space
        while sitting tens to hundreds of edges apart along the surface -- an armpit, a
        limb/torso junction, any fold. Those pairs need a geometric query, which is what
        this provides.

        Sizing expectation, so this is not mistaken for the main event: measured on the
        exported armadillo (2026-08-06), genuine folds are only 0.2% of the total
        flicker potential -- the bulk of what term (b) was previously missing turned out
        to be ordinary same-sheet corner neighbors, and widening face_adjacency_pairs()
        to vertex adjacency is what addresses that (17.2%). What folds do have is the
        highest per-pair color spread of any class (median |dC| 0.0059, p90 0.0338 = 8.7x
        the same-face class) concentrated into a few small regions, so they can matter
        visually far more than 0.2% suggests -- but if flicker persists, this term is not
        where the volume is.

        Reuses detect_fold_safe_radius()'s existing nearest-non-1-ring-face search rather
        than doing its own spatial query -- that function already computes exactly this
        relation for the radius cap, it was simply discarding the indices. One pair per
        folded face (that face and whatever it faces across the gap) is enough to tie the
        two sheets together: terms (a)/(b) already pull each sheet internally consistent,
        so a single link per fold face propagates through them to the whole neighborhood.

        Independent of enable_fold_detection on purpose -- see the call site in
        populate_modules. Skipped entirely (empty tensors) when the color loss is off,
        unlike face_adj_a/b which is always built: that one is a cheap edge hash, this
        one is a chunked all-pairs centroid search, too expensive to run for nothing.

        Args:
            is_fold, nearest_foreign: detect_fold_safe_radius() outputs, passed in when
                the caller already ran it (populate_modules) so it isn't run twice.
                Both None -> recomputed here (append_from_mesh / load path).
        """
        device = self.mesh_faces.device
        if self.config.color_consistency_lambda <= 0:
            _empty = torch.zeros(0, dtype=torch.long, device=device)
            self.face_overlap_a, self.face_overlap_b = _empty, _empty.clone()
            return

        if is_fold is None or nearest_foreign is None:
            _ring_radius = face_ring_radius(self.mesh_faces, self.mesh_faces_verts)
            _, is_fold, nearest_foreign = detect_fold_safe_radius(
                self.mesh_faces, self.mesh_faces_verts, _ring_radius,
                self.config.fold_safety_frac, self.config.fold_ratio,
            )

        src = torch.nonzero(is_fold & (nearest_foreign >= 0)).reshape(-1)
        dst = nearest_foreign[src]
        # Both sides of a fold usually nominate each other, which would emit the same
        # pair twice and silently double its weight. Canonicalize to (lo, hi) and dedupe.
        pairs = torch.stack([torch.minimum(src, dst), torch.maximum(src, dst)], dim=-1)
        pairs = torch.unique(pairs, dim=0) if pairs.numel() > 0 else pairs.reshape(0, 2)
        self.face_overlap_a = pairs[:, 0].contiguous()
        self.face_overlap_b = pairs[:, 1].contiguous()

    def _compute_color_consistency(self):
        """
        Color-disagreement penalty over spatially-overlapping Gaussians -- the |dC| factor
        of the a1*a2*|dC| flicker term. See config.color_consistency_lambda for the
        measurements that motivated this.

        Computed against per-face opacity-weighted MEAN colors rather than over explicit
        pairs. A pairwise form would need a spatial neighbor search every step; going
        through the face mean is O(N + E) with plain scatter ops and pulls the same
        cluster of overlapping Gaussians together, since minimizing each member's squared
        deviation from their common mean is exactly minimizing their pairwise squared
        differences (up to a constant factor).

        Three terms, because overlap happens at three scales -- measured on the exported
        armadillo model, 47.8% of overlapping Gaussian pairs sit on DIFFERENT faces, so
        an own-face-only penalty would miss about half of them:

          (a) within a face: each Gaussian vs its own face's mean.
          (b) across a shared vertex: neighboring faces' means vs each other. Base
              Gaussians alone have a footprint ~2.6-2.8x their own triangle's area, so a
              face's Gaussians routinely overlap every face around its corners, not just
              the 3 across its edges. See face_adjacency_pairs() for the measurement that
              moved this from edge- to vertex-adjacency on 2026-08-06: after the first
              training run the covered classes sat at median |dC| 0.0011/0.0019 while the
              corner neighbors this was missing sat at 0.0056 (5.3x, 6.2x at p90) and
              carried 17.2% of the total flicker potential.
          (c) across a fold: faces close in 3D but far apart along the surface (see
              _set_face_overlap_pairs). A small population -- 0.2% of the flicker
              potential on this mesh -- but the highest per-pair color spread of any
              class (p90 8.7x the same-face class) and concentrated in a few places
              (armpits, limb junctions) instead of spread thin, which is what can make
              it visible out of proportion to its share of the total.

        Each term is normalized by its OWN weight sum rather than concatenating all pairs
        into one list: fold pairs number in the thousands against ~630k vertex-adjacent
        pairs, so a shared denominator would dilute term (c) to nothing.

        Everything is opacity-weighted: a nearly-transparent Gaussian cannot produce a
        visible jump when it swaps depth order, so its color shouldn't be constrained.
        Weights come from self.opacities (the property, i.e. the base-layer floor applied)
        so they match what actually renders. Gradient flows through the colors only --
        opacity is detached, since this term is about making colors agree, not about
        pushing opacities down to dodge the penalty (opacity_reg_lambda and the coverage
        machinery own that decision, and letting this term fight them would be a silent
        tug-of-war).
        """
        eps = 1e-8
        gs_face_idx = self.gaussians_to_mesh_indices
        n_faces = self.mesh_faces.shape[0]

        w = torch.sigmoid(self.opacities).squeeze(-1).detach()  # (N,), see docstring
        c = self.gauss_params["features_dc"]  # (N, 3), DC term only

        w_sum = torch.zeros(n_faces, device=w.device, dtype=w.dtype).scatter_add(0, gs_face_idx, w)
        wc_sum = torch.zeros(n_faces, 3, device=c.device, dtype=c.dtype).scatter_add(
            0, gs_face_idx.unsqueeze(-1).expand(-1, 3), w.unsqueeze(-1) * c
        )
        face_mean = wc_sum / w_sum.clamp(min=eps).unsqueeze(-1)  # (F, 3)

        # (a) within-face spread
        dev = (c - face_mean[gs_face_idx]).pow(2).sum(dim=-1)  # (N,)
        within = (w * dev).sum() / w.sum().clamp(min=eps)

        # (b) across-edge spread. Faces with no Gaussians have face_mean == 0 from the
        # clamp above, which is not a real color -- weighting each pair by the product of
        # the two faces' opacity mass makes those contribute ~0 instead of dragging a
        # neighbor toward black.
        fa, fb = self.face_adj_a, self.face_adj_b
        pair_w = (w_sum[fa] * w_sum[fb]).detach()
        pair_dev = (face_mean[fa] - face_mean[fb]).pow(2).sum(dim=-1)
        across = (pair_w * pair_dev).sum() / pair_w.sum().clamp(min=eps)

        # (c) across-fold spread. Same opacity-mass weighting as (b) for the same reason
        # (a face with no Gaussians has face_mean == 0, which is not a real color), but
        # separately normalized -- see the docstring.
        oa, ob = self.face_overlap_a, self.face_overlap_b
        if oa.numel() > 0:
            fold_w = (w_sum[oa] * w_sum[ob]).detach()
            fold_dev = (face_mean[oa] - face_mean[ob]).pow(2).sum(dim=-1)
            fold = (fold_w * fold_dev).sum() / fold_w.sum().clamp(min=eps)
        else:
            # No folds detected (or a mesh with none) -- keep the return type/graph shape
            # identical instead of branching in the caller.
            fold = torch.zeros((), device=c.device, dtype=c.dtype)

        return within + across + fold

    def _compute_coverage_density(self):
        """
        Point-sampled coverage: measure the Gaussian density actually received at every
        mesh vertex and every face centroid (all in the face's own 2D plane, where both
        the Gaussians and the sample points live). Position-aware, unlike a per-face
        footprint *area* budget: Gaussians piling up at the face center cannot satisfy
        this while the triangle corners stay bare.

        Shared by get_loss_dict() (coverage_lambda, needs grad -- called without a
        no_grad() wrapper here on purpose, so it inherits whatever grad mode the caller
        is in) and refinement_after() (coverage_densify_scale, called from inside that
        function's own torch.no_grad() block). Returns the raw per-vertex/per-face-centroid
        density (vert_cov, cent_cov), not yet compared against coverage_target -- callers
        do their own ReLU(coverage_target - cov) as needed, since get_loss_dict needs the
        mean over all points while refinement_after needs it per-Gaussian.

        Includes the vertex filler layer now (2026-08-05): its contribution at its own
        anchor point (exact, distance 0), plus its cross-contribution to the face
        centroid check point (a vertex Gaussian sized to reach its full 1-ring
        neighborhood can genuinely brighten its own face's centroid too, via plain
        3D-distance Gaussian falloff, see below). Before this, gs_face_idx only spanned
        base+detail, so this function was structurally blind to the vertex layer -- a
        vertex already fully covered by it still reported as bare to coverage_lambda /
        coverage_rescue_thresh / coverage_densify_scale, which kept recruiting more
        detail Gaussians (and spawning coverage-rescue ones) on top of a hole that was
        already filled. Observed in practice on the bunny mesh: detail grew from 3/face
        at seed to 9.1/face by the end of training. (A face-centroid filler layer existed
        alongside the vertex layer 2026-07-27 through 2026-08-06 and was included here
        too when it was still present; removed at the user's request to test whether
        base+detail+vertex alone reach 100% coverage -- see its removal note in
        populate_modules.)
        """
        gs_face_idx = self.gaussians_to_mesh_indices
        # self.opacities (the property), not the raw parameter: base-layer rows render at
        # their floored opacity, and coverage should measure what actually renders --
        # identical to the raw value whenever the base layer is off.
        cov_opacity = torch.sigmoid(self.opacities).squeeze(-1)
        cov_scales = torch.exp(self.scales)
        # Conservative isotropic radius so anisotropy can't fake coverage. Superseded by a
        # real anisotropic distance when config.anisotropic_coverage_density is on (see
        # below and that config's docstring): collapsing an ellipse to its narrow axis
        # cannot score a plate that legitimately covers its triangle along the wide one,
        # which is what keeps forcing the base layer back to circular.
        sigma_min = torch.minimum(cov_scales[:, 0], cov_scales[:, 1]).clamp(min=1e-12)
        # Clamped to each Gaussian's own face triangle (see _clip_means_2d_to_face), NOT
        # the raw gauss_params["means_2d"] directly (2026-07-29 fix): means_2d is an
        # unconstrained parameter that can drift outside its face -- most concretely
        # right after reassign_gaussians_to_nearest_face()/finalize_face_assignment(),
        # which store the UNCLAMPED projection (see _reproject_to_new_faces()) and rely
        # on downstream readers to clip it the same way `means` does. Coverage previously
        # read the raw value directly, so a drifted Gaussian's contribution to
        # vert_cov/cent_cov was measured from a position that didn't match where it
        # actually renders -- systematically wrong distances (and therefore wrong
        # coverage_lambda gradient / coverage_rescue_thresh / coverage_densify_scale
        # decisions) for exactly the Gaussians whose face assignment just changed.
        pos_2d = self._clip_means_2d_to_face(self.gauss_params["means_2d"], gs_face_idx)  # (N, 2), face-plane coords
        corners_2d = torch.stack([
            self.mesh_triangles_2d_coords_a[gs_face_idx],
            self.mesh_triangles_2d_coords_b[gs_face_idx],
            self.mesh_triangles_2d_coords_c[gs_face_idx],
        ], dim=1)  # (N, 3, 2)
        centroid_2d = corners_2d.mean(dim=1)  # (N, 2)

        if self.config.anisotropic_coverage_density:
            # Evaluate the actual 2D Gaussian rather than an isotropic stand-in for it:
            # rotate each offset into the Gaussian's own frame and take the Mahalanobis
            # distance (u/sx)^2 + (v/sy)^2, so the wide axis gets credit for the reach it
            # really has and the narrow one still scores low toward a corner it misses.
            #
            # _inplane_theta() is the same angle the quats property renders with, base pin
            # included -- see its docstring on why both must read one source. Out-of-plane
            # tilt is ignored here, exactly as the isotropic branch already ignores it:
            # this whole function works in the face's 2D plane, and cone_coef keeps the
            # tilt small enough for that projection to stay a good approximation.
            _theta = self._inplane_theta()
            _cos = torch.cos(_theta).unsqueeze(1)  # (N, 1)
            _sin = torch.sin(_theta).unsqueeze(1)  # (N, 1)
            _sx = cov_scales[:, 0].clamp(min=1e-12).unsqueeze(1)  # (N, 1)
            _sy = cov_scales[:, 1].clamp(min=1e-12).unsqueeze(1)  # (N, 1)

            _d_corner = corners_2d - pos_2d.unsqueeze(1)  # (N, 3, 2)
            _u = _d_corner[..., 0] * _cos + _d_corner[..., 1] * _sin  # (N, 3)
            _v = -_d_corner[..., 0] * _sin + _d_corner[..., 1] * _cos  # (N, 3)
            # Already dimensionless (a squared Mahalanobis distance), so unlike the
            # isotropic branch below there is no further division by a sigma here.
            d2_corner = (_u / _sx) ** 2 + (_v / _sy) ** 2  # (N, 3)

            _d_cent = centroid_2d - pos_2d  # (N, 2)
            _uc = _d_cent[:, 0] * _cos.squeeze(1) + _d_cent[:, 1] * _sin.squeeze(1)  # (N,)
            _vc = -_d_cent[:, 0] * _sin.squeeze(1) + _d_cent[:, 1] * _cos.squeeze(1)  # (N,)
            d2_centroid = (_uc / _sx.squeeze(1)) ** 2 + (_vc / _sy.squeeze(1)) ** 2  # (N,)

            dens_corner = cov_opacity.unsqueeze(1) * torch.exp(-0.5 * d2_corner)
            dens_centroid = cov_opacity * torch.exp(-0.5 * d2_centroid)
        else:
            d2_corner = ((pos_2d.unsqueeze(1) - corners_2d) ** 2).sum(dim=-1)  # (N, 3)
            d2_centroid = ((pos_2d - centroid_2d) ** 2).sum(dim=-1)  # (N,)
            dens_corner = cov_opacity.unsqueeze(1) * torch.exp(
                -0.5 * d2_corner / sigma_min.unsqueeze(1) ** 2
            )
            dens_centroid = cov_opacity * torch.exp(-0.5 * d2_centroid / sigma_min ** 2)

        # A vertex is covered if Gaussians from ANY adjacent face reach it, so
        # accumulate corner contributions by global vertex id.
        vert_cov = torch.zeros(
            self.mesh_verts.shape[0], device=dens_corner.device, dtype=dens_corner.dtype
        ).scatter_add(0, self.mesh_faces[gs_face_idx].reshape(-1), dens_corner.reshape(-1))
        cent_cov = torch.zeros(
            self.mesh_faces.shape[0], device=dens_centroid.device, dtype=dens_centroid.dtype
        ).scatter_add(0, gs_face_idx, dens_centroid)

        # The whole vertex-layer contribution is skipped when the layer does not exist
        # (config.use_vertex_layer). Both parts below read vertex_gauss_params, and
        # leaving them in would credit coverage to a population that is not rendered --
        # which would keep floors released on faces nothing is actually holding up.
        if self.vertex_gauss_params is not None:
            # Vertex filler layer (see docstring above): sits EXACTLY at its own check point
            # by construction (self.vertex_positions == self.mesh_verts, fixed, never
            # trained) -- so its Gaussian kernel evaluates to exp(0) = 1 at that point and its
            # contribution reduces to its own rendered opacity, no distance computation
            # needed.
            vertex_opacity = torch.sigmoid(self.vertex_gauss_params["opacities"]).squeeze(-1)  # (V,)
            vert_cov = vert_cov + vertex_opacity

            # Cross-contribution: a face's vertex Gaussians reaching into its own centroid
            # (2026-08-05). The addition just above only credits the vertex layer at its OWN
            # anchor point -- but a vertex Gaussian sized to reach its full 1-ring
            # neighborhood can genuinely brighten its own face's centroid too, and skipping
            # that undercounts real coverage there. Distance is plain 3D world distance, not
            # the face-local 2D coords used above: a mesh vertex is shared by many faces each
            # with a different local plane, so there's no single face-plane to project into,
            # and since both points lie exactly on the mesh surface, world distance is
            # already a good proxy for surface distance between them.
            face_centroid_pos = self.mesh_faces_verts.mean(dim=1)  # (F, 3)
            tri_verts = self.mesh_verts[self.mesh_faces]  # (F, 3, 3), this face's 3 corner positions
            d2_v2c = ((tri_verts - face_centroid_pos.unsqueeze(1)) ** 2).sum(dim=-1)  # (F, 3)

            v_scales = torch.exp(self._vertex_scales())  # (V, 3)
            v_opacity_per_corner = vertex_opacity[self.mesh_faces]  # (F, 3)
            if self.config.anisotropic_coverage_density:
                # Same correction the base layer gets just above, applied to the vertex
                # layer's cross-contribution: measure the real elliptical reach instead of
                # collapsing it to the narrow axis. Without this, anisotropic_vertex_floor is
                # self-defeating -- v_sigma_min below scores an elongated filler by the axis it
                # is THIN on, reports it as under-covering the centroid it actually reaches,
                # and update_coverage_floor_mask() then keeps vertex_needs_floor set, which
                # holds the row at a floor whose whole purpose was to stop being circular.
                _vids = self.mesh_faces  # (F, 3), the vertex id behind each corner
                # Vertex -> centroid, still the plain 3D offset the isotropic branch uses (see
                # its note on why world distance is the right proxy here), just resolved into
                # the vertex's own frame so each component can be divided by the sigma that
                # actually governs it.
                _off = face_centroid_pos.unsqueeze(1) - tri_verts  # (F, 3, 3)
                _u = (_off * self.vertex_axis_x[_vids]).sum(dim=-1)  # (F, 3)
                _t = (_off * self.vertex_axis_y[_vids]).sum(dim=-1)  # (F, 3)
                _w = (_off * self.vertex_normals[_vids]).sum(dim=-1)  # (F, 3), out-of-plane
                # Into the Gaussian's own axes: _vertex_inplane_theta() is the same angle
                # _vertex_quats() renders with, pin included -- see its docstring on why both
                # must read one source.
                _th = self._vertex_inplane_theta()[_vids]  # (F, 3)
                _cos, _sin = torch.cos(_th), torch.sin(_th)
                _u_rot = _u * _cos + _t * _sin
                _t_rot = -_u * _sin + _t * _cos
                _sx = v_scales[:, 0].clamp(min=1e-12)[_vids]  # (F, 3)
                _sy = v_scales[:, 1].clamp(min=1e-12)[_vids]  # (F, 3)
                # The out-of-plane residual is charged against the NARROW in-plane axis, not
                # against the z thickness. Dividing by a face_flat_coef-thin z would make any
                # curvature at all read as zero coverage, and thickness has never been part of
                # this layer's reach guarantee; charging it at the conservative in-plane rate
                # keeps curvature costing what it costs in the isotropic branch today, while
                # letting the two in-plane axes get the credit they have earned. On a flat
                # 1-ring _w is ~0 and this reduces exactly to the in-plane ellipse.
                _s_min = torch.minimum(_sx, _sy)
                # Already a squared Mahalanobis distance, so no further division by a sigma.
                d2_v2c_maha = (_u_rot / _sx) ** 2 + (_t_rot / _sy) ** 2 + (_w / _s_min) ** 2  # (F, 3)
                dens_v2c = v_opacity_per_corner * torch.exp(-0.5 * d2_v2c_maha)  # (F, 3)
            else:
                v_sigma_min = torch.minimum(v_scales[:, 0], v_scales[:, 1]).clamp(min=1e-12)  # (V,)
                v_sigma_per_corner = v_sigma_min[self.mesh_faces]  # (F, 3)
                dens_v2c = v_opacity_per_corner * torch.exp(-0.5 * d2_v2c / v_sigma_per_corner ** 2)  # (F, 3)
            cent_cov = cent_cov + dens_v2c.sum(dim=1)

        return vert_cov, cent_cov

    @torch.no_grad()
    def update_coverage_floor_mask(self):
        """
        Recompute which faces/vertices still need their coverage floor engaged (see
        config.adaptive_coverage_floor). No-op when that config is off.

        The measurement is taken with EVERY floor released. That is the whole point and it
        is what makes dropping a floor safe: self.face_needs_floor / self.vertex_needs_floor
        are cleared first, so _compute_coverage_density() -- which reads self.scales and
        self.opacities, both of which route through _base_floor_mask() -- reports the
        coverage the mesh would have if no base plate and no vertex filler were being
        propped up. A face keeps its floor unless it clears coverage_floor_target on its
        own merits, so the floor can only ever be dropped where it was doing nothing.

        Deliberately NOT a fixed point: releasing floors can only lower coverage, so one
        pass is the pessimistic answer, not an approximation of some converged one. Two
        neighbouring faces can each be covered only because the other's floor is on, and
        this measurement (both released at once) correctly refuses to drop either.

        Called at the END of refinement_after(), after every population change for the
        cycle has landed, so the mask the next refine_every steps render against matches
        the Gaussians that actually exist. Also called from load_state_dict() so a resumed
        or exported model reconstructs the same rendered sizes instead of falling back to
        the conservative all-floors-on initial state -- without that, every export would
        silently reinstate ~num_faces full-face opaque plates that training had released.
        """
        if not self.config.adaptive_coverage_floor:
            return

        prev_face, prev_vertex = self.face_needs_floor, self.vertex_needs_floor
        try:
            self.face_needs_floor = torch.zeros_like(prev_face)
            self.vertex_needs_floor = torch.zeros_like(prev_vertex)
            vert_cov, cent_cov = self._compute_coverage_density()
        except Exception:
            # Never leave the model in the all-released state on failure -- that is the
            # one configuration that can silently open holes everywhere at once.
            self.face_needs_floor, self.vertex_needs_floor = prev_face, prev_vertex
            raise

        tgt = self.config.coverage_floor_target
        vert_short = vert_cov < tgt  # (num_verts,)
        # A face needs its floor if its centroid is short OR any of its 3 corners is --
        # same 4 check points coverage_rescue_thresh uses, and for the same reason: a
        # face's own Gaussians cluster toward its interior, so a comfortably covered
        # centroid routinely masks still-bare corners.
        self.face_needs_floor = (cent_cov < tgt) | vert_short[self.mesh_faces].any(dim=1)
        self.vertex_needs_floor = vert_short

        n_f = int(self.face_needs_floor.sum().item())
        n_v = int(self.vertex_needs_floor.sum().item())
        CONSOLE.log(
            f"adaptive coverage floor: {n_f}/{self.face_needs_floor.shape[0]} faces "
            f"({100.0 * n_f / max(self.face_needs_floor.shape[0], 1):.1f}%) and "
            f"{n_v}/{self.vertex_needs_floor.shape[0]} vertices still floored "
            f"(target {tgt})"
        )

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        accumulation = outputs["accumulation"]
        # Compute mean only over transparent pixels so the loss stays strong
        # even when only a small fraction of pixels are transparent.
        transparent_penalty = torch.nn.functional.relu(0.95 - accumulation)
        transparent_mask = transparent_penalty > 0
        # After stop_split_at, no new Gaussians are created and only culling of
        # now-redundant ones brings the point count back down. Keeping acm_lambda
        # active here fights that natural pruning (it keeps pushing opacities up
        # so overlapping redundant Gaussians never drop below cull_alpha_thresh),
        # so the point count never settles. Gate it off once splitting has stopped.
        if self.config.stop_acm_after_split and self.step >= self.config.stop_split_at:
            effective_acm_lambda = 0.0
        else:
            effective_acm_lambda = self.config.acm_lambda
        if transparent_mask.any():
            loss_acm = transparent_penalty[transparent_mask].mean() * effective_acm_lambda
        else:
            loss_acm = transparent_penalty.sum()  # zero, keeps graph alive

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        if "mesh_depth" in batch:
            if batch["mesh_depth"].shape[0] != gt_img.shape[0] and batch["mesh_depth"].shape[1] != gt_img.shape[1]:
                downscale_factor = int(torch.floor(torch.tensor([float(batch["mesh_depth"].shape[0]) / float(gt_img.shape[0]) + 0.5])))
            else:
                downscale_factor = 1
            # print("0 mesh_depth: ", batch["mesh_depth"].shape)
            # print("0 gt_img: ", gt_img.shape)
            # print("downscale_factor: ", downscale_factor)
            mesh_depth = batch["mesh_depth"].unsqueeze(-1)
            if downscale_factor > 1:
                mesh_depth = resize_image(mesh_depth, downscale_factor)
            mesh_depth = mesh_depth.to(self.device)
            # print("1 mesh_depth: ", mesh_depth.shape)
            # print("1 gt_img: ", gt_img.shape)
            assert mesh_depth.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            predicted_depth = outputs["depth"]
            L1_depth = torch.nn.functional.relu(torch.abs(mesh_depth.reshape(predicted_depth.shape) - predicted_depth) - 0.).mean() * self.config.mesh_depth_lambda
        else:
            L1_depth = 0

        grazing_weight = outputs.get("grazing_weight")
        l1_map = torch.abs(gt_img - pred_img)
        if self.config.grazing_weight_enabled and grazing_weight is not None:
            w = grazing_weight.clamp(min=self.config.grazing_weight_floor) ** self.config.grazing_weight_power
            Ll1 = (l1_map * w).sum() / (w.expand_as(l1_map).sum() + 1e-8)
        else:
            Ll1 = l1_map.mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization:
            scale_exp = torch.exp(self.scales[:, :2])
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio, device=self.device),
                )
                - self.config.max_gauss_ratio
            )
            if self.config.scale_reg_exclude_base:
                # A floored base plate's elongation is dictated by its triangle, not chosen
                # by the optimizer, so penalising it here just fights the floor -- and on a
                # mesh with one base row per face the base layer dominates this mean badly
                # enough to bury the detail rows the regulariser is actually for. See
                # config.scale_reg_exclude_base. _base_floor_mask(), not is_base: a
                # released base row renders at its raw size and belongs in the mean.
                _reg_rows = ~self._base_floor_mask()
                scale_reg = scale_reg[_reg_rows]
            # Guard the degenerate all-excluded case: mean() over an empty tensor is nan,
            # which would silently poison the total loss for the rest of the run.
            scale_reg = 1.0 * scale_reg.mean() if scale_reg.numel() > 0 else torch.tensor(
                0.0, device=self.device
            )
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        opacity = torch.sigmoid(self.gauss_params["opacities"])
        opacity_reg = (opacity * (1.0 - opacity)).mean() * self.config.opacity_reg_lambda

        if self.config.coverage_lambda > 0:
            vert_cov, cent_cov = self._compute_coverage_density()
            tgt = self.config.coverage_target
            coverage_reg = (
                torch.nn.functional.relu(tgt - vert_cov).mean()
                + torch.nn.functional.relu(tgt - cent_cov).mean()
            ) * self.config.coverage_lambda
        else:
            coverage_reg = torch.tensor(0.0, device=self.device)

        if self.config.color_consistency_lambda > 0:
            color_reg = self._compute_color_consistency() * self.config.color_consistency_lambda
        else:
            color_reg = torch.tensor(0.0, device=self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss + loss_acm + L1_depth,
            "scale_reg": scale_reg,
            "opacity_reg": opacity_reg,
            "coverage_reg": coverage_reg,
            "color_reg": color_reg,
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        predicted_depth = outputs["depth"]
        accumulation = outputs["accumulation"]

        depth_im = cv2.applyColorMap(np.clip((predicted_depth / predicted_depth.max()).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        depth_im = torch.from_numpy(np.array(depth_im, dtype=np.float32)[..., ::-1] / 255.)

        accu_im = cv2.applyColorMap(np.clip(accumulation.cpu().numpy() * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        accu_im = torch.from_numpy(np.array(accu_im, dtype=np.float32)[..., ::-1] / 255.)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "depth_im": depth_im, "accu_im": accu_im}

        if "mesh_depth" in batch:
            mesh_depth_im = cv2.applyColorMap(np.clip((batch["mesh_depth"].to(predicted_depth.device) / predicted_depth.max()).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
            mesh_depth_im = torch.from_numpy(np.array(mesh_depth_im, dtype=np.float32)[..., ::-1] / 255.)
            images_dict["mesh_depth_im"] = mesh_depth_im


        return metrics_dict, images_dict

    def export_splatfacto_on_mesh(self):
        # NOTE: self.scales (the clamped property), not gauss_params["scales"] (the
        # raw, unclamped optimizer parameter). During training every render goes
        # through the property, so training always respects min_scale_frac/
        # upper_scale -- but the raw parameter itself is never bounded, only its
        # *interpretation* is. Exporting the raw value here meant the exported/final
        # result could have Gaussians smaller (or larger) than min_scale_frac/
        # upper_scale ever intended, even though training looked correct the whole
        # time. Both are the same log-space representation, so this is a drop-in fix.
        #
        # Captured BEFORE finalize_face_assignment() below, deliberately -- this is the
        # scale as it was actually clamped and rendered for every step of training
        # (under whichever face each Gaussian was assigned at birth/split/dup), not a
        # value reinterpreted under a face it was never trained against. Reassigning
        # face changes which face's xyz_radius bounds self.scales' clamp; if the new
        # face has a smaller xyz_radius, reading self.scales AFTER relabeling would
        # clamp an already-well-trained Gaussian down to the new, smaller ceiling for
        # no reason other than the relabel itself (this was the "still tiny points"
        # regression seen 2026-07-20 when finalize_face_assignment() was called before
        # this capture). Reading it first sidesteps that entirely: the exported size is
        # exactly what training produced, and the face label below is still corrected
        # for accuracy -- the two no longer have to trade off against each other here.
        scales_out = self.scales.detach().cpu()
        # Same what-you-render-is-what-you-save rationale as scales_out: base-layer rows'
        # rendered opacity is floored by the opacities property, and exporting the raw
        # parameter instead would let a base Gaussian whose raw logit drifted below the
        # floor come out transparent in the export -- reintroducing exactly the holes the
        # base layer exists to prevent. (Unlike scales, opacity doesn't depend on face
        # indices, but capture it here alongside scales_out for the same
        # before-finalize_face_assignment discipline.)
        opacities_out = self.opacities.detach().cpu()

        # One-shot-at-export-only face reclassification (see finalize_face_assignment()
        # docstring, option (A) discussed with the user): fixes gaussians_to_mesh_indices
        # to reflect each Gaussian's true nearest face instead of the birth/split/dup-
        # inherited one, without touching training (no periodic reassignment mid-training
        # -- that's config.reassign_face_every / reassign_gaussians_to_nearest_face(),
        # deliberately left disabled/unused for now).
        self.finalize_face_assignment()

        export_dict = {
            "means_2d": self.gauss_params["means_2d"].detach().cpu(),
            "normal_elevates": self.gauss_params["normal_elevates"].detach().cpu(),
            "scales": scales_out,
            "quats": self.gauss_params["quats"].detach().cpu(),
            "features_dc": self.gauss_params["features_dc"].detach().cpu(),
            "features_rest": self.gauss_params["features_rest"].detach().cpu(),
            "opacities": opacities_out,

            "mesh_faces_verts": self.mesh_faces_verts.cpu(),
            "normals": self.normals.cpu(),
            "radius": self.radius.cpu(),
            "xyz_radius": self.xyz_radius.cpu(),
            "faces_quats": self.faces_quats.cpu(),
            "mesh_verts": self.mesh_verts.cpu(),
            "mesh_faces": self.mesh_faces.cpu(),

            "faces_axis_x": self.faces_axis_x.cpu(),
            "faces_axis_y": self.faces_axis_y.cpu(),

            "mesh_triangles_edge_ab": self.mesh_triangles_edge_ab.cpu(),
            "mesh_triangles_edge_bc": self.mesh_triangles_edge_bc.cpu(),
            "mesh_triangles_edge_ca": self.mesh_triangles_edge_ca.cpu(),

            "mesh_triangles_edge_len_a": self.mesh_triangles_edge_len_a.cpu(),
            "mesh_triangles_edge_len_b": self.mesh_triangles_edge_len_b.cpu(),
            "mesh_triangles_edge_len_c": self.mesh_triangles_edge_len_c.cpu(),

            "mesh_triangles_2d_coords_a": self.mesh_triangles_2d_coords_a.cpu(),
            "mesh_triangles_2d_coords_b": self.mesh_triangles_2d_coords_b.cpu(),
            "mesh_triangles_2d_coords_c": self.mesh_triangles_2d_coords_c.cpu(),

            # Captured AFTER finalize_face_assignment(): these three are exactly what
            # that call updates (face label + the new face's local-frame reprojection of
            # the same world-space position), so they must reflect the post-relabel
            # state to stay mutually consistent with each other and with scales_out's
            # world-space size.
            "gaussians_to_mesh_indices": self.gaussians_to_mesh_indices.cpu(),
            "is_base": self.is_base.cpu(),
            "is_rescue": self.is_rescue.cpu(),
        }
        return export_dict

    def load_splatfacto_on_mesh(self, load_dict):
        self.gauss_params["means_2d"] = torch.nn.Parameter(load_dict["means_2d"].cuda())
        self.gauss_params["normal_elevates"] = torch.nn.Parameter(load_dict["normal_elevates"].cuda())
        self.gauss_params["scales"] = torch.nn.Parameter(load_dict["scales"].cuda())
        self.gauss_params["quats"] = torch.nn.Parameter(load_dict["quats"].cuda())
        self.gauss_params["features_dc"] = torch.nn.Parameter(load_dict["features_dc"].cuda())
        self.gauss_params["features_rest"] = torch.nn.Parameter(load_dict["features_rest"].cuda())
        self.gauss_params["opacities"] = torch.nn.Parameter(load_dict["opacities"].cuda())

        self.mesh_faces_verts = load_dict["mesh_faces_verts"].cuda()
        # Deterministic recompute from the loaded mesh (same function populate_modules
        # uses), so older som.pt files without a stored cv_radius work identically.
        self.cv_radius = face_centroid_vertex_radius(self.mesh_faces_verts).reshape(-1, 1)
        # No irregular-face widening recomputed here (see cv_radius_base_floor in
        # populate_modules) -- this load path (splat_merge.py) never validated base-layer
        # machinery anyway; plain mean-based value only needs to exist for the scales
        # property's gather to stay in bounds.
        self.cv_radius_base_floor = self.cv_radius
        # Same treatment for base_floor_reaches_corners' reference (see populate_modules).
        # No 99th-pct clamp recomputed here for the same reason cv_radius isn't widened
        # above -- this load path (splat_merge.py) only needs the value to exist and be in
        # bounds for the scales property's gather.
        self.base_floor_xy = self.cv_radius_base_floor.expand(-1, 2).contiguous()
        self.normals = load_dict["normals"].cuda()
        self.radius = load_dict["radius"].cuda()
        self.xyz_radius = load_dict["xyz_radius"].cuda()
        self.faces_quats = load_dict["faces_quats"].cuda()
        self.mesh_verts = load_dict["mesh_verts"].cuda()
        self.mesh_faces = load_dict["mesh_faces"].cuda()
        # Derived from mesh_faces, so recomputed rather than persisted (same treatment as
        # cv_radius just above). Needed by color_consistency_lambda.
        self.face_adj_a, self.face_adj_b = face_adjacency_pairs(self.mesh_faces)
        # Likewise derived, not persisted -- keeps som.pt forward/backward compatible
        # (no new key) and this path is only ever reached by splat_merge.py-style
        # assembly, where color_consistency_lambda is at its 0 default and this returns
        # empty tensors immediately without running the search.
        self._set_face_overlap_pairs()

        self.faces_axis_x = load_dict["faces_axis_x"].cuda()
        self.faces_axis_y = load_dict["faces_axis_y"].cuda()
        self.mesh_triangles_edge_ab = load_dict["mesh_triangles_edge_ab"].cuda()
        self.mesh_triangles_edge_bc = load_dict["mesh_triangles_edge_bc"].cuda()
        self.mesh_triangles_edge_ca = load_dict["mesh_triangles_edge_ca"].cuda()
        self.mesh_triangles_edge_len_a = load_dict["mesh_triangles_edge_len_a"].cuda()
        self.mesh_triangles_edge_len_b = load_dict["mesh_triangles_edge_len_b"].cuda()
        self.mesh_triangles_edge_len_c = load_dict["mesh_triangles_edge_len_c"].cuda()
        self.mesh_triangles_2d_coords_a = load_dict["mesh_triangles_2d_coords_a"].cuda()
        self.mesh_triangles_2d_coords_b = load_dict["mesh_triangles_2d_coords_b"].cuda()
        self.mesh_triangles_2d_coords_c = load_dict["mesh_triangles_2d_coords_c"].cuda()
        # Deterministic recompute from the loaded mesh, same treatment as cv_radius above
        # (and it has to sit here, below the 2D coords it reads, rather than beside the
        # other radii). No 99th-pct clamp for the same reason given there: this path only
        # needs the values to exist and be in bounds for the scales / _inplane_theta
        # gathers. Recomputed rather than persisted so older som.pt files keep loading.
        _steiner_axes, _steiner_theta = steiner_ellipse_axes(
            self.mesh_triangles_2d_coords_a,
            self.mesh_triangles_2d_coords_b,
            self.mesh_triangles_2d_coords_c,
            max_aspect=self.config.base_floor_max_aspect,
        )
        self.base_ellipse_theta = _steiner_theta.reshape(-1).contiguous()
        self.base_floor_xy_aniso = (
            _steiner_axes / max(self.config.base_floor_corner_sigma, 1e-6)
        ).contiguous()
        self.gaussians_to_mesh_indices = load_dict["gaussians_to_mesh_indices"].cuda()
        if "is_base" in load_dict:
            self.is_base = load_dict["is_base"].cuda()
        else:
            # som.pt exported before the base layer existed: everything non-base.
            self.is_base = torch.zeros(
                self.gaussians_to_mesh_indices.shape[0], dtype=torch.bool, device="cuda"
            )
        if "is_rescue" in load_dict:
            self.is_rescue = load_dict["is_rescue"].cuda()
        else:
            # som.pt exported before coverage-rescue ceiling tracking existed.
            self.is_rescue = torch.zeros(
                self.gaussians_to_mesh_indices.shape[0], dtype=torch.bool, device="cuda"
            )
