#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 11.
#
# Baseline is run_table_gs10.sh. Lines are marked against gs10's value.
#
# ---------------------------------------------------------------------------------------
# WHAT gs10's EXPORT MEASURED  (scripts/analyze_splat_ply.py on table_gs10/export_ply)
#
#   [3]  floored rows: aspect median 1.163, circles 0.6%   <- gs8 was 96.8% circles
#   [3b] floored rows 1.163 vs their faces' 1.168          <- the ellipse is tracking
#   [2]  mesh vertices  >= target 97.53%, median 3.18      <- vertices are 3x oversupplied
#   [2]  face centroids >= target 18.65%, q25/q50 = 0.950/0.950
#   [3]  free/detail rows: aspect median 1.835, q90 23.63, q99 25.00, circles 19.4%
#   [4]  on-screen sigma: face-based median 2.50 px, VERTEX LAYER median 4.52 px
#   [1]  area coverage 93.17% at 2.19x redundancy (gs8: 92.56% at 1.53x)
#
# READ THOSE TOGETHER AND THREE SEPARATE PROBLEMS FALL OUT, EACH WITH ITS OWN CAUSE:
#
# (a) THE ELLIPTICAL BASE FLOOR WORKED. 96.8% circles -> 0.6%, and floored aspect now
#     tracks the triangles (1.163 vs 1.168). Nothing here needs changing. The circles the
#     user still sees are NOT these rows -- see (b) and (c).
#
# (b) THE REMAINING CIRCLES ARE THE VERTEX LAYER, WHICH anisotropic_base_floor NEVER
#     TOUCHED. Its size is self.vertex_log_scales, built as vertex_radius.repeat(1, 3) --
#     ONE scalar per vertex into both in-plane axes -- and _vertex_scales() derives its
#     ceiling AND its floor from that, so every row at either bound has sx == sy by
#     definition. Measured over the 707,579 vertex rows: median in-plane aspect 1.000,
#     59.1% below 1.001. That layer is also the blurriest population in the model
#     (sigma_px median 4.52 vs the face-based 2.50) while vertex coverage is already
#     3x oversupplied -- so shrinking its footprint is close to free.
#
# (c) COVERAGE IS SHORT AT FACE CENTROIDS, NOT AT VERTICES. Vertices clear target at
#     97.53%. Centroids pile up EXACTLY on 0.950, which is base_opacity_floor: the lone
#     base plate contributes alpha*exp(0) at its own centroid and nothing else reaches it.
#     And coverage_rescue_thresh is 0.9 -- so that entire pile sits just ABOVE the rescue
#     threshold and never qualifies for repair. The one mechanism still enabled is blind
#     to the exact population that is short.
#
# (d) THE NEEDLE CULL HAS NEVER FIRED, IN ANY RUN. cull_gaussians() culls on in-plane
#     aspect above max_gauss_ratio * 4, but in-plane aspect is bounded by
#     upper_scale / min_scale_frac, and no run has ever set those so the cull is reachable:
#         gs8 : cull 5.0*4 = 20:1   vs bound 0.5/0.05 = 10:1   -> unreachable
#         gs10: cull 10.0*4 = 40:1  vs bound 0.5/0.02 = 25:1   -> unreachable
#     gs10's own note ("moving from 20:1 to 40:1") tracked the cull moving but not that it
#     had moved past the bound. Hence free/detail q90 23.63 and q99 pinned at exactly
#     25.00: those rows are clamped by the bound, and nothing has ever removed them.
#
# WHAT THIS RUN DOES NOT CHANGE, DELIBERATELY:
#   - coverage_lambda / coverage_densify_scale stay 0. gs8 turned them off for a good
#     reason ("its only way to raise coverage is sigma+ and alpha+, on EVERY gaussian").
#     Turning them back on to chase (c) would buy coverage by making everything bigger and
#     more opaque -- i.e. by reintroducing the blur this run is trying to remove.
#   - min_scale_frac stays 0.02. It sets the 25:1 bound, but gs10 chose it to let detail
#     shrink below 2.5 px and resolve text, which matters more. (d) is fixed by moving the
#     cull under the bound instead of raising the bound.
#   - coverage_floor_target stays 1.0. It did its job: 1,261,295 of 2,520,223 face-based
#     rows are still floored, i.e. half were released, against gs8 where 91.9% stayed
#     floored.
#   - min_vertex_scale_frac stays 0.15. It only binds on rows still marked
#     vertex_needs_floor, and with vertices at 97.53% of target almost none are.
#
# Requires, applied in order:
#   adaptive_coverage_floor.patch, anisotropic_base_floor.patch, and the vertex-layer
#   anisotropy change (one_ring_ellipse_axes / anisotropic_vertex_floor) already in
#   splat/nerfstudio/models/splatfacto_on_mesh_uc.py.
#
# Usage (on the DGX, inside the drawer_splat env):
#   bash scripts/run_table_gs11.sh
#   bash scripts/run_table_gs11.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs11}"

MESH="${DATA_DIR}/table_sdf_ds1/texture_mesh/mesh_table-only-400k_world.obj"
AREA=1e-4

EXTRA_INFO_DIR="${DATA_DIR}/gs_extra_info"
EXTRA_INFO="${EXTRA_INFO_DIR}/${EXP}.pt"
mkdir -p "${EXTRA_INFO_DIR}"

cd "${SPLAT_DIR}"

if [[ "${1:-}" != "--export-only" ]]; then
PYTHONPATH="${SPLAT_DIR}" python nerfstudio/scripts/train.py splatfacto_on_mesh_uc \
    --vis tensorboard \
    --output-dir "${DATA_DIR}" \
    --experiment-name "${EXP}" \
    --max-num-iterations 30000 \
    \
    `# ==== ELLIPTICAL VERTEX FLOOR -- fixes (b) ===============================` \
    `# The vertex-layer counterpart of anisotropic_base_floor, which only ever rewrote   ` \
    `# the base layer's floor. Sizes each vertex Gaussian to its own 1-ring's ellipse     ` \
    `# instead of one radius on both axes, and pins its in-plane rotation to that         ` \
    `# ellipse's major axis so the (x, y) bounds describe the ellipse they are meant to.  ` \
    `#                                                                                    ` \
    `# NOT the Steiner formula: that rests on d_i^T M^-1 d_i == 2, an identity specific   ` \
    `# to THREE offsets summing to zero. On an n-gon 1-ring it does not hold, so an       ` \
    `# ellipse built from M alone would reach some neighbours and miss others -- silently ` \
    `# losing the reach guarantee the current isotropic radius provides. M therefore sets ` \
    `# only the SHAPE and ORIENTATION, and the ellipse is then scaled so the farthest     ` \
    `# neighbour lands exactly on its boundary: the same guarantee scatter_reduce(amax)   ` \
    `# on edge length gives today, kept exact. Verified on the function's own source:     ` \
    `# an isotropic hexagonal ring returns aspect 1.0000 and semi-axis 1.0 (i.e. today's  ` \
    `# circumscribing circle, unchanged); a 3x-stretched ring returns exactly 3.0000; a   ` \
    `# near-collinear ring caps at exactly 8:1; every case encloses all neighbours with    ` \
    `# the farthest on the boundary. On a flat square-grid mesh the ellipse covers the     ` \
    `# identical 1-ring at 0.577x the isotropic circle's footprint -- the mesh's own       ` \
    `# triangulation makes a 1-ring anisotropic (neighbours at 1, 1, sqrt2) even where the ` \
    `# surface is not, so this bites on ordinary meshes, not just on stretched ones.       ` \
    `# WATCH the startup line 'anisotropic vertex floor: ... footprint area vs the 1-ring  ` \
    `# circle Nx' for what it actually buys on THIS mesh before reading anything else.     ` \
    --pipeline.model.anisotropic_vertex_floor True `# NEW` \
    \
    `# Insurance against boundary/crease vertices, whose 1-ring is genuinely close to     ` \
    `# collinear and would otherwise yield an arbitrarily elongated ellipse -- i.e. the   ` \
    `# needle this run is elsewhere trying to remove. Matters more here than on a         ` \
    `# triangle: the mesh's triangle aspect q99 is 1.68, but a boundary vertex's ring has ` \
    `# no such bound. The cap only ever WIDENS the minor axis, so the reaches-every-      ` \
    `# neighbour guarantee survives it. Startup log prints the real hit rate.             ` \
    --pipeline.model.vertex_floor_max_aspect 8.0 `# NEW` \
    \
    `# Unchanged from gs10 but now load-bearing for TWO layers. Without it the coverage   ` \
    `# metric collapses every Gaussian to min(sx, sy), so an elliptical vertex filler is  ` \
    `# scored by the axis it is THIN on, its vertex reads short, vertex_needs_floor stays ` \
    `# set, and the floor holds it at the size it was supposed to stop being. Measured on ` \
    `# gs10: at the mesh vertices the anisotropic metric reads 3.527 against the isotropic` \
    `# 3.180, +10.9% -- and that is BEFORE the vertex layer is elliptical.                ` \
    --pipeline.model.anisotropic_coverage_density True \
    \
    `# ==== NEEDLES -- fixes (d) ===============================================` \
    `# Back to gs8's value, purely to put the cull under the bound: 5.0 * 4 = 20:1 vs the ` \
    `# 25:1 that upper_scale/min_scale_frac allows, so for the first time the cull can     ` \
    `# actually fire. It lands on exactly the right population -- gs10's free/detail rows  ` \
    `# have q90 23.63 and q99 pinned at 25.00 -- and base rows are exempt from culling by  ` \
    `# construction (culls &= ~is_base), so the coverage guarantee is untouched.           ` \
    `#                                                                                     ` \
    `# gs10 raised this to 10.0 believing 5.0 pushed "a large part of the detail layer     ` \
    `# back toward round". It does not: scale_reg is a HINGE, max(ratio, max_gauss_ratio)  ` \
    `# - max_gauss_ratio, which is identically zero for every row below the threshold. At  ` \
    `# a detail median of 1.835 a 5:1 hinge touches nothing near the median, only the tail.` \
    `# And it cannot fight the elliptical floors either: floored base rows are excluded by ` \
    `# scale_reg_exclude_base, the vertex layer is not in self.scales at all, and the mesh ` \
    `# triangle aspect q99 is 1.68 -- so no legitimate ellipse on this mesh comes near 5:1.` \
    `# If needles survive, go to 3.0 (cull 12:1) before touching min_scale_frac.           ` \
    --pipeline.model.max_gauss_ratio 5.0 `# gs10: 10.0, gs8: 5.0` \
    \
    `# ==== CENTROID COVERAGE -- fixes (c) =====================================` \
    `# THE single highest-leverage number in this file. gs10's face-centroid coverage      ` \
    `# piles up exactly on 0.950 (q25 AND q50), which is one base plate contributing       ` \
    `# base_opacity_floor at its own centroid with nothing else reaching. At a rescue      ` \
    `# threshold of 0.9 that entire pile scores as ALREADY FINE and never qualifies for    ` \
    `# repair -- the deterministic rescue, which gs8 made the primary coverage mechanism,  ` \
    `# has been structurally blind to the only population that is actually short. Moving   ` \
    `# to 1.0 matches coverage_target and makes the 0.950 pile qualify. Rescue spawns at   ` \
    `# whichever of {centroid, 3 corners} is worst, so it will land where the deficit is.  ` \
    --pipeline.model.coverage_rescue_thresh 1.0 `# gs10: 0.9` \
    \
    `# Must move WITH the line above, per coverage_rescue_thresh's own docstring: the cap  ` \
    `# exists precisely because raising the threshold toward coverage_target makes the     ` \
    `# spawn unbounded. gs10 did the opposite of a pair -- it raised the threshold to 0.9  ` \
    `# and cut the cap to a fifth of the code default, so demand went up while supply went ` \
    `# down. cap = frac * num_points, so at 0.05 and ~3.2M points that is ~160k faces per  ` \
    `# cycle, served worst-deficit-first; the rest are not dropped, they qualify again next ` \
    `# cycle. WATCH the log line 'coverage_rescue_spawn: spawning N (M qualified)'. If N    ` \
    `# equals the cap every single cycle for the whole run, raise this again; if VRAM is    ` \
    `# the binding constraint instead, lower it and accept slower convergence.              ` \
    --pipeline.model.coverage_rescue_max_frac 0.05 `# gs10: 0.01, gs8: 0.002 (code default 0.05)` \
    \
    `# ==== unchanged from gs10 =================================================` \
    --pipeline.model.anisotropic_base_floor True \
    --pipeline.model.scale_reg_exclude_base True \
    --pipeline.model.base_floor_corner_sigma 1.0 \
    --pipeline.model.base_floor_max_aspect 8.0 \
    --pipeline.model.base_floor_reaches_corners False \
    --pipeline.model.coverage_floor_target 1.0 \
    --pipeline.model.init_copies_per_face 4 \
    --pipeline.model.min_scale_frac 0.02 \
    --pipeline.model.adaptive_coverage_floor True \
    --pipeline.model.coverage_rescue_every 100 \
    --pipeline.model.opacity_reset_value 0.10 \
    --pipeline.model.opacity_reset_min_value 0.075 \
    --pipeline.model.coverage_lambda 0.0 \
    --pipeline.model.coverage_densify_scale 0.0 \
    --pipeline.model.cull_at_floor_gaussians False \
    --pipeline.model.opacity_reg_lambda 0.0 \
    --pipeline.model.base_opacity_floor 0.95 \
    --pipeline.model.elevate_coef 0.5 \
    --pipeline.model.gaussian_save_extra_info_path "${EXTRA_INFO}" \
    --pipeline.model.mesh_area_to_subdivide ${AREA} \
    --pipeline.model.upper_scale 0.5 \
    --pipeline.model.min_vertex_scale_frac 0.15 \
    --pipeline.model.use_base_layer True \
    --pipeline.model.copies_scale_shrink_power 0.3 \
    --pipeline.model.coverage_target 1.0 \
    --pipeline.model.curvature_densify_scale 5.0 \
    --pipeline.model.densify_grad_thresh 0.0004 \
    --pipeline.model.densify_size_thresh_frac 0.5 \
    --pipeline.model.dup_size_shrink_factor 1.2 \
    --pipeline.model.n_split_samples 3 \
    --pipeline.model.stop_split_at 25000 \
    --pipeline.model.refine_every 100 \
    --pipeline.model.warmup_length 500 \
    --pipeline.model.reset_alpha_every 30 \
    --pipeline.model.reassign_face_every 3000 \
    --pipeline.model.continue_cull_post_densification True \
    --pipeline.model.cull_alpha_thresh 0.05 \
    --pipeline.model.cull_scale_thresh 0.1 \
    --pipeline.model.init_opacity 0.9 \
    --pipeline.model.use_scale_regularization True \
    --pipeline.model.face_flat_coef 0.05 \
    --pipeline.model.unconstrained_scale True \
    --pipeline.model.unconstrained_elevate True \
    --pipeline.model.enable_fold_detection False \
    --pipeline.model.acm_lambda 0.0 \
    --pipeline.model.stop_acm_after_split True \
    --pipeline.model.mesh_depth_lambda 0.0 \
    --pipeline.model.sh_degree 3 \
    --pipeline.model.num_downscales 2 \
    --pipeline.model.background_color random \
    --pipeline.model.rasterize_mode antialiased \
    --pipeline.model.grazing_weight_enabled True \
    --pipeline.model.grazing_weight_power 2.0 \
    --pipeline.model.grazing_weight_floor 0.1 \
    \
    panoptic-data \
    --data "${DATA_DIR}" \
    --mesh_gauss_path "${MESH}" \
    --mesh_area_to_subdivide ${AREA} \
    --mesh_depth False \
    --downscale_factor 1 \
    --num_max_image 50 \
    --orientation_method none \
    --center_poses False \
    --auto_scale_poses False
fi

PYTHONPATH="${SPLAT_DIR}" python nerfstudio/scripts/exporter.py gaussian-splat \
    --load-config "${DATA_DIR}/${EXP}/config.yml" \
    --output-dir "${DATA_DIR}/${EXP}/export_ply"

echo "=== done -> ${DATA_DIR}/${EXP}/export_ply ==="
