#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 13.
#
# Baseline is run_table_gs12.sh. Lines are marked against gs12's value.
#
# ---------------------------------------------------------------------------------------
# WHAT gs12 SETTLED  (scripts/analyze_splat_ply.py, gs12 vs the scale-1.0 baseline)
#
#   [4] vertex layer sigma_px  6.29 -> 3.73    = 0.593x, matching vertex_radius_scale 0.6
#   [2] mesh vertices >=target 95.46% -> 94.35%   cost of that: 1.1 points
#   [2] face centroids >=target 45.61% -> 48.85%  went UP; the predicted loss did not happen
#   [4] face-based sigma_px    1.55 -> 1.56       untouched, nothing leaked
#   wall clock                 >8h -> 4h30m       from the I/O settings, not the model
#
# So shrinking the vertex layer is close to free and the layer is still the blurriest
# population by a wide margin: 3.73 px against the face-based 1.56 px, where resolving a
# 2 px text stroke needs sigma <~1 px. Going further is the obvious next step, and the
# coverage it spends is measured, not guessed.
#
# ---------------------------------------------------------------------------------------
# THE COVERAGE NUMBER, AND WHY NOTHING SO FAR HAS MOVED IT
#
# "Gaussian area coverage" has sat at 93.17 -> 91.34 -> 90.89% across gs10/11/12 and the
# user wants it at 100%. Three things had to be established before touching it:
#
# (1) It is NOT the vertex layer. Area coverage groups by mesh_face_idx, and vertex rows
#     carry -1, so that layer contributes nothing to it. Shrinking it cannot lower the
#     number, and gs13's move to 0.4 is free with respect to coverage.
#
# (2) It is NOT unobserved geometry. Every one of the 1,395,229 faces is inside at least
#     one camera frustum, and the area-starved ones are seen front-facing by a median of
#     18 of the 50 cameras -- statistically the same as the 20 for all faces. There is no
#     unreachable region putting a ceiling under 100%.
#
# (3) It IS the floor-release population, and the deficit is concentrated there. A floored
#     base plate is its face's Steiner ellipse, whose area is 2.418x the triangle's for
#     every triangle shape -- exactly the per-face ratio median the analysis reports
#     (2.418, q25 2.297). 79.0% of faces are still floored, so they cap at 1.0 and
#     contribute fully. Solving 0.79 * 1.0 + 0.21 * x = 0.9134 gives x = 0.588: the 21%
#     whose floor was RELEASED average 59% of their own area, and that is the whole gap.
#
# The reason it never moved: nothing in the training loop optimises it. coverage_lambda
# (off since gs8) and coverage_rescue_thresh both measure DENSITY AT FOUR POINTS -- the
# centroid and the three corners. A face can pass all four with its interior empty, which
# is precisely what a released face does. The reported number has been a spectator.
#
# What this run does NOT do about it: add an area term to coverage_floor_target. That
# would re-engage the floor on those faces, restoring a plate as large as the whole
# triangle -- which is what took on-screen sigma from gs10's 2.50 px down to 1.56 px in
# the first place. It would buy the number back by reintroducing the blur, i.e. undo
# gs11 and gs12. Rescue SPAWNS Gaussians instead, so the area arrives at detail size.
# ---------------------------------------------------------------------------------------
#
# Usage (inside the drawer_splat env):
#   nohup bash scripts/run_table_gs13.sh > data/table/table_gs13.log 2>&1 &
#   tail -f data/table/table_gs13.log        # Ctrl-C leaves the run alive
#   bash scripts/run_table_gs13.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs13}"

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
    `# ==== I/O (unchanged from gs12, took wall clock >8h -> 4h30m) ============` \
    --steps-per-save 10000 \
    --steps-per-eval-image 500 \
    --steps-per-eval-all-images 30000 \
    \
    `# ==== CHANGE 1 -- push the vertex layer further ==========================` \
    `# 0.6 cost 1.1 points of vertex coverage (95.46 -> 94.35%) and bought 6.29 -> 3.73 px.` \
    `# The cost was far below linear, so 0.4 is the next step on the same curve; expect     ` \
    `# roughly 2.5 px. Note what the analysis footnote says about that 94.35%: it EXCLUDES  ` \
    `# the vertex layer's own contribution, so face-based Gaussians alone already carry     ` \
    `# 94% of vertices to target and this layer is insurance for the remaining few percent  ` \
    `# -- while being the blurriest population in the model. That asymmetry is what makes   ` \
    `# spending vertex coverage for sharpness the right trade here.                          ` \
    `# If mesh vertices falls materially below 94%, come back to 0.5 rather than pushing on. ` \
    --pipeline.model.vertex_radius_scale 0.4 `# gs12: 0.6` \
    \
    `# ==== CHANGE 2 -- make coverage rescue see the area deficit ==============` \
    `# The rescue's qualification test becomes "4-point density OR area below its own face  ` \
    `# area", so the 21% of faces carrying the entire gap finally qualify for repair. 1.0    ` \
    `# asks every face to cover its own area outright, which is the 100% target stated       ` \
    `# directly rather than approached by proxy.                                              ` \
    `#                                                                                        ` \
    `# Cost estimate before running, so the population is not a surprise: the gap is 9.1% of  ` \
    `# 49.47 total mesh area = 4.51, and a detail Gaussian contributes pi*major*minor*alpha    ` \
    `# ~ 6.3e-6, so closing it takes on the order of 718k new Gaussians -- about +13% on       ` \
    `# gs12's 5.48M. Rate-limited by coverage_rescue_max_frac (0.05 of current count per       ` \
    `# cycle) either way, so it arrives gradually rather than all at once.                     ` \
    `#                                                                                          ` \
    `# WATCH 'coverage_rescue_spawn: spawning N (M qualified)'. M should jump sharply from      ` \
    `# gs12 (the area-short faces are new qualifiers) and then FALL over the run as they are    ` \
    `# filled. M still at the cap near step 29000 means the rescue never caught up and the       ` \
    `# final number will land short of 100%; raise coverage_rescue_max_frac next, not this.      ` \
    --pipeline.model.coverage_rescue_area_target 1.0 `# NEW (0.0 = the historical 4-point-only test)` \
    \
    `# The first attempt at this run OOMed at step 3101 -- at HALF resolution, on a config  ` \
    `# whose predecessor fit at FULL resolution in the same 32 GB with 5.48M rows. Cause:   ` \
    `# coverage_rescue_max_frac caps at 5% of the CURRENT population, so once the cap binds  ` \
    `# every cycle the population compounds -- 1.05^26 = 3.6x over the ~26 cycles from       ` \
    `# warmup to step 3100, on a ~7M seed. The fraction was never the problem while the      ` \
    `# 4-point test kept qualifiers few and culling kept pace; the area target qualifies far ` \
    `# more faces, spawning outruns culling, and the compounding term takes over.            ` \
    `#                                                                                        ` \
    `# An absolute per-cycle count makes the worst case linear and knowable: 2500 x 300       ` \
    `# cycles = at most +0.75M, i.e. 6.23M against gs12's 5.48M. Sized against the measured   ` \
    `# deficit, not by feel -- closing the 9.1% area gap needs ~718k rows. Resolution is NOT  ` \
    `# a lever here and is not being touched: full res is the whole point of the project.     ` \
    --pipeline.model.coverage_rescue_max_count 2500 `# NEW (0 = fraction cap only, the old behaviour)` \
    \
    `# ==== unchanged from gs12 =================================================` \
    --pipeline.model.anisotropic_vertex_floor True \
    --pipeline.model.vertex_floor_max_aspect 8.0 \
    --pipeline.model.anisotropic_coverage_density True \
    --pipeline.model.max_gauss_ratio 5.0 \
    --pipeline.model.coverage_rescue_thresh 1.0 \
    --pipeline.model.coverage_rescue_max_frac 0.05 \
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
