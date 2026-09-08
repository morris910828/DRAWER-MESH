#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 12.
#
# Baseline is run_table_gs11.sh. Lines are marked against gs11's value.
#
# ---------------------------------------------------------------------------------------
# WHAT gs11 ACHIEVED  (scripts/analyze_splat_ply.py, gs11 vs gs10)
#
#   free/detail aspect q90   23.63 -> 4.14      needles gone
#   free/detail aspect q99   25.00 -> 4.88      no longer pinned to upper_scale/min_scale_frac
#   face centroids >= target 18.65% -> 45.61%   the 0.950 pile broke up (q50 0.950 -> 0.999)
#   face-based sigma_px q50   2.50 -> 1.55 px   the big sharpness win
#   gaussians                3.23M -> 5.46M
#
#   Regressions to keep an eye on, NOT acted on here (one change set per run):
#   area coverage 93.17% -> 91.34%, centroid short(<0.9x) 5.52% -> 11.32%, faces under
#   half their own area 7.47% -> 9.03%. More faces cleared target and more fell badly
#   short at the same time -- the population nearly doubled and churned. If gs12 does not
#   move these, the next lever is coverage_rescue_max_frac, not anything in this file.
#
# ---------------------------------------------------------------------------------------
# WHAT gs11 DID NOT FIX, AND WHY THE EXPORT HID IT
#
# The user's report after gs11 was "still very blurry, and the vertex Gaussians are all
# circles". Both observations were correct. Neither meant what it looked like.
#
# THE CIRCLES WERE AN EXPORT BUG, NOT A TRAINING RESULT. exporter.py capped every
# population at 3x the 99th percentile of upper_scale * xyz_radius -- a FACE-derived
# bound -- including the vertex layer, whose size comes from exp(vertex_log_scales), the
# vertex's 1-ring reach across several faces. That cap sat at 0.00622 against a vertex
# ceiling whose median is 0.01025, so it clipped 89.5% of the layer, and clipped BOTH
# in-plane axes on 56.9% of it. Two axes clipped to one value is sx == sy: a perfect
# circle, written at export time. Read back from the checkpoint instead, gs11's vertex
# layer has aspect median 1.2055 with 0.2% circles -- anisotropic_vertex_floor had been
# working since gs11 and the ply could not show it. Fixed in exporter.py; the vertex and
# centroid blocks now derive their own caps. Nothing about training changed, so
# re-exporting gs11 (--export-only) shows the corrected picture without retraining.
#
# THE BLUR IS THE VERTEX LAYER, AND THE CEILING IS WHAT HOLDS IT UP. With the export
# distortion removed, gs11's true numbers are:
#     vertex layer TRUE minor-axis median   0.00663
#     face-based        minor-axis median   0.00111     -> the vertex layer is ~6x
#     vertex rows still visible (alpha>0.05)  324,432 of 707,579
# gs11 took the face-based population from 2.50 px to 1.55 px on screen and left this
# layer untouched, so it is now the only blurry population and stands out badly.
#
# It is the CEILING, not the floor. Training had already pushed the layer from its 0.01025
# ceiling down to 0.00875 and stopped, and releasing every floor changes the rendered
# aspect from 1.2051 to 1.2055 and nothing else -- min_vertex_scale_frac is not the lever
# and never was. The ceiling was a hardcoded local (_VERTEX_RADIUS_SAFETY_MARGIN = 1.0)
# with no way to lower it; it is now config.vertex_radius_scale.
# ---------------------------------------------------------------------------------------
#
# Usage (inside the drawer_splat env):
#   bash scripts/run_table_gs12.sh
#   bash scripts/run_table_gs12.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs12}"

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
    `# ==== I/O COST -- no effect on the model, added 2026-08-28 ===============` \
    `# A run was writing ~65 GB to the mounted volume and taking >8h wall clock against  ` \
    `# ~3.8h of actual Trainer.train_iteration time. Almost all of that write volume is   ` \
    `# throwaway.                                                                          ` \
    `#                                                                                     ` \
    `# The checkpoint is 3.78 GB and save_only_latest_checkpoint is true, so every save     ` \
    `# but the last is written and then deleted. At 2000 that was 15 writes / 56.8 GB for   ` \
    `# one surviving file. 10000 keeps two mid-run resume points and drops ~45 GB.          ` \
    --steps-per-save 10000 `# was 2000` \
    \
    `# 300 full-resolution eval renders, each a 69.8 MB uncompressed RGB frame at          ` \
    `# 5568x4176, is what made tfevents 6.8 GB. This also drives how often Test PSNR       ` \
    `# refreshes in the terminal, so it is a feedback setting as much as an I/O one --      ` \
    `# 2000 was tried first and starved the run of visible progress. 500 keeps 60 of them   ` \
    `# (PSNR every ~500 steps, still frequent enough to watch) at roughly 1/6 the write      ` \
    `# volume. Do not raise this further without saying so: the number you give up is the    ` \
    `# only per-run quality signal visible before the export.                                ` \
    --steps-per-eval-image 500 `# was 100` \
    \
    `# The all-images pass profiles at 7.16 s per call and rendered all 50 eval frames at   ` \
    `# full res, 30 times. Once at the end is what the numbers are actually read from --     ` \
    `# unlike the single-image eval above, nothing watches this mid-run.                     ` \
    --steps-per-eval-all-images 30000 `# was 1000` \
    \
    `# ==== THE ONLY TRAINING CHANGE IN THIS RUN ===============================` \
    `# Shrink the vertex layer's size ceiling to 60% of the 1-ring reach. This is the    ` \
    `# one population gs11 left blurry, at ~6x the face-based minor axis, and the ceiling ` \
    `# is the only thing holding it there.                                                ` \
    `#                                                                                     ` \
    `# Nearly free in coverage, from the definition rather than from a guess: a vertex      ` \
    `# Gaussian's contribution to vert_cov AT ITS OWN ANCHOR is opacity * exp(0), which     ` \
    `# does not depend on its size at all (_compute_coverage_density). So this cannot       ` \
    `# reduce coverage at the vertex each Gaussian is anchored to. It only reduces the      ` \
    `# cross-contribution to nearby face CENTROIDS -- a wide, low-opacity smear, i.e.        ` \
    `# coverage bought with blur, the same trade gs8 rejected when it set coverage_lambda   ` \
    `# to 0. And there is room: gs11 measured mesh vertices at 95.46% of target with a       ` \
    `# median of 3.107 against a target of 1.0, roughly 3x oversupplied.                     ` \
    `#                                                                                       ` \
    `# What this GIVES UP, stated plainly: below 1.0 the layer no longer reaches every       ` \
    `# 1-ring neighbour, which is the property the isotropic radius -- and the ellipse built ` \
    `# on it -- was constructed to guarantee. 0.6 is chosen against the 3x oversupply, not   ` \
    `# derived from anything. If the mesh-vertices coverage row falls materially below       ` \
    `# gs11's 95.46%, come back up to 0.8 rather than pushing further down.                  ` \
    --pipeline.model.vertex_radius_scale 0.6 `# NEW (was the hardcoded 1.0)` \
    \
    `# ==== unchanged from gs11 =================================================` \
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
