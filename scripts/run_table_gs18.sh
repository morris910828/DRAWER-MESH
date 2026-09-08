#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 18.
#
# Baseline is run_table_gs16.sh (= gs12 restored, the sharpest configuration so far).
# ONE flag differs: the training resolution drops to a quarter, 5568x4176 -> 1392x1044.
#
# ---------------------------------------------------------------------------------------
# WHY, AND WHY IT IS NOT A RETREAT
#
# The mesh and the image resolution have to match, and right now they are out by 4x per
# axis. Measured, by projecting every face into every camera that sees it:
#
#     triangle edge on screen at 5568 px wide : 6.99 px (median)
#     a base plate spans its own face, so its sigma is edge/sqrt(3) : 4.04 px
#     text strokes in these photos            : ~2 px
#
# A single Gaussian has one colour. A 4 px plate laid over a 2 px stroke can only converge
# to the average of the stroke and the paper around it. No amount of training fixes that,
# and no parameter in this file changes the plate's size -- it is the face's size, and the
# face's size comes from the mesh.
#
# Solving for the width at which the plate stops blurring anything (sigma <= 1 px):
#
#     6.99 * (W / 5568) / sqrt(3) <= 1   ->   W <= 1378 px
#
#         downscale 1   5568 px   triangle 7.00 px   plate sigma 4.04 px
#         downscale 2   2784 px   triangle 3.50 px   plate sigma 2.02 px
#         downscale 4   1392 px   triangle 1.75 px   plate sigma 1.01 px   <- this run
#
# The independent check: a vanilla 3DGS run of this scene exists at data/table2 and looks
# clean. Its renders are 2782x1043, which is a GT|render pair, so it trained at 1391x1043
# -- 5568/4 to the pixel. That was never an efficiency difference, and the earlier "300k
# Gaussians vs 5.48M and sharper" framing here was wrong: per pixel of image the two are
# within 15% of each other.
#
#     vanilla 3DGS   300,000 GS over  1.45 Mpx = 0.207 GS/px
#     DRAWER gs12  5,480,079 GS over 23.3  Mpx = 0.236 GS/px
#
# So this run puts DRAWER at the resolution its own mesh is built for. If the result is
# comparable to data/table2's, the mesh/resolution mismatch was the whole story and the
# choice from here is a real one: this resolution with this mesh, or full resolution with
# a mesh cropped small enough to afford ~16x the face density over what is kept.
#
# WHAT THIS COSTS. The model's detail ceiling drops with the resolution -- viewed at 4K it
# will be soft, because nothing in it was ever asked to represent 4K detail. That is the
# honest trade, and it is the same trade the vanilla comparison has been making silently.
#
# ---------------------------------------------------------------------------------------
# READING THE NUMBERS AFTERWARDS -- IMPORTANT
#
# analyze_splat_ply.py's [4] projects with transforms.json, which is FULL resolution, so
# every sigma_px it prints is in 5568-px units regardless of what the run trained at.
# Divide by 4 for this run's own scale:
#
#     reported 4.0 px  ==  1.0 px at 1392   (a base plate; the target)
#     reported 1.56 px ==  0.39 px at 1392  (gs12's face-based median)
#
# Same for the painted-area split by sigma: the ">3 px" bucket at 5568 units is ">0.75 px"
# at this run's scale, so that particular cut stops meaning what it meant for gs12-gs17.
# Compare gs18 against gs12 in the SAME units and then divide, rather than reading either
# in isolation.
#
# Coverage is unaffected by resolution -- analyze_gap_coverage.py works on the mesh, not on
# any camera. gs11 baseline: 99.507 / 99.989 / 99.998% at thresholds 0.50 / 0.10 / 0.01.
# ---------------------------------------------------------------------------------------
#
# Usage (inside the drawer_splat env):
#   nohup bash scripts/run_table_gs18.sh > data/table/table_gs18.log 2>&1 &
#   tail -f data/table/table_gs18.log        # Ctrl-C leaves the run alive
#   bash scripts/run_table_gs18.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs18}"

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
    `# 1.0, not gs12's 0.6 -- see the header. gs12 ran before the scale reached the    ` \
    `# 1-ring ellipse, so its 0.6 only ever touched the z column and its vertex layer   ` \
    `# rendered at full reach (identical in-plane size to gs11's, which had no flag).    ` \
    `# With that fixed, 1.0 is what reproduces gs12; 0.6 would be a new change.          ` \
    --pipeline.model.vertex_radius_scale 1.0 `# gs12 file said 0.6, gs12 BEHAVED as 1.0` \
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
    `# 4, not 1: 1392x1044. This is the whole run -- see the header for the arithmetic ` \
    `# that picks 4 (plate sigma 4.04 px -> 1.01 px) and for why every sigma_px the     ` \
    `# analysis prints afterwards has to be divided by 4 to be read at this scale.      ` \
    --downscale_factor 4 `# gs16: 1` \
    --num_max_image 50 \
    --orientation_method none \
    --center_poses False \
    --auto_scale_poses False
fi

PYTHONPATH="${SPLAT_DIR}" python nerfstudio/scripts/exporter.py gaussian-splat \
    --load-config "${DATA_DIR}/${EXP}/config.yml" \
    --output-dir "${DATA_DIR}/${EXP}/export_ply"

echo "=== done -> ${DATA_DIR}/${EXP}/export_ply ==="
