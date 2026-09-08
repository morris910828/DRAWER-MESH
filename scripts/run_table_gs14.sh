#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 14.
#
# Baseline is run_table_gs12.sh (NOT gs13 -- see below). One flag differs.
#
# ---------------------------------------------------------------------------------------
# WHY gs13 IS NOT THE BASELINE, AND WHY THE GOAL CHANGED
#
# gs13 chased "Gaussian area coverage" to 100%. That number turned out to measure the
# wrong thing, and chasing it was actively harmful.
#
# It is an AREA BUDGET: sum of pi*sx*sy*alpha per face, capped at the face's own area. Two
# consequences. It is blind to WHERE the coverage sits -- a face with everything piled at
# its centroid scores like one covered evenly. And it is weighted by size x opacity, which
# is the definition of a blurry Gaussian, so pushing it up pushes the model toward large
# opaque plates. Across gs10->gs13 the metric and on-screen sharpness moved in opposite
# directions every single time. gs13 took it from 90.89% to 95.39% by returning 13% of
# faces to their full-face plate, and the population halved (5.48M -> 2.98M) as the detail
# underneath stopped earning its opacity and was culled.
#
# The actual requirement -- every point on every triangle reached by at least one Gaussian,
# no gaps -- was never measured until scripts/analyze_gap_coverage.py was written for it.
# Measured on gs11, whose area budget reads 91.34%:
#
#     threshold 0.50 (solidly covered)          99.507% covered, 0.493% gap
#     threshold 0.10 (visibly covered)          99.989% covered, 0.011% gap
#     threshold 0.01 (rasterizer writes)        99.998% covered, 0.002% gap
#
#     1,008 of 1,395,229 faces show any gap at all (0.072%), 31 of them over half. Those
#     faces are statistically ordinary -- same size, same shape, MORE Gaussians than
#     average (4 vs 3), fully opaque -- which points at the tool's own 2-ring omission
#     rather than a real defect. The measurement is conservative, so the truth is better.
#
# So there is no coverage problem, and there never was. The wall that made every attempt at
# sharpness look like it cost coverage was an artifact of the metric.
#
# ---------------------------------------------------------------------------------------
# WHAT THIS RUN CHANGES, AND THE LOOP IT IS AIMED AT
#
# With the false constraint gone, the only remaining problem is blur, and the mechanism is
# known. A floored base plate spans its whole face -- ~7 px across on screen, sigma ~4 px
# against 2 px text strokes -- and renders at base_opacity_floor, 0.95. That plate already
# explains its patch of the image, at the grey mean of everything under it. So:
#
#   1. the plate is opaque and covers the face
#   2. it explains the image well enough that the residual is small
#   3. detail Gaussians beneath it earn little opacity
#   4. reset_alpha_every knocks opacity to 0.10 and cull_alpha_thresh removes what does
#      not earn it back
#   5. the face now has no detail, so the coverage check keeps its floor engaged
#   6. back to 1
#
# The correlation across runs is exact: gs10 0.81 detail/face at 91.9% floored, gs12 2.42
# at 79.0%, gs13 0.63 at 92.0%.
#
# Dropping the floor to 0.2 leaves the plate exactly where it is geometrically -- it still
# spans its face, so it still prevents holes -- while leaving 80% of the residual for
# detail Gaussians to earn. Contribution at the plate centre becomes 0.20 and at its
# corners 0.20*exp(-0.5) = 0.12, both still above the 0.10 "visibly covered" threshold, so
# a face carrying nothing but its plate is still covered by the measure that matters.
#
# Under the OLD metric this change is suicide: area budget is proportional to alpha, so it
# would cut the reported number to roughly a fifth. That is exactly the point -- the number
# it destroys is the one that was never measuring the requirement.
# ---------------------------------------------------------------------------------------
#
# Usage (inside the drawer_splat env):
#   nohup bash scripts/run_table_gs14.sh > data/table/table_gs14.log 2>&1 &
#   tail -f data/table/table_gs14.log        # Ctrl-C leaves the run alive
#   bash scripts/run_table_gs14.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs14}"

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
    `# ==== RESCUE RECALIBRATION -- the second OOM, and its real cause ==========` \
    `# The first OOM here was blamed on the seed size. That was wrong. The log shows   ` \
    `# split/dup producing 258 rows in the same cycle the coverage rescue produced      ` \
    `# 483,463, with 800,895 faces qualifying and the population at 9.73M.              ` \
    `#                                                                                  ` \
    `# Cause: coverage_rescue_thresh is compared against a density that a floored face  ` \
    `# gets almost entirely from its base plate's opacity. At base_opacity_floor 0.95   ` \
    `# such a face scored ~0.95 at its centroid and a threshold of 1.0 caught only the   ` \
    `# genuinely thin ones. At 0.2 the same face scores 0.2, so essentially EVERY face   ` \
    `# fails -- and coverage_rescue_max_frac then spawns 5% of the current population    ` \
    `# every cycle, which compounds. The threshold has to move with the opacity floor;   ` \
    `# leaving it at 1.0 turns the rescue into an unbounded producer.                    ` \
    `#                                                                                   ` \
    `# 0.15 sits just under what a bare plate now scores (0.20), so the rescue catches   ` \
    `# faces that really have nothing and ignores the ones a plate is already holding.   ` \
    `# Sanity-check it against the log: qualified should be thousands, not ~800k.        ` \
    --pipeline.model.coverage_rescue_thresh 0.15 `# gs12: 1.0 -- moved with base_opacity_floor` \
    \
    `# Absolute per-cycle ceiling as well, so the fraction cap can never compound again ` \
    `# even if the threshold is mis-set. max_gaussians is now enforced here too, but     ` \
    `# this bounds the RATE rather than the total.                                       ` \
    --pipeline.model.coverage_rescue_max_count 2500 `# NEW` \
    --pipeline.model.coverage_rescue_max_frac 0.05 \
    --pipeline.model.anisotropic_base_floor True \
    --pipeline.model.scale_reg_exclude_base True \
    --pipeline.model.base_floor_corner_sigma 1.0 \
    --pipeline.model.base_floor_max_aspect 8.0 \
    --pipeline.model.base_floor_reaches_corners False \
    --pipeline.model.coverage_floor_target 1.0 \
    `# ==== POPULATION BRAKES, added after this run OOMed once ==================` \
    `# 4 seeds 6.98M rows (1.4M base + 4/face) -- already ABOVE what gs12 FINISHED     ` \
    `# with (5.48M). gs12 only fit because its opaque plates starved the detail beneath ` \
    `# them of opacity and ~1.5M rows were culled away early. Lowering base_opacity_    ` \
    `# floor to let that detail survive is the whole point of this run, so that culling ` \
    `# pressure is gone and the seed alone no longer fits. 2 seeds 4.19M and leaves     ` \
    `# room to densify. gs10 raised this to 4 because detail could not otherwise take   ` \
    `# over a face from its plate -- at alpha 0.2 it no longer has to out-compete one.   ` \
    --pipeline.model.init_copies_per_face 2 `# gs12: 4` \
    \
    `# Hard ceiling so a population runaway can never cost another 4.5 hours. Sized at  ` \
    `# gs12's measured 5.48M, which is what actually fit in 32 GB at full resolution.   ` \
    `# Densification pauses on reaching it; culling, rescue and floor bookkeeping carry ` \
    `# on. If the log says it was reached, the answer is a smaller seed or a higher     ` \
    `# densify_grad_thresh -- NOT a bigger ceiling.                                     ` \
    --pipeline.model.max_gaussians 5500000 `# NEW` \
    --pipeline.model.min_scale_frac 0.02 \
    --pipeline.model.adaptive_coverage_floor True \
    --pipeline.model.coverage_rescue_every 100 \
    --pipeline.model.opacity_reset_value 0.10 \
    --pipeline.model.opacity_reset_min_value 0.075 \
    --pipeline.model.coverage_lambda 0.0 \
    --pipeline.model.coverage_densify_scale 0.0 \
    --pipeline.model.cull_at_floor_gaussians False \
    --pipeline.model.opacity_reg_lambda 0.0 \
    `# ==== THE ONLY CHANGE IN THIS RUN ========================================` \
    `# 0.95 -> 0.2. The plate keeps its full-face SIZE, so the no-gap guarantee is    ` \
    `# intact; what it gives up is the share of the image it was explaining, which is  ` \
    `# what lets the detail layer underneath earn opacity instead of being culled.     ` \
    `# See the header for the loop this is aimed at.                                   ` \
    `#                                                                                 ` \
    `# If detail-per-face does not rise well above gs12's 2.42, the loop was not the   ` \
    `# binding mechanism and 0.2 was the wrong lever -- do NOT just push it lower       ` \
    `# without saying why. Verify no-gap with scripts/analyze_gap_coverage.py, never    ` \
    `# with area coverage.                                                              ` \
    --pipeline.model.base_opacity_floor 0.2 `# gs12: 0.95` \
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
