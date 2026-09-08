#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 17.
#
# Baseline is run_table_gs16.sh (= gs12 restored). This removes BOTH mandatory coverage
# layers and lets densification decide where Gaussians go, which is what vanilla 3DGS does.
#
# ---------------------------------------------------------------------------------------
# THE COMPARISON THAT PROMPTED THIS
#
# A vanilla 3DGS run of the same table exists at data/table2/table2/ply, and it is sharp:
#
#     vanilla 3DGS    300,000 Gaussians    0.75 GB    PSNR 23.23 / SSIM 0.846
#     DRAWER gs12   5,480,079 Gaussians   ~32 GB      visibly blurrier
#
# 18x the Gaussians and 40x the memory for a worse image. Whatever is wrong, it is not the
# budget -- which retires the earlier "short by 3.6x, crop the mesh" conclusion outright.
#
# WHERE THE BUDGET GOES. Weighting each Gaussian by the screen area it paints
# (pi * sigma_x_px * sigma_y_px * alpha) on one training camera, gs11:
#
#     sigma <= 1 px    23.5% of the count    0.9% of the painted area
#     sigma >  3 px    32.1% of the count   86.1% of the painted area
#
#     face-based    4,715,865 rows   69.6% of paint   sigma median 1.55 px
#     vertex layer    702,813 rows   30.4% of paint   sigma median 6.29 px
#
# The sharp Gaussians are nearly invisible: everything at or below 1 px totals 0.9% of the
# painted area, so at most ~3.5% of a 23.3 Mpx frame can carry 1 px detail even if none of
# it overlaps. The image is painted by the big ones. The [4] sigma median of 1.55 px that
# earlier runs were judged on is a median BY COUNT, which weights a Gaussian that paints
# nothing the same as one that paints a whole triangle -- it was the wrong statistic.
#
# WHAT THE BIG ONES ARE. Two populations that exist by fiat, not because training chose
# them: one plate per face floored to that face's own size, and one Gaussian per mesh
# vertex sized to its 1-ring. 1.40M + 0.71M = 2.11M rows, 44% of the budget, placed
# uniformly over the surface whether or not there is anything there to resolve. They
# cannot be sharp: the mesh's triangles are ~7 px on screen, so a plate spanning one is
# ~4 px against 2 px text strokes, and a single Gaussian has one colour.
#
# Both exist as coverage insurance. use_base_layer's own default is False -- every run
# since gs8 turned it on by hand. The vertex layer had no switch at all until now.
#
# ---------------------------------------------------------------------------------------
# WHY BOTH GO AT ONCE, AND HOW TO READ THE RESULT
#
# One change per run is the rule here and this breaks it deliberately: the two layers are
# one mechanism, and the vertex layer alone paints 30% of the frame, so switching off only
# the base layer would likely show no signal and prove nothing. The decision tree is:
#
#   sharp AND scripts/analyze_gap_coverage.py holds       -> the layers were the problem
#   sharp BUT gaps open                                   -> turn ONE back on and re-measure;
#                                                            the tool's --no-vertex-layer flag
#                                                            separates their contributions
#   not sharp                                             -> the layers were not the problem,
#                                                            and the constraint is elsewhere
#
# Coverage is measurable now and must be judged ONLY with analyze_gap_coverage.py. Baseline
# to beat, gs11 with both layers on: 99.507 / 99.989 / 99.998% at thresholds 0.50/0.10/0.01.
# Excluding just the vertex layer from that test gave 95.725 / 99.153 / 99.745%, so expect
# some loss -- the question is how much, not whether.
#
# Ignore "Gaussian area coverage" entirely. It is sum(pi*sx*sy*alpha), blind to placement
# and weighted by opacity, i.e. a blur metric; gs14 read 79.15% on it while measuring
# 100.000% actual coverage.
# ---------------------------------------------------------------------------------------
#
# Usage (inside the drawer_splat env):
#   nohup bash scripts/run_table_gs17.sh > data/table/table_gs17.log 2>&1 &
#   tail -f data/table/table_gs17.log        # Ctrl-C leaves the run alive
#   bash scripts/run_table_gs17.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs17}"

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
    `# 3, not gs16's 4. With use_base_layer off there is no base row per face, so 4   ` \
    `# copies seed 4 x 1,395,229 = 5.58M -- ABOVE the 5.5M ceiling below. The first    ` \
    `# attempt at this run did exactly that: every spawn was blocked from step 0, so   ` \
    `# faces emptied by reassignment were never refilled and cull_gaussians()'s        ` \
    `# no-empty-face assert fired at step 24999 with 44,296 of them. 3 seeds 4.19M and ` \
    `# leaves 1.31M of headroom for the rescue.                                        ` \
    --pipeline.model.init_copies_per_face 3 `# gs16: 4 -- must stay under max_gaussians` \
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
    `# ==== THE CHANGE: drop both mandatory coverage layers =====================` \
    `# use_base_layer False is this flag's OWN DEFAULT -- every run since gs8 set it   ` \
    `# True by hand. Off, no face is handed a plate the size of itself, and the        ` \
    `# base_floor_*, adaptive_coverage_floor, base_opacity_floor and                    ` \
    `# scale_reg_exclude_base settings below all become no-ops (left in place so the    ` \
    `# diff against gs16 stays one idea rather than a rewrite).                          ` \
    --pipeline.model.use_base_layer False `# gs16: True -- and False is the default` \
    \
    `# New switch (the layer was hardcoded before). 702,813 rows painting 30.4% of the  ` \
    `# frame at a median 6.29 px, against face-based 1.55 px. Off, min_scale_frac is     ` \
    `# the only remaining scale floor and every Gaussian is one densification chose.      ` \
    --pipeline.model.use_vertex_layer False `# NEW switch; True was the hardcoded behaviour` \
    \
    `# ==== brakes, because this is unexplored territory =======================` \
    `# With no plate underneath them, faces score far lower on the 4-point density the  ` \
    `# rescue tests, so many more will qualify -- the same coupling that made gs14 spawn ` \
    `# 483,463 rows in one cycle. coverage_rescue_thresh stays at gs12's 1.0 and these   ` \
    `# two bound it instead. With the vertex layer gone, num_points IS the total, so     ` \
    `# 5.5M here really is the ~5.48M that fit in 32 GB (unlike gs14's, which counted     ` \
    `# face-based only and so allowed 6.21M).                                             ` \
    --pipeline.model.max_gaussians 5500000 `# NEW` \
    --pipeline.model.coverage_rescue_max_count 6000 `# NEW` \
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
