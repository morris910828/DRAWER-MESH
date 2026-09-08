#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 15.
#
# Baseline is run_table_gs14.sh. Four flags differ, one of them the point of the run.
#
# ---------------------------------------------------------------------------------------
# WHAT gs14 ESTABLISHED, INCLUDING BY FAILING
#
# gs14 dropped base_opacity_floor 0.95 -> 0.2 to stop the full-face base plate from
# pre-explaining its patch of the image. It also changed init_copies_per_face and
# coverage_rescue_thresh in the same run -- three changes at once, which normally makes a
# result unattributable. The OOM log from its first attempt happens to separate them:
#
#     Splitting        255 / 9726680
#     Duplicating        3 / 9726680
#     coverage_rescue_spawn: 483463 gaussian(s) (800895 faces qualified)
#
# In one cycle the rescue produced 483,463 rows and densification produced 258. THE
# COVERAGE RESCUE IS THIS MODEL'S DETAIL PRODUCER, by roughly 1900 to 1. Densification is
# almost inert here. That was not known before and it explains gs14's result outright:
#
#     run    rescue_thresh   detail/face   face-based sigma_px
#     gs12   1.0             2.42          1.56
#     gs14   0.15            0.34          3.60
#
# Lowering the threshold to 0.15 switched the producer off. With detail gone, 70% of
# face-based rows are base plates (gs12: 24%), and a plate's inherent sigma on this mesh is
# ~4 px -- which is what 3.60 px measures. base_opacity_floor was not the cause: the plates
# did go faint, 17.85% of rows sit exactly on the 0.2 floor.
#
# gs14 also settled the coverage question, at the thresholds that correspond to "no holes":
#
#     threshold 0.50   77.954% covered      (expected: a lone plate now scores 0.12 at its
#                                            corners rather than 0.58)
#     threshold 0.10   99.999% covered      BETTER than gs11's 99.989%
#     threshold 0.01  100.000% covered      no gap at all
#
# So faint plates cost nothing in real coverage. What they cost was the detail producer,
# and only because the rescue threshold was moved with them.
#
# ---------------------------------------------------------------------------------------
# WHAT THIS RUN DOES
#
# Put the producer back on while keeping the faint plates, and spend the memory budget on
# rescue-placed Gaussians instead of uniform seed copies.
#
# The combination -- faint plates AND an aggressive rescue -- is exactly what OOMed, and
# for a reason worth stating: a faint plate scores 0.2 at its own centroid, so against a
# threshold of 1.0 essentially every face qualifies, and coverage_rescue_max_frac then
# spawns 5% of the CURRENT population per cycle, which compounds. It is safe now because
# both bounds exist: coverage_rescue_max_count caps the per-cycle rate absolutely, and
# max_gaussians caps the total and is enforced on the rescue path too (it originally gated
# only densification, which is how a run reached 9.7M rows).
#
# Budget, in the units max_gaussians actually counts (face-based rows only -- the 707,579
# vertex rows are a separate fixed dict and are NOT included):
#
#     ceiling                4.80M    (gs12 finished at 4.77M face-based and fit in 32 GB)
#     seed  1 copy/face      2.79M    (1.40M base + 1.40M detail)
#     headroom for rescue    2.01M    at 6000/cycle x 300 cycles = 1.80M, inside it
#
# Seed copies are spread uniformly over every face whether it needs them or not; rescue
# rows are placed at whichever of a face's four check points is worst. Same budget, aimed
# rather than sprayed.
# ---------------------------------------------------------------------------------------
#
# Usage (inside the drawer_splat env):
#   nohup bash scripts/run_table_gs15.sh > data/table/table_gs15.log 2>&1 &
#   tail -f data/table/table_gs15.log        # Ctrl-C leaves the run alive
#   bash scripts/run_table_gs15.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs15}"

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
    `# ==== THE POINT OF THIS RUN ==============================================` \
    `# Back to 1.0. At 0.15 the rescue -- this model's only real detail producer --   ` \
    `# stopped firing and face-based sigma went from gs12's 1.56 px to 3.60 px. With  ` \
    `# plates now faint (0.2), a face carrying only its plate scores 0.2 here, so a    ` \
    `# threshold of 1.0 means "keep adding detail until something other than the plate ` \
    `# is holding this face up" -- which is precisely the intent.                       ` \
    `#                                                                                  ` \
    `# WATCH the first cycles: qualified will be large (most faces DO score only 0.2)   ` \
    `# and that is expected now. What must stay bounded is SPAWNED, which the two caps  ` \
    `# below hold at 6000/cycle regardless of how many qualify.                          ` \
    --pipeline.model.coverage_rescue_thresh 1.0 `# gs14: 0.15, gs12: 1.0` \
    \
    `# Absolute per-cycle ceiling as well, so the fraction cap can never compound again ` \
    `# even if the threshold is mis-set. max_gaussians is now enforced here too, but     ` \
    `# this bounds the RATE rather than the total.                                       ` \
    --pipeline.model.coverage_rescue_max_count 6000 `# gs14: 2500 -- fills the headroom below` \
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
    --pipeline.model.init_copies_per_face 1 `# gs14: 2 -- budget moved to the rescue` \
    \
    `# Hard ceiling so a population runaway can never cost another 4.5 hours. Sized at  ` \
    `# gs12's measured 5.48M, which is what actually fit in 32 GB at full resolution.   ` \
    `# Densification pauses on reaching it; culling, rescue and floor bookkeeping carry ` \
    `# on. If the log says it was reached, the answer is a smaller seed or a higher     ` \
    `# densify_grad_thresh -- NOT a bigger ceiling.                                     ` \
    `# Corrected: max_gaussians counts self.num_points, which is the FACE-BASED        ` \
    `# population only -- the 707,579 vertex rows are a separate fixed dict. 5.5M       ` \
    `# therefore allowed 6.21M total, 13% above the 5.48M that actually fit. 4.80M      ` \
    `# matches gs12's 4.77M face-based rows, which is the measured budget.               ` \
    --pipeline.model.max_gaussians 4800000 `# gs14: 5500000 (mis-sized: wrong units)` \
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
