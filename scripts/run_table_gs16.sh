#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 16.
#
# This is run_table_gs12.sh restored. gs12 is the sharpest run to date and gs13, gs14 and
# gs15 were all worse than it; this returns to that configuration as the base to build on.
# ONE flag differs from gs12's file, and it exists to make this run match what gs12
# ACTUALLY DID rather than what its file said -- see below.
#
# ---------------------------------------------------------------------------------------
# WHY vertex_radius_scale IS 1.0 HERE AND 0.6 IN gs12'S FILE
#
# gs12 ran against a version of populate_modules where config.vertex_radius_scale was
# applied to self.vertex_radius but NOT to the 1-ring ellipse that overwrites the in-plane
# columns of _vertex_xyz_radius. With anisotropic_vertex_floor on -- which gs12 had -- the
# scale therefore governed only the z column and left the footprint untouched.
#
# Measured, not inferred: gs11 (no such flag, so 1.0) and gs12 (0.6) exported vertex layers
# with identical in-plane sizes, major median 0.00866 and minor 0.00652 in both. gs12's
# 1.56 px was produced with the vertex layer at FULL 1-ring reach.
#
# The bug is fixed now, so re-running gs12's file verbatim would apply a 0.6 the original
# never applied, and would not be gs12. 1.0 reproduces it.
#
# ---------------------------------------------------------------------------------------
# WHAT THE gs13-gs15 DETOUR ESTABLISHED, SO IT IS NOT REPEATED
#
# Sharpness is set entirely by how many detail Gaussians survive per face, and that comes
# from the SEED (init_copies_per_face), not from the coverage rescue -- rescue-spawned rows
# are born into an already-converged solution, fail to earn opacity, and are culled. One
# cycle of the OOMed gs14 shows the split: rescue produced 483,463 rows, densification 258,
# and the population still fell.
#
#     run   floor  copies  thresh   detail/face   sigma_px
#     gs12  0.95   4       1.0      2.42          1.56
#     gs10  0.95   2       0.9      0.81          2.50
#     gs14  0.2    2       0.15     0.34          3.60
#     gs15  0.2    1       1.0      0.13          3.87
#
#     sigma_px = 2.283 * (detail/face)^-0.431      (fit on the two floor-0.95 runs)
#
# Reaching sigma 1.0 px needs 6.8 detail/face, i.e. ~11 seed copies and 17.1M face-based
# rows against a measured 32 GB budget of 4.8M -- short by 3.6x. Independently: the mesh's
# triangles are ~7 px on screen and would need to be ~1.73 px, i.e. 22.7M faces. Same order
# from two directions. More parameter tuning on the full mesh cannot close that.
#
# Coverage is NOT the constraint and never was. Measured with scripts/analyze_gap_coverage.py
# (the union test, which is the actual "no gaps" requirement), gs11/gs14/gs15 all sit at
# 99.998-100.000% at the rasterizer threshold. The "Gaussian area coverage" number that
# gs13 chased is sum(pi*sx*sy*alpha) -- blind to placement, weighted by opacity, and
# therefore a blur metric. Ignore it.
#
# So this run is a baseline, not an attempt to beat 1.56 px by tuning. The next real move
# is to cut the surface area the same budget has to cover.
# ---------------------------------------------------------------------------------------
#
# Usage (inside the drawer_splat env):
#   nohup bash scripts/run_table_gs16.sh > data/table/table_gs16.log 2>&1 &
#   tail -f data/table/table_gs16.log        # Ctrl-C leaves the run alive
#   bash scripts/run_table_gs16.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs16}"

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
