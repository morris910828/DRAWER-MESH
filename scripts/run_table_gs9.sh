#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 9.
#
# Baseline is run_table_gs8.sh. Changes target the two symptoms reported after gs8:
# coverage still short in places, and Gaussians rendering as circles rather than ellipses.
# Every line that differs from gs8 is marked, with gs8's value.
#
# Requires adaptive_coverage_floor.patch (19 hunks) to be applied first:
#   cd /workspace/DRAWER-MESH && git apply -v adaptive_coverage_floor.patch
#
# Usage (on the DGX, inside the drawer_splat env):
#   bash scripts/run_table_gs9.sh
#   bash scripts/run_table_gs9.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs9}"

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
    `# ==== COVERAGE ============================================================` \
    `# Bug fix, not a tuning knob: base floor was min(cv_radius, xyz_radius) and    ` \
    `# xyz_radius is already divided by init_copies_per_face**copies_scale_shrink_  ` \
    `# power, so the min() always won and every base plate sat at 0.812 x cv_radius ` \
    `# (measured: all 1,497,524 of them, none above). It reached its own farthest   ` \
    `# corner at 1.30 sigma = 0.43 of peak. With this on: 1.0 x cv_radius, farthest ` \
    `# corner at 1.06 sigma = 0.57 of peak. +33% where the gaps actually are.       ` \
    --pipeline.model.base_floor_reaches_corners True `# NEW` \
    \
    `# The release bar. gs8 dropped a face's floor at 0.9; the metric is a SUM over ` \
    `# contributing Gaussians, so 0.9 means "just barely, by one Gaussian" -- no    ` \
    `# margin for the released plate shrinking afterwards. 1.5 demands redundancy.  ` \
    --pipeline.model.coverage_floor_target 1.5 `# gs8: 0.9` \
    \
    `# Let repair keep up with damage. Raise further if the log shows 'qualified'   ` \
    `# stuck at the cap every cycle.                                                ` \
    --pipeline.model.coverage_rescue_max_frac 0.01 `# gs8: 0.002` \
    \
    `# More detail Gaussians per face from the start, so a face can actually earn   ` \
    `# its floor release. Only affects DETAIL copies now: with                      ` \
    `# base_floor_reaches_corners the base plate no longer keys off the             ` \
    `# copies_scale_shrink_power divisor, so raising N no longer weakens coverage.  ` \
    `# Costs ~2.8M extra seed Gaussians -- drop back to 2 if VRAM is tight.         ` \
    --pipeline.model.init_copies_per_face 4 `# gs8: 2` \
    \
    `# ==== ELLIPTICITY =========================================================` \
    `# Aspect ratio is capped at upper_scale / min_scale_frac because both bounds  ` \
    `# are per-axis and isotropic. gs6 measurement: detail rows' p90 AND p99 sit at ` \
    `# exactly 10.000 = 0.5 / 0.05, i.e. >10% are clamped by this and not by any    ` \
    `# optimiser preference. 0.02 raises the cap to 25:1.                           ` \
    --pipeline.model.min_scale_frac 0.02 `# gs8: 0.05` \
    \
    `# use_scale_regularization penalises aspect > max_gauss_ratio with weight 1.0. ` \
    `# Detail median is already 2.47, so at 5.0 the reg was actively pushing a large ` \
    `# fraction of the detail layer back toward round. Note the needle cull follows  ` \
    `# it at max_gauss_ratio * 4, so this moves that from 20:1 to 40:1.              ` \
    --pipeline.model.max_gauss_ratio 10.0 `# gs8: 5.0` \
    \
    `# ==== unchanged from gs8 ==================================================` \
    --pipeline.model.adaptive_coverage_floor True \
    --pipeline.model.coverage_rescue_every 100 \
    --pipeline.model.coverage_rescue_thresh 0.9 \
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
