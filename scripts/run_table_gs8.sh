#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 8.
#
# Baseline is table_gs6's config.yml, with the adaptive-coverage-floor changes from
# adaptive_coverage_floor.patch turned on. Every line that differs from table_gs6 is
# marked CHANGED or NEW below, with the old value, so the run is diffable by eye.
#
# Requires adaptive_coverage_floor.patch to be applied first:
#   cd /workspace/DRAWER-MESH && git apply -v adaptive_coverage_floor.patch
#
# Usage (on the DGX, inside the drawer_splat env):
#   bash scripts/run_table_gs8.sh
#   bash scripts/run_table_gs8.sh --export-only    # skip training, just export the ply

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs8}"

MESH="${DATA_DIR}/table_sdf_ds1/texture_mesh/mesh_table-only-400k_world.obj"
AREA=1e-4

# Per-run extra_info path. table_gs6 left this at null, which resolves to
# <mesh_dir>/gaussian_on_mesh_extra_info.pt -- a path keyed to the MESH, not to the run.
# Any later run (or mesh re-extraction) writing that same file silently invalidates every
# earlier checkpoint built on it, which is exactly what broke the table_gs6 export:
#   RuntimeError: The size of tensor a (2863560) must match the size of tensor b (4119740)
# Giving each run its own file makes checkpoints and their face bindings inseparable.
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
    `# ---- NEW: adaptive coverage floor (patch block 1) --------------------------` \
    --pipeline.model.adaptive_coverage_floor True \
    --pipeline.model.coverage_floor_target 0.9 \
    \
    `# ---- NEW: per-cycle coverage repair (patch block 2) ------------------------` \
    --pipeline.model.coverage_rescue_every 100 \
    --pipeline.model.coverage_rescue_max_frac 0.002 \
    \
    `# ---- NEW: opacity reset now configurable (patch block 3) -------------------` \
    `#      was hardcoded [0.5, 0.8]; these are vanilla splatfacto's values.       ` \
    --pipeline.model.opacity_reset_value 0.10 \
    --pipeline.model.opacity_reset_min_value 0.075 \
    \
    `# ---- CHANGED from table_gs6 ------------------------------------------------` \
    --pipeline.model.coverage_rescue_thresh 0.9 `# was 0.2  -- now the primary coverage mechanism, not a rare backstop` \
    --pipeline.model.coverage_lambda 0.0        `# was 1.0  -- soft loss; its only way to raise coverage is sigma+ and alpha+, on EVERY gaussian` \
    --pipeline.model.coverage_densify_scale 0.0 `# was 5.0  -- inert once coverage_lambda is 0; zeroed for clarity` \
    --pipeline.model.cull_at_floor_gaussians False `# was True -- deleted exactly the sub-pixel gaussians that draw text` \
    --pipeline.model.opacity_reg_lambda 0.0     `# was 0.05 -- binarised alpha; anti-aliased text needs partial alpha` \
    --pipeline.model.base_opacity_floor 0.95    `# was 0.5  -- now only applies to floored faces, so a real guarantee is cheap` \
    --pipeline.model.elevate_coef 0.5           `# was 3.0  -- +-3 face radii off-surface was the white fringe on the sign edge` \
    --pipeline.model.gaussian_save_extra_info_path "${EXTRA_INFO}" `# was null -- see note above` \
    \
    `# ---- unchanged from table_gs6 ----------------------------------------------` \
    --pipeline.model.mesh_area_to_subdivide ${AREA} \
    --pipeline.model.upper_scale 0.5 \
    --pipeline.model.min_scale_frac 0.05 \
    --pipeline.model.min_vertex_scale_frac 0.15 \
    --pipeline.model.use_base_layer True \
    --pipeline.model.init_copies_per_face 2 \
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
    --pipeline.model.max_gauss_ratio 5.0 \
    --pipeline.model.face_flat_coef 0.05 \
    --pipeline.model.unconstrained_scale True \
    --pipeline.model.unconstrained_elevate True \
    --pipeline.model.enable_fold_detection False \
    --pipeline.model.acm_lambda 0.0 \
    --pipeline.model.stop_acm_after_split True \
    --pipeline.model.mesh_depth_lambda 0.0 \
    `# freeze_geometry_at_step left at its -1 default: tyro can read a bare -1 as a flag.` \
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

# ---- export ---------------------------------------------------------------------
PYTHONPATH="${SPLAT_DIR}" python nerfstudio/scripts/exporter.py gaussian-splat \
    --load-config "${DATA_DIR}/${EXP}/config.yml" \
    --output-dir "${DATA_DIR}/${EXP}/export_ply"

echo "=== done -> ${DATA_DIR}/${EXP}/export_ply ==="
