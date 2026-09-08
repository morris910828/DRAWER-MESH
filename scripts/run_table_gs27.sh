#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 27.
#
# THIS FILE NOW MATCHES WHAT WAS ACTUALLY RUN as table_gs27 (single flag from gs25:
# base_opacity_floor 0.95 -> 0.40, base_floor_corner_sigma left at 1.0). An earlier
# version of this file experimented with a different pair of values
# (base_opacity_floor 0.90 / corner_sigma 1.3) before that run happened -- that attempt
# was renamed to run_table_gs28.sh so its EXP name doesn't collide with (and overwrite)
# this one's real output at data/table/table_gs27/.
#
# RESULT (measured on the real export, 2026-09-05):
#   * face-based gaussians per face: median 1.0 -> 3.0 (q90 7.0). Detail recovered
#     hugely (889k -> 3,508,649) from JUST the opacity drop, size untouched. This
#     isolates opacity, not size, as the bigger lever for keeping detail alive.
#   * gap coverage @ 0.01: 99.999% (0.001% short of the 100.000% hard requirement --
#     far better than gs26's 99.994%, which came from also shrinking the plate).
#   * screen overlap 3.19x (gs25) -> 2.19x. Real de-haze, though less than gs26's 1.08x
#     (which additionally shrank the plate).
#   * base layer still renders at alpha ~0.40 -- full-size, dim. Expect the gs26
#     regression (opaque surfaces like the monitor rendering see-through) to still be
#     present here, since it was an opacity effect, not a size effect.
#
# See run_table_gs28.sh for the follow-up: raise base_opacity_floor from this 0.40
# partway back up, aiming to keep most of the detail recovery while making opaque
# surfaces solid again and closing the last 0.001% coverage gap.
#
# Usage (inside the drawer_splat env):
#   bash scripts/run_table_gs27.sh                # train + export
#   bash scripts/run_table_gs27.sh --export-only  # export from the last checkpoint only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs27}"

# SDF_EXP must match the suffix used by run_table_sdf.sh / run_table_post.sh.
SDF_EXP="${SDF_EXP:-table_sdf_ds1}"
MESH="${DATA_DIR}/${SDF_EXP}/texture_mesh/mesh_table-only-400k_world.obj"
AREA=1e-4

LOG_DIR="${DATA_DIR}/gs_logs"
LOG="${LOG_DIR}/${EXP}_$(date +%Y%m%d_%H%M%S).log"
EXTRA_INFO="${DATA_DIR}/gs_extra_info/${EXP}.pt"
CKPT_DIR="${DATA_DIR}/${EXP}/nerfstudio_models"
mkdir -p "${LOG_DIR}" "${DATA_DIR}/gs_extra_info"

# ── Pre-flight: masks ──────────────────────────────────────────────────────────────
MASK_DIR="${DATA_DIR}/mask"
MASK_SRC="${MASK_SRC:-${DATA_DIR}/gs_masks/masks}"
if [[ ! -d "${MASK_DIR}" ]]; then
    [[ -d "${MASK_SRC}" ]] || { echo "ERROR: no masks at ${MASK_DIR} or ${MASK_SRC}"; exit 1; }
    ln -sfn "${MASK_SRC}" "${MASK_DIR}"
    echo "[pre-flight] linked ${MASK_DIR} -> ${MASK_SRC}"
fi
N_MASK=$(find -L "${MASK_DIR}" -name '*.png' | wc -l)
N_IMG=$(find "${DATA_DIR}/images" -name '*.JPG' | wc -l)
[[ "${N_MASK}" -eq "${N_IMG}" ]] || { echo "ERROR: ${N_MASK} masks vs ${N_IMG} images"; exit 1; }

cd "${SPLAT_DIR}"

echo "[pre-flight] dry-running the dataparser..."
PYTHONPATH="${SPLAT_DIR}" python - "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path
from nerfstudio.data.dataparsers.panoptic_dataparser import PanopticDataParserConfig
cfg = PanopticDataParserConfig(data=Path(sys.argv[1]), downscale_factor=1,
                               mesh_gauss_path=None, orientation_method="none",
                               center_poses=False, auto_scale_poses=False, num_max_image=50)
out = cfg.setup().get_dataparser_outputs(split="train")
n = 0 if out.mask_filenames is None else len(out.mask_filenames)
print(f"[pre-flight] dataparser resolved {n} train mask(s)")
if n == 0:
    sys.exit("ERROR: would train UNMASKED -- aborting.")
PY

LOAD_ARG=()
if compgen -G "${CKPT_DIR}/step-*.ckpt" > /dev/null; then
    echo "[pre-flight] resuming from $(ls -1 "${CKPT_DIR}"/step-*.ckpt | tail -1)"
    LOAD_ARG=(--load-dir "${CKPT_DIR}")
fi

if [[ "${1:-}" != "--export-only" ]]; then
PYTHONPATH="${SPLAT_DIR}" python nerfstudio/scripts/train.py splatfacto_on_mesh_uc \
    --vis tensorboard \
    --output-dir "${DATA_DIR}" \
    --experiment-name "${EXP}" \
    --max-num-iterations 30000 \
    "${LOAD_ARG[@]}" \
    --steps-per-save 10000 \
    --steps-per-eval-image 500 \
    --steps-per-eval-all-images 30000 \
    --pipeline.model.max_gaussians 0 \
    --pipeline.model.vertex_radius_scale 0.5 \
    --pipeline.model.anisotropic_vertex_floor True \
    --pipeline.model.vertex_floor_max_aspect 8.0 \
    --pipeline.model.anisotropic_coverage_density True \
    --pipeline.model.max_gauss_ratio 5.0 \
    --pipeline.model.coverage_rescue_thresh 0.5 \
    --pipeline.model.coverage_rescue_max_frac 0.02 \
    --pipeline.model.coverage_rescue_max_count 0 \
    --pipeline.model.coverage_rescue_every 100 \
    --pipeline.model.anisotropic_base_floor True \
    --pipeline.model.scale_reg_exclude_base True \
    --pipeline.model.base_floor_corner_sigma 1.0 \
    --pipeline.model.base_floor_max_aspect 8.0 \
    --pipeline.model.base_floor_reaches_corners False \
    --pipeline.model.base_opacity_floor 0.40 \
    --pipeline.model.coverage_floor_target 1.0 \
    --pipeline.model.coverage_target 1.0 \
    --pipeline.model.coverage_lambda 0.0 \
    --pipeline.model.coverage_densify_scale 0.0 \
    --pipeline.model.adaptive_coverage_floor True \
    --pipeline.model.init_copies_per_face 4 \
    --pipeline.model.copies_scale_shrink_power 0.3 \
    --pipeline.model.min_scale_frac 0.02 \
    --pipeline.model.min_vertex_scale_frac 0.15 \
    --pipeline.model.upper_scale 0.5 \
    --pipeline.model.use_base_layer True \
    --pipeline.model.cull_at_floor_gaussians False \
    --pipeline.model.cull_screen_size_min 0.0 \
    --pipeline.model.opacity_reg_lambda 0.0 \
    --pipeline.model.opacity_reset_value 0.10 \
    --pipeline.model.opacity_reset_min_value 0.075 \
    --pipeline.model.elevate_coef 0.5 \
    --pipeline.model.detail_elevate_min_frac=-1.0 \
    --pipeline.model.gaussian_save_extra_info_path "${EXTRA_INFO}" \
    --pipeline.model.mesh_area_to_subdivide ${AREA} \
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
    panoptic-data \
    --data "${DATA_DIR}" \
    --mesh_gauss_path "${MESH}" \
    --mesh_area_to_subdivide ${AREA} \
    --mesh_depth False \
    --downscale_factor 1 \
    --num_max_image 50 \
    --orientation_method none \
    --center_poses False \
    --auto_scale_poses False \
    2>&1 | tee -a "${LOG}"
fi

PYTHONPATH="${SPLAT_DIR}" python nerfstudio/scripts/exporter.py gaussian-splat \
    --load-config "${DATA_DIR}/${EXP}/config.yml" \
    --output-dir "${DATA_DIR}/${EXP}/export_ply" 2>&1 | tee -a "${LOG}"

echo "=== done -> ${DATA_DIR}/${EXP}/export_ply  (log: ${LOG}) ==="
