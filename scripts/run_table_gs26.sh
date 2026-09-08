#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 26.
#
# gs25's recipe, with one intervention (two coupled flags) on the BASE PLATE.
#
# WHY: gs25 was the first genuinely-masked run with honest (uncapped) refinement.
# Analysis of its export (analyze_layer_smear / analyze_gap_coverage):
#   * base plate paints 83.6% of the screen at sigma 4.17 px, alpha 0.950
#   * detail layer collapsed 3.65M -> 0.89M; median face ends with ONE face-based
#     Gaussian (just the plate). gs20 only "had" more detail because its 5.5M ceiling
#     froze refinement, leaving dead seed rows uncounted-as-culled.
#   * gap coverage still 100.000% -- held ENTIRELY by the opaque plates.
# Mechanism (traced in splatfacto_on_mesh_uc.py): base_opacity_floor 0.95 -> plate is
# opaque -> shields the 3 detail Gaussians on its face from photometric gradient ->
# opacity reset every 3000 steps slams them to ~0.09 -> they can't climb back -> culled
# (base is never culled: `culls & ~is_base`) -> face has only its plate -> adaptive
# floor measures the face as bare without the plate -> keeps the plate floored -> loop.
# coverage_floor_target 1.0 -> 0.7 alone (the old plan) cannot break this: by the time
# the release check runs, the detail that would trigger it is already gone.
#
# THE CHANGE (both flags are "shrink+dim the plate so it stops being a shield"):
#   base_opacity_floor    0.95 -> 0.40   detail behind the plate now gets real gradient
#   base_floor_corner_sigma 1.0 -> 2.0   floored plates reach their corners at 2 sigma
#                                        instead of 1 -> every floored plate ~2x smaller
#                                        in-plane (4.17 px -> ~2.1 px)
# Nothing else changes from gs25.
#
# PASS/FAIL (analyze after export -- BOTH required, per the project's hard constraints):
#   1. analyze_gap_coverage @ 0.01 threshold == 100.000%  (@ 0.10 >= 99.9%)
#      if not: base_opacity_floor was too low -> gs27 at 0.6, and/or corner_sigma 1.5
#   2. analyze_layer_smear: base smear% < 35%, detail count > 2M, median face-based/face >= 2
#   3. inside-mask PSNR (compare_masked_psnr) beats gs20's ~20.8 dB
#
# NOTE: mesh geometry is a separate ceiling (table_gs_experiments.md 4.8 -- table top
# ~5 mm off, legs badly misaligned). This fix targets the table top / objects / text;
# the legs stay bad until Stage 1 (SDF) is redone.
#
# Usage (inside the drawer_splat env):
#   bash scripts/run_table_gs26.sh                # train + export
#   bash scripts/run_table_gs26.sh --export-only  # export from the last checkpoint only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs26}"

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
# At downscale 1 the dataparser reads each frame's mask_path (./mask/<name>) as
# <data>/mask/<name>. If mask/ is missing, link it to the full-res set.
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

# The count check above does not prove the dataparser will pick the masks up; run it.
# panoptic_dataparser raises if any mask_path fails to resolve, so this fails loud.
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

# ── Resume from the newest checkpoint if this run was interrupted ───────────────────
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
    --pipeline.model.base_floor_corner_sigma 2.0 \
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
