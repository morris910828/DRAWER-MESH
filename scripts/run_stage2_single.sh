#!/usr/bin/env bash
# Stage-2 Gaussian Splatting for a single dataset.
# Depends on Stage-1 outputs: <DATA_DIR>/<DATA_NAME>_sdf_recon/texture_mesh/mesh-clean-simplify.obj
#
# Usage:
#   bash scripts/run_stage2_single.sh <DATA_DIR>
#
# Examples:
#   bash scripts/run_stage2_single.sh /opt/disk/drawer_dataset/single_joint/V0001/frame_0000
#   bash scripts/run_stage2_single.sh /opt/disk/drawer_dataset/studio/home
#
# Steps:
#   2b. GS training           → <DATA_NAME>_gs_masked/
#   2c. export PLY            → export_ply/splat.ply
#   2d. gs_to_world           → export_ply/splat_world.ply (with SH rotation)
#   2e. save mesh_depth masks → gs_masks/

set -euo pipefail

# ── args ─────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <DATA_DIR>"
    echo "  DATA_DIR  absolute path to the dataset (Stage-1 must be complete)"
    exit 1
fi

DATA_DIR="$(realpath "$1")"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: DATA_DIR not found: ${DATA_DIR}"
    exit 1
fi

DATA_NAME="$(basename "${DATA_DIR}")"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &>/dev/null && pwd)"
SPLAT_DIR="${REPO_ROOT}/splat"
SDF_OUT_DIR="${DATA_DIR}/${DATA_NAME}_sdf_recon"
GS_EXP="${DATA_NAME}_gs_masked"
GS_OUT_DIR="${DATA_DIR}/${GS_EXP}"
MESH_OBJ="${SDF_OUT_DIR}/texture_mesh/mesh-clean-simplify.obj"
TIMING_FILE="${GS_OUT_DIR}/timing.json"

# ── conda ─────────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawer_splat

# ── helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

time_step() {
    local name="$1"; shift
    local start_ts; start_ts=$(date -u '+%Y-%m-%dT%H:%M:%S')
    local start_s; start_s=$(date +%s)
    "$@"
    local elapsed=$(( $(date +%s) - start_s ))
    python "${REPO_ROOT}/tools/record_timing.py" \
        --out "${TIMING_FILE}" --name "${name}" --start "${start_ts}" --sec "${elapsed}"
}

# ── pre-check ─────────────────────────────────────────────────────────────────
log "========================================================"
log "Dataset : ${DATA_NAME}"
log "DATA_DIR: ${DATA_DIR}"
log "GS OUT  : ${GS_OUT_DIR}"
log "========================================================"

if [[ ! -f "${MESH_OBJ}" ]]; then
    echo "ERROR: texture mesh not found: ${MESH_OBJ}"
    echo "       Run run_stage1_single.sh first."
    exit 1
fi

mkdir -p "${GS_OUT_DIR}"

# ── 2b. GS training ───────────────────────────────────────────────────────────
log "Step 2b: GS training → ${GS_OUT_DIR}"
GS_CKPT_DIR="${GS_OUT_DIR}/nerfstudio_models"
GS_LOAD_ARG=()
if [[ -d "${GS_CKPT_DIR}" ]] && compgen -G "${GS_CKPT_DIR}/step-*.ckpt" > /dev/null; then
    log "  Checkpoint found in ${GS_CKPT_DIR}, will resume"
    GS_LOAD_ARG=(--trainer.load-dir "${GS_CKPT_DIR}")
fi
cd "${SPLAT_DIR}"
time_step step_2b_gs_train python nerfstudio/scripts/train.py splatfacto_on_mesh_uc \
    --vis tensorboard \
    --output-dir "${DATA_DIR}" \
    --experiment-name "${GS_EXP}" \
    --pipeline.model.mesh_area_to_subdivide 2e-5 \
    --pipeline.model.acm_lambda 1.0 \
    --pipeline.model.elevate_coef 1.5 \
    --pipeline.model.upper_scale 1.5 \
    --pipeline.model.continue_cull_post_densification True \
    --pipeline.model.mesh_depth_lambda 1.0 \
    --pipeline.model.reset_alpha_every 30 \
    --pipeline.model.use_scale_regularization True \
    --pipeline.model.max_gauss_ratio 1.5 \
    --max-num-iterations 30000 \
    "${GS_LOAD_ARG[@]}" \
    panoptic-data \
    --data "${DATA_DIR}" \
    --mesh_gauss_path "${MESH_OBJ}" \
    --mesh_area_to_subdivide 2e-5 \
    --mesh_depth True \
    --downscale_factor 1 \
    --num_max_image 2000

# ── 2c. export PLY ────────────────────────────────────────────────────────────
EXPORT_DIR="${GS_OUT_DIR}/export_ply"
if [[ ! -f "${EXPORT_DIR}/splat.ply" ]]; then
    log "Step 2c: export GS → ${EXPORT_DIR}/splat.ply"
    mkdir -p "${EXPORT_DIR}"
    CONFIG_PATH=$(find "${GS_OUT_DIR}" -name "config.yml" | sort | tail -1)
    cd "${SPLAT_DIR}"
    time_step step_2c_export_ply python nerfstudio/scripts/exporter.py gaussian-splat \
        --load-config "${CONFIG_PATH}" \
        --output-dir  "${EXPORT_DIR}"
else
    log "Step 2c: splat.ply already exists, skipping"
fi

# ── 2d. convert GS → COLMAP world coordinates ────────────────────────────────
DATAPARSER_TRANSFORM=$(find "${GS_OUT_DIR}" -name "dataparser_transforms.json" | sort | tail -1)
if [[ ! -f "${EXPORT_DIR}/splat_world.ply" ]]; then
    log "Step 2d: convert GS → COLMAP world coordinates (with SH rotation)"
    if [[ -z "${DATAPARSER_TRANSFORM}" ]]; then
        log "WARNING: dataparser_transforms.json not found, skipping world conversion"
    else
        python "${REPO_ROOT}/scripts/gs_to_world.py" \
            --input     "${EXPORT_DIR}/splat.ply" \
            --transform "${DATAPARSER_TRANSFORM}" \
            --output    "${EXPORT_DIR}/splat_world.ply" \
            --timing    "${TIMING_FILE}"
    fi
else
    log "Step 2d: splat_world.ply already exists, skipping"
fi

# ── 2e. save mesh_depth masks ─────────────────────────────────────────────────
MASKS_DIR="${DATA_DIR}/gs_masks"
if [[ ! -d "${MASKS_DIR}/masks" ]] && [[ -n "${DATAPARSER_TRANSFORM}" ]]; then
    log "Step 2e: save mesh_depth masks → ${MASKS_DIR}"
    python "${REPO_ROOT}/scripts/save_mesh_depth_masks.py" \
        --data_dir      "${DATA_DIR}" \
        --mesh_path     "${MESH_OBJ}" \
        --dataparser_tf "${DATAPARSER_TRANSFORM}" \
        --out_dir       "${MASKS_DIR}" \
        --timing        "${TIMING_FILE}"
else
    log "Step 2e: gs_masks/ already exists or dataparser_transforms.json not found, skipping"
fi

log "========================================================"
log "=== ${DATA_NAME} complete ==="
log "========================================================"
