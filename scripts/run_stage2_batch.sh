#!/usr/bin/env bash
# Stage-2 batch: Gaussian Splatting on mesh for all joint* and home datasets.
# Depends on Stage-1 outputs: texture_mesh/mesh-clean-simplify.obj
# Runs sequentially to avoid GPU contention.
#
# Steps per dataset:
#   2a. GS training           → <DATA_NAME>_gs_masked/
#       mask derived on-the-fly from mesh_depth (mesh_depth > 0 = foreground)
#   2b. export PLY            → export_ply/splat.ply
#   2c. gs_to_world.py        → export_ply/splat_world.ply (COLMAP coords, with SH rotation)
#   2d. save mesh_depth masks → gs_masks/ (visualization / downstream)
#
# Usage:
#   bash scripts/run_stage2_batch.sh
#   DATA_ROOT=/opt/disk/drawer_dataset/autourdf_studio/V0000 bash scripts/run_stage2_batch.sh

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/opt/disk/drawer_dataset/studio}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &>/dev/null && pwd)"
SPLAT_DIR="${REPO_ROOT}/splat"

# studio datasets
DATASETS=(
    home
    joint1_-180
    joint2_-50
    # joint1_-90
    # joint2_-90
    # joint3_-90
    # joint4_-90
    # joint5_60
    # joint6_90
)

# autourdf_studio/V0000 datasets
# Usage: DATA_ROOT=/opt/disk/drawer_dataset/autourdf_studio/V0000 bash scripts/run_stage2_batch.sh
# DATASETS=(
#     frame_0000
#     frame_0001
#     frame_0002
#     frame_0003
#     frame_0004
#     frame_0005
#     frame_0006
# )

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawer_splat

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Record elapsed time for a shell-level step.
# Usage: time_step <step_name> <cmd...>
# Requires TIMING_FILE to be set in the calling scope.
time_step() {
    local name="$1"; shift
    local start_ts; start_ts=$(date -u '+%Y-%m-%dT%H:%M:%S')
    local start_s; start_s=$(date +%s)
    "$@"
    local elapsed=$(( $(date +%s) - start_s ))
    python "${REPO_ROOT}/tools/record_timing.py" \
        --out "${TIMING_FILE}" --name "${name}" --start "${start_ts}" --sec "${elapsed}"
}

for DATA_NAME in "${DATASETS[@]}"; do
    DATA_DIR="${DATA_ROOT}/${DATA_NAME}"
    SDF_OUT_DIR="${DATA_DIR}/${DATA_NAME}_sdf_recon"
    GS_EXP="${DATA_NAME}_gs_masked"
    GS_OUT_DIR="${DATA_DIR}/${GS_EXP}"
    MESH_OBJ="${SDF_OUT_DIR}/texture_mesh/mesh-clean-simplify.obj"
    TIMING_FILE="${GS_OUT_DIR}/timing.json"

    log "========================================================"
    log "Dataset: ${DATA_NAME}"
    log "========================================================"

    # ── pre-check ────────────────────────────────────────────────────────────
    if [[ ! -f "${MESH_OBJ}" ]]; then
        log "SKIP: texture mesh not found: ${MESH_OBJ}"
        log "      (run run_stage1_batch.sh first)"
        continue
    fi

    # ── 2a. (選用) 預計算 mesh silhouette masks ────────────────────────────────
    # render_mesh_masks.py 對每個訓練相機渲染 mesh 深度遮罩，儲存至
    # <DATA_DIR>/masks/ 並更新 transforms.json 的 mask_path 欄位。
    # 目前已改為在 loss 計算時直接從 mesh_depth（dataparser 初始化時由
    # nvdiffrast 預計算）衍生遮罩，不需要額外的 PNG 檔案，故此步驟停用。
    #
    # if [[ ! -d "${DATA_DIR}/masks" ]]; then
    #     log "Step 2a: render mesh masks → ${DATA_DIR}/masks/"
    #     conda run -n drawer_splat python "${REPO_ROOT}/sdf/scripts/render_mesh_masks.py" \
    #         --data_dir  "${DATA_DIR}" \
    #         --mesh_path "${MESH_OBJ}" \
    #         --method    depth \
    #         --dilate    10 \
    #         --scale     0.5
    # else
    #     log "Step 2a: masks/ already exists, skipping"
    # fi

    # ── 2b. GS training ───────────────────────────────────────────────────────
    if [[ ! -f "${GS_OUT_DIR}/config.yml" ]]; then
        log "Step 2b: GS training → ${GS_OUT_DIR}"
        mkdir -p "${GS_OUT_DIR}"
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
            panoptic-data \
            --data "${DATA_DIR}" \
            --mesh_gauss_path "${MESH_OBJ}" \
            --mesh_area_to_subdivide 2e-5 \
            --mesh_depth True \
            --downscale_factor 1 \
            --num_max_image 2000
    else
        log "Step 2b: config.yml already exists, skipping GS training"
    fi

    # ── 2c. export PLY ────────────────────────────────────────────────────────
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
        log "Step 2c: splat.ply already exists, skipping export"
    fi

    # ── 2d. convert GS to COLMAP world coordinates ────────────────────────────
    # gs_to_world.py transforms: xyz, quaternions, log-scales, AND SH degree 1-3
    # (f_rest_* rotated via Wigner D-matrices — omitting this causes fuzzy edges
    #  in world-space viewers because view-dependent color is in training frame).
    # --no-sh-rotation flag exists but must NOT be used.
    DATAPARSER_TRANSFORM=$(find "${GS_OUT_DIR}" -name "dataparser_transforms.json" | sort | tail -1)
    if [[ ! -f "${EXPORT_DIR}/splat_world.ply" ]]; then
        log "Step 2d: convert GS → COLMAP world coordinates (with SH rotation)"
        if [[ -z "${DATAPARSER_TRANSFORM}" ]]; then
            log "WARNING: dataparser_transforms.json not found, skipping world conversion"
        else
            conda run -n drawer_splat python "${REPO_ROOT}/scripts/gs_to_world.py" \
                --input     "${EXPORT_DIR}/splat.ply" \
                --transform "${DATAPARSER_TRANSFORM}" \
                --output    "${EXPORT_DIR}/splat_world.ply" \
                --timing    "${TIMING_FILE}"
        fi
    else
        log "Step 2d: splat_world.ply already exists, skipping"
    fi

    # ── 2e. save mesh_depth masks (for visualization & downstream use) ────────
    # mask = mesh-clean-simplify.obj (GS training space) + GS-normalized cameras + no flip
    # 與 panoptic_dataparser 訓練時 mesh_depth > 0 的計算方式完全一致
    MASKS_DIR="${DATA_DIR}/gs_masks"
    if [[ ! -d "${MASKS_DIR}/masks" ]] && [[ -n "${DATAPARSER_TRANSFORM}" ]]; then
        log "Step 2e: save mesh_depth masks → ${MASKS_DIR}"
        conda run -n drawer_splat python "${REPO_ROOT}/scripts/save_mesh_depth_masks.py" \
            --data_dir      "${DATA_DIR}" \
            --mesh_path     "${MESH_OBJ}" \
            --dataparser_tf "${DATAPARSER_TRANSFORM}" \
            --out_dir       "${MASKS_DIR}" \
            --timing        "${TIMING_FILE}"
    else
        log "Step 2e: gs_masks/ already exists or dataparser_transforms.json not found, skipping"
    fi

    log "=== ${DATA_NAME} complete ==="
done

log "========================================================"
log "All datasets finished."
log "========================================================"
