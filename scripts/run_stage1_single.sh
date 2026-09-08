#!/usr/bin/env bash
# Stage-1 SDF reconstruction for a single dataset.
#
# Usage:
#   bash scripts/run_stage1_single.sh <DATA_DIR> [DOWNSCALE_FACTOR]
#
# Examples:
#   bash scripts/run_stage1_single.sh /opt/disk/drawer_dataset/single_joint/V0001/frame_0000
#   bash scripts/run_stage1_single.sh /opt/disk/drawer_dataset/studio/home 2
#
# Steps:
#   0.  colmap_to_drawer  → transforms.json  (skipped if already exists)
#   1a. Marigold depth + normal              (skipped if already exists)
#   1b. BakedSDF training                    (skipped if config.yml exists)
#   1c. extract mesh (raw, --simplify False) (skipped if mesh.ply exists)
#   1d. filter interior mesh                 (skipped if mesh-clean.ply exists)
#   1e. simplify cleaned mesh                (skipped if mesh-clean-simplify.ply exists)
#   1f. ply_to_world (all 3 PLY → world)     (skipped if world files exist)
#   1g. texture baking on mesh-clean-simplify (skipped if .obj exists)
#   1h. mesh_to_world (OBJ → world)          (skipped if world.obj exists)
#   1i. save_pose                            (skipped if pose.pkl exists)

set -euo pipefail

# ── args ─────────────────────────────────────────────────────────────────────
# if [[ $# -lt 1 ]]; then
#     echo "Usage: bash $0 <DATA_DIR> [DOWNSCALE_FACTOR]"
#     echo "  DATA_DIR         absolute path to the dataset (must contain images/)"
#     echo "  DOWNSCALE_FACTOR integer ≥1, default 1"
#     exit 1
# fi

# DATA_DIR="$(realpath "$1")"
# DOWNSCALE_FACTOR="${2:-1}"
DATA_DIR="/workspace/DRAWER-MESH/data/table2"
DOWNSCALE_FACTOR=4
# panoptic scale_factor: >1 scales the object up into the [-1,1] linear foreground
# (finer mesh) at the cost of clipping the scene periphery. Default 1.0 = original.
# The SAME value is passed to world-conversion (--extra-scale) so COLMAP coords stay correct.
SCALE_FACTOR="${SCALE_FACTOR:-1.0}"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: DATA_DIR not found: ${DATA_DIR}"
    exit 1
fi

DATA_NAME="$(basename "${DATA_DIR}")"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &>/dev/null && pwd)"
SDF_OUT_DIR="${DATA_DIR}/${DATA_NAME}_sdf_recon"
TIMING_FILE="${SDF_OUT_DIR}/timing.json"

# ── conda ─────────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawer_sdf

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

# ── start ─────────────────────────────────────────────────────────────────────
log "========================================================"
log "Dataset : ${DATA_NAME}"
log "DATA_DIR: ${DATA_DIR}"
log "OUT_DIR : ${SDF_OUT_DIR}"
log "========================================================"

mkdir -p "${SDF_OUT_DIR}"

# ── 0. colmap_to_drawer ───────────────────────────────────────────────────────
NEEDS_COLMAP=false
[[ ! -f "${DATA_DIR}/transforms.json" ]] && NEEDS_COLMAP=true
if [[ "${DOWNSCALE_FACTOR}" -gt 1 && ! -d "${DATA_DIR}/images_${DOWNSCALE_FACTOR}" ]]; then
    NEEDS_COLMAP=true
fi
if [[ "${NEEDS_COLMAP}" == "true" ]]; then
    log "Step 0: colmap_to_drawer → transforms.json (downscale=${DOWNSCALE_FACTOR})"
    python "${REPO_ROOT}/scripts/colmap_to_drawer.py" \
        --data_dir  "${DATA_DIR}" \
        --downscale "${DOWNSCALE_FACTOR}" \
        --timing    "${TIMING_FILE}"
else
    log "Step 0: transforms.json and images_${DOWNSCALE_FACTOR}/ already exist, skipping"
fi

# image dir for Marigold
if [[ "${DOWNSCALE_FACTOR}" -eq 1 ]]; then
    MARIGOLD_INPUT="${DATA_DIR}/images"
else
    MARIGOLD_INPUT="${DATA_DIR}/images_${DOWNSCALE_FACTOR}"
fi

# ── 1a. Marigold depth + normal ───────────────────────────────────────────────
if [[ ! -f "${DATA_DIR}/marigold_ft/depth/0001.npy" ]]; then
    log "Step 1a: Marigold depth"
    cd "${REPO_ROOT}/marigold"
    time_step step_1a_marigold_depth python run.py \
        --checkpoint "GonzaloMG/marigold-e2e-ft-depth" \
        --modality depth \
        --input_rgb_dir "${MARIGOLD_INPUT}" \
        --output_dir    "${DATA_DIR}/marigold_ft"

    log "Step 1a: Marigold normals"
    time_step step_1a_marigold_normals python run.py \
        --checkpoint "GonzaloMG/marigold-e2e-ft-normals" \
        --modality normals \
        --input_rgb_dir "${MARIGOLD_INPUT}" \
        --output_dir    "${DATA_DIR}/marigold_ft"

    log "Step 1a: read_marigold"
    time_step step_1a_read_marigold python read_marigold.py \
        --data_dir "${DATA_DIR}/marigold_ft"
else
    log "Step 1a: Marigold output already exists, skipping"
fi

ln -sfn "${DATA_DIR}/marigold_ft/depth"  "${DATA_DIR}/depth"
ln -sfn "${DATA_DIR}/marigold_ft/normal" "${DATA_DIR}/normal"

# ── 1b. BakedSDF training ─────────────────────────────────────────────────────
CKPT_DIR="${SDF_OUT_DIR}/sdfstudio_models"
# max-num-iterations=50001 → loop range(0,50001) → last step=50000
FINAL_CKPT="${CKPT_DIR}/step-00050000.ckpt"
if [[ -f "${FINAL_CKPT}" ]]; then
    log "Step 1b: training complete (step-00050000.ckpt exists), skipping"
else
    log "Step 1b: BakedSDF training → ${SDF_OUT_DIR}"
    export PYTHONHTTPSVERIFY=0   # allow model weight downloads behind corporate proxy/cert
    LOAD_DIR_ARG=()
    if [[ -d "${CKPT_DIR}" ]] && compgen -G "${CKPT_DIR}/step-*.ckpt" > /dev/null; then
        log "  Checkpoint found in ${CKPT_DIR}, will resume"
        LOAD_DIR_ARG=(--trainer.load-dir "${CKPT_DIR}")
    fi
    cd "${REPO_ROOT}/sdf"
    time_step step_1b_sdf_train python scripts/train.py bakedsdf --vis tensorboard \
        --output-dir "${DATA_DIR}" \
        --experiment-name "${DATA_NAME}_sdf_recon" \
        --trainer.steps-per-eval-image 2000 \
        --trainer.steps-per-eval-all-images 50001 \
        --trainer.max-num-iterations 50001 \
        --trainer.steps-per-eval-batch 50001 \
        --trainer.gradient-clipping-val 1.0 \
        --trainer.save-best-checkpoint True \
        --trainer.save-only-latest-checkpoint True \
        --optimizers.fields.optimizer.eps 1e-8 \
        --optimizers.field-background.optimizer.eps 1e-8 \
        --optimizers.proposal-networks.optimizer.eps 1e-8 \
        --optimizers.fields.scheduler.max-steps 50000 \
        --optimizers.field-background.scheduler.max-steps 50000 \
        --optimizers.proposal-networks.scheduler.max-steps 50000 \
        --pipeline.model.eikonal-anneal-max-num-iters 50000 \
        --pipeline.model.beta-anneal-max-num-iters 50000 \
        --pipeline.model.sdf-field.bias 1.5 \
        --pipeline.model.sdf-field.inside-outside True \
        --pipeline.model.eikonal-loss-mult 0.01 \
        --pipeline.model.num-neus-samples-per-ray 24 \
        --pipeline.datamanager.train-num-rays-per-batch 4096 \
        --machine.num-gpus 1 \
        --pipeline.model.scene-contraction-norm inf \
        --pipeline.model.mono-normal-loss-mult 0.2 \
        --pipeline.model.mono-depth-loss-mult 1.0 \
        --pipeline.model.near-plane 1e-6 \
        --pipeline.model.far-plane 100 \
        "${LOAD_DIR_ARG[@]}" \
        panoptic-data \
        --data "${DATA_DIR}" \
        --panoptic_data False \
        --mono_normal_data True \
        --mono_depth_data True \
        --panoptic_segment False \
        --downscale_factor "${DOWNSCALE_FACTOR}" \
        --num_max_image 2000 \
        --scale_factor "${SCALE_FACTOR}"
fi

# ── 1c. extract mesh ──────────────────────────────────────────────────────────
if [[ ! -f "${SDF_OUT_DIR}/mesh.ply" ]]; then
    log "Step 1c: extract mesh"
    cd "${REPO_ROOT}/sdf"
    time_step step_1c_extract_mesh python scripts/extract_mesh.py \
        --load-config "${SDF_OUT_DIR}/config.yml" \
        --output-path "${SDF_OUT_DIR}/mesh.ply" \
        --bounding-box-min -1.0 -1.0 -1.0 \
        --bounding-box-max  1.0  1.0  1.0 \
        --resolution 2048 \
        --marching_cube_threshold 0.0035 \
        --create_visibility_mask True \
        --simplify-mesh False
else
    log "Step 1c: mesh.ply already exists, skipping"
fi

# ── 1c-best. extract mesh from best checkpoint (only if save-best produced one) ─
BEST_CKPT="$(ls -t "${CKPT_DIR}"/best-step-*.ckpt 2>/dev/null | head -1)"
if [[ -n "${BEST_CKPT}" && ! -f "${SDF_OUT_DIR}/mesh_best.ply" ]]; then
    log "Step 1c-best: extract mesh from ${BEST_CKPT##*/} (best PSNR)"
    cd "${REPO_ROOT}/sdf"
    time_step step_1c_extract_mesh_best python scripts/extract_mesh.py \
        --load-config "${SDF_OUT_DIR}/config.yml" \
        --checkpoint_path "${BEST_CKPT}" \
        --output-path "${SDF_OUT_DIR}/mesh_best.ply" \
        --bounding-box-min -1.0 -1.0 -1.0 \
        --bounding-box-max  1.0  1.0  1.0 \
        --resolution 2048 \
        --marching_cube_threshold 0.0035 \
        --create_visibility_mask True \
        --simplify-mesh False
elif [[ -z "${BEST_CKPT}" ]]; then
    log "Step 1c-best: no best-step-*.ckpt found, skipping best mesh"
else
    log "Step 1c-best: mesh_best.ply already exists, skipping"
fi

# ── 1d. filter interior mesh ──────────────────────────────────────────────────
if [[ ! -f "${SDF_OUT_DIR}/mesh-clean.ply" ]]; then
    log "Step 1d: filter interior mesh → mesh-clean.ply"
    python "${REPO_ROOT}/scripts/filter_interior_mesh.py" \
        --input  "${SDF_OUT_DIR}/mesh.ply" \
        --output "${SDF_OUT_DIR}/mesh-clean.ply" \
        --n_samples 30 \
        --timing "${TIMING_FILE}"
else
    log "Step 1d: mesh-clean.ply already exists, skipping"
fi

# ── 1e. simplify cleaned mesh ─────────────────────────────────────────────────
if [[ ! -f "${SDF_OUT_DIR}/mesh-clean-simplify.ply" ]]; then
    log "Step 1e: simplify mesh-clean.ply → mesh-clean-simplify.ply"
    python "${REPO_ROOT}/scripts/simplify_mesh.py" \
        --input  "${SDF_OUT_DIR}/mesh-clean.ply" \
        --output "${SDF_OUT_DIR}/mesh-clean-simplify.ply" \
        --target_faces 1000000 \
        --timing "${TIMING_FILE}"
else
    log "Step 1e: mesh-clean-simplify.ply already exists, skipping"
fi

# ── 1f. convert PLY → world coordinates ──────────────────────────────────────
if [[ ! -f "${SDF_OUT_DIR}/mesh-clean-simplify_world.ply" ]]; then
    log "Step 1f: convert PLY → COLMAP world coordinates"
    WORLD_INPUTS=()
    [[ -f "${SDF_OUT_DIR}/mesh.ply" ]]                && WORLD_INPUTS+=("${SDF_OUT_DIR}/mesh.ply")
    [[ -f "${SDF_OUT_DIR}/mesh-clean.ply" ]]          && WORLD_INPUTS+=("${SDF_OUT_DIR}/mesh-clean.ply")
    [[ -f "${SDF_OUT_DIR}/mesh-clean-simplify.ply" ]] && WORLD_INPUTS+=("${SDF_OUT_DIR}/mesh-clean-simplify.ply")
    python "${REPO_ROOT}/scripts/ply_to_world.py" \
        --transform "${DATA_DIR}/transforms.json" \
        --inputs    "${WORLD_INPUTS[@]}" \
        --suffix    _world \
        --outdir    "${SDF_OUT_DIR}" \
        --timing    "${TIMING_FILE}" \
        --extra-scale "${SCALE_FACTOR}"
else
    log "Step 1f: world PLY files already exist, skipping"
fi

# ── 1g. texture baking ────────────────────────────────────────────────────────
if [[ ! -f "${SDF_OUT_DIR}/texture_mesh/mesh-clean-simplify.obj" ]]; then
    log "Step 1g: bake texture on mesh-clean-simplify.ply"
    mkdir -p "${SDF_OUT_DIR}/texture_mesh"
    cd "${REPO_ROOT}/sdf"
    time_step step_1g_texture python scripts/texture.py \
        --load-config "${SDF_OUT_DIR}/config.yml" \
        --output-dir  "${SDF_OUT_DIR}/texture_mesh" \
        --input_mesh_filename "${SDF_OUT_DIR}/mesh-clean-simplify.ply" \
        --target_num_faces 300000
else
    log "Step 1g: texture mesh already exists, skipping"
fi

# ── 1h. texture mesh OBJ → world coordinates ─────────────────────────────────
if [[ ! -f "${SDF_OUT_DIR}/texture_mesh/mesh-clean-simplify_world.obj" ]]; then
    log "Step 1h: convert texture mesh OBJ → COLMAP world coordinates"
    python "${REPO_ROOT}/scripts/mesh_to_world.py" \
        --input     "${SDF_OUT_DIR}/texture_mesh/mesh-clean-simplify.obj" \
        --transform "${DATA_DIR}/transforms.json" \
        --output    "${SDF_OUT_DIR}/texture_mesh/mesh-clean-simplify_world.obj" \
        --timing    "${TIMING_FILE}" \
        --extra-scale "${SCALE_FACTOR}"
else
    log "Step 1h: texture mesh world OBJ already exists, skipping"
fi

# ── 1i. save pose ─────────────────────────────────────────────────────────────
if [[ ! -f "${DATA_DIR}/pose.pkl" ]]; then
    log "Step 1i: save pose"
    cd "${REPO_ROOT}/sdf"
    time_step step_1i_save_pose python scripts/save_pose.py \
        --ckpt_dir "${SDF_OUT_DIR}" \
        --save_dir "${DATA_DIR}"
else
    log "Step 1i: pose.pkl already exists, skipping"
fi

log "========================================================"
log "=== ${DATA_NAME} complete ==="
log "========================================================"
