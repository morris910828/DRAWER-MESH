#!/usr/bin/env bash
# Batch Stage-1 SDF reconstruction for all joint* and home datasets.
# Runs sequentially to avoid GPU contention.
#
# Usage:
#   bash scripts/run_stage1_batch.sh
#   DATA_ROOT=/opt/disk/drawer_dataset/autourdf_studio/V0000 bash scripts/run_stage1_batch.sh
#
# DATA_ROOT can be overridden via environment variable.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &>/dev/null && pwd)"
# 將圖片downscale的數字，變成1/n倍大
DOWNSCALE_FACTOR=1
# panoptic scale_factor: >1 zooms object into [-1,1] foreground (finer mesh, clips periphery).
# Global for all datasets in this batch; same value passed to world-conversion. Default 1.0.
SCALE_FACTOR="${SCALE_FACTOR:-1.0}"

# studio datasets
# DATA_ROOT="${DATA_ROOT:-/opt/disk/drawer_dataset/studio}"
# DATASETS=(
    # home
    # joint1_-180
    # joint2_-50
    # joint3_-90
    # joint4_-90
    # joint5_60
    # joint6_90
# )

# autourdf_studio/V0000 datasets (each frame is a separate pose)
# Usage: DATA_ROOT=/opt/disk/drawer_dataset/autourdf_studio/V0000 bash scripts/run_stage1_batch.sh
TRAJECTORIES=(
    V0001
    V0002
    V0003
    V0004
)
DATASETS=(
    frame_0000
    frame_0001
    frame_0002
    frame_0003
    frame_0004
    frame_0005
    frame_0006
    frame_0007
    frame_0008
    frame_0009
)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawer_sdf

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Record elapsed time for a shell-level step (nerfstudio scripts, marigold, etc.)
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

for TRAJECTORY in "${TRAJECTORIES[@]}"; do
    DATA_ROOT="/opt/disk/drawer_dataset/single_joint/${TRAJECTORY}"

    if [[ ! -d "${DATA_ROOT}" ]]; then
        log "SKIP trajectory ${TRAJECTORY}: ${DATA_ROOT} not found"
        continue
    fi

    for DATA_NAME in "${DATASETS[@]}"; do
        DATA_DIR="${DATA_ROOT}/${DATA_NAME}"
        SDF_OUT_DIR="${DATA_DIR}/${DATA_NAME}_sdf_recon"
        TIMING_FILE="${SDF_OUT_DIR}/timing.json"

        log "========================================================"
        log "Dataset: ${DATA_NAME}"
        log "========================================================"

        # ── 0. colmap_to_drawer: generate transforms.json ────────────────────────
        if [[ ! -f "${DATA_DIR}/transforms.json" ]]; then
            log "Step 0: colmap_to_drawer → transforms.json"
            conda run -n drawer_sdf python "${REPO_ROOT}/scripts/colmap_to_drawer.py" \
                --data_dir "${DATA_DIR}" \
                --downscale "${DOWNSCALE_FACTOR}" \
                --timing "${TIMING_FILE}"
        else
            log "Step 0: transforms.json already exists, skipping"
        fi

        # image dir for Marigold: images/ when downscale=1, images_N/ otherwise
        if [[ "${DOWNSCALE_FACTOR}" -eq 1 ]]; then
            MARIGOLD_INPUT="${DATA_DIR}/images"
        else
            MARIGOLD_INPUT="${DATA_DIR}/images_${DOWNSCALE_FACTOR}"
        fi

        # ── 1a. Marigold depth + normal ───────────────────────────────────────────
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

        # ── 1b. BakedSDF training ─────────────────────────────────────────────────
        if [[ ! -f "${SDF_OUT_DIR}/config.yml" ]]; then
            log "Step 1b: BakedSDF training → ${SDF_OUT_DIR}"
            mkdir -p "${SDF_OUT_DIR}"
            cd "${REPO_ROOT}/sdf"
            time_step step_1b_sdf_train python scripts/train.py bakedsdf --vis tensorboard \
                --output-dir "${DATA_DIR}" \
                --experiment-name "${DATA_NAME}_sdf_recon" \
                --trainer.steps-per-eval-image 2000 \
                --trainer.steps-per-eval-all-images 250001 \
                --trainer.max-num-iterations 250001 \
                --trainer.steps-per-eval-batch 250001 \
                --trainer.gradient-clipping-val 1.0 \
                --trainer.save-best-checkpoint True \
                --trainer.save-only-latest-checkpoint True \
                --optimizers.fields.optimizer.eps 1e-8 \
                --optimizers.field-background.optimizer.eps 1e-8 \
                --optimizers.proposal-networks.optimizer.eps 1e-8 \
                --optimizers.fields.scheduler.max-steps 250000 \
                --optimizers.field-background.scheduler.max-steps 250000 \
                --optimizers.proposal-networks.scheduler.max-steps 250000 \
                --pipeline.model.eikonal-anneal-max-num-iters 250000 \
                --pipeline.model.beta-anneal-max-num-iters 250000 \
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
                panoptic-data \
                --data "${DATA_DIR}" \
                --panoptic_data False \
                --mono_normal_data True \
                --mono_depth_data True \
                --panoptic_segment False \
                --downscale_factor "${DOWNSCALE_FACTOR}" \
                --num_max_image 2000 \
                --scale_factor "${SCALE_FACTOR}"
        else
            log "Step 1b: config.yml already exists, skipping BakedSDF training"
        fi

        # ── 1c. extract mesh (raw, no simplify) ──────────────────────────────────
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

        # ── 1c-best. extract mesh from best checkpoint (if save-best produced one) ─
        BEST_CKPT="$(ls -t "${SDF_OUT_DIR}/sdfstudio_models"/best-step-*.ckpt 2>/dev/null | head -1)"
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
        fi

        # ── 1d. filter interior mesh (mesh-in-mesh) ───────────────────────────────
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

        # ── 1e. simplify cleaned mesh ─────────────────────────────────────────────
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

        # ── 1f. convert mesh to COLMAP world coordinates ──────────────────────────
        if [[ ! -f "${SDF_OUT_DIR}/mesh-clean-simplify_world.ply" ]]; then
            log "Step 1f: convert PLY → COLMAP world coordinates"
            WORLD_INPUTS=()
            [[ -f "${SDF_OUT_DIR}/mesh.ply" ]]                 && WORLD_INPUTS+=("${SDF_OUT_DIR}/mesh.ply")
            [[ -f "${SDF_OUT_DIR}/mesh-clean.ply" ]]           && WORLD_INPUTS+=("${SDF_OUT_DIR}/mesh-clean.ply")
            [[ -f "${SDF_OUT_DIR}/mesh-clean-simplify.ply" ]]  && WORLD_INPUTS+=("${SDF_OUT_DIR}/mesh-clean-simplify.ply")
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

        # ── 1h. convert texture mesh OBJ to COLMAP world coordinates ─────────────
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

        if [[ ! -f "${DATA_DIR}/pose.pkl" ]]; then
            log "Step 1i: save pose"
            cd "${REPO_ROOT}/sdf"
            time_step step_1i_save_pose python scripts/save_pose.py \
                --ckpt_dir "${SDF_OUT_DIR}" \
                --save_dir "${DATA_DIR}"
        else
            log "Step 1i: pose.pkl already exists, skipping"
        fi

        log "=== ${DATA_NAME} complete ==="
    done
done

log "========================================================"
log "All datasets finished."
log "========================================================"
