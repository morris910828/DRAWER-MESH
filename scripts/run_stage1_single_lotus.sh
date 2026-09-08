#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 SDF reconstruction for a single dataset — Lotus-2 prior variant.
#
# IDENTICAL to scripts/run_stage1_single.sh EXCEPT step 1a: the depth/normal
# priors come from Lotus-2 instead of Marigold. Everything from 1b (training)
# onward — training args, mesh naming (mesh.ply=last, mesh_best.ply=best),
# downstream clean/simplify/world/texture/pose — is the same as the Marigold run.
#
# Lotus-2 raw output is not directly usable, so step 1a-conv fixes it:
#   (1) depth : Lotus emits DISPARITY [0,1] (near=high) → depth = 1/(disp+eps).
#   (2) normal: Lotus axis convention differs from Marigold by a fixed global
#               negation (perm=(0,1,2) signs=(-1,-1,-1)); lotus_convert.py applies
#               it directly (hardcoded — no brute-force, no Marigold reference).
#
# Only difference vs the Marigold run's output location: writes to
#   <DATA_DIR>/<name>_sdf_recon_lotus   (so it never clobbers the Marigold run).
#
# Usage:
#   bash scripts/run_stage1_single_lotus.sh              # uses DATA_DIR set below
#   bash scripts/run_stage1_single_lotus.sh <DATA_DIR> [DOWNSCALE_FACTOR]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── args / config ────────────────────────────────────────────────────────────
DATA_DIR="${1:-/opt/disk/drawer_dataset/robot_hand/gopro_2fps}"
DOWNSCALE_FACTOR="${2:-1}"
# panoptic scale_factor: >1 scales the object up into the [-1,1] linear foreground
# (finer mesh) at the cost of clipping the scene periphery. Default 1.0 = original.
# The SAME value is passed to world-conversion (--extra-scale) so COLMAP coords stay correct.
SCALE_FACTOR="${SCALE_FACTOR:-1.0}"
DATA_DIR="$(realpath "${DATA_DIR}")"

# Lotus-2 external tool (checkpoints + conda env live here)
EXT="/opt/disk/external_tools"
LOTUS_ENV="${EXT}/envs/lotus2"
LOTUS_REPO="${EXT}/Lotus-2"
LOTUS_CKPT="${EXT}/checkpoints/Lotus-2"
FLUX_CKPT="${EXT}/checkpoints/FLUX.1-dev"
DISP_EPS="0.02"                       # depth = 1/(disparity + DISP_EPS)

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: DATA_DIR not found: ${DATA_DIR}"; exit 1
fi

DATA_NAME="$(basename "${DATA_DIR}")"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &>/dev/null && pwd)"
# distinct output dir so it never collides with the Marigold run
EXP_NAME="${DATA_NAME}_sdf_recon_lotus"
SDF_OUT_DIR="${DATA_DIR}/${EXP_NAME}"
TIMING_FILE="${SDF_OUT_DIR}/timing.json"

LOTUS2_DIR="${DATA_DIR}/lotus2"         # raw Lotus-2 output (depth/ + normal/)
LOTUS_DEPTH="${DATA_DIR}/lotus_depth"   # converted, marigold-format depth (.npy)
LOTUS_NORMAL="${DATA_DIR}/lotus_normal" # converted, marigold-format normal (.png)

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
log "Dataset : ${DATA_NAME}   (Lotus-2 prior variant)"
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

# image dir for the prior model
if [[ "${DOWNSCALE_FACTOR}" -eq 1 ]]; then
    IMG_DIR="${DATA_DIR}/images"
else
    IMG_DIR="${DATA_DIR}/images_${DOWNSCALE_FACTOR}"
fi

# ── 1a. Lotus-2 depth + normal priors (the ONLY difference vs Marigold) ────────
LOTUS_DEPTH_NPY="${LOTUS2_DIR}/depth/depth_npy"
LOTUS_NORMAL_NPY="${LOTUS2_DIR}/normal/normal_npy"
N_IMGS=$(ls "${IMG_DIR}" | wc -l)
N_DEPTH=$(find "${LOTUS_DEPTH_NPY}" -name "*.npy" 2>/dev/null | wc -l)
N_NORM=$(find "${LOTUS_NORMAL_NPY}" -name "*.npy" 2>/dev/null | wc -l)
if [[ "${N_DEPTH}" -lt "${N_IMGS}" || "${N_NORM}" -lt "${N_IMGS}" ]]; then
    log "Step 1a: Lotus-2 inference (depth + normal) on ${N_IMGS} images"
    export HF_HOME="${EXT}/.hf" TORCH_HOME="${EXT}/.torch" TMPDIR="${EXT}/.tmp"
    for task in depth normal; do
        log "  Lotus-2 ${task}"
        time_step "step_1a_lotus_${task}" \
        conda run --prefix "${LOTUS_ENV}" --no-capture-output \
            python "${LOTUS_REPO}/infer.py" \
                --pretrained_model_name_or_path="${FLUX_CKPT}" \
                --core_predictor_model_path="${LOTUS_CKPT}/lotus-2_core_predictor_${task}.safetensors" \
                --lcm_model_path="${LOTUS_CKPT}/lotus-2_lcm_${task}.safetensors" \
                --detail_sharpener_model_path="${LOTUS_CKPT}/lotus-2_detail_sharpener_${task}.safetensors" \
                --input_dir="${IMG_DIR}" \
                --output_dir="${LOTUS2_DIR}/${task}" \
                --seed=0 --task_name="${task}"
    done
else
    log "Step 1a: Lotus-2 raw priors complete (depth=${N_DEPTH} normal=${N_NORM}/${N_IMGS}), skipping"
fi

# ── 1a-conv. Lotus-2 → SDF dataparser format (disparity→depth + hardcoded normal axis) ─
N_CONV_D=$(find "${LOTUS_DEPTH}"  -name "*.npy" 2>/dev/null | wc -l)
N_CONV_N=$(find "${LOTUS_NORMAL}" -name "*.png" 2>/dev/null | wc -l)
if [[ "${N_CONV_D}" -lt "${N_IMGS}" || "${N_CONV_N}" -lt "${N_IMGS}" ]]; then
    log "Step 1a-conv: convert Lotus-2 output → depth/*.npy + normal/*.png (hardcoded axis)"
    time_step step_1a_lotus_convert python "${REPO_ROOT}/marigold/lotus_convert.py" \
        --img_dir    "${IMG_DIR}" \
        --lotus2_dir "${LOTUS2_DIR}" \
        --out_depth  "${LOTUS_DEPTH}" \
        --out_normal "${LOTUS_NORMAL}" \
        --disp_eps   "${DISP_EPS}"
else
    log "Step 1a-conv: converted priors complete (depth=${N_CONV_D} normal=${N_CONV_N}/${N_IMGS}), skipping"
fi

# symlink the priors the dataparser reads (→ Lotus instead of Marigold)
ln -sfn "${LOTUS_DEPTH}"  "${DATA_DIR}/depth"
ln -sfn "${LOTUS_NORMAL}" "${DATA_DIR}/normal"

# ═════════════════════════════════════════════════════════════════════════════
# From here on (1b–1i) this is byte-for-byte the same as run_stage1_single.sh,
# only EXP_NAME carries the _lotus suffix.
# ═════════════════════════════════════════════════════════════════════════════

# ── 1b. BakedSDF training ─────────────────────────────────────────────────────
CKPT_DIR="${SDF_OUT_DIR}/sdfstudio_models"
# max-num-iterations=250001 → loop range(0,250001) → last step=250000
FINAL_CKPT="${CKPT_DIR}/step-000250000.ckpt"
if [[ -f "${FINAL_CKPT}" ]]; then
    log "Step 1b: training complete (step-000250000.ckpt exists), skipping"
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
        --experiment-name "${EXP_NAME}" \
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
log "=== ${DATA_NAME} complete (Lotus-2 variant) ==="
log "========================================================"
