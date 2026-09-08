#!/usr/bin/env bash
# Stage 1: SDF reconstruction (Marigold pre-pass + BakedSDF training + mesh/texture/pose).
# Adapted from scripts/run_stage1_sdf.sh for RTX 5090 / drawer_sdf env / studio_dataset paths.
#
# Usage:  bash scripts/run_stage1_sdf_5090.sh [DATA_NAME]
#         DATA_NAME defaults to cs_kitchen.

set -euo pipefail

DATA_NAME="${1:-cs_kitchen}"
DATA_ROOT="/media/user/EAA4244BA4241D17/studio_dataset"
DATA_DIR="${DATA_ROOT}/${DATA_NAME}"
OUT_DIR="${DATA_ROOT}/sdf_outputs/${DATA_NAME}"
IMAGE_DIR="images_2"
DOWNSCALE_FACTOR=2

if [[ ! -d "${DATA_DIR}/${IMAGE_DIR}" ]]; then
    echo "ERROR: ${DATA_DIR}/${IMAGE_DIR} not found" >&2
    exit 1
fi
mkdir -p "${OUT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawer_sdf

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &> /dev/null && pwd)"

# ---------------------------------------------------------------------------
# Stage 1a: Marigold monocular depth + normal priors
# ---------------------------------------------------------------------------
echo "=== Stage 1a: Marigold depth ==="
cd "${REPO_ROOT}/marigold"
python run.py \
    --checkpoint "GonzaloMG/marigold-e2e-ft-depth" \
    --modality depth \
    --input_rgb_dir "${DATA_DIR}/${IMAGE_DIR}" \
    --output_dir "${DATA_DIR}/marigold_ft"

echo "=== Stage 1a: Marigold normal ==="
python run.py \
    --checkpoint "GonzaloMG/marigold-e2e-ft-normals" \
    --modality normals \
    --input_rgb_dir "${DATA_DIR}/${IMAGE_DIR}" \
    --output_dir "${DATA_DIR}/marigold_ft"

echo "=== Stage 1a: read_marigold ==="
python read_marigold.py --data_dir "${DATA_DIR}/marigold_ft"

# Symlinks expected by the dataparser
ln -sfn "${DATA_DIR}/marigold_ft/depth" "${DATA_DIR}/depth"
ln -sfn "${DATA_DIR}/marigold_ft/normal" "${DATA_DIR}/normal"

# ---------------------------------------------------------------------------
# Stage 1b: BakedSDF training (250k iterations)
# ---------------------------------------------------------------------------
echo "=== Stage 1b: BakedSDF training ==="
cd "${REPO_ROOT}/sdf"

# Use tensorboard instead of wandb to avoid login requirement.
python scripts/train.py bakedsdf --vis tensorboard \
    --output-dir "${OUT_DIR}" --experiment-name "${DATA_NAME}_sdf_recon" \
    --trainer.steps-per-eval-image 2000 --trainer.steps-per-eval-all-images 250001 \
    --trainer.max-num-iterations 250001 --trainer.steps-per-eval-batch 250001 \
    --optimizers.fields.scheduler.max-steps 250000 \
    --optimizers.field-background.scheduler.max-steps 250000 \
    --optimizers.proposal-networks.scheduler.max-steps 250000 \
    --pipeline.model.eikonal-anneal-max-num-iters 250000 \
    --pipeline.model.beta-anneal-max-num-iters 250000 \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm inf \
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
    --num_max_image 2000

SDF_DIR="${OUT_DIR}/${DATA_NAME}_sdf_recon"

# ---------------------------------------------------------------------------
# Stage 1c: extract mesh, bake texture, save poses
# ---------------------------------------------------------------------------
echo "=== Stage 1c: extract mesh ==="
python scripts/extract_mesh.py --load-config "${SDF_DIR}/config.yml" \
    --output-path "${SDF_DIR}/mesh.ply" \
    --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
    --resolution 2048 --marching_cube_threshold 0.0035 \
    --create_visibility_mask True --simplify-mesh True

echo "=== Stage 1c: bake texture ==="
mkdir -p "${SDF_DIR}/texture_mesh"
python scripts/texture.py --load-config "${SDF_DIR}/config.yml" \
    --output-dir "${SDF_DIR}/texture_mesh" \
    --input_mesh_filename "${SDF_DIR}/mesh-simplify.ply" \
    --target_num_faces 300000

echo "=== Stage 1c: save pose ==="
python scripts/save_pose.py \
    --ckpt_dir "${SDF_DIR}" \
    --save_dir "${DATA_DIR}"

echo
echo "=== Stage 1 complete: ${SDF_DIR} ==="
