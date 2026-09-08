#!/usr/bin/env bash
# Stage-1 SDF reconstruction for data/table, centred on the middle table.
#
# Usage (on the DGX):
#   bash scripts/run_table_sdf.sh 1     # full resolution 5568x4176
#   bash scripts/run_table_sdf.sh 2     # half resolution 2784x2088
#
# Each factor writes to its own experiment dir, so the two runs never collide.
#
# The mesh bounding box below was measured from the COLMAP sparse cloud and
# mapped through the dataparser's own normalisation (auto_orient 'up' +
# centre + auto-scale 1/4.8477).  It covers the middle table, everything on
# it, all four legs, and a slice of floor underneath:
#     floor plane      z = -0.861
#     tabletop plane   z = -0.180   (0.73 m above the floor)

set -euo pipefail

DOWNSCALE="${1:-2}"
DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &>/dev/null && pwd)"
EXP="table_sdf_ds${DOWNSCALE}"
OUT_DIR="${DATA_DIR}/${EXP}"

# Rendering a whole eval image costs several GB at full resolution (5568x4176 =
# 23.2M pixels x rgb/normal/depth/accumulation), which OOMs a 32 GB card even
# though training itself only needs ~0.1 s and a fraction of that memory. So at
# downscale 1 the periodic eval renders are switched off and geometry is checked
# on the extracted mesh instead; at lower resolutions they stay on.
if [[ "${DOWNSCALE}" -eq 1 ]]; then
    EVAL_EVERY=999999
    EVAL_ALL_EVERY=999999
else
    EVAL_EVERY=2000
    EVAL_ALL_EVERY=5000
fi

# table bounding box in training coordinates (small margin already added)
BB_MIN=(-0.71 -0.96 -0.96)
BB_MAX=( 0.79  0.55  0.41)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawer_sdf
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "dataset=${DATA_DIR}  downscale=${DOWNSCALE}  exp=${EXP}"

# ── downscaled images (only needed when DOWNSCALE > 1) ───────────────────────
if [[ "${DOWNSCALE}" -gt 1 && ! -d "${DATA_DIR}/images_${DOWNSCALE}" ]]; then
    log "building images_${DOWNSCALE}/"
    python - "$DATA_DIR" "$DOWNSCALE" <<'PY'
import sys, os
from pathlib import Path
from PIL import Image
data, f = Path(sys.argv[1]), int(sys.argv[2])
dst = data / f"images_{f}"; dst.mkdir(exist_ok=True)
for src in sorted((data / "images").iterdir()):
    if src.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
    out = dst / src.name
    if out.exists(): continue
    im = Image.open(src)
    im.resize((im.width // f, im.height // f), Image.LANCZOS).save(out, quality=95)
print("done:", len(list(dst.iterdir())), "images")
PY
fi

# ── Marigold cues are already present as depth/ and normal/ symlinks ─────────
ln -sfn "${DATA_DIR}/marigold_ft/depth"  "${DATA_DIR}/depth"
ln -sfn "${DATA_DIR}/marigold_ft/normal" "${DATA_DIR}/normal"

# ── BakedSDF training ────────────────────────────────────────────────────────
CKPT_DIR="${OUT_DIR}/sdfstudio_models"
if [[ -f "${CKPT_DIR}/step-00050000.ckpt" ]]; then
    log "training already complete, skipping"
else
    LOAD_DIR_ARG=()
    if [[ -d "${CKPT_DIR}" ]] && compgen -G "${CKPT_DIR}/step-*.ckpt" > /dev/null; then
        log "resuming from existing checkpoint"
        LOAD_DIR_ARG=(--trainer.load-dir "${CKPT_DIR}")
    fi
    cd "${REPO_ROOT}/sdf"
    python scripts/train.py bakedsdf --vis tensorboard \
        --output-dir "${DATA_DIR}" \
        --experiment-name "${EXP}" \
        --trainer.steps-per-eval-image 2000 \
        --trainer.steps-per-eval-all-images 5000 \
        --trainer.steps-per-eval-batch 50001 \
        --trainer.max-num-iterations 50001 \
        --trainer.steps-per-save 5000 \
        --trainer.gradient-clipping-val 1.0 \
        --trainer.save-best-checkpoint False \
        --trainer.save-only-latest-checkpoint False \
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
        --pipeline.model.scene-contraction-norm inf \
        --pipeline.model.mono-normal-loss-mult 0.2 \
        --pipeline.model.mono-depth-loss-mult 1.0 \
        --pipeline.model.near-plane 1e-6 \
        --pipeline.model.far-plane 100 \
        --pipeline.datamanager.train-num-rays-per-batch 4096 \
        --machine.num-gpus 1 \
        "${LOAD_DIR_ARG[@]}" \
        panoptic-data \
        --data "${DATA_DIR}" \
        --panoptic_data False \
        --panoptic_segment False \
        --mono_normal_data True \
        --mono_depth_data True \
        --downscale_factor "${DOWNSCALE}" \
        --num_max_image 50
fi

# ── extract mesh, cropped to the middle table ────────────────────────────────
cd "${REPO_ROOT}/sdf"
if [[ ! -f "${OUT_DIR}/mesh_table.ply" ]]; then
    log "extracting table mesh (bbox ${BB_MIN[*]} .. ${BB_MAX[*]})"
    python scripts/extract_mesh.py \
        --load-config "${OUT_DIR}/config.yml" \
        --output-path "${OUT_DIR}/mesh_table.ply" \
        --bounding-box-min "${BB_MIN[@]}" \
        --bounding-box-max "${BB_MAX[@]}" \
        --resolution 2048 \
        --marching_cube_threshold 0.0035 \
        --create_visibility_mask True \
        --simplify-mesh False
fi

# full-scene mesh as a fallback / for context
if [[ ! -f "${OUT_DIR}/mesh_full.ply" ]]; then
    log "extracting full-scene mesh"
    python scripts/extract_mesh.py \
        --load-config "${OUT_DIR}/config.yml" \
        --output-path "${OUT_DIR}/mesh_full.ply" \
        --bounding-box-min -1.0 -1.0 -1.0 \
        --bounding-box-max  1.0  1.0  1.0 \
        --resolution 2048 \
        --marching_cube_threshold 0.0035 \
        --create_visibility_mask True \
        --simplify-mesh False
fi

log "=== ${EXP} done -> ${OUT_DIR}/mesh_table.ply ==="
