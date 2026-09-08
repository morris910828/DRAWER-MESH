#!/usr/bin/env bash
# Post-processing for the table mesh produced by run_table_sdf.sh.
#
# Usage:
#   bash scripts/run_table_post.sh 2              # CPU steps only (safe while another training runs)
#   bash scripts/run_table_post.sh 2 --texture    # also bake texture -> .obj  (needs a free GPU)
#
# Steps 1-3 are pure CPU, so they can run while the next SDF training occupies
# the GPU.  Texture baking loads the trained model and needs GPU memory, so it
# is opt-in.
#
# Do NOT overwrite run_table_sdf.sh while it is still running -- bash keeps
# reading the script file as it executes.

set -euo pipefail

DOWNSCALE="${1:-2}"
DO_TEXTURE=false
[[ "${2:-}" == "--texture" ]] && DO_TEXTURE=true

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. &>/dev/null && pwd)"
EXP="table_sdf_ds${DOWNSCALE}"
OUT_DIR="${DATA_DIR}/${EXP}"
MESH="${OUT_DIR}/mesh_table.ply"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawer_sdf
log() { echo "[$(date '+%H:%M:%S')] $*"; }

[[ -f "${MESH}" ]] || { echo "ERROR: ${MESH} not found -- has the SDF run finished?"; exit 1; }
log "post-processing ${MESH}"

# ── 1. drop interior faces ───────────────────────────────────────────────────
CLEAN="${OUT_DIR}/mesh_table-clean.ply"
if [[ ! -f "${CLEAN}" ]]; then
    log "1/5 filter interior faces"
    python "${REPO_ROOT}/scripts/filter_interior_mesh.py" \
        --input "${MESH}" --output "${CLEAN}" --n_samples 30
else
    log "1/5 ${CLEAN##*/} exists, skipping"
fi

# ── 2. simplify ──────────────────────────────────────────────────────────────
# The raw mesh is ~28.6M faces and the GS stage wants ~100k, i.e. a 286:1
# reduction.  Two settings matter for the table legs:
#   target_faces        final budget (override with TARGET_FACES=... )
#   min_component_faces default is 10000, which on a 100k-face mesh would delete
#                       any component up to 10% of the whole model -- the legs
#                       would be prime candidates.  Scaled down accordingly.
SIMP="${OUT_DIR}/mesh_table-clean-simplify.ply"
TARGET_FACES="${TARGET_FACES:-100000}"
MIN_COMP="${MIN_COMP:-200}"
if [[ ! -f "${SIMP}" ]]; then
    log "2/5 simplify -> ${TARGET_FACES} faces (min component ${MIN_COMP})"
    python "${REPO_ROOT}/scripts/simplify_mesh.py" \
        --input "${CLEAN}" --output "${SIMP}" \
        --target_faces "${TARGET_FACES}" \
        --min_component_faces "${MIN_COMP}"
else
    log "2/5 ${SIMP##*/} exists, skipping"
fi

# ── 3. training space -> COLMAP world ────────────────────────────────────────
if [[ ! -f "${OUT_DIR}/mesh_table-clean-simplify_world.ply" ]]; then
    log "3/5 PLY -> world coordinates"
    WORLD_INPUTS=()
    for f in "${MESH}" "${CLEAN}" "${SIMP}"; do
        [[ -f "$f" ]] && WORLD_INPUTS+=("$f")
    done
    python "${REPO_ROOT}/scripts/ply_to_world.py" \
        --transform "${DATA_DIR}/transforms.json" \
        --inputs    "${WORLD_INPUTS[@]}" \
        --suffix    _world \
        --outdir    "${OUT_DIR}" \
        --extra-scale 1.0
else
    log "3/5 world PLY exists, skipping"
fi

if [[ "${DO_TEXTURE}" != "true" ]]; then
    log "done (CPU steps only). Re-run with --texture once the GPU is free."
    exit 0
fi

# ── 4. texture baking (GPU) ──────────────────────────────────────────────────
cd "${REPO_ROOT}/sdf"
if [[ ! -f "${OUT_DIR}/texture_mesh/mesh_table-clean-simplify.obj" ]]; then
    log "4/5 bake texture"
    mkdir -p "${OUT_DIR}/texture_mesh"
    python scripts/texture.py \
        --load-config "${OUT_DIR}/config.yml" \
        --output-dir  "${OUT_DIR}/texture_mesh" \
        --input_mesh_filename "${SIMP}" \
        --target_num_faces 300000
else
    log "4/5 textured OBJ exists, skipping"
fi

# ── 5. textured OBJ -> world, plus pose.pkl ──────────────────────────────────
OBJ="${OUT_DIR}/texture_mesh/mesh_table-clean-simplify.obj"
if [[ -f "${OBJ}" && ! -f "${OUT_DIR}/texture_mesh/mesh_table-clean-simplify_world.obj" ]]; then
    log "5/5 OBJ -> world coordinates"
    python "${REPO_ROOT}/scripts/mesh_to_world.py" \
        --input     "${OBJ}" \
        --transform "${DATA_DIR}/transforms.json" \
        --output    "${OUT_DIR}/texture_mesh/mesh_table-clean-simplify_world.obj" \
        --extra-scale 1.0
fi

if [[ ! -f "${DATA_DIR}/pose.pkl" ]]; then
    log "save_pose"
    python scripts/save_pose.py --ckpt_dir "${OUT_DIR}" --save_dir "${DATA_DIR}"
fi

log "=== done -> ${OUT_DIR}/texture_mesh/mesh_table-clean-simplify_world.obj ==="
