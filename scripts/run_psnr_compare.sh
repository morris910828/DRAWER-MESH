#!/usr/bin/env bash
# Full-resolution masked-PSNR comparison: gs12 vs gs19 vs gs20 (+ gs24).
#
# Renders every train + eval view at native 5568x4176 for each checkpoint and
# reports PSNR inside the mask (comparable to vanilla's 26.46 dB), outside, and
# over the whole frame -- one ruler for all runs.
#
# gs19 trained at downscale 4; it is force-rendered at downscale 1 here so it is
# judged on the same ground truth. A low inside-mask number for gs19 is the
# honest, expected result.
#
# Run on the DGX inside the drawer_splat env:
#     bash scripts/run_psnr_compare.sh
#
set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/psnr_compare}"
EXPS="${EXPS:-table_gs12 table_gs19 table_gs20}"   # add table_gs24 to include the latest

# ── PRE-FLIGHT: masks must resolve at <data>/mask (downscale-1 location) ─────────
MASK_DIR="${DATA_DIR}/mask"
if [[ ! -d "${MASK_DIR}" ]]; then
    if [[ -d "${DATA_DIR}/gs_masks/masks" ]]; then
        echo "[pre-flight] linking ${MASK_DIR} -> gs_masks/masks"
        ln -sfn "${DATA_DIR}/gs_masks/masks" "${MASK_DIR}"
    else
        echo "ERROR: no masks at ${DATA_DIR}/gs_masks/masks"; exit 1
    fi
fi
N_MASK=$(find -L "${MASK_DIR}" -name '*.png' | wc -l)
N_IMG=$(find "${DATA_DIR}/images" -name '*.JPG' | wc -l)
[[ "${N_MASK}" -eq "${N_IMG}" ]] || { echo "ERROR: ${N_MASK} masks vs ${N_IMG} images"; exit 1; }
echo "[pre-flight] ${N_MASK} masks resolved at ${MASK_DIR}"

# warn if the masks predate the mesh they should describe (experiments file 4.6)
MESH="${DATA_DIR}/table_sdf_ds1/texture_mesh/mesh_table-only-400k_world.obj"
if [[ -f "${MESH}" ]]; then
    if [[ "$(find -L "${MASK_DIR}" -name '*.png' -newer "${MESH}" | wc -l)" -eq 0 ]]; then
        echo "[warn] every mask is OLDER than ${MESH##*/} -- they may have been cut from a"
        echo "       different mesh (see experiments file 4.6). Relative PSNR is still valid;"
        echo "       absolute numbers vs vanilla are only as good as the mask."
    fi
fi

cd "${SPLAT_DIR}"
PYTHONPATH="${SPLAT_DIR}" python "${SPLAT_DIR}/../scripts/compare_masked_psnr.py" \
    --exp ${EXPS} \
    --data "${DATA_DIR}" \
    --out "${OUT_DIR}" \
    --splits eval train \
    --cache-device cpu \
    --save-renders \
    2>&1 | tee "${OUT_DIR%/}_$(date +%Y%m%d_%H%M).log"
