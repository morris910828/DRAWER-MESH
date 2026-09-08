#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 22.
#
# gs21 re-run with the population ceiling raised. Two flags differ from gs20:
#   coverage_floor_target 1.0 -> 0.7   (the thing being tested, unchanged from gs21)
#   max_gaussians    5.5M -> 6.5M      (gs21 was killed at 69% because of this)
#
# WHY gs21 WAS ABANDONED
#
# At step 20590 its log read:
#     max_gaussians reached (5,580,916 >= 5,500,000) -- densification paused
#     coverage_rescue_spawn: spawning 0 gaussian(s) (205888 face(s) qualified)
#     adaptive coverage floor: 888561/1395229 faces (63.7%) still floored
#
# The flag under test was working -- floored fell from gs19's 80.1% to 63.7%, which is
# exactly what releasing base plates looks like. But releasing a plate lowers that
# face's coverage, and coverage_rescue is the ONLY mechanism that restores it here
# (coverage_lambda is 0). The ceiling froze densification and rescue together, so
# 205,888 faces were flagged as needing repair and got zero Gaussians. Coverage was
# being spent without anything buying it back.
#
# 5.5M came from gs12, which had neither a shrunken vertex layer nor released base
# floors -- a different budget shape entirely. Measured headroom: gs19 reached 6.02M
# at downscale 4, gs20 reached 5.52M at downscale 1 without OOM, and gs21 was sitting
# at 5.58M at downscale 1 still running. 6.5M leaves room for the rescue that gs21
# was denied. If this OOMs, cut init_copies_per_face or raise densify_grad_thresh --
# NOT the resolution.
#
# This ceiling has now cost two runs: gs20 (densification frozen, face sigma rose
# 1.16 -> 1.33 px) and gs21 (rescue frozen). The experiments file already recorded
# the same failure once before -- gs17 hit an assert at step 24999 because the cap
# blocked rescue's non-optional job of filling empty faces.
#
# ---------------------------------------------------------------------------------------
# WHY: the base layer paints three quarters of the frame at 4 px.
#
# Measured on gs20's own export -- opacity-weighted screen smear (pi*sx_px*sy_px*alpha)
# for camera GOPR0236, split by layer:
#
#     layer      count       share   sigma_px   alpha    SMEAR SHARE
#     base     1,128,796     20.6%     4.17     0.950       74.3%
#     detail   3,654,538     66.6%     1.17     0.996       15.2%
#     vertex     702,813     12.8%     3.08     0.170       10.5%
#
# By sigma bucket: sigma <= 1 px is 23.6% of the rows and paints 1.2% of the frame;
# sigma 3-10 px is 25.6% of the rows and paints 80.9%.
#
# 3.65M detail Gaussians already sit at 1.17 px -- fine enough for the 2 px text strokes.
# They contribute 15% of the picture. What is actually on screen is 1.13M base plates at
# 4.17 px and opacity 0.950.
#
# base measures 4.17 px because the mesh triangles measure 3.87 px (median centroid-to-
# corner on the same export) and the floor pins base at >= 0.7 of the face radius. Its
# alpha median is 0.950 -- exactly base_opacity_floor, so not one plate ever earned its
# way off the bound. From the adaptive_coverage_floor docstring: a Gaussian has ONE
# colour, so a plate spanning both black stroke and white paper can only converge to the
# grey mean. Training its colour harder cannot fix that; only letting it SHRINK can.
#
# THIS IS NOT A NEW MECHANISM -- IT IS A THRESHOLD NOBODY EVER TUNED.
#
# adaptive_coverage_floor already drops a face's plate once that face is covered without
# it. update_coverage_floor_mask() re-measures every refine_every steps with EVERY floor
# released (base plates AND vertex fillers at once), so a floor is dropped only where it
# was demonstrably doing nothing. The gate is coverage_floor_target, whose own docstring
# reads:
#
#     "Lower it toward ~0.7 to let more faces go floor-free (sharper, more gap risk);
#      raise it toward 1.0 to keep more floors (safer, blurrier). Never tuned."
#
# The default is 0.9. gs19 and gs20 both set 1.0 -- the blurriest end of that sentence.
# gs19's final log line reported 80.1% of faces still floored.
#
# 0.7 is the value that docstring names. The risk is bounded on both sides: the release
# measurement is deliberately pessimistic (single pass, all floors released together, so
# two faces that only cover each other are both correctly refused), and coverage_rescue
# still runs every 100 steps to refill any face that ends up genuinely empty.
#
# WHAT TO WATCH, in order:
#   1. base's smear share -- scratchpad/whopaints.py against the export.
#      74.3% now; expect well under 50%. This is the whole point of the run.
#   2. the run's own final coverage line. gs19 printed 99.991% at alpha >= 0.01.
#      If it drops below ~99.9%, come back up to 0.8.
#   3. face-based sigma_px in analyze_splat_ply [4]. The median must not RISE. Plates
#      shrinking down into the detail size range is the intended outcome.
#
# Everything else is gs20 verbatim, so any change in (1) is attributable to this flag.
# ---------------------------------------------------------------------------------------
#
# Usage (inside the drawer_splat env):
#   bash scripts/run_table_gs22.sh
#   bash scripts/run_table_gs22.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs22}"

# SDF_EXP must match the EXP_SUFFIX given to run_table_sdf.sh / run_table_post.sh,
# otherwise this trains against one mesh while the masks describe another.
SDF_EXP="${SDF_EXP:-table_sdf_ds1}"
MESH="${DATA_DIR}/${SDF_EXP}/texture_mesh/mesh_table-only-400k_world.obj"
AREA=1e-4

EXTRA_INFO_DIR="${DATA_DIR}/gs_extra_info"
EXTRA_INFO="${EXTRA_INFO_DIR}/${EXP}.pt"
mkdir -p "${EXTRA_INFO_DIR}"

# ── PRE-FLIGHT: the masks must resolve BEFORE 4 hours of GPU time ────────────────────
# gs19 ran unmasked and only said so in a log line that scrolled past. This exits
# instead. At downscale 1 the dataparser resolves <data>/ + mask_path = <data>/mask/.
MASK_DIR="${DATA_DIR}/mask"
if [[ ! -d "${MASK_DIR}" ]]; then
    if [[ -d "${DATA_DIR}/gs_masks/masks" ]]; then
        echo "[pre-flight] linking ${MASK_DIR} -> gs_masks/masks"
        ln -sfn "${DATA_DIR}/gs_masks/masks" "${MASK_DIR}"
    else
        echo "ERROR: no masks. Expected ${DATA_DIR}/gs_masks/masks (50 files, 5568x4176)."
        exit 1
    fi
fi
N_MASK=$(find -L "${MASK_DIR}" -name '*.png' | wc -l)
N_IMG=$(find "${DATA_DIR}/images" -name '*.JPG' | wc -l)
if [[ "${N_MASK}" -ne "${N_IMG}" ]]; then
    echo "ERROR: ${N_MASK} masks vs ${N_IMG} images -- refusing to train unmasked."
    exit 1
fi
echo "[pre-flight] ${N_MASK} masks resolved at ${MASK_DIR}"

# 遮罩必須比 mesh 新。gs_masks 曾經比 mesh 早 31 小時 -- 它是用一顆更早、簡化更凶的
# mesh 切的，把立牌上緣整片切掉而沒有留下任何跡象，gs20 就是戴著那份遮罩訓練的。
NEWEST_MASK=$(find -L "${MASK_DIR}" -name '*.png' -newer "${MESH}" | head -1)
if [[ -z "${NEWEST_MASK}" ]]; then
    echo "ERROR: every mask is OLDER than ${MESH##*/}"
    echo "       they were cut from a different mesh. Re-cut them first:"
    echo "       python scripts/save_mesh_depth_masks.py --data_dir ${DATA_DIR} \\"
    echo "           --mesh_path ${MESH} \\"
    echo "           --dataparser_tf ${DATA_DIR}/table_gs19/dataparser_transforms.json \\"
    echo "           --out_dir ${DATA_DIR}/gs_masks"
    exit 1
fi
echo "[pre-flight] masks are newer than the mesh"

cd "${SPLAT_DIR}"

if [[ "${1:-}" != "--export-only" ]]; then
PYTHONPATH="${SPLAT_DIR}" python nerfstudio/scripts/train.py splatfacto_on_mesh_uc \
    --vis tensorboard \
    --output-dir "${DATA_DIR}" \
    --experiment-name "${EXP}" \
    --max-num-iterations 30000 \
    \
    `# ==== I/O (unchanged from gs19) =========================================` \
    --steps-per-save 10000 \
    --steps-per-eval-image 500 \
    --steps-per-eval-all-images 30000 \
    \
    `# ==== CHANGE 1 of 2 in the model: shrink the vertex layer ===============` \
    `# 1.0 -> 0.5. See cause 3 in the header: this layer is 12% of the rows and    ` \
    `# ~78% of the painted area at 6.01 px median, sitting on top of a face layer   ` \
    `# that is already at 1.16 px.                                                  ` \
    --pipeline.model.vertex_radius_scale 0.5 `# gs19: 1.0` \
    \
    `# ==== CHANGE 2 of 2: population ceiling ==================================` \
    `# gs19 finished with 6.02M rows at downscale 4. At full resolution the render  ` \
    `# and gradient buffers are 16x larger per frame, and two earlier full-res runs  ` \
    `# died of CUDA OOM hours in. gs12 is the proven point on a 32 GB card: 5.48M     ` \
    `# rows at downscale 1. Cap below that and let culling do the rest.               ` \
    `#                                                                                ` \
    `# If this binds every cycle, the fix is a smaller seed (init_copies_per_face) or  ` \
    `# a higher densify_grad_thresh -- NOT a lower resolution and NOT a higher cap.    ` \
    --pipeline.model.max_gaussians 6500000 `# gs20/gs21: 5.5M -- froze rescue, see header` \
    \
    `# ==== unchanged from gs19 =================================================` \
    --pipeline.model.anisotropic_vertex_floor True \
    --pipeline.model.vertex_floor_max_aspect 8.0 \
    --pipeline.model.anisotropic_coverage_density True \
    --pipeline.model.max_gauss_ratio 5.0 \
    --pipeline.model.coverage_rescue_thresh 1.0 \
    --pipeline.model.coverage_rescue_max_frac 0.05 \
    --pipeline.model.anisotropic_base_floor True \
    --pipeline.model.scale_reg_exclude_base True \
    --pipeline.model.base_floor_corner_sigma 1.0 \
    --pipeline.model.base_floor_max_aspect 8.0 \
    --pipeline.model.base_floor_reaches_corners False \
    `# ==== THE ONLY CHANGE FROM gs20 -- see the header =================` \
    --pipeline.model.coverage_floor_target 0.7 `# gs20: 1.0, default 0.9` \
    --pipeline.model.init_copies_per_face 4 \
    --pipeline.model.min_scale_frac 0.02 \
    --pipeline.model.adaptive_coverage_floor True \
    --pipeline.model.coverage_rescue_every 100 \
    --pipeline.model.opacity_reset_value 0.10 \
    --pipeline.model.opacity_reset_min_value 0.075 \
    --pipeline.model.coverage_lambda 0.0 \
    --pipeline.model.coverage_densify_scale 0.0 \
    --pipeline.model.cull_at_floor_gaussians False \
    --pipeline.model.opacity_reg_lambda 0.0 \
    --pipeline.model.base_opacity_floor 0.95 \
    --pipeline.model.elevate_coef 0.5 \
    --pipeline.model.gaussian_save_extra_info_path "${EXTRA_INFO}" \
    --pipeline.model.mesh_area_to_subdivide ${AREA} \
    --pipeline.model.upper_scale 0.5 \
    --pipeline.model.min_vertex_scale_frac 0.15 \
    --pipeline.model.use_base_layer True \
    --pipeline.model.copies_scale_shrink_power 0.3 \
    --pipeline.model.coverage_target 1.0 \
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
    \
    panoptic-data \
    --data "${DATA_DIR}" \
    --mesh_gauss_path "${MESH}" \
    --mesh_area_to_subdivide ${AREA} \
    --mesh_depth False \
    `# ==== THE CHANGE THAT MATTERS: back to full resolution ==================` \
    `# 1, not gs18/gs19's 4. 5568x4176, where a 2 px text stroke is 2 px.        ` \
    `# Every sigma_px the analysis prints afterwards is then read directly, with   ` \
    `# no division -- unlike gs18/gs19, whose numbers had to be divided by 4.       ` \
    --downscale_factor 1 `# gs19: 4` \
    --num_max_image 50 \
    --orientation_method none \
    --center_poses False \
    --auto_scale_poses False
fi

PYTHONPATH="${SPLAT_DIR}" python nerfstudio/scripts/exporter.py gaussian-splat \
    --load-config "${DATA_DIR}/${EXP}/config.yml" \
    --output-dir "${DATA_DIR}/${EXP}/export_ply"

echo "=== done -> ${DATA_DIR}/${EXP}/export_ply ==="
