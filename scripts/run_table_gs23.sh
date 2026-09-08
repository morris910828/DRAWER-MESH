#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 23.
#
# Two changes from gs20. One is a new hypothesis; the other is fixing a number that was
# simply wrong and invalidated gs20 as a baseline.
#
# ---------------------------------------------------------------------------------------
# 1. detail_elevate_min_frac 0.0  (NEW -- this is what the run tests)
#
# Half the detail layer is buried behind the base plate and contributes nothing.
# Measured on gs20's export, offsets along the face normal in units of face radius:
#
#     base    1,136,576 rows   p10 -0.000  p50 -0.000  p90 +0.000   (pinned to the face)
#     detail  3,679,132 rows   p10 -0.500  p50 -0.000  p90 +0.500   (both caps saturated)
#
#     50.9% of detail sits BEHIND the plate.  49.1% in front.
#
# "Behind" is well defined here: the mesh is closed and consistently wound -- signed volume
# +3.90, and 100.00% of 2,063,187 shared edges have same-facing normals -- so the outward
# normal points towards every camera that can see that face, whatever the viewpoint.
#
# Those buried rows carry a median alpha of 0.996 and a median size of 0.3299 face radii,
# identical to the ones in front. They are not a size-based division of labour; they are
# half the detail budget behind an opacity-0.950 wall. And the state is self-sustaining: a
# Gaussian hidden behind an opaque plate barely moves any pixel, so it gets almost no
# gradient, so nothing pushes it out. The clean 50/50 is the initial random distribution
# frozen in place, not something the optimiser chose.
#
# Confining detail to the outward side roughly doubles the effective detail population at
# zero memory cost. Coverage is untouched -- _compute_coverage_density() measures in the
# face plane, which the normal offset does not change.
#
# ---------------------------------------------------------------------------------------
# 2. max_gaussians 0  (no ceiling -- it has cost three runs in a row)
#
#     gs20  5.5M -- BELOW its own seed of 1,395,229 faces x 4 = 5,580,916, so it hit the
#                   cap on its first refine and densification NEVER RAN. That is why its
#                   face sigma regressed from gs19's 1.16 px to 1.33 px.
#     gs21  5.5M -- deadlocked at step 18890: rescue reported "spawning 0" against 205,888
#                   faces flagged for repair.
#     gs22  6.5M -- same deadlock at step 6590, only sooner.
#
# A ceiling is a guess about memory made before the run starts. Getting it wrong costs the
# whole run; being without one costs at most the hour since the last checkpoint (saved every
# 10000 steps). Those are not symmetric, and the guess has now been wrong three times.
#
# The right lever for memory is quality, not headcount: delete rows that paint nothing. That
# is cull_screen_size_min, added for this purpose but still disabled here -- the threshold
# has to be read off the max_2Dsize quantiles the cull log now prints, and at step 5290 of
# gs22 the distribution had not yet separated (p1 = p10 = 1.08e-03). Read those quantiles
# late in THIS run and set it in the next one.
#
# If this OOMs: resume from the last checkpoint with a lower init_copies_per_face or a
# higher densify_grad_thresh. Never the resolution.
#
# ---------------------------------------------------------------------------------------
# coverage_floor_target stays at 1.0, i.e. gs20/gs19's value, NOT gs21/gs22's 0.7. That
# experiment is unfinished and it pushes the population up (releasing plates costs coverage,
# which rescue then has to buy back). Testing it on top of a detail layer that has just
# doubled in effectiveness would confound both. It is the next run, not this one.
#
# WHAT TO WATCH
#   1. face-layer sigma_px in analyze_splat_ply [4] -- gs19 got 1.16 px, gs20 regressed to
#      1.33 px with densification dead. This run should beat 1.16.
#   2. eval PSNR inside the mask -- the only number comparable to vanilla's 26.46 dB.
#   3. the text and the remote-control buttons, by eye. That is the actual goal.
# ---------------------------------------------------------------------------------------
#
# Usage (inside the drawer_splat env):
#   bash scripts/run_table_gs23.sh
#   bash scripts/run_table_gs23.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs23}"

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
    `# ==== THE HYPOTHESIS UNDER TEST -- see section 1 of the header ====` \
    --pipeline.model.detail_elevate_min_frac 0.0 `# was -1.0 (symmetric)` \
    \
    `# ==== CHANGE 2 of 2: population ceiling ==================================` \
    `# gs19 finished with 6.02M rows at downscale 4. At full resolution the render  ` \
    `# and gradient buffers are 16x larger per frame, and two earlier full-res runs  ` \
    `# died of CUDA OOM hours in. gs12 is the proven point on a 32 GB card: 5.48M     ` \
    `# rows at downscale 1. Cap below that and let culling do the rest.               ` \
    `#                                                                                ` \
    `# If this binds every cycle, the fix is a smaller seed (init_copies_per_face) or  ` \
    `# a higher densify_grad_thresh -- NOT a lower resolution and NOT a higher cap.    ` \
    --pipeline.model.max_gaussians 0 `# 0 = no ceiling. gs20 5.5M / gs21 5.5M / gs22 6.5M all failed because of it` \
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
    --pipeline.model.coverage_floor_target 1.0 \
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
