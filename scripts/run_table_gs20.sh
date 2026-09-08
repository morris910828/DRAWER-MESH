#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 20.
#
# ---------------------------------------------------------------------------------------
# WHY: gs18 and gs19 were blurry for a reason that has nothing to do with the Gaussians.
#
# THREE INDEPENDENT CAUSES, all fixed here. Each one alone is enough to produce the
# blur that was observed, so fixing any two of them would still have failed.
#
# 1. downscale_factor 4 -- THE TRAINING IMAGES HAD NO TEXT IN THEM
#
#    Text strokes in these photos are ~2 px wide at 5568x4176. At downscale 4 the
#    training images are 1392x1044, where the same stroke is 0.5 px. The model cannot
#    learn detail that is not in its supervision, no matter how small its Gaussians are:
#    gs19's face layer reached a median on-screen sigma of 1.16 px -- finer than gs11's
#    1.55 px, the best of the series -- and still rendered blurry text.
#
#    gs18 picked downscale 4 from `tri_px * (W/5568) / sqrt(3) <= 1`, solving 6.99 px
#    triangles to W <= 1378. That arithmetic is right and its conclusion was backwards:
#    it says THE MESH IS TOO COARSE FOR THE PHOTOS, and the fix is to subdivide the mesh,
#    not to throw away the photos. Downscaling only makes both sides equally blurry.
#
#    Full resolution is a hard requirement for this dataset. If this run runs out of
#    memory, cut the population (max_gaussians below), never the resolution.
#
# 2. The masks were never found -- NO RUN IN THIS PROJECT HAS EVER TRAINED MASKED
#
#    The files exist on the DGX, one directory level away from where the dataparser
#    looks. _get_fname() (panoptic_dataparser.py:801-803) is rigid:
#        downscale_factor > 1  ->  <data>/masks_<N>/<name>
#        downscale_factor == 1 ->  <data>/ + transforms.json's mask_path  (./mask/<name>)
#    and the masks are under <data>/gs_masks/. At downscale 1 this run needs
#    <data>/mask/, which the pre-flight below creates as a symlink to gs_masks/masks
#    (already full resolution, 5568x4176, mode L) and then verifies.
#
#    Cost of it being absent, measured on gs18: eval PSNR 19.47 dB inside the mask vs
#    11.64 dB over the frame. Every Gaussian is anchored to the mesh and structurally
#    cannot draw background, so 72% of each frame was an unwinnable loss term whose
#    gradient pulls Gaussians outward and larger -- i.e. directly towards blur.
#
# 3. vertex_radius_scale 1.0 -- 12% OF THE POPULATION COVERS 78% OF THE SCREEN
#
#    Measured on gs19's own export (scripts/analyze_splat_ply.py):
#        face-based    5,254,112 gaussians   sigma_px q10/50/90 = 0.39 / 1.16 / 4.24
#        vertex layer    702,813 gaussians   sigma_px q10/50/90 = 3.62 / 6.01 / 9.45
#    Smear area goes as sigma^2, so at the medians the vertex layer paints roughly
#    3.5x more of the frame than all 5.25M face Gaussians combined. The sharp layer is
#    already there; it is being covered up.
#
#    Shrinking it is nearly free, from the definition rather than a guess: a vertex
#    Gaussian's coverage contribution AT ITS OWN ANCHOR is opacity * exp(0), independent
#    of its size (_compute_coverage_density). Size only buys cross-contribution to nearby
#    face centroids -- a wide, low-opacity smear, which is coverage bought with blur.
#    And there is room: gs19 measured mesh-vertex coverage at median 3.088 against a
#    target of 1.0, ~3x oversupplied. 0.5 leaves ~1.5x.
#
#    If the run's final "mesh vertices" coverage row drops materially below gs19's
#    94.13%, come back up to 0.7 rather than pushing further down.
# ---------------------------------------------------------------------------------------
#
# Usage (inside the drawer_splat env):
#   bash scripts/run_table_gs20.sh
#   bash scripts/run_table_gs20.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs20}"

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
    \
    `# ==== CHANGE 2 of 2: population ceiling ==================================` \
    `# gs19 finished with 6.02M rows at downscale 4. At full resolution the render  ` \
    `# and gradient buffers are 16x larger per frame, and two earlier full-res runs  ` \
    `# died of CUDA OOM hours in. gs12 is the proven point on a 32 GB card: 5.48M     ` \
    `# rows at downscale 1. Cap below that and let culling do the rest.               ` \
    `#                                                                                ` \
    `# If this binds every cycle, the fix is a smaller seed (init_copies_per_face) or  ` \
    `# a higher densify_grad_thresh -- NOT a lower resolution and NOT a higher cap.    ` \
    --pipeline.model.max_gaussians 5500000 `# gs19: unset` \
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
