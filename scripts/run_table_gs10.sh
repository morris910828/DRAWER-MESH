#!/usr/bin/env bash
# Gaussian-splat-on-mesh training for data/table, run 10.
#
# Baseline is run_table_gs8.sh, the last run that actually executed. gs9 was written but
# never run; this supersedes it. Lines are marked against gs8's value, and where gs10
# differs from what gs9 proposed, that is marked too.
#
# ---------------------------------------------------------------------------------------
# WHAT gs8's EXPORT MEASURED  (scripts/analyze_splat_ply.py on table_gs8/export_ply)
#
#   [3]  1,282,316 rows at the base floor, in-plane aspect median 1.000, 96.8% under 1.001
#   [2]  face-centroid coverage piled up exactly on 0.950 (q5/q25/q50 all 0.950)
#   [1]  area coverage 92.56% at 1.53x redundancy, 8.39% of faces under half their area
#   [4]  on-screen sigma median 2.64 px against roughly 2 px text strokes
#   [3b] the mesh's own triangles: aspect median 1.17, q90 1.39, q99 1.68
#
# ROOT CAUSE OF THE CIRCLES: self.base_floor_xy is cv_radius_base_floor.expand(-1, 2) --
# ONE scalar per face written into both the x and y bound -- so any base row the floor
# binds has sx == sy by definition, not by the optimizer's choice. min_scale_frac and
# max_gauss_ratio (gs9's ellipticity fix) cannot reach it: floored base rows take a
# different branch of the scales property entirely, which neither value feeds.
#
# WHAT THIS RUN WILL AND WILL NOT FIX. Measured on gs8's own mesh, not assumed:
#   - base plates become elliptical and aligned to their triangles. But this mesh's
#     triangles are nearly equilateral (aspect median 1.17), so the ellipses are mild.
#     Expect the shape symptom to go away, not a dramatic visual change.
#   - corner coverage improves from 1.063 sigma (0.569 of peak) to 1.000 sigma (0.607),
#     about +7% of Gaussian value at every triangle corner on the mesh.
#   - base footprint area lands at 0.991x the current circle, so overlap and blur are
#     essentially UNCHANGED by the ellipse alone on this mesh. The appearance problem is
#     addressed by the floor-release settings below instead, not by the ellipse.
# ---------------------------------------------------------------------------------------
#
# Requires anisotropic_base_floor.patch, applied on top of adaptive_coverage_floor.patch:
#   cd /workspace/DRAWER-MESH && git apply -v anisotropic_base_floor.patch
#
# Usage (on the DGX, inside the drawer_splat env):
#   bash scripts/run_table_gs10.sh
#   bash scripts/run_table_gs10.sh --export-only

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/DRAWER-MESH/data/table}"
SPLAT_DIR="${SPLAT_DIR:-/workspace/DRAWER-MESH/splat}"
EXP="${EXP:-table_gs10}"

MESH="${DATA_DIR}/table_sdf_ds1/texture_mesh/mesh_table-only-400k_world.obj"
AREA=1e-4

EXTRA_INFO_DIR="${DATA_DIR}/gs_extra_info"
EXTRA_INFO="${EXTRA_INFO_DIR}/${EXP}.pt"
mkdir -p "${EXTRA_INFO_DIR}"

cd "${SPLAT_DIR}"

if [[ "${1:-}" != "--export-only" ]]; then
PYTHONPATH="${SPLAT_DIR}" python nerfstudio/scripts/train.py splatfacto_on_mesh_uc \
    --vis tensorboard \
    --output-dir "${DATA_DIR}" \
    --experiment-name "${EXP}" \
    --max-num-iterations 30000 \
    \
    `# ==== ELLIPTICAL BASE FLOOR ==============================================` \
    `# Floors each base plate to its face Steiner circumellipse -- the minimum-area  ` \
    `# ellipse through all three vertices -- instead of one isotropic radius on both  ` \
    `# axes, and pins the plate in-plane rotation to that ellipse major axis so the   ` \
    `# (x, y) bounds describe the ellipse they are meant to. Verified numerically on  ` \
    `# 120k synthetic triangles: every vertex lands at Mahalanobis^2 = 1 (max error   ` \
    `# 2.7e-6) and ellipse area / triangle area = 2.4184 for EVERY shape, matching     ` \
    `# theory 4pi/3sqrt3. On THIS mesh that buys +7% corner value at 0.99x the area.   ` \
    --pipeline.model.anisotropic_base_floor True `# NEW` \
    \
    `# Without this the metric collapses every Gaussian to min(sx, sy), so an ellipse ` \
    `# covering its triangle scores as a circle of its NARROW axis, the face reads as  ` \
    `# under-covered, the floor re-engages and squeezes the plate back to round. That  ` \
    `# loop returns the base layer to circles regardless of the scale bounds. Measured ` \
    `# gain on gs8 is modest on its own -- floor-release rate at target 1.0 goes from  ` \
    `# 15.19% to 17.84% -- but the loop has to be cut for the line above to hold.      ` \
    --pipeline.model.anisotropic_coverage_density True `# NEW` \
    \
    `# scale_reg penalises in-plane aspect above max_gauss_ratio at weight 1.0 across  ` \
    `# ALL rows. With one base row per face the base layer dominates that mean, so     ` \
    `# leaving it on both fights the new floor and drowns out the detail rows the term ` \
    `# is actually for.                                                                ` \
    --pipeline.model.scale_reg_exclude_base True `# NEW` \
    \
    `# Vertices land exactly on the 1-sigma ellipse, 0.607 of peak. Lower toward 0.85  ` \
    `# if gs10 export still shows short corners; it grows every floored plate, so go    ` \
    `# back up if blur is the bigger complaint.                                         ` \
    --pipeline.model.base_floor_corner_sigma 1.0 `# NEW` \
    \
    `# Insurance against marching-cubes slivers only. This mesh q99 triangle aspect is ` \
    `# 1.68, so nothing comes close to 8:1; the cap only ever WIDENS the minor axis, so ` \
    `# corner coverage survives it. The startup log prints the real hit rate.           ` \
    --pipeline.model.base_floor_max_aspect 8.0 `# NEW` \
    \
    `# Superseded: anisotropic_base_floor answers the same question (how far must a    ` \
    `# base plate reach to cover its own face) per direction rather than with a single  ` \
    `# radius, and reaches the same 1.000 sigma at 0.877x its footprint area.           ` \
    --pipeline.model.base_floor_reaches_corners False `# gs8: False, gs9 proposed True` \
    \
    `# ==== FLOOR RELEASE -- this is what the appearance depends on ============` \
    `# gs9 proposed 1.5. Unreachable by construction: the metric is a SUM and a lone   ` \
    `# base plate contributes at most its opacity floor, 0.95, so 1.5 keeps essentially ` \
    `# every face floored -- the state gs8 was already stuck in, where 91.9% of         ` \
    `# face-based rows still render at floor size. Since a floored plate is as large as  ` \
    `# its triangle (2.64 px on screen, against 2 px text strokes), that is the single   ` \
    `# biggest reason detail is not resolving. 1.0 asks a face to clear what its base     ` \
    `# alone scores before the floor is dropped, without asking the impossible.           ` \
    `# WATCH the log line 'adaptive coverage floor: N/M faces still floored'. If it       ` \
    `# never falls below ~90%, lower this to 0.9 before changing anything else.           ` \
    --pipeline.model.coverage_floor_target 1.0 `# gs8: 0.9, gs9 proposed 1.5` \
    \
    `# gs9 proposed 4 and gs9 was right, for a reason gs9 did not state: a floor can    ` \
    `# only be released if the detail layer covers the face WITHOUT it, and gs8 carried  ` \
    `# just 733,741 detail rows over 1,395,229 faces -- 0.53 per face, nowhere near      ` \
    `# enough to take over. Costs roughly 2.8M extra seed Gaussians; drop back to 2 if   ` \
    `# VRAM is tight, but then expect the floor-release rate to stay low.                ` \
    --pipeline.model.init_copies_per_face 4 `# gs8: 2` \
    \
    `# Let repair keep up with damage. Raise further if the log shows 'qualified' stuck ` \
    `# at the cap every cycle.                                                           ` \
    --pipeline.model.coverage_rescue_max_frac 0.01 `# gs8: 0.002` \
    \
    `# ==== DETAIL LAYER (resolution) ==========================================` \
    `# Aspect is capped at upper_scale / min_scale_frac because both bounds are per-axis ` \
    `# and isotropic; gs8 detail rows sit at exactly 10.000 for both p90 and p99, i.e.   ` \
    `# clamped rather than converged. 0.02 raises that to 25:1 and, more importantly,    ` \
    `# lets a detail Gaussian shrink far enough below 2.64 px to resolve text at all.    ` \
    --pipeline.model.min_scale_frac 0.02 `# gs8: 0.05` \
    \
    `# Detail median aspect is already 1.834, so at 5.0 the regulariser was pushing a   ` \
    `# large part of the detail layer back toward round. The needle cull follows it at   ` \
    `# max_gauss_ratio * 4, moving from 20:1 to 40:1; base rows are never culled.        ` \
    --pipeline.model.max_gauss_ratio 10.0 `# gs8: 5.0` \
    \
    `# ==== unchanged from gs8 ==================================================` \
    --pipeline.model.adaptive_coverage_floor True \
    --pipeline.model.coverage_rescue_every 100 \
    --pipeline.model.coverage_rescue_thresh 0.9 \
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
    --downscale_factor 1 \
    --num_max_image 50 \
    --orientation_method none \
    --center_poses False \
    --auto_scale_poses False
fi

PYTHONPATH="${SPLAT_DIR}" python nerfstudio/scripts/exporter.py gaussian-splat \
    --load-config "${DATA_DIR}/${EXP}/config.yml" \
    --output-dir "${DATA_DIR}/${EXP}/export_ply"

echo "=== done -> ${DATA_DIR}/${EXP}/export_ply ==="
echo
echo "Check it with:"
echo "  python scripts/analyze_splat_ply.py ${DATA_DIR}/${EXP}/export_ply/splat.ply \\"
echo "      --transforms ${DATA_DIR}/transforms.json"
echo
echo "What to look for, against gs8:"
echo "  [3b] floored rows aspect should track their faces (gs8: 1.000 vs 1.168)"
echo "  [3]  'at base floor' count should DROP -- that is the floor being released"
echo "  [4]  on-screen sigma median should fall below gs8 2.64 px"
echo "  [2]  centroid coverage should stop piling up exactly on 0.950"
