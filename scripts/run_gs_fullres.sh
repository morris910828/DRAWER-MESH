#!/bin/bash
set -e

DATA_DIR="/opt/disk/drawer_dataset/studio/studio_h200ver"
SDF_DIR="${DATA_DIR}/studio_h200ver_sdf_recon"
SPLAT_DIR="/workspace/DRAWER/splat"

MESH_OBJ="${SDF_DIR}/texture_mesh/mesh-simplify.obj"
GS_EXP="studio_h200ver_gs_eroded"
EXTRA_INFO_DIR="${DATA_DIR}/gs_extra_info_eroded"

mkdir -p ${EXTRA_INFO_DIR}

echo "=============================="
echo "Step 1: GS training (eroded masks, fullres)"
echo "  mesh:    ${MESH_OBJ}"
echo "  output:  ${DATA_DIR}/${GS_EXP}"
echo "  masks:   masks_eroded/ (8px erosion, 3703x2083)"
echo "=============================="

cd ${SPLAT_DIR}
conda run -n drawer_splat python nerfstudio/scripts/train.py splatfacto_on_mesh_uc \
    --vis tensorboard \
    --output-dir ${DATA_DIR} \
    --experiment-name ${GS_EXP} \
    --pipeline.model.mesh_area_to_subdivide 2e-5 \
    --pipeline.model.acm_lambda 1.0 \
    --pipeline.model.elevate_coef 1.5 \
    --pipeline.model.upper_scale 1.5 \
    --pipeline.model.continue_cull_post_densification True \
    --pipeline.model.gaussian_save_extra_info_path ${EXTRA_INFO_DIR}/${GS_EXP}.pt \
    --pipeline.model.mesh_depth_lambda 1.0 \
    --pipeline.model.reset_alpha_every 30 \
    --pipeline.model.use_scale_regularization True \
    --pipeline.model.max_gauss_ratio 1.5 \
    --max-num-iterations 30000 \
    panoptic-data \
    --data ${DATA_DIR} \
    --mesh_gauss_path ${MESH_OBJ} \
    --mesh_area_to_subdivide 2e-5 \
    --mesh_depth True \
    --downscale_factor 1 \
    --num_max_image 2000

echo "=============================="
echo "Step 2: Export Gaussian Splat to .ply"
echo "=============================="

CONFIG_PATH=$(find ${DATA_DIR}/${GS_EXP} -name "config.yml" | sort | tail -1)
echo "Using config: ${CONFIG_PATH}"

EXPORT_DIR="${DATA_DIR}/${GS_EXP}/export_ply"
mkdir -p ${EXPORT_DIR}

cd ${SPLAT_DIR}
conda run -n drawer_splat python nerfstudio/scripts/exporter.py gaussian-splat \
    --load-config ${CONFIG_PATH} \
    --output-dir ${EXPORT_DIR}

echo "=============================="
echo "Done. PLY exported to: ${EXPORT_DIR}"
echo "=============================="
