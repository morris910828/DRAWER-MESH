#!/usr/bin/env bash
# DRAWER SDF environment for RTX 5090 / Blackwell sm_120 / CUDA 12.8.
# Replaces env.sh (which is pinned to PyTorch 2.1.2+cu118 and does not run on sm_120).
#
# Usage:  bash sdf/env_5090.sh   (run from the repo root)
# Assumes: CUDA 12.8 toolkit at /usr/local/cuda-12.8, gcc 11+, conda available.

set -euo pipefail

# -- toolchain -----------------------------------------------------------------
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"          # sm_120 (RTX 5090)
export TCNN_CUDA_ARCHITECTURES=120
export MAX_JOBS=8                            # cap parallel nvcc to avoid OOM

# -- conda env -----------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create --name drawer_sdf -y python=3.11
conda activate drawer_sdf

# NOTE: setuptools<81 — newer setuptools dropped pkg_resources, which
# tiny-cuda-nn's setup.py still imports.
python -m pip install -U pip wheel "setuptools<81" ninja packaging

# -- PyTorch (cu128, native sm_120) --------------------------------------------
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Quick sanity check before slow source builds
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "device:", torch.cuda.get_device_name(0))
PY

# -- source builds (need TORCH_CUDA_ARCH_LIST set) -----------------------------
# tiny-cuda-nn: master CMakeLists has LATEST_SUPPORTED_CUDA_ARCHITECTURE=120 for CUDA 12.8
pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# nvdiffrast: pure source build, sm_120 OK
pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git"

# pytorch3d: no cu128 wheels yet, build from source
pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# -- prebuilt cu128 wheels -----------------------------------------------------
# torch-scatter for torch 2.8 / cu128
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# kaolin 0.18.0 official cu128 wheel
pip install kaolin==0.18.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

# xformers intentionally NOT installed in drawer_sdf:
#   - Stage 1 (Marigold + BakedSDF) does not require it.
#   - Without --no-deps, xformers force-upgrades torch (e.g. 0.0.35 pulls torch 2.11),
#     which breaks the ABI of tiny-cuda-nn / nvdiffrast / pytorch3d we just built.
#   - If you ever need it: pip install --no-deps xformers==0.0.33.post1
#     (0.0.33.post1 is built against torch 2.8 ABI; later releases require torch ≥2.9).

# -- python deps (relaxed bounds vs the original env.sh) -----------------------
pip install torchmetrics[image]
pip install accelerate diffusers tokenizers transformers
pip install omegaconf tabulate pandas
pip install scikit-learn
pip install "imageio[ffmpeg]"
pip install hydra-core hydra-submitit-launcher visdom
pip install transformations

# Kept (still consumed by sdfstudio fork code)
pip install torchtyping "typeguard<3"
pip install --upgrade tyro

# Dropped vs original:
#   - functorch  (merged into torch>=2.0; nothing imports it in sdf/)

# -- this repo (sdfstudio fork) ------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
pip install --no-build-isolation -e "${SCRIPT_DIR}"

echo
echo "=== drawer_sdf install complete ==="
python - <<'PY'
import torch, tinycudann, nvdiffrast.torch as nvd, pytorch3d, kaolin, torch_scatter, nerfstudio
print("torch        :", torch.__version__)
print("tinycudann   :", tinycudann.__version__ if hasattr(tinycudann,"__version__") else "(loaded)")
print("nvdiffrast   :", "(loaded)")
print("pytorch3d    :", pytorch3d.__version__)
print("kaolin       :", kaolin.__version__)
print("torch_scatter:", torch_scatter.__version__)
print("nerfstudio   :", nerfstudio.__version__ if hasattr(nerfstudio,"__version__") else "(loaded)")
x = torch.zeros(1024, device="cuda"); x += 1
print("cuda kernel  :", x.sum().item())  # must equal 1024.0
PY
