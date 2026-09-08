#!/usr/bin/env bash
# Resume drawer_sdf install from the point where env_5090.sh failed
# (setuptools >=81 removed pkg_resources, breaking tiny-cuda-nn's setup.py).
# Idempotent — safe to re-run.

set -euo pipefail

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
export TCNN_CUDA_ARCHITECTURES=120
export MAX_JOBS=8

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawer_sdf

# Downgrade setuptools so pkg_resources is importable
pip install -U "setuptools<81"
python -c "import pkg_resources; print('pkg_resources OK')"

# Source builds
pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git"
pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Prebuilt cu128 wheels
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install kaolin==0.18.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
pip install xformers --index-url https://download.pytorch.org/whl/cu128

# Python deps
pip install torchmetrics[image]
pip install accelerate diffusers tokenizers transformers
pip install omegaconf tabulate pandas
pip install scikit-learn
pip install "imageio[ffmpeg]"
pip install hydra-core hydra-submitit-launcher visdom
pip install transformations
pip install torchtyping "typeguard<3"
pip install --upgrade tyro

# This repo (sdfstudio fork)
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
