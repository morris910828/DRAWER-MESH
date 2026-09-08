#!/usr/bin/env bash
# Finalize drawer_sdf install: skip visdom, complete remaining deps,
# remove xformers (ABI-incompatible 0.0.35 was pulled in earlier),
# install the sdfstudio fork, then smoke-test.
# Idempotent.

set -euo pipefail

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drawer_sdf

# Drop xformers — its 0.0.35 wheel needs torch 2.11; we are on 2.8.0+cu128
pip uninstall -y xformers || true

# Pin torch back to the cu128 build to be safe (no-op if already there)
pip install --force-reinstall --no-deps \
    torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Remaining deps (visdom skipped — not used; --vis wandb is configured)
pip install hydra-core hydra-submitit-launcher
pip install transformations
pip install torchtyping "typeguard<3"
pip install --upgrade tyro

# This repo (sdfstudio fork)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
pip install --no-build-isolation -e "${SCRIPT_DIR}"

echo
echo "=== drawer_sdf smoke test ==="
python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
print("cuda runtime   :", torch.version.cuda)
print("device         :", torch.cuda.get_device_name(0))

import tinycudann as tcnn
print("tinycudann     :", getattr(tcnn, "__version__", "(loaded)"))

import nvdiffrast.torch as nvd
print("nvdiffrast     : (loaded)")

import pytorch3d
print("pytorch3d      :", pytorch3d.__version__)

import kaolin
print("kaolin         :", kaolin.__version__)

import torch_scatter
print("torch_scatter  :", torch_scatter.__version__)

import nerfstudio
print("nerfstudio     :", getattr(nerfstudio, "__version__", "(loaded)"))

# Real CUDA kernel exercise on sm_120
x = torch.zeros(1024, device="cuda")
x += 1
assert x.sum().item() == 1024.0, "cuda kernel sanity check failed"
print("cuda kernel    : OK (1024.0)")

# Exercise tiny-cuda-nn on the GPU
import tinycudann as tcnn
enc = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2,
                        "log2_hashmap_size": 14, "base_resolution": 16, "per_level_scale": 1.5})
inp = torch.rand(8, 3, device="cuda")
out = enc(inp)
print("tcnn HashGrid  : output", tuple(out.shape), out.dtype)

# Exercise pytorch3d on GPU
from pytorch3d.transforms import euler_angles_to_matrix
R = euler_angles_to_matrix(torch.zeros(2, 3, device="cuda"), "XYZ")
print("pytorch3d eul  :", tuple(R.shape))

print()
print("=== ALL SMOKE TESTS PASSED ===")
PY
