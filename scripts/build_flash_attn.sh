#!/bin/bash
#SBATCH --job-name=gloss_build_flash_attn
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --output=./logs/slurm/build/%x_%j.out
#SBATCH --error=./logs/slurm/build/%x_%j.err
#SBATCH --mail-user=hiren.madhu@yale.edu
#SBATCH --mail-type=END,FAIL

# Build flash-attn 2.8.3 from SOURCE for this cluster (el8 / glibc 2.28), matching the in-place
# torch 2.8.0+cu128 (cxx11abi=TRUE). Prebuilt wheels need glibc 2.32 -> won't load. Build for BOTH
# A40 (sm_86) and H100 (sm_90) so the cached encoder + any unbiased attention path run on either.
# Adapted from ../HypPostTraining/scripts/build_flash_attn.sh.
#   sbatch scripts/build_flash_attn.sh
# Success = "FLASH_BUILD_OK" at the end of the .out.
#
# NOTE: flash-attn is OFF the Phase 0-2 critical path. HALOS relational attention uses additive biases
# (B_time + B_hop) that flash-attn's API can't take, so we use PyTorch SDPA there. Flash-attn only speeds
# the one-time Qwen3-Embedding-4B caching pass + any future unbiased full-attention path.

mkdir -p ./logs/slurm/build
ml load CUDA/12.1.1
ml load GCC/12.2.0
cd /gpfs/milgram/project/ying_rex/hm638/gloss
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"          # uv

export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}
echo "CUDA_HOME=$CUDA_HOME"; nvcc --version | tail -2
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cxx11abi', torch._C._GLIBCXX_USE_CXX11_ABI)"

# Remove any stale prebuilt .so so it can't be picked up.
rm -f .venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so

# Clone source (+ cutlass submodule) at the pinned tag.
BUILD=/tmp/fa_build_${SLURM_JOB_ID:-manual}
rm -rf "$BUILD"; mkdir -p "$BUILD"; cd "$BUILD"
git clone --depth 1 -b v2.8.3 --recursive https://github.com/Dao-AILab/flash-attention.git 2>&1 | tail -3
cd flash-attention
echo "cutlass present: $(ls csrc/cutlass/include >/dev/null 2>&1 && echo yes || echo NO)"

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE    # MUST match torch 2.8 (abiTRUE)
export FLASH_ATTN_CUDA_ARCHS="86;90"           # A40 (sm_86) + H100 (sm_90)
export MAX_JOBS=16
export NVCC_THREADS=2

echo "=== compiling flash-attn 2.8.3 from source (sm_86;sm_90, abiTRUE)... ~30-60 min ==="
cd /gpfs/milgram/project/ying_rex/hm638/gloss
uv pip install --python .venv/bin/python3 --no-build-isolation --no-deps --reinstall \
    "$BUILD/flash-attention"
echo "uv exit=$?"

echo "=== verify import + a real flash kernel on GPU ==="
python3 - <<'PY'
import torch, flash_attn
from flash_attn import flash_attn_func
print("flash_attn", flash_attn.__version__, "imported (no glibc error)")
q = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q, causal=True)
assert o.shape == q.shape and torch.isfinite(o).all()
print("flash kernel ran, out", tuple(o.shape))
print("FLASH_BUILD_OK")
PY
rm -rf "$BUILD"
