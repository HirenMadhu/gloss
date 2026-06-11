#!/bin/bash -l
# Build flash-attn 2.8.3 from SOURCE on the CURRENT (interactive A40) node — no SLURM.
# Same recipe as scripts/build_flash_attn.sh but MAX_JOBS=8 (this node has 8 CPUs) and a login shell
# so the module system is available. Run in the background:
#   bash -l scripts/build_flash_attn_local.sh > logs/flash_build_local.out 2>&1 &
# Success = "FLASH_BUILD_OK" at the end.
set -uo pipefail
mkdir -p ./logs
ml load CUDA/12.1.1 GCC/12.2.0
cd /gpfs/milgram/project/ying_rex/hm638/gloss
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}
echo "CUDA_HOME=$CUDA_HOME"; nvcc --version | tail -2
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'abi', torch._C._GLIBCXX_USE_CXX11_ABI)"

rm -f .venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so

BUILD=/tmp/fa_build_local_$$
rm -rf "$BUILD"; mkdir -p "$BUILD"; cd "$BUILD"
git clone --depth 1 -b v2.8.3 --recursive https://github.com/Dao-AILab/flash-attention.git 2>&1 | tail -3
cd flash-attention
echo "cutlass present: $(ls csrc/cutlass/include >/dev/null 2>&1 && echo yes || echo NO)"

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
export FLASH_ATTN_CUDA_ARCHS="80;90"
export MAX_JOBS=8
export NVCC_THREADS=2

echo "=== compiling flash-attn 2.8.3 from source (sm_86;sm_90, abiTRUE, MAX_JOBS=8)... ~30-60 min ==="
cd /gpfs/milgram/project/ying_rex/hm638/gloss
uv pip install --python .venv/bin/python3 --no-build-isolation --no-deps --reinstall "$BUILD/flash-attention"
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
