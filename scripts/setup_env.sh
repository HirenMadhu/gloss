#!/bin/bash
# Staged uv install for gloss/HALOS. Torch + PyG-ecosystem + relbench come from custom indexes, so we
# install in stages with explicit index URLs rather than one `uv pip install -e .` (which would pull the
# CPU torch wheel from PyPI). Run from repo root with the venv already created (`uv venv --python 3.12 .venv`).
#   bash scripts/setup_env.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python

echo "=== [1/4] torch 2.8.0 (+cu128) ==="
uv pip install --python "$PY" torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# NOTE: prebuilt pyg_lib / torch_scatter wheels need glibc >=2.29/2.32; this cluster is el8 (glibc 2.28),
# so they fail to load and PyG auto-disables them. We rely on PyG 2.8 native ops
# (`torch_geometric.utils.scatter`) + our own leakage-safe temporal sampler instead. If neighbor-sampling
# speed ever matters, build pyg_lib/torch_scatter FROM SOURCE on el8 (same approach as build_flash_attn.sh).

echo "=== [2/3] relational/tabular + text + audit + harness ==="
uv pip install --python "$PY" \
    torch_geometric==2.8.0 \
    relbench==2.1.2 pytorch-frame==0.3.0 \
    "sentence-transformers>=3.0" "transformers>=4.51.2" \
    "lightgbm>=4.0" "shap>=0.45" "scikit-learn<=1.6.1" \
    numpy pandas pyarrow duckdb pooch pyyaml tqdm \
    pytorch-lightning "hydra-core>=1.3" wandb torchmetrics \
    pytest matplotlib ipdb

echo "=== [3/3] gloss (editable, no deps) ==="
uv pip install --python "$PY" --no-deps -e .

echo "=== import smoke check ==="
"$PY" - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import torch, torch_geometric, torch_frame, relbench, sentence_transformers, lightgbm, shap
import pytorch_lightning as pl
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__, "| pytorch_frame", torch_frame.__version__, "| pl", pl.__version__)
print("ENV_OK")
PY
