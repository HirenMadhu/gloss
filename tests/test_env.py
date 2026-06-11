"""Environment smoke test — the 'test whatever is required' gate.

Asserts the provisioned .venv can import the whole HALOS stack and that torch is the expected
CUDA / cxx11-ABI build (so the from-source flash-attn build will be ABI-compatible). flash-attn
itself is xfail until `scripts/build_flash_attn.sh` succeeds (it is off the Phase 0-3 critical path;
its only consumer is the frozen Qwen encoder).
"""
import importlib

import pytest

# (import name, human label) for every dependency Phases 0-3 rely on.
REQUIRED = [
    ("torch", "torch"),
    ("torch_geometric", "torch_geometric"),
    ("torch_frame", "pytorch-frame"),
    ("relbench", "relbench"),
    ("sentence_transformers", "sentence-transformers"),
    ("transformers", "transformers"),
    ("lightgbm", "lightgbm"),
    ("shap", "shap"),
    ("sklearn", "scikit-learn"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
    ("tqdm", "tqdm"),
    ("pytorch_lightning", "pytorch-lightning"),
    ("hydra", "hydra-core"),
    ("wandb", "wandb"),
    ("torchmetrics", "torchmetrics"),
]


@pytest.mark.parametrize("module_name,label", REQUIRED, ids=[lbl for _, lbl in REQUIRED])
def test_required_dependency_imports(module_name, label):
    mod = importlib.import_module(module_name)
    assert mod is not None, f"{label} imported as None"


def test_torch_build_is_cuda12_cxx11abi():
    import torch

    # cu128 build → torch.version.cuda == "12.8"; tolerate point releases within 12.x.
    assert torch.version.cuda is not None, "torch is not a CUDA build"
    assert torch.version.cuda.startswith("12."), f"unexpected CUDA: {torch.version.cuda}"
    # Must be the cxx11-ABI=True build so the from-source flash-attn links correctly.
    assert torch._C._GLIBCXX_USE_CXX11_ABI is True, "torch is not the cxx11-ABI build"


def test_pyg_version_supports_torch28():
    import torch_geometric

    major, minor = (int(x) for x in torch_geometric.__version__.split(".")[:2])
    assert (major, minor) >= (2, 6), f"torch_geometric too old: {torch_geometric.__version__}"


@pytest.mark.xfail(reason="flash-attn is built from source on SLURM; off the Phase 0-3 critical path", strict=False)
def test_flash_attn_available():
    import flash_attn  # noqa: F401
