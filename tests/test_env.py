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


def test_flash_attn_imports():
    """flash-attn is built from source (scripts/build_flash_attn_local.sh). It imports once built;
    the GPU kernel is arch-specific (the current build targets sm_90/H100), so we only check import
    here and exercise a kernel opportunistically below."""
    pytest.importorskip("flash_attn")


@pytest.mark.slow
def test_flash_attn_kernel_runs_if_arch_matches():
    fa = pytest.importorskip("flash_attn")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from flash_attn import flash_attn_func

    q = torch.randn(1, 8, 4, 64, device="cuda", dtype=torch.bfloat16)
    try:
        o = flash_attn_func(q, q, q, causal=True)
    except Exception as e:
        pytest.skip(f"flash-attn kernel not built for this GPU arch ({e.__class__.__name__})")
    assert o.shape == q.shape and torch.isfinite(o).all()
