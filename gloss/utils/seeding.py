"""Global seeding — one call to make a run reproducible (spec §0: 'global seeds')."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 0, *, deterministic_torch: bool = False) -> int:
    """Seed python / numpy / torch (CPU+CUDA). Returns the seed for logging.

    `deterministic_torch=True` also forces deterministic cuDNN/algorithms (slower; use in tests
    that assert exact numerical invariances, e.g. scale-equivariance).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_torch:
        # cuBLAS picks a reduction split per call from a shared workspace; without a pinned workspace
        # its GEMMs stay non-reproducible and `use_deterministic_algorithms` raises on them. Read once
        # when the cuBLAS handle is created (first matmul), so it must be set before any CUDA work —
        # which is why it lives here and not at the call site.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:  # torch always present in this project, but keep utils import-light
        pass
    return seed
