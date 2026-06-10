"""Global, reproducible seeding — implementation.md §0 ('global seed everywhere')."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 0, deterministic: bool = False) -> int:
    """Seed python, numpy, and torch (CPU+CUDA). Returns the seed for logging.

    ``deterministic=True`` also forces deterministic cuDNN — slower, used in tests / audits where
    bit-reproducibility matters more than speed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:  # torch optional for pure-data utilities
        pass
    return seed


def new_rng(seed: int) -> np.random.Generator:
    """A fresh, independent numpy Generator (preferred over global state for samplers)."""
    return np.random.default_rng(seed)
