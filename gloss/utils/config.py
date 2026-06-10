"""Config loading glue.

Configs live in ``configs/*.yaml`` with the schema of implementation.md §8. We use OmegaConf (ships
with hydra-core) so plain ``load_config`` works in standalone scripts (proxy/audit) while Hydra drives
the Lightning training entry points. Env-var expansion routes large caches to scratch (see §A of the plan).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name_or_path: str, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config by bare name (resolved under ``configs/``) or explicit path.

    ``overrides`` is a dotlist (e.g. ``["docs.regime=placebo", "seed=1"]``) applied on top.
    """
    path = Path(name_or_path)
    if not path.exists():
        cand = CONFIG_DIR / name_or_path
        path = cand if cand.exists() else CONFIG_DIR / f"{name_or_path}.yaml"
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg  # type: ignore[return-value]


def scratch_dir(sub: str = "") -> Path:
    """Resolve a writable scratch directory (NOT home quota): ``$GLOSS_SCRATCH`` or ``~/scratch60``."""
    root = Path(os.environ.get("GLOSS_SCRATCH", Path.home() / "scratch60"))
    d = root / sub if sub else root
    d.mkdir(parents=True, exist_ok=True)
    return d


def to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
