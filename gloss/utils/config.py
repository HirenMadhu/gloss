"""Config loading — thin OmegaConf helpers so standalone scripts and Hydra share one schema.

Scripts merge a named config (e.g. configs/rel-f1.yaml) over configs/default.yaml, then apply
CLI dotlist overrides. We deliberately avoid a hard Hydra dependency at import time so unit tests
can build configs in-memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(
    name: str = "default",
    *,
    overrides: Sequence[str] | None = None,
    merge_default: bool = True,
) -> DictConfig:
    """Load ``configs/<name>.yaml`` merged over ``configs/default.yaml`` + dotlist overrides.

    >>> cfg = load_config("rel-f1", overrides=["docs.regime=null"])
    """
    default_path = CONFIGS_DIR / "default.yaml"
    cfg = OmegaConf.load(default_path) if (merge_default and default_path.exists()) else OmegaConf.create({})
    if name and name != "default":
        named = CONFIGS_DIR / f"{name}.yaml"
        if not named.exists():
            raise FileNotFoundError(f"config not found: {named}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(named))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg  # type: ignore[return-value]


def to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
