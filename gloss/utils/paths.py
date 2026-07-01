"""Single source of truth for the cached-artifact locations.

Two caches live outside the repo tree on scratch (see ``scripts/env.sh``):
  * the **graph-materialization cache** (``build_gloss_graph`` writes a fast-load bundle per dataset), and
  * the **frozen schema cache** (per-column name embeddings, keyed by the frozen encoder label).

Both default to repo-relative ``data/`` so hermetic tests and env-less local runs keep working, but are
redirected to scratch when ``GLOSS_GRAPH_CACHE`` / ``GLOSS_SCHEMA_CACHE`` are set. Resolving them here (not
in each caller) guarantees prep, the ablation runner, and finetune all read/write the *same* location.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def graph_cache_dir(dataset: str) -> Path:
    """Directory holding ``dataset``'s materialized graph bundle."""
    root = os.environ.get("GLOSS_GRAPH_CACHE") or (REPO / "data" / "graph_cache")
    return Path(root) / dataset


def schema_cache_path(dataset: str, safe_encoder: str) -> Path:
    """File holding ``dataset``'s frozen ``[C, d_text]`` name table for a given (path-safe) encoder label."""
    root = os.environ.get("GLOSS_SCHEMA_CACHE") or (REPO / "data" / "schema_cache")
    return Path(root) / dataset / f"name_emb_{safe_encoder}.pt"
