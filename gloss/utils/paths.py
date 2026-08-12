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


#: The cell-VALUE text encoder whose graph caches live at the un-suffixed root. Historically the only
#: one: ``build_gloss_graph`` defaulted to ``HashTextEmbedder`` and no caller ever overrode it, so every
#: cache written before 2026-08-12 holds 32-d pseudo-random vectors for free-text cells.
LEGACY_VALUE_ENCODER = "hash"


def graph_cache_dir(dataset: str, text_encoder: str = LEGACY_VALUE_ENCODER) -> Path:
    """Directory holding ``dataset``'s materialized graph bundle, keyed by its cell-text encoder.

    ``text_encoder`` names the encoder used for free-text **cell values** (not column names — those
    live in :func:`schema_cache_path`). It is part of the path because it changes the *content* of
    every text column: a bundle built with ``hash`` and one built with ``minilm`` are different data,
    not different formats, and silently reusing one for the other would move every measured result.

    ``hash`` keeps the historical un-suffixed layout so every previously written cache — and therefore
    every reported grid/leaderboard number — stays readable at exactly the path it was written to.
    """
    root = os.environ.get("GLOSS_GRAPH_CACHE") or (REPO / "data" / "graph_cache")
    base = Path(root) if text_encoder == LEGACY_VALUE_ENCODER else Path(root) / text_encoder
    return base / dataset


def ckpt_dir(run: str = "") -> Path:
    """Root for pretraining checkpoints (``$GLOSS_CKPT_DIR``, default scratch — never ``$HOME``).

    A 520M-param model's AdamW state is ~7 GB per checkpoint and home's quota is ~125 GiB, so the
    default deliberately points at scratch60 rather than the repo or the home cache.
    """
    root = os.environ.get("GLOSS_CKPT_DIR") or (REPO / "data" / "ckpt")
    return Path(root) / run if run else Path(root)


def schema_cache_path(dataset: str, safe_encoder: str) -> Path:
    """File holding ``dataset``'s frozen ``[C, d_text]`` name table for a given (path-safe) encoder label."""
    root = os.environ.get("GLOSS_SCHEMA_CACHE") or (REPO / "data" / "schema_cache")
    return Path(root) / dataset / f"name_emb_{safe_encoder}.pt"
