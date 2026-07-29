"""build_schema_cache.py — precompute + cache the per-dataset column-name embeddings (the frozen table
MoRE's router routes on), so the ablation array does **no** LM forward passes at train time.

    .venv/bin/python scripts/build_schema_cache.py                       # rel-f1, rel-trial, rel-event (Qwen)
    .venv/bin/python scripts/build_schema_cache.py --datasets rel-f1     # one dataset

The default list is the three leaderboard databases (``eval/ablation.py:LEADERBOARD_TASKS``), not rel-stack:
rel-stack has no ``LEADERBOARD_TASKS`` entry and nothing in this repo runs it.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rel-f1", "rel-trial", "rel-event"])
    ap.add_argument("--encoder", default="qwen", help="'qwen' | registry label | HF model id")
    ap.add_argument("--d-text", type=int, default=2560)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from gloss.data.graph import build_gloss_graph
    from gloss.text.schema import (
        build_column_name_embeddings,
        build_role_name_embeddings,
        build_table_name_embeddings,
    )
    from gloss.train.finetune import _name_encoder

    for ds in args.datasets:
        bundle = build_gloss_graph(ds)
        # ONE EmbeddingCache per (dataset, encoder), content-hash keyed per (kind, text) — so the
        # three tables share a cache file and re-running only encodes strings that are missing.
        enc = _name_encoder(ds, encoder=args.encoder, d_text=args.d_text)

        col = build_column_name_embeddings(bundle, enc, kind="query")
        # P0.4: the row level's counterparts. Name-DERIVED so an unseen schema works on day one —
        # `n_tables` and K index frozen data, never a weight shape (changes.md §0).
        tab = build_table_name_embeddings(bundle, enc, kind="query")
        role = build_role_name_embeddings(bundle, enc, kind="query")

        print(f"{ds}: col {tuple(col.shape)}  table {tuple(tab.shape)}  role {tuple(role.shape)}"
              f"  (encoder={args.encoder}, K={bundle.num_roles})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
