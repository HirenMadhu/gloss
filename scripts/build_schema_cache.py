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
    from gloss.train.finetune import name_embeddings

    for ds in args.datasets:
        bundle = build_gloss_graph(ds)
        emb = name_embeddings(bundle, ds, encoder=args.encoder, d_text=args.d_text)
        print(f"{ds}: name_emb {tuple(emb.shape)} cached (encoder={args.encoder})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
