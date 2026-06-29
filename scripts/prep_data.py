"""prep_data.py — one-time prep before the ablation array.

For each dataset: download it, materialize the official entity-task tables, populate the
**graph-materialization cache** (so the array's per-job ``build_gloss_graph`` is a fast load, not a full
rebuild of millions of rows), and build the frozen **schema cache** (column-name embeddings). After this
the ablation array does no downloads, no LM forward passes, and no heavy graph rebuilds.

The frozen encoder (e.g. ``harrier`` = Gemma-3-27B, ~51GB) is built **once** and reused across datasets:
loading it per-dataset blew the SLURM ``--mem`` cap (the mmap'd weights accumulate in the cgroup's
page-cache accounting). Per-dataset cache files are still written separately (keyed by encoder label).

    .venv/bin/python scripts/prep_data.py                       # rel-f1, rel-trial, rel-event
"""
from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rel-f1", "rel-trial", "rel-event"])
    ap.add_argument("--encoder", default="qwen")
    ap.add_argument("--d-text", type=int, default=2560)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph
    from gloss.eval.ablation import entity_tasks
    from gloss.text.cache import EmbeddingCache, HashEncoder, QwenEncoder, make_text_encoder
    from gloss.text.schema import build_column_name_embeddings

    # Build the (expensive) frozen encoder ONCE and reuse it across datasets; the underlying model loads
    # lazily on first use. Per-dataset cache files mirror finetune._name_encoder's path so the ablation
    # array reads exactly these (it never reloads the model — every column name is already cached).
    if args.encoder == "hash":
        base_encoder = HashEncoder(dim=args.d_text)
    elif args.encoder == "qwen":
        base_encoder = QwenEncoder("Qwen/Qwen3-Embedding-4B")
    else:
        base_encoder = make_text_encoder(args.encoder)
    safe = args.encoder.replace("/", "__")

    for ds in args.datasets:
        print(f"[{ds}] downloading dataset ...", flush=True)
        get_dataset(ds, download=True).get_db(upto_test_timestamp=False)
        for name in entity_tasks(ds):
            task = get_task(ds, name, download=True)
            for split in ("train", "val", "test"):
                task.get_table(split)
            print(f"  [{ds}/{name}] task tables ready", flush=True)
        cache_dir = str(REPO / "data" / "graph_cache" / ds)
        bundle = build_gloss_graph(ds, cache_dir=cache_dir)
        print(f"  [{ds}] graph cache -> {cache_dir}", flush=True)
        cache_path = REPO / "data" / "schema_cache" / ds / f"name_emb_{safe}.pt"
        enc = EmbeddingCache(base_encoder, cache_path)
        emb = build_column_name_embeddings(bundle, enc, kind="query")
        print(f"[{ds}] schema cache: name_emb {tuple(emb.shape)} -> {cache_path} "
              f"(encoder={args.encoder})", flush=True)
        # Free this dataset's graph/embeddings (the encoder/model is kept, reused next loop).
        del emb, bundle, enc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    print("PREP_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
