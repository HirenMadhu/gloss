"""prep_data.py — one-time prep before the ablation array.

For each dataset: download it, materialize the official entity-task tables, populate the
**graph-materialization cache** (so the array's per-job ``build_gloss_graph`` is a fast load, not a full
rebuild of millions of rows), and build the frozen **schema cache** (column-name embeddings). After this
the ablation array does no downloads, no LM forward passes, and no heavy graph rebuilds.

The frozen encoder (e.g. ``harrier`` = Gemma-3-27B, ~51GB) is built **once** and reused across datasets:
loading it per-dataset blew the SLURM ``--mem`` cap (the mmap'd weights accumulate in the cgroup's
page-cache accounting). Per-dataset cache files are still written separately (keyed by encoder label).

**There are two independent text encoders here and they do different jobs.** ``--encoder`` embeds
column / table / role **names** — tens of strings per DB, memoized to ``$GLOSS_SCHEMA_CACHE``, and the
table the MoE router routes on. ``--text-encoder`` embeds free-text **cell values** — ~96M strings
across the 7 DBs, stored densely inside the materialized TensorFrames, and therefore keyed into their
own graph cache. Until 2026-08-12 the second one was never set and silently fell back to
``HashTextEmbedder``: every cached graph held 32-d pseudo-random vectors where roughly half of each
schema's columns live (rel-trial 47/102, rel-stack 15/32, rel-f1 13/45).

    .venv/bin/python scripts/prep_data.py                       # rel-f1, rel-trial, rel-event
    .venv/bin/python scripts/prep_data.py --datasets rel-f1 --encoder qwen \
        --text-encoder minilm --no-download                     # real cell text, d=384
"""
from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _report_value_dims(bundle, ds: str) -> None:
    """Print the embedded width of each text column, so a wrong cell-text encoder is visible here.

    ``EMB_DIM=32`` means the hash fallback ran: the free-text cells are pseudo-random vectors. That is
    the exact failure this flag exists to fix, and it is invisible downstream (shapes are all valid),
    so it gets asserted at prep time instead of being discovered after a pretraining run.
    """
    from torch_frame.data.stats import StatType

    seen: dict[int, int] = {}
    for nt in bundle.node_types:
        tf = bundle.data[nt].tf
        for st, cols in tf.col_names_dict.items():
            if str(st).rsplit(".", 1)[-1] not in ("embedding", "text_embedded"):
                continue
            for c in cols:
                d = bundle.col_stats_dict[nt][c].get(StatType.EMB_DIM)
                if d is not None:
                    seen[int(d)] = seen.get(int(d), 0) + 1
    if seen:
        print(f"  [{ds}] text-column EMB_DIM histogram: {seen}", flush=True)
    else:
        print(f"  [{ds}] no free-text columns", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rel-f1", "rel-trial", "rel-event"])
    ap.add_argument("--encoder", default="qwen",
                    help="frozen encoder for column/table/role NAMES (the router's schema table)")
    ap.add_argument("--text-encoder", default="hash",
                    help="frozen encoder for free-text CELL VALUES ('minilm' | 'qwen' | HF id | "
                         "'hash'). Keys its own graph cache, so it never clobbers another encoder's.")
    ap.add_argument("--text-batch-size", type=int, default=8192,
                    help="strings per torch_frame embedder call; the value pass is GPU-bound over "
                         "tens of millions of short strings, so the 256 name-table default is far "
                         "too small here.")
    ap.add_argument("--d-text", type=int, default=2560)
    ap.add_argument("--no-download", action="store_true",
                    help="assume the DB and its task tables are already cached. Worth using on this "
                         "cluster: rel-avito / rel-stack / rel-event are SYMLINKS out of the scratch "
                         "root into ~/.cache (or ~/scratch60/relbench), so a re-download writes "
                         "through the link into home, whose quota is ~125 GiB.")
    args = ap.parse_args()
    dl = not args.no_download
    warnings.filterwarnings("ignore")

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph
    from gloss.eval.ablation import entity_tasks
    from gloss.text.cache import EmbeddingCache, HashEncoder, QwenEncoder, make_text_encoder
    from gloss.text.schema import build_column_name_embeddings
    from gloss.utils.paths import graph_cache_dir, schema_cache_path

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
        print(f"[{ds}] loading dataset (download={dl}) ...", flush=True)
        get_dataset(ds, download=dl).get_db(upto_test_timestamp=False)
        for name in entity_tasks(ds):
            task = get_task(ds, name, download=dl)
            for split in ("train", "val", "test"):
                task.get_table(split)
            print(f"  [{ds}/{name}] task tables ready", flush=True)
        cache_dir = str(graph_cache_dir(ds, args.text_encoder))
        bundle = build_gloss_graph(ds, cache_dir=cache_dir, text_encoder=args.text_encoder,
                                   text_batch_size=args.text_batch_size)
        print(f"  [{ds}] graph cache -> {cache_dir} (cell-text encoder={args.text_encoder})",
              flush=True)
        _report_value_dims(bundle, ds)
        cache_path = schema_cache_path(ds, safe)
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
