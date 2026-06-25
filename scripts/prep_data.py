"""prep_data.py — one-time prep before the ablation array.

For each dataset: download it (rel-stack is not cached), materialize the official entity-task tables,
populate the **graph-materialization cache** (so the array's per-job ``build_gloss_graph`` is a fast
load, not a full rebuild of millions of rows), and build the frozen **Qwen schema cache** (column-name
embeddings). After this the ablation array does no downloads, no LM forward passes, and no heavy graph
rebuilds.

    .venv/bin/python scripts/prep_data.py                       # rel-f1, rel-stack, rel-trial
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["rel-f1", "rel-stack", "rel-trial"])
    ap.add_argument("--encoder", default="qwen")
    ap.add_argument("--d-text", type=int, default=2560)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    from gloss.data.graph import build_gloss_graph
    from gloss.eval.ablation import entity_tasks
    from gloss.train.finetune import name_embeddings

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
        emb = name_embeddings(bundle, ds, encoder=args.encoder, d_text=args.d_text)
        print(f"[{ds}] schema cache: name_emb {tuple(emb.shape)} (encoder={args.encoder})", flush=True)
    print("PREP_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
