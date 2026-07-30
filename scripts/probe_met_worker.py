"""probe_met_worker.py — capture the MultiEmbeddingTensor offset layout that the graph.py patch misses.

`graph.py:_patch_multiembedding_offset` normalises two observed layouts and deliberately falls through
to torch_frame's `assert self.offset[0] == 0` for anything else. Four runs of the two-level grid
(29029522_{8,34}, 29029526_{16,33}) died on that fall-through, all inside DataLoader workers.

`probe_met_offset.py` tried to `print()` the offending layout, which is why it never told us anything:
worker stdout is discarded. **Exceptions, unlike prints, do propagate** — the worker re-raises them in
the parent with the original traceback. So this probe replaces the fall-through assert with a
RuntimeError carrying the numbers, and runs with workers on.

    .venv/bin/python scripts/probe_met_worker.py --dataset rel-event --task user-repeat --num-workers 8
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def install_loud_fallthrough() -> None:
    """Make the un-normalisable case raise its (k, T, w, numel) instead of a bare assert."""
    from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor as _MET

    inner = _MET.validate          # already wrapped by graph.py's patch

    def validate(self):
        off = self.offset
        if off is not None and off.numel() and int(off[0]) != 0:
            k = int(off[0])
            T = int(off[-1]) - k
            w = int(self.values.shape[1])
            raise RuntimeError(
                "MET_FALLTHROUGH "
                f"numel={int(off.numel())} n_cols={int(off.numel()) - 1} k={k} T={T} "
                f"values={tuple(self.values.shape)} w-T={w - T} w-(k+T)={w - (k + T)} "
                f"offset_head={off[:6].tolist()} offset_tail={off[-4:].tolist()} "
                f"col_dims={(off[1:] - off[:-1]).tolist()[:8]}"
            )
        return inner(self)

    _MET.validate = validate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rel-event")
    ap.add_argument("--task", default="user-repeat")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from relbench.tasks import get_task

    from gloss.data.collate import to_cell_batch
    from gloss.data.graph import build_gloss_graph, make_loader
    from gloss.utils.seeding import seed_everything

    install_loud_fallthrough()     # AFTER graph.py's patch, so we only see what it could not fix

    bundle = build_gloss_graph(args.dataset)
    task = get_task(args.dataset, args.task, download=True)
    seed_everything(args.seed)
    loader = make_loader(bundle, task, args.split, batch_size=args.batch_size,
                         num_workers=args.num_workers, shuffle=args.shuffle)

    n_ok = 0
    for i, raw in enumerate(loader):
        if i >= args.batches:
            break
        to_cell_batch(raw, bundle, task.entity_table, seq_len=512, max_fk=5)
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"  {n_ok} batches OK", flush=True)

    print(f"=== {args.dataset}/{args.task} [{args.split}] workers={args.num_workers}: "
          f"{n_ok} batches OK, no fall-through reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
