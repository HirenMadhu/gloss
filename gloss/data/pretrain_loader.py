"""Task-free seed sampling for masked-cell pretraining.

Every loader in this repo until now was driven by a RelBench **task table**: `make_loader` calls
`get_node_train_table_input(task.get_table(split), task)` and an `AttachTargetTransform` puts `y` on
the entity store. Pretraining has no labels, so seeds must come from ordinary table rows instead.

Everything else is deliberately identical to finetuning — the same `NeighborSampler(time_attr='time',
temporal_strategy='last', disjoint=True)`, the same `to_cell_batch`. RT does the same ("the input
sequence to the model is constructed in the same way as for forecasting tasks"), and it is what makes
a pretrained trunk usable downstream without a distribution shift in how context is drawn.

**One table per batch.** `to_cell_batch` takes a single `entity_table`, and `row_is_root` /
`is_seed_cell` both key off it, so a mixed batch would have no well-defined root. Instead each batch
draws all its seeds from one table and :class:`RoundRobinLoader` interleaves the per-table streams.

**Seed tables are weighted by *maskable* rows, not raw rows.** Measured: rel-f1's `drivers` (the
entity table for every driver task) has zero maskable columns, and so do 7 of rel-trial's 15 tables —
including `facilities_studies` at 1.87M rows, about half that database. Weighting by raw row count
would spend half of rel-trial's budget on sequences that cannot produce a seed target. Such tables are
still sampled as *neighbours*, where they carry the relational structure they are there for.

**Temporal safety is enforced here, not assumed.** The graph is materialized with
`upto_test_timestamp=False`, i.e. the whole database, so nothing stops a sampler from reaching a
test-period row. Train seed times are capped at `val_timestamp`; the sampler's `row_time <= seed_time`
rule then makes every cell in the batch train-period by construction. Without this cap every
downstream number would be contaminated, silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

#: Seeds are drawn from timestamps at or before this fraction of the train window for `train`, and
#: from the tail of it for `val` — a held-out slice in *time*, matching how RelBench splits.
VAL_TAIL_FRACTION = 0.1


@dataclass
class SeedTable:
    """One table's eligible seed rows and the weight its stream gets in the mixture."""

    node_type: str
    index: Tensor        # long [n] — row ids eligible as seeds in this split
    time: Tensor         # long [n] — seed timestamp for each (own row time, or a sampled cutoff)
    weight: float        # sampling weight = number of maskable rows
    timed: bool

    def __len__(self) -> int:
        return int(self.index.numel())


def maskable_tables(bundle, spec) -> dict[str, int]:
    """``node_type -> number of maskable columns``. Tables at 0 are never used as seeds."""
    out: dict[str, int] = {}
    for nt in bundle.node_types:
        tid = bundle.node_type_id[nt]
        out[nt] = int(((spec.table == tid) & spec.maskable).sum())
    return out


def _time_bounds(bundle) -> tuple[int, int, int]:
    """``(min_time, val_ts, test_ts)`` in unix seconds, from the RelBench dataset metadata."""
    from relbench.datasets import get_dataset

    ds = get_dataset(bundle.dataset_name, download=False)
    val = int(ds.val_timestamp.timestamp())
    test = int(ds.test_timestamp.timestamp())
    lo = None
    for nt in bundle.node_types:
        st = bundle.data[nt]
        if "time" in st and st.time.numel():
            t = st.time[st.time > 0]
            if t.numel():
                m = int(t.min())
                lo = m if lo is None else min(lo, m)
    return (lo if lo is not None else 0), val, test


def build_seed_tables(bundle, spec, split: str = "train", *,
                      generator: torch.Generator | None = None) -> list[SeedTable]:
    """Eligible seed rows per table for ``split`` ∈ {train, val}, weighted by maskable rows.

    A *timed* table's seed time is the row's own timestamp — that is the moment the row came into
    existence, so its own context is exactly the causal history available then. An *untimed* (static)
    table has no such moment, so a cutoff is drawn uniformly from the split's window; without one the
    row would have no causal context at all and `seed_time` would be undefined.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    lo, val_ts, _test_ts = _time_bounds(bundle)
    cut = int(lo + (1.0 - VAL_TAIL_FRACTION) * (val_ts - lo))     # train | val boundary
    t_lo, t_hi = (lo, cut) if split == "train" else (cut, val_ts)

    per_table = maskable_tables(bundle, spec)
    out: list[SeedTable] = []
    for nt in bundle.node_types:
        w = per_table.get(nt, 0)
        if w == 0:
            continue                       # no maskable column -> never a seed, still a neighbour
        store = bundle.data[nt]
        n = store.num_nodes
        if n == 0:
            continue
        if "time" in store:
            t = store.time.to(torch.long)
            keep = (t > 0) & (t >= t_lo) & (t <= t_hi)
            idx = keep.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            out.append(SeedTable(nt, idx, t[idx], float(w) * float(idx.numel()), True))
        else:
            idx = torch.arange(n, dtype=torch.long)
            span = max(t_hi - t_lo, 1)
            draw = torch.randint(0, span, (n,), generator=generator, dtype=torch.long) + t_lo
            out.append(SeedTable(nt, idx, draw, float(w) * float(n), False))
    if not out:
        raise ValueError(
            f"{bundle.dataset_name}/{split}: no table has both a maskable column and an eligible "
            f"row in [{t_lo}, {t_hi}]. Nothing can be pretrained on this split.")
    return out


def make_pretrain_loader(bundle, seed_table: SeedTable, *, num_neighbors=None, batch_size: int = 64,
                         shuffle: bool = True, num_workers: int = 0, max_seeds: int | None = None):
    """A label-free temporal loader over one table's rows. Same sampler as `graph.make_loader`.

    No `transform`, so nothing attaches `y`; `to_cell_batch` leaves `target`/`has_target` zeroed,
    which it already handles. `seed_time` still arrives on the entity store — PyG's `NodeLoader`
    writes it from `input_time` (`node_loader.py`), independent of any task machinery.
    """
    from relbench.modeling.loader import CustomNodeLoader, NeighborSampler

    idx, t = seed_table.index, seed_table.time
    if max_seeds is not None and idx.numel() > max_seeds:
        pick = torch.randperm(idx.numel())[:max_seeds]
        idx, t = idx[pick], t[pick]
    sampler = NeighborSampler(
        bundle.data,
        num_neighbors=list(num_neighbors or [12, 12]),
        time_attr="time",
        temporal_strategy="last",
        disjoint=True,
    )
    return CustomNodeLoader(
        bundle.data,
        node_sampler=sampler,
        input_nodes=(seed_table.node_type, idx),
        input_time=t,
        transform=None,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


class RoundRobinLoader:
    """Interleaves per-table loaders, drawing each batch's table ∝ its maskable-row weight.

    Yields ``(node_type, raw_batch)`` so the training step knows which ``entity_table`` to collate
    against. Length is a *step budget*, not an epoch: the corpus is ~110M candidate seeds across the
    7 DBs, so "an epoch" is not a meaningful unit here and RT likewise budgets in steps (50k).

    The table schedule is drawn from a seeded generator and depends only on the step index, so every
    DDP rank picks the same table at the same step without communicating. That matters more than it
    looks: with per-DB/per-table encoders, ranks disagreeing on the table would leave different
    parameters unused on different ranks, which DDP cannot reconcile.
    """

    def __init__(self, loaders: dict[str, object], weights: dict[str, float], *, steps: int,
                 seed: int = 0):
        if not loaders:
            raise ValueError("RoundRobinLoader needs at least one table loader")
        self.names = sorted(loaders)
        self.loaders = loaders
        w = torch.tensor([max(weights.get(n, 0.0), 0.0) for n in self.names], dtype=torch.double)
        if float(w.sum()) <= 0:
            w = torch.ones_like(w)
        self.probs = w / w.sum()
        self.steps = int(steps)
        self.seed = seed

    def table_schedule(self) -> list[str]:
        """The (deterministic) table drawn at each step — same on every rank, given the seed."""
        g = torch.Generator().manual_seed(self.seed)
        pick = torch.multinomial(self.probs, self.steps, replacement=True, generator=g)
        return [self.names[int(i)] for i in pick]

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        iters = {n: iter(l) for n, l in self.loaders.items()}
        for name in self.table_schedule():
            try:
                batch = next(iters[name])
            except StopIteration:                       # that table's stream ran out -> restart it
                iters[name] = iter(self.loaders[name])
                try:
                    batch = next(iters[name])
                except StopIteration:
                    continue                            # genuinely empty; skip rather than hang
            yield name, batch


def build_pretrain_stream(bundle, spec, *, split: str = "train", steps: int, batch_size: int = 64,
                          num_neighbors=None, num_workers: int = 0, seed: int = 0,
                          max_seeds_per_table: int | None = None,
                          key_prefix: str = "") -> RoundRobinLoader:
    """The whole thing: eligible seeds -> per-table loaders -> one weighted round-robin stream.

    ``key_prefix`` namespaces the stream keys as ``"<dataset>/<table>"`` so several databases can be
    merged into one mixture without their table names colliding (rel-f1 and rel-stack both have
    ``users``-like tables, and `to_cell_batch` needs the *table*, the trainer needs the *database*).
    """
    tables = build_seed_tables(bundle, spec, split,
                               generator=torch.Generator().manual_seed(seed))
    loaders, weights = {}, {}
    for st in tables:
        key = f"{key_prefix}{st.node_type}"
        loaders[key] = make_pretrain_loader(
            bundle, st, num_neighbors=num_neighbors, batch_size=batch_size,
            shuffle=(split == "train"), num_workers=num_workers, max_seeds=max_seeds_per_table)
        weights[key] = st.weight
    return RoundRobinLoader(loaders, weights, steps=steps, seed=seed)


def split_key(key: str) -> tuple[str, str]:
    """``"rel-f1/results"`` -> ``("rel-f1", "results")``; a bare table -> ``("", table)``."""
    return tuple(key.split("/", 1)) if "/" in key else ("", key)  # type: ignore[return-value]


def build_multi_pretrain_stream(bundles: dict, specs: dict, *, split: str = "train", steps: int,
                                batch_size: int = 64, num_neighbors=None, num_workers: int = 0,
                                seed: int = 0, max_seeds_per_table: int | None = None,
                                dataset_weights: dict | None = None) -> RoundRobinLoader:
    """One mixture over several databases, keyed ``"<dataset>/<table>"``.

    Table weights stay proportional to **maskable rows** within a database and are then scaled by
    ``dataset_weights`` (default 1.0 each). Without that scaling the mixture is dominated by whichever
    database is largest — rel-event alone is 44.8M rows against rel-f1's 0.1M, a 450x ratio, so the
    small databases would contribute essentially nothing over a 50k-step budget. Making it an explicit
    knob keeps the choice visible instead of implicit in the row counts.
    """
    loaders, weights = {}, {}
    for ds, bundle in bundles.items():
        sub = build_pretrain_stream(bundle, specs[ds], split=split, steps=1,
                                    batch_size=batch_size, num_neighbors=num_neighbors,
                                    num_workers=num_workers, seed=seed,
                                    max_seeds_per_table=max_seeds_per_table,
                                    key_prefix=f"{ds}/")
        scale = float((dataset_weights or {}).get(ds, 1.0))
        total = sum(sub.probs.tolist()) or 1.0
        for i, name in enumerate(sub.names):
            loaders[name] = sub.loaders[name]
            weights[name] = float(sub.probs[i]) / total * scale
    return RoundRobinLoader(loaders, weights, steps=steps, seed=seed)
