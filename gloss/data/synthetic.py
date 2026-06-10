"""Planted-ground-truth synthetic generator (implementation.md §7) — existence proof + audit validation.

The crux of the falsifiability argument. We emit a real relbench :class:`~relbench.base.Database`
(``entity`` + ``event``, FK ``event.entity_id -> entity.id``) with **twin columns** ``col_X``/``col_Y``:
identical distribution and identical topological role, differing only in **one documented fact** — a coded
*sign*. The label depends on the documented sign of the *causal* twin and is **not inferable from values or
topology alone**.

Why a single fixed schema is not enough (stress-test #4): if the causal column were always ``col_X`` with a
fixed sign, a no-doc model would simply learn that correlation from labels and documentation would be
redundant. The honest existence proof is **cross-schema transfer**: across schema *variants* the causal
twin and its sign are permuted, so the only invariant, transferable signal that says *which* column matters
and *with what sign* is the DocCard. Hence the generator exposes ``causal_col`` / ``sign`` knobs and
:func:`make_variants` builds a family; a values+structure model cannot generalize across variants, a
doc-reading one can. This is the data-processing-inequality separation the audit is validated against.

**Synthetic DocCards are programmatic** (template-encoded from ``planted_truth``); Claude never sees the
sign — see ``gloss.data.doccards.render_synthetic_card`` and stress-test #3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from relbench.base import Database, Table

from gloss.data.relbench_graph import HeteroTemporalGraph, TaskBundle

TWINS = ("col_X", "col_Y")
_EPOCH = pd.Timestamp("2015-01-01")


@dataclass
class PlantedTruth:
    """Ground truth the audit scores attribution precision/recall against."""

    causal_col: str           # which twin drives the label ("col_X" or "col_Y")
    decoy_col: str            # the other twin (documented as inert)
    sign: int                 # +1 / -1, documented in the causal column's card, NOT in values
    alpha: float              # effect strength
    lag_days: int             # history window the label aggregates over
    variant_id: int = 0

    def as_dict(self) -> dict:
        return {**self.__dict__}


@dataclass
class SyntheticTask:
    """A relbench-EntityTask-compatible duck type (only the fields Phase 0-2 use)."""

    dataset: str
    task_name: str
    entity_table: str
    entity_col: str
    time_col: str
    target_col: str
    task_type: str
    _tables: dict[str, pd.DataFrame]
    planted_truth: PlantedTruth = field(default=None)  # type: ignore[assignment]

    def get_table(self, split: str):
        from types import SimpleNamespace

        return SimpleNamespace(df=self._tables[split])


def make_synthetic_db(
    seed: int = 0,
    num_entities: int = 400,
    events_per_entity: int = 30,
    causal_col: Literal["col_X", "col_Y"] = "col_X",
    sign: int = 1,
    alpha: float = 2.5,
    lag_days: int = 180,
    twin_noise: float = 1.0,
    variant_id: int = 0,
) -> tuple[Database, SyntheticTask, PlantedTruth]:
    """Build one schema variant: Database + seeds (train/val/test) + planted truth.

    The label at a seed is ``y ~ Bernoulli(sigmoid(alpha * sign * mean(causal_col over last lag_days)))``.
    ``col_X`` and ``col_Y`` are i.i.d. ``N(0, twin_noise)`` — exchangeable in distribution and topology.
    """
    rng = np.random.default_rng(seed)
    decoy = TWINS[1] if causal_col == TWINS[0] else TWINS[0]

    # --- entity table (static dimension; no time) ---
    ent_ids = np.arange(num_entities)
    entity_df = pd.DataFrame(
        {"id": ent_ids, "ent_bias": rng.normal(0, 1, num_entities).astype("float32")}
    )

    # --- event table (timestamped, FK -> entity) ---
    rows = []
    eid = 0
    for ent in ent_ids:
        n = events_per_entity + int(rng.integers(-5, 6))
        n = max(n, 5)
        # event days spread over ~3 years
        days = np.sort(rng.integers(0, 3 * 365, size=n))
        cx = rng.normal(0, twin_noise, n).astype("float32")
        cy = rng.normal(0, twin_noise, n).astype("float32")
        aux = rng.normal(0, 1, n).astype("float32")
        for k in range(n):
            rows.append((eid, ent, _EPOCH + pd.Timedelta(days=int(days[k])), cx[k], cy[k], aux[k]))
            eid += 1
    event_df = pd.DataFrame(rows, columns=["evt_id", "entity_id", "date", "col_X", "col_Y", "aux"])

    db = Database({
        "entity": Table(entity_df, fkey_col_to_pkey_table={}, pkey_col="id", time_col=None),
        "event": Table(event_df, fkey_col_to_pkey_table={"entity_id": "entity"},
                       pkey_col="evt_id", time_col="date"),
    })

    # --- seeds + labels (label depends ONLY on the documented causal col + sign) ---
    causal_vals = event_df[causal_col].to_numpy()
    ev_ent = event_df["entity_id"].to_numpy()
    ev_day = (event_df["date"] - _EPOCH).dt.days.to_numpy()
    # group event indices per entity for fast history lookup
    order = np.argsort(ev_ent, kind="stable")
    ev_ent_s, ev_day_s, ev_val_s = ev_ent[order], ev_day[order], causal_vals[order]
    starts = np.searchsorted(ev_ent_s, ent_ids, side="left")
    ends = np.searchsorted(ev_ent_s, ent_ids, side="right")

    seed_rows = []
    for ent, lo, hi in zip(ent_ids, starts, ends):
        if hi - lo < 6:
            continue
        days_e, vals_e = ev_day_s[lo:hi], ev_val_s[lo:hi]
        # a few seed times per entity, each after some history accrues
        for q in (0.5, 0.7, 0.9):
            st_day = int(np.quantile(days_e, q))
            window = (days_e <= st_day) & (days_e > st_day - lag_days)
            if window.sum() < 2:
                continue
            m = float(vals_e[window].mean())
            p = 1.0 / (1.0 + np.exp(-(alpha * sign * m)))
            y = int(rng.random() < p)
            seed_rows.append((_EPOCH + pd.Timedelta(days=st_day), int(ent), y))
    seeds = pd.DataFrame(seed_rows, columns=["date", "id", "target"]).sort_values("date").reset_index(drop=True)

    # time-based split (relbench style)
    n = len(seeds)
    tr, va = int(0.6 * n), int(0.8 * n)
    tables = {"train": seeds.iloc[:tr].copy(), "val": seeds.iloc[tr:va].copy(), "test": seeds.iloc[va:].copy()}

    planted = PlantedTruth(causal_col, decoy, int(sign), float(alpha), int(lag_days), int(variant_id))
    task = SyntheticTask(
        dataset="synthetic", task_name=f"twin-sign-v{variant_id}", entity_table="entity",
        entity_col="id", time_col="date", target_col="target",
        task_type="binary_classification", _tables=tables, planted_truth=planted,
    )
    return db, task, planted


def make_synthetic_bundle(**kwargs) -> tuple[TaskBundle, PlantedTruth]:
    """One variant as a :class:`TaskBundle` (shares the Phase-0 sampler/collate path)."""
    db, task, planted = make_synthetic_db(**kwargs)
    graph, reg = HeteroTemporalGraph.build(db)
    bundle = TaskBundle(
        dataset="synthetic", task_name=task.task_name, db=db, task=task, graph=graph, registry=reg,
        entity_table="entity", entity_col="id", time_col="date", target_col="target",
        task_type=task.task_type,
    )
    return bundle, planted


def make_variants(
    num_variants: int = 6, base_seed: int = 0, **kwargs
) -> list[tuple[TaskBundle, PlantedTruth]]:
    """A family of variants with the causal twin + sign permuted — the transfer existence proof.

    Across variants, values/topology are exchangeable; only the (programmatic) DocCard says which twin is
    causal and with what sign. A values+structure model cannot generalize across the family; a doc model can.
    """
    rng = np.random.default_rng(base_seed)
    out = []
    for v in range(num_variants):
        causal = TWINS[v % 2]                       # alternate causal twin
        sgn = int(rng.choice([-1, 1]))              # random documented sign
        bundle, planted = make_synthetic_bundle(
            seed=base_seed + 100 + v, causal_col=causal, sign=sgn, variant_id=v, **kwargs)
        out.append((bundle, planted))
    return out


def make_synthetic_dualfk(
    seed: int = 0, num_users: int = 200, num_tx: int = 2000
) -> tuple[Database, dict]:
    """Dual-FK schema for the FK-role test (stress-test #6): ``transaction.buyer_id`` and
    ``transaction.seller_id`` both reference ``users.id``. The label depends on the *buyer*'s history, so
    swapping the two FK roles must change predictions and ``name_only`` cannot tell them apart.
    Returns (db, planted) where planted names the causal role ("buyer_id").
    """
    rng = np.random.default_rng(seed)
    users = pd.DataFrame({"id": np.arange(num_users),
                          "trust": rng.normal(0, 1, num_users).astype("float32")})
    buyer = rng.integers(0, num_users, num_tx)
    seller = rng.integers(0, num_users, num_tx)
    days = np.sort(rng.integers(0, 2 * 365, num_tx))
    amount = rng.normal(0, 1, num_tx).astype("float32")
    tx = pd.DataFrame({
        "tx_id": np.arange(num_tx),
        "buyer_id": buyer, "seller_id": seller,
        "date": _EPOCH + pd.to_timedelta(days, unit="D"),
        "amount": amount,
    })
    db = Database({
        "users": Table(users, fkey_col_to_pkey_table={}, pkey_col="id", time_col=None),
        "transaction": Table(
            tx, fkey_col_to_pkey_table={"buyer_id": "users", "seller_id": "users"},
            pkey_col="tx_id", time_col="date"),
    })
    planted = {"causal_fk_role": "buyer_id", "decoy_fk_role": "seller_id"}
    return db, planted
