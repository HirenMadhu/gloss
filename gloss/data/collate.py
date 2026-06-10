"""Subgraph -> TokenBatch (implementation.md §4.2).

A *cell* (one value at a (row, column)) is the token unit, à la RT. We pack ``B`` sampled subgraphs into
padded ``[B, T_max]`` per-cell tensors with a ``pad_mask`` and a ``seg_id`` (so attention never crosses
subgraphs). Pairwise geometry (``dt_ij``, ``hop_ij``, ``mask_kind_ij``) is **not** materialized as a dense
``[B, T, T]`` tensor here — at ``max_cells=4096`` that is infeasible (stress-test #5). It is exposed as
per-segment builder methods the Phase-3 blocked attention calls on demand.

Value encoding is intentionally minimal (numeric scalar + hashed-categorical id): the real type-specific
encoding is the Phase-3 tokenizer's job. This module only needs to be shape-correct and leakage-faithful.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from gloss.data.relbench_graph import Subgraph


def _hash_cat(value) -> int:
    """Deterministic non-negative 63-bit hash for a categorical/string cell (0 reserved for null)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    h = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return (int.from_bytes(h, "big") & ((1 << 63) - 1)) or 1


@dataclass
class TokenBatch:
    """Padded per-cell tensors. All ``[B, T_max]`` unless noted. ``pad_mask`` is True for real cells."""

    value_num: torch.Tensor       # float32, NaN where the cell is non-numeric/null
    value_cat: torch.Tensor       # int64, hashed categorical id (0 = numeric/null)
    col_global_id: torch.Tensor   # int64 -> DocCard text-cache gather + per-column stats
    node_type_id: torch.Tensor    # int64 (table id)
    fk_role_id: torch.Tensor      # int64 (0 = not an FK cell) -- fixes RT dual-FK ambiguity
    table_id: torch.Tensor        # int64 (== node_type_id; kept distinct per the §4.2 contract)
    row_local_id: torch.Tensor    # int64, which row within the batch this cell belongs to
    hop: torch.Tensor             # int64, BFS hop of the row
    mask_kind: torch.Tensor       # int64, how the row was reached (MASK_SEED/FEATURE/NEIGHBOR)
    row_time: torch.Tensor        # int64 ns (TIME_MIN for timeless rows)
    seed_time: torch.Tensor       # int64 ns (per subgraph, broadcast to its cells)
    is_self_label: torch.Tensor   # int64 {0,1}
    seg_id: torch.Tensor          # int64, which subgraph (0..B-1)
    pad_mask: torch.Tensor        # bool, True = real cell

    @property
    def batch_size(self) -> int:
        return self.value_num.shape[0]

    @property
    def max_cells(self) -> int:
        return self.value_num.shape[1]

    def to(self, device) -> "TokenBatch":
        return TokenBatch(**{k: v.to(device) for k, v in self.__dict__.items()})

    def shapes(self) -> dict[str, tuple]:
        return {k: tuple(v.shape) for k, v in self.__dict__.items()}

    # --- pairwise geometry, built lazily per segment (NOT stored dense) ------------------------------
    def segment_dt(self, b: int) -> torch.Tensor:
        """``[n, n]`` |dt| in seconds among the real cells of subgraph ``b`` (timeless -> 0 distance)."""
        m = self.pad_mask[b]
        rt = self.row_time[b][m].to(torch.float64)
        # timeless rows (TIME_MIN) contribute 0 temporal distance, not a huge one
        rt = torch.where(rt <= torch.finfo(torch.float64).min / 2, torch.nan, rt)
        dt = (rt[:, None] - rt[None, :]).abs() / 1e9
        return torch.nan_to_num(dt, nan=0.0)


def collate_subgraphs(
    subgraphs: list[Subgraph],
    db,
    registry,
    self_label_col_global_id: int | None = None,
    max_cells: int | None = None,
) -> TokenBatch:
    """Pack subgraphs into a padded :class:`TokenBatch`. ``db``/``registry`` from the TaskBundle."""
    per_seed_cells = []  # list of dict-of-lists
    for sg in subgraphs:
        cols: dict[str, list] = {k: [] for k in (
            "value_num", "value_cat", "col_global_id", "node_type_id", "fk_role_id", "table_id",
            "row_local_id", "hop", "mask_kind", "row_time", "is_self_label")}
        for row_local, r in enumerate(sg.rows):
            tbl = db.table_dict[r.table]
            df_row = tbl.df.iloc[r.pos]
            tid = registry.table_to_id[r.table]
            for c in tbl.df.columns:
                cg = registry.col_global_id[(r.table, c)]
                v = df_row[c]
                is_num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                cols["value_num"].append(float(v) if is_num and not pd.isna(v) else float("nan"))
                cols["value_cat"].append(0 if is_num else _hash_cat(v))
                cols["col_global_id"].append(cg)
                cols["node_type_id"].append(tid)
                cols["fk_role_id"].append(registry.fk_role_id.get((r.table, c), 0))
                cols["table_id"].append(tid)
                cols["row_local_id"].append(row_local)
                cols["hop"].append(r.hop)
                cols["mask_kind"].append(r.mask_kind)
                cols["row_time"].append(r.row_time_ns)
                cols["is_self_label"].append(
                    1 if (self_label_col_global_id is not None and cg == self_label_col_global_id) else 0)
        per_seed_cells.append((sg.seed_time_ns, cols))

    B = len(subgraphs)
    lengths = [len(c["value_num"]) for _, c in per_seed_cells]
    T = max_cells or (max(lengths) if lengths else 1)
    T = max(T, 1)

    def pad_int(key):
        out = torch.zeros(B, T, dtype=torch.int64)
        for b, (_, c) in enumerate(per_seed_cells):
            n = min(len(c[key]), T)
            if n:
                out[b, :n] = torch.tensor(c[key][:n], dtype=torch.int64)
        return out

    value_num = torch.full((B, T), float("nan"), dtype=torch.float32)
    seed_time = torch.zeros(B, T, dtype=torch.int64)
    seg_id = torch.zeros(B, T, dtype=torch.int64)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    for b, (st, c) in enumerate(per_seed_cells):
        n = min(len(c["value_num"]), T)
        if n:
            value_num[b, :n] = torch.tensor(c["value_num"][:n], dtype=torch.float32)
        seed_time[b, :n] = st
        seg_id[b, :n] = b
        pad_mask[b, :n] = True

    return TokenBatch(
        value_num=value_num,
        value_cat=pad_int("value_cat"),
        col_global_id=pad_int("col_global_id"),
        node_type_id=pad_int("node_type_id"),
        fk_role_id=pad_int("fk_role_id"),
        table_id=pad_int("table_id"),
        row_local_id=pad_int("row_local_id"),
        hop=pad_int("hop"),
        mask_kind=pad_int("mask_kind"),
        row_time=pad_int("row_time"),
        seed_time=seed_time,
        is_self_label=pad_int("is_self_label"),
        seg_id=seg_id,
        pad_mask=pad_mask,
    )
