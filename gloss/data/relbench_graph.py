"""DB -> heterogeneous temporal graph + leakage-safe temporal neighbor sampler (Phase 0).

implementation.md §4.2 / §6-Phase-0. We deliberately do **not** use PyG ``NeighborLoader`` (it needs
``pyg_lib``/``torch_sparse``, whose prebuilt wheels don't load on this el8 / glibc-2.28 cluster). Instead
we build our own index over a relbench :class:`~relbench.base.Database` and sample with the single
normative invariant:

    **a context cell is valid iff ``row_time <= seed_time``**  (tested by ``tests/test_leakage.py``).

Static dimension tables (``time_col is None``, e.g. ``drivers``) are timeless schema metadata and are
valid at any seed time. Each foreign-key column gets a distinct ``fk_role_id`` so two FKs into the same
table are distinguishable — the RT limitation DocCards/biases target (CLAUDE.md, §4.2).

The same API runs on synthetic databases (``gloss.data.synthetic`` emits a real ``Database``), so Phase-2
proxy + Phase-3 model share one sampling path across synthetic and real data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from relbench.base import Database, EntityTask

# Sentinel for "timeless" rows (static dimension tables): smaller than any real timestamp, so the
# leakage check ``row_time <= seed_time`` always passes for them.
TIME_MIN = np.iinfo(np.int64).min

# How a sampled row was reached, which drives the RT attention-mask kind in collate.
MASK_SEED = 0      # the seed/root row
MASK_FEATURE = 1   # reached via F->P (this row's foreign key points at it): parent context
MASK_NEIGHBOR = 2  # reached via P->F (it points back at an ancestor): child/event history


def _to_int64_ns(s: pd.Series) -> np.ndarray:
    """Datetime column -> int64 ns; NaT -> TIME_MIN (treated as timeless/always-valid)."""
    arr = pd.to_datetime(s).to_numpy("datetime64[ns]").astype("int64")
    arr[arr == np.iinfo(np.int64).min] = TIME_MIN  # NaT already maps to min, keep explicit
    return arr


def _timestamp_to_ns(ts) -> int:
    # ``.value`` is ALWAYS nanoseconds since epoch, regardless of the input's datetime resolution
    # (us/ns). Using ``.astype("int64")`` would silently return microseconds for datetime64[us] inputs
    # and break the leakage comparison against _to_int64_ns (which forces ns).
    return int(pd.Timestamp(ts).value)


# --------------------------------------------------------------------------------------------------
# Schema registry: stable global ids for tables / columns / fk-roles
# --------------------------------------------------------------------------------------------------
@dataclass
class SchemaRegistry:
    """Deterministic global ids derived from a Database schema (sorted for reproducibility)."""

    table_to_id: dict[str, int]
    # (table, column) -> col_global_id ; gathered against the DocCard text cache at train time.
    col_global_id: dict[tuple[str, str], int]
    col_of_global: list[tuple[str, str]]
    # (table, fkey_col) -> fk_role_id >= 1 ; 0 means "not an FK cell".
    fk_role_id: dict[tuple[str, str], int]
    fk_role_of_id: list[tuple[str, str] | None]

    @classmethod
    def build(cls, db: "Database") -> "SchemaRegistry":
        tables = sorted(db.table_dict.keys())
        table_to_id = {t: i for i, t in enumerate(tables)}
        col_global_id: dict[tuple[str, str], int] = {}
        col_of_global: list[tuple[str, str]] = []
        fk_role_id: dict[tuple[str, str], int] = {}
        fk_role_of_id: list[tuple[str, str] | None] = [None]  # id 0 reserved
        for t in tables:
            tbl = db.table_dict[t]
            for c in tbl.df.columns:
                key = (t, c)
                col_global_id[key] = len(col_of_global)
                col_of_global.append(key)
                if c in tbl.fkey_col_to_pkey_table:
                    fk_role_id[key] = len(fk_role_of_id)
                    fk_role_of_id.append(key)
        return cls(table_to_id, col_global_id, col_of_global, fk_role_id, fk_role_of_id)

    @property
    def num_tables(self) -> int:
        return len(self.table_to_id)

    @property
    def num_cols(self) -> int:
        return len(self.col_of_global)

    @property
    def num_fk_roles(self) -> int:
        return len(self.fk_role_of_id)


# --------------------------------------------------------------------------------------------------
# Sampled subgraph containers
# --------------------------------------------------------------------------------------------------
@dataclass
class SampledRow:
    table: str
    pos: int                 # row position within the table df
    hop: int
    row_time_ns: int
    mask_kind: int           # MASK_SEED / MASK_FEATURE / MASK_NEIGHBOR
    parent_idx: int          # index into Subgraph.rows of the row we expanded from (-1 for seed)
    via_fk_role: int         # fk_role_id of the edge used (0 if none / seed)


@dataclass
class Subgraph:
    entity_table: str
    entity_id: object
    seed_time_ns: int
    label: object
    rows: list[SampledRow] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)


# --------------------------------------------------------------------------------------------------
# The graph index
# --------------------------------------------------------------------------------------------------
class HeteroTemporalGraph:
    """Precomputed PK/FK indices + per-table time arrays over a relbench ``Database``."""

    def __init__(self, db: "Database", registry: SchemaRegistry):
        self.db = db
        self.reg = registry
        self.row_time: dict[str, np.ndarray] = {}
        self.pk_pos: dict[str, dict] = {}                      # table -> {pkey_value: pos}
        self.fwd: dict[tuple[str, str], np.ndarray] = {}        # (table, fkcol) -> child_pos->parent_pos
        # parent_table -> list of (child_table, fkcol, {parent_pos: np.array(child_pos)})
        self.rev: dict[str, list[tuple[str, str, dict]]] = {t: [] for t in db.table_dict}

        for t, tbl in db.table_dict.items():
            df = tbl.df.reset_index(drop=True)
            self.row_time[t] = (
                _to_int64_ns(df[tbl.time_col]) if tbl.time_col else
                np.full(len(df), TIME_MIN, dtype=np.int64)
            )
            if tbl.pkey_col is not None:
                self.pk_pos[t] = {v: i for i, v in enumerate(df[tbl.pkey_col].to_numpy())}

        for t, tbl in db.table_dict.items():
            df = tbl.df.reset_index(drop=True)
            for fkcol, parent in tbl.fkey_col_to_pkey_table.items():
                pk = self.pk_pos.get(parent, {})
                parent_pos = np.fromiter(
                    (pk.get(v, -1) for v in df[fkcol].to_numpy()), dtype=np.int64, count=len(df)
                )
                self.fwd[(t, fkcol)] = parent_pos
                # reverse adjacency: group child positions by parent position (skip dangling -1)
                child_pos = np.arange(len(df), dtype=np.int64)
                valid = parent_pos >= 0
                groups: dict[int, np.ndarray] = {}
                if valid.any():
                    order = np.argsort(parent_pos[valid], kind="stable")
                    pp = parent_pos[valid][order]
                    cp = child_pos[valid][order]
                    bounds = np.searchsorted(pp, np.unique(pp), side="left")
                    uniq = np.unique(pp)
                    for k, start in zip(uniq, bounds):
                        end = pp.searchsorted(k, side="right")
                        groups[int(k)] = cp[start:end]
                self.rev[parent].append((t, fkcol, groups))

    @classmethod
    def build(cls, db: "Database") -> tuple["HeteroTemporalGraph", SchemaRegistry]:
        reg = SchemaRegistry.build(db)
        return cls(db, reg), reg


# --------------------------------------------------------------------------------------------------
# The leakage-safe temporal neighbor sampler
# --------------------------------------------------------------------------------------------------
class TemporalNeighborSampler:
    """BFS over FK/PK edges from a seed entity row, keeping only rows with ``row_time <= seed_time``.

    ``num_neighbors[hop]`` caps the number of P->F child rows sampled per parent per hop (RT-style fanout).
    F->P parent rows are always followed (there is at most one per FK and they are schema context).
    ``max_cells`` bounds total cells (rows x columns) so batches stay bounded.
    """

    def __init__(
        self,
        graph: HeteroTemporalGraph,
        num_neighbors: list[int],
        max_cells: int = 4096,
        seed: int = 0,
    ):
        self.g = graph
        self.num_neighbors = list(num_neighbors)
        self.max_cells = max_cells
        self.rng = np.random.default_rng(seed)
        self._ncols = {t: graph.db.table_dict[t].df.shape[1] for t in graph.db.table_dict}

    def sample(self, entity_table: str, entity_id, seed_time, label=None) -> Subgraph:
        seed_ns = _timestamp_to_ns(seed_time) if seed_time is not None else np.iinfo(np.int64).max
        sg = Subgraph(entity_table, entity_id, seed_ns, label)
        seed_pos = self.g.pk_pos.get(entity_table, {}).get(entity_id)
        if seed_pos is None:
            return sg  # dangling entity (filtered upstream normally); empty subgraph

        visited: set[tuple[str, int]] = set()
        cells = 0

        def add(table: str, pos: int, hop: int, mask: int, parent_idx: int, fk_role: int) -> int:
            nonlocal cells
            key = (table, pos)
            if key in visited:
                return -1
            if cells + self._ncols[table] > self.max_cells and len(sg.rows) > 0:
                return -1
            visited.add(key)
            sg.rows.append(
                SampledRow(table, pos, hop, int(self.g.row_time[table][pos]), mask, parent_idx, fk_role)
            )
            cells += self._ncols[table]
            return len(sg.rows) - 1

        root = add(entity_table, seed_pos, 0, MASK_SEED, -1, 0)
        frontier = [root]
        for hop in range(len(self.num_neighbors)):
            k = self.num_neighbors[hop]
            next_frontier: list[int] = []
            for ridx in frontier:
                if ridx < 0:
                    continue
                r = sg.rows[ridx]
                tbl = self.g.db.table_dict[r.table]
                # F->P: follow this row's foreign keys to parent rows (schema/feature context)
                for fkcol, parent in tbl.fkey_col_to_pkey_table.items():
                    ppos = int(self.g.fwd[(r.table, fkcol)][r.pos])
                    if ppos < 0:
                        continue
                    if self.g.row_time[parent][ppos] > seed_ns:
                        continue
                    role = self.g.reg.fk_role_id.get((r.table, fkcol), 0)
                    ni = add(parent, ppos, hop + 1, MASK_FEATURE, ridx, role)
                    if ni >= 0:
                        next_frontier.append(ni)
                # P->F: sample child rows that point at this row (event/history), time-filtered
                for ct, cfk, groups in self.g.rev[r.table]:
                    cand = groups.get(r.pos)
                    if cand is None or len(cand) == 0:
                        continue
                    ctime = self.g.row_time[ct][cand]
                    cand = cand[ctime <= seed_ns]
                    if len(cand) == 0:
                        continue
                    if len(cand) > k:
                        cand = self.rng.choice(cand, size=k, replace=False)
                    role = self.g.reg.fk_role_id.get((ct, cfk), 0)
                    for cp in cand:
                        ni = add(ct, int(cp), hop + 1, MASK_NEIGHBOR, ridx, role)
                        if ni >= 0:
                            next_frontier.append(ni)
            frontier = next_frontier
        return sg


# --------------------------------------------------------------------------------------------------
# Convenience loader
# --------------------------------------------------------------------------------------------------
@dataclass
class TaskBundle:
    """Everything Phase 0-2 needs for one (dataset, task)."""

    dataset: str
    task_name: str
    db: "Database"
    task: "EntityTask"
    graph: HeteroTemporalGraph
    registry: SchemaRegistry
    entity_table: str
    entity_col: str
    time_col: str
    target_col: str
    task_type: str

    def make_sampler(self, num_neighbors, max_cells=4096, seed=0) -> TemporalNeighborSampler:
        return TemporalNeighborSampler(self.graph, num_neighbors, max_cells, seed)


def load_task_bundle(dataset: str, task_name: str, download: bool = True) -> TaskBundle:
    """Load a RelBench (dataset, task), build the graph + registry. Adapts to the installed API."""
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    ds = get_dataset(dataset, download=download)
    db = ds.get_db()
    task = get_task(dataset, task_name, download=download)
    graph, reg = HeteroTemporalGraph.build(db)
    return TaskBundle(
        dataset=dataset,
        task_name=task_name,
        db=db,
        task=task,
        graph=graph,
        registry=reg,
        entity_table=task.entity_table,
        entity_col=task.entity_col,
        time_col=task.time_col,
        target_col=task.target_col,
        task_type=str(task.task_type),
    )
