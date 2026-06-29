"""Phase 0 substrate: RelBench DB -> heterogeneous temporal graph + leakage-safe temporal sampler.

We *reuse* relbench (`make_pkey_fkey_graph`, `get_node_train_table_input`) and PyG's temporal
`NeighborSampler` (leakage-safe via `time_attr`, count/order-based so global time-rescale leaves the
sampled set unchanged — a precondition for scale-equivariance). On top we build the stable per-DB
**vocabularies** HALOS needs:

* ``node_type_id``  — table name -> int.
* ``fk_role_id``    — FK edge relation (``f2p_<col>`` / ``rev_f2p_<col>``) -> int. Two FKs into one
  table are distinct relations, so they get distinct ids (this is how HALOS disambiguates dual FKs).
* ``metapath_id``   — reserved {PAD:0, SELF:1, MULTIHOP:2} + one id per directed relation (>=3).

Cell *features* are pytorch-frame ``TensorFrame``s on each node store; embedding the text columns
needs a ``TextEmbedderConfig``. For Phase-0 substrate work we default to a cheap deterministic
``HashTextEmbedder`` (no model download, hermetic tests); swap in a real encoder later via config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

# Reserved metapath tokens (shared with the bias generator's E_metapath embedding in Phase 3).
MP_PAD, MP_SELF, MP_MULTIHOP = 0, 1, 2
FK_NONE = 0  # fk_role_id for self / non-adjacent / >1-hop pairs


def canonical_relation(rel: str) -> str:
    """Map a PyG relation name to its undirected FK identity (the fkey column).

    relbench names FK edges ``f2p_<col>`` and their reverse ``rev_f2p_<col>`` (also ``p2f_``). The FK
    *role* (buyer vs seller) is the column and is direction-independent, so both map to ``<col>``.
    """
    r = rel[len("rev_"):] if rel.startswith("rev_") else rel
    for p in ("f2p_", "p2f_"):
        if r.startswith(p):
            return r[len(p):]
    return r


def is_forward_relation(rel: str) -> bool:
    """True for the canonical (non-reverse) direction; collate adds symmetry itself, so it processes
    only forward relations to avoid double-writing the pairwise matrices."""
    return not rel.startswith("rev_")


class HashTextEmbedder:
    """Deterministic, dependency-free text embedder for substrate/dev work.

    Maps each string to a fixed pseudo-random unit-ish vector via a seeded RNG on its hash. Meaningless
    semantically, but stable and fast — enough to materialize node features for shape/leakage tests and
    the --dry-run. Real cell-text semantics are a later concern (swap via `build_gloss_graph(..., text_embedder=...)`).
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def __call__(self, sentences: list[str]) -> Tensor:  # noqa: D401
        out = torch.empty(len(sentences), self.dim)
        for i, s in enumerate(sentences):
            g = torch.Generator().manual_seed(hash(str(s)) & 0x7FFFFFFF)
            out[i] = torch.rand(self.dim, generator=g)
        return out


@dataclass
class GraphBundle:
    """Everything Phase 0 hands downstream: the graph, stats, and the per-DB vocabularies."""

    dataset_name: str
    data: "object"  # torch_geometric.data.HeteroData (avoid hard import in annotations)
    col_stats_dict: dict
    node_types: list[str]
    edge_types: list[tuple[str, str, str]]
    node_type_id: dict[str, int]
    fk_role_id: dict[str, int] = field(default_factory=dict)       # relation name -> id (>=1)
    metapath_id: dict[str, int] = field(default_factory=dict)      # relation name -> id (>=3)

    @property
    def num_node_types(self) -> int:
        return len(self.node_type_id)

    @property
    def num_fk_roles(self) -> int:
        return 1 + len(self.fk_role_id)  # + FK_NONE

    @property
    def num_metapaths(self) -> int:
        return 3 + len(self.metapath_id)  # + reserved PAD/SELF/MULTIHOP

    def relation_fk_role(self, rel: str) -> int:
        return self.fk_role_id.get(canonical_relation(rel), FK_NONE)

    def relation_metapath(self, rel: str) -> int:
        return self.metapath_id.get(canonical_relation(rel), MP_MULTIHOP)


def _build_vocabs(node_types, edge_types) -> tuple[dict, dict, dict]:
    """Vocabs keyed by the **canonical** FK column, so forward/reverse share a role and two FKs into
    one table stay distinct. ``fk_role_id``/``metapath_id`` map ``<col> -> int``."""
    node_type_id = {nt: i for i, nt in enumerate(sorted(node_types))}
    cols = sorted({canonical_relation(rel) for (_, rel, _) in edge_types})
    fk_role_id = {c: i + 1 for i, c in enumerate(cols)}          # 0 reserved (FK_NONE)
    metapath_id = {c: i + 3 for i, c in enumerate(cols)}         # 0,1,2 reserved
    return node_type_id, fk_role_id, metapath_id


def build_gloss_graph(
    dataset_name: str = "rel-f1",
    *,
    text_embedder: Callable[[list[str]], Tensor] | None = None,
    text_embed_dim: int = 32,
    text_batch_size: int = 256,
    cache_dir: str | None = None,
    download: bool = False,
) -> GraphBundle:
    """Load a RelBench dataset and build the HALOS graph bundle (graph + stats + vocabs)."""
    from relbench.datasets import get_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from torch_frame.config.text_embedder import TextEmbedderConfig

    dataset = get_dataset(dataset_name, download=download)
    db = dataset.get_db(upto_test_timestamp=False)
    stype_dict = get_stype_proposal(db)
    embedder = text_embedder if text_embedder is not None else HashTextEmbedder(text_embed_dim)
    cfg = TextEmbedderConfig(text_embedder=embedder, batch_size=text_batch_size)
    data, col_stats = make_pkey_fkey_graph(db, stype_dict, text_embedder_cfg=cfg, cache_dir=cache_dir)

    node_types = list(data.node_types)
    edge_types = list(data.edge_types)
    node_type_id, fk_role_id, metapath_id = _build_vocabs(node_types, edge_types)
    return GraphBundle(
        dataset_name=dataset_name,
        data=data,
        col_stats_dict=col_stats,
        node_types=node_types,
        edge_types=edge_types,
        node_type_id=node_type_id,
        fk_role_id=fk_role_id,
        metapath_id=metapath_id,
    )


_BOOL_STR_TARGET = {"t": 1.0, "f": 0.0, "true": 1.0, "false": 0.0, "True": 1.0, "False": 0.0,
                    True: 1.0, False: 0.0, "1": 1.0, "0": 0.0, 1: 1.0, 0: 0.0}


def coerce_binary_target(table, task):
    """Map Postgres-style boolean-string targets (``'t'``/``'f'``) to ``{1.0, 0.0}`` for binary tasks.

    Some RelBench binary tasks (rel-trial ``eligibilities-adult`` / ``eligibilities-child`` /
    ``studies-has_dmc``) store the target column as object ``'t'``/``'f'`` strings, which RelBench's own
    ``get_node_train_table_input`` (``astype(float)``) and metrics (``roc_auc`` with ``pos_label=1``)
    cannot consume. No-op for numeric targets and non-binary tasks; leaves unmapped/NaN values as-is
    (e.g. the masked test table)."""
    from relbench.base import TaskType

    col = getattr(task, "target_col", None)
    if task.task_type != TaskType.BINARY_CLASSIFICATION or col is None or col not in table.df.columns:
        return table
    s = table.df[col]
    if s.dtype != object and s.dtype != bool:
        return table  # already numeric
    new_df = table.df.copy()
    new_df[col] = s.map(lambda v: _BOOL_STR_TARGET.get(v, v)).astype("float64")
    return type(table)(df=new_df, fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                       pkey_col=table.pkey_col, time_col=table.time_col)


def make_loader(
    bundle: GraphBundle,
    task,
    split: str = "train",
    *,
    num_neighbors: list[int] | None = None,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
):
    """A leakage-safe, per-seed-**disjoint** temporal loader yielding sampled HeteroData minibatches.

    `disjoint=True` gives each sampled node a ``batch`` (= seed index) and puts the seed time on the
    entity store as ``seed_time`` — exactly what `collate.to_gloss_batch` needs to build per-seed dense
    subgraphs. `temporal_strategy='last'` + `time_attr='time'` enforce ``row_time <= seed_time``.
    """
    from relbench.modeling.graph import get_node_train_table_input
    from relbench.modeling.loader import CustomNodeLoader, NeighborSampler

    num_neighbors = num_neighbors or [12, 12]
    table = coerce_binary_target(task.get_table(split), task)
    inp = get_node_train_table_input(table, task)
    sampler = NeighborSampler(
        bundle.data,
        num_neighbors=num_neighbors,
        time_attr="time",
        temporal_strategy="last",
        disjoint=True,
    )
    return CustomNodeLoader(
        bundle.data,
        node_sampler=sampler,
        input_nodes=inp.nodes,
        input_time=inp.time,
        transform=inp.transform,   # AttachTargetTransform -> seed store carries `y` (labels)
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
