"""Phase 0 substrate: RelBench DB -> heterogeneous temporal graph + leakage-safe temporal sampler.

We *reuse* relbench (`make_pkey_fkey_graph`, `get_node_train_table_input`) and PyG's temporal
`NeighborSampler` (leakage-safe via `time_attr`, count/order-based so global time-rescale leaves the
sampled set unchanged — a precondition for scale-equivariance). On top we build the stable per-DB
**vocabularies** HALOS needs:

* ``node_type_id``  — table name -> int.
* ``fk_role_id``    — FK **role triple** ``(child_table, fk_column, parent_table)`` -> int in ``[1, K]``
  (changes.md P0.1). Keyed on the triple, not on the column name: two FK columns that merely *share a
  name* in different child tables are different relations and must not collide (``raceId`` merges 5
  relations on rel-f1, ``nct_id`` merges 10 on rel-trial under the old column-only keying). A relation
  and its reverse (``f2p_<col>`` / ``rev_f2p_<col>``) share one role — the role is direction-free.
* ``metapath_id``   — reserved {PAD:0, SELF:1, MULTIHOP:2} + one id per role triple (>=3).

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


_MET_REREAD_TRIES = 3

#: Per-process tally of which repair path fired. **Not** aggregated across DataLoader workers (each
#: is its own process and its counters die with it), so this is only readable under
#: ``num_workers=0`` — which is exactly the configuration you reproduce this bug in.
MET_REPAIRS = {"reread": 0, "rebase": 0}


def met_repair_counts() -> dict[str, int]:
    """Copy of :data:`MET_REPAIRS`. A non-zero ``rebase`` means embeddings were *rewritten*."""
    return dict(MET_REPAIRS)


def _met_failed_checks(off, values, num_cols) -> list[str]:
    """torch_frame's four ``MultiEmbeddingTensor.validate`` asserts, named and evaluated one by one.

    Upstream ``validate`` is a bare ``assert`` chain, so a failure says only "an assert fired". Naming
    them is what produced the diagnosis in the docstring below: on 2026-08-08 four jobs died there
    against tensors that satisfy **all four**.
    """
    bad: list[str] = []
    n = 0 if num_cols is None else int(num_cols)
    if off is None or off.numel() == 0 or int(off.reshape(-1)[0]) != 0:
        bad.append("offset[0]==0")
    # upstream uses `len(offset)`, which raises TypeError on a 0-dim tensor rather than asserting —
    # so read size(0) defensively and let the `offset.ndim==1` check below name that case instead.
    if off is None or off.ndim < 1 or off.shape[0] != n + 1:
        bad.append("len(offset)==num_cols+1")
    if off is None or off.ndim != 1:
        bad.append("offset.ndim==1")
    if values is None or not (values.ndim == 2 or values.numel() == 0):
        bad.append("values.ndim==2 or values.numel()==0")
    return bad


def _patch_multiembedding_offset() -> None:
    """Survive a ``MultiEmbeddingTensor`` that fails torch_frame's ``validate`` under ``num_workers>0``.

    **Re-read before repairing.** The original patch assumed the leading ``offset`` was genuinely
    shifted by a constant and rebased it. The 2026-08-08 failures falsify that premise: four jobs
    (``29054792_7``, ``29054796_1``, ``29054798_{1,5}``) raised out of ``validate`` while the dumped
    layout satisfied every one of the four asserts —

    ``offset=[0,32,64,96,128] num_cols=4 values=(512,128)`` and three like it.

    An assert cannot fail on a tensor that satisfies it, so the tensor's **storage changed between
    the assert and the dump**: a DataLoader shared-memory reuse race (which also explains why this
    is exclusive to ``num_workers>0`` and why the stored TensorFrames are clean on disk). That makes
    the transient garbage read, not a shifted offset, the likely cause of the *original* symptom too
    — in which case every rebase this patch performed silently rewrote correct embeddings.

    So the order is now: **check → re-read → rebase → raise**, and re-reading is tried first because
    it cannot corrupt anything (it reads the same storage a second time), whereas rebasing can. The
    rebase is kept as a last resort but is now narrowed to the case where ``offset[0]==0`` is the
    *only* surviving failure, and it bumps :data:`MET_REPAIRS` so a run can tell whether it happened.

    Column ``j``'s embedding is ``values[:, offset[j]:offset[j+1]]``, so when a rebase is genuinely
    warranted, subtracting ``offset[0]`` and keeping the referenced value-columns is content-preserving.
    Anything that survives all three stages raises a **diagnostic** ``RuntimeError`` naming the failed
    checks — an exception is the only thing that crosses the worker boundary; ``print`` does not.
    Forked DataLoader workers inherit this patch (applied at import, before any loader is built)."""
    try:
        from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor as _MET
    except Exception:
        return
    if getattr(_MET, "_gloss_offset_patch", False):
        return
    _orig_validate = _MET.validate

    def _validate(self):
        nc = getattr(self, "num_cols", None)
        first = _met_failed_checks(self.offset, self.values, nc)
        if not first:
            return _orig_validate(self)          # the overwhelmingly common path, untouched

        # --- 1. RE-READ. A shared-memory race resolves in microseconds, so a fresh read of the same
        # storage is usually already correct. `.clone()` is the re-read; it can only observe the
        # buffer a second time, never alter it, which is why it is tried before any repair.
        for _ in range(_MET_REREAD_TRIES):
            off = self.offset
            if off is None:
                break
            fresh = off.clone()
            if not _met_failed_checks(fresh, self.values, nc):
                # Pin the good read: `_row_index_select` forwards `offset` by reference, so leaving
                # the racy tensor in place would hand the same hazard to every child.
                object.__setattr__(self, "offset", fresh)
                MET_REPAIRS["reread"] += 1
                return _orig_validate(self)

        # --- 2. REBASE (last resort). Only when a persistently non-zero base is the *sole* problem;
        # any other broken invariant means the tensor is not merely shifted and a rebase would be a
        # guess. T = true total dim = offset[-1] - k. Two value layouts have ever been observed:
        #   * values width == T      : buffer already correct, just rebase offset
        #   * values width == k + T  : k orphan leading value-columns, drop them then rebase
        off = self.offset
        bad = _met_failed_checks(off, self.values, nc)
        if not bad:
            # Healed between the loop's `.clone()` and this read — same race, one beat later.
            MET_REPAIRS["reread"] += 1
            return _orig_validate(self)
        if bad == ["offset[0]==0"] and off is not None and off.numel():
            k = int(off[0])
            if off.numel() == 1:
                # ZERO-COLUMN MET: `len(offset) == num_cols + 1` with a 1-element offset means
                # num_cols == 0, so no column's embedding can be addressed and rebasing to [0] cannot
                # lose data — there is none. `values` is untouched and still satisfies its check.
                # This case killed 29030571_{15,25,71} before it was handled.
                object.__setattr__(self, "offset", off - k)
                MET_REPAIRS["rebase"] += 1
                return _orig_validate(self)
            T = int(off[-1]) - k
            w = int(self.values.shape[1])
            if w == k + T:
                object.__setattr__(self, "values", self.values[:, k:])
                object.__setattr__(self, "offset", off - k)
                MET_REPAIRS["rebase"] += 1
                return _orig_validate(self)
            if w == T:
                object.__setattr__(self, "offset", off - k)
                MET_REPAIRS["rebase"] += 1
                return _orig_validate(self)
            raise RuntimeError(
                "MultiEmbeddingTensor offset could not be rebased (unrecognised layout): "
                f"n_cols={int(off.numel()) - 1} k={k} T={T} values={tuple(self.values.shape)} "
                f"w-T={w - T} w-(k+T)={w - (k + T)} "
                f"col_dims={(off[1:] - off[:-1]).tolist()[:8]}. "
                f"survived {_MET_REREAD_TRIES} re-reads, so this is not the shared-memory race. "
                "Rerun with --num-workers 0 to get past it, and extend _patch_multiembedding_offset "
                "for the layout above once it is understood. Do NOT guess a repair — a wrong rebase "
                "silently corrupts embeddings instead of crashing."
            )

        # --- 3. GIVE UP, loudly and with the layout attached.
        raise RuntimeError(
            "MultiEmbeddingTensor.validate() failed and neither a re-read nor a rebase fixed it. "
            f"failed_checks_on_entry={first} failed_checks_after_{_MET_REREAD_TRIES}_rereads={bad} "
            f"offset={None if off is None else off.tolist()[:8]} "
            f"dtype={None if off is None else off.dtype} "
            f"shape={None if off is None else tuple(off.shape)} "
            f"values={None if self.values is None else tuple(self.values.shape)} "
            f"num_cols={nc} num_rows={getattr(self, 'num_rows', None)} "
            f"repairs_so_far={met_repair_counts()}. "
            "An EMPTY failed_checks list means the tensor was valid by the time it was inspected — "
            "that is the shared-memory race, and the re-read loop above is what should have caught "
            "it. Do NOT guess a repair from this message alone."
        )

    _MET.validate = _validate
    _MET._gloss_offset_patch = True


_patch_multiembedding_offset()


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


def canonical_edge_role(edge_type: tuple[str, str, str]) -> tuple[str, str, str]:
    """PyG edge type -> the direction-free FK **role triple** ``(child_table, fk_column, parent_table)``.

    ``(child, "f2p_<col>", parent)`` and its reverse ``(parent, "rev_f2p_<col>", child)`` both map to
    ``(child, <col>, parent)``, so forward and reverse share a role id (changes.md P0.1). The triple —
    not the column name — is the identity: the same column name in two child tables is two roles.
    """
    src, rel, dst = edge_type
    col = canonical_relation(rel)
    return (src, col, dst) if is_forward_relation(rel) else (dst, col, src)


def role_name(role: tuple[str, str, str]) -> str:
    """The P0.4 name string for a role triple (fed to the frozen name encoder)."""
    child, col, parent = role
    return f"table {child} column {col} references table {parent}"


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
    # (child_table, fk_column, parent_table) -> id
    fk_role_id: dict[tuple[str, str, str], int] = field(default_factory=dict)   # >=1 (0 = FK_NONE)
    metapath_id: dict[tuple[str, str, str], int] = field(default_factory=dict)  # >=3 (0,1,2 reserved)

    @property
    def num_node_types(self) -> int:
        return len(self.node_type_id)

    @property
    def num_fk_roles(self) -> int:
        return 1 + len(self.fk_role_id)  # + FK_NONE

    @property
    def num_roles(self) -> int:
        """K — the number of real FK roles (ids ``1..K``), excluding ``FK_NONE``."""
        return len(self.fk_role_id)

    @property
    def num_metapaths(self) -> int:
        return 3 + len(self.metapath_id)  # + reserved PAD/SELF/MULTIHOP

    @property
    def role_triples(self) -> list[tuple[str, str, str]]:
        """Role triples ordered by id, so ``role_triples[i]`` is role ``i + 1`` (P0.4 name tables)."""
        return [t for t, _ in sorted(self.fk_role_id.items(), key=lambda kv: kv[1])]

    # ---- the triple is the identity; prefer these two over the name-based helpers ----
    def edge_fk_role(self, edge_type: tuple[str, str, str]) -> int:
        """Role id of a PyG edge type ``(src, rel, dst)`` — unambiguous in both directions."""
        return self.fk_role_id.get(canonical_edge_role(edge_type), FK_NONE)

    def edge_metapath(self, edge_type: tuple[str, str, str]) -> int:
        return self.metapath_id.get(canonical_edge_role(edge_type), MP_MULTIHOP)

    def _roles_for_relation(self, rel: str) -> list[tuple[str, str, str]]:
        col = canonical_relation(rel)
        return sorted(t for t in self.fk_role_id if t[1] == col)

    def relation_fk_role(self, rel: str) -> int:
        """Role id from a relation *name* alone. Only defined when the name is unambiguous.

        A name (``f2p_<col>``) identifies a role only if that FK column occurs in exactly one child
        table; otherwise the schema has several roles behind the name and the caller must say *which*
        edge type it means — use :meth:`edge_fk_role` with the ``(src, rel, dst)`` triple. Raising is
        deliberate: silently returning one of several ids is how the pre-P0.1 vocabulary lost 9 of
        rel-f1's 13 roles.
        """
        matches = self._roles_for_relation(rel)
        if not matches:
            return FK_NONE
        if len(matches) > 1:
            raise ValueError(
                f"relation {rel!r} is ambiguous: {len(matches)} roles share this FK column "
                f"({matches}). Use edge_fk_role((src, rel, dst)) instead."
            )
        return self.fk_role_id[matches[0]]

    def relation_metapath(self, rel: str) -> int:
        """Metapath id from a relation name alone; same ambiguity contract as :meth:`relation_fk_role`."""
        matches = self._roles_for_relation(rel)
        if not matches:
            return MP_MULTIHOP
        if len(matches) > 1:
            raise ValueError(
                f"relation {rel!r} is ambiguous: {len(matches)} roles share this FK column "
                f"({matches}). Use edge_metapath((src, rel, dst)) instead."
            )
        return self.metapath_id[matches[0]]


def _build_vocabs(node_types, edge_types) -> tuple[dict, dict, dict]:
    """Vocabs keyed by the FK **role triple** ``(child_table, fk_column, parent_table)`` (changes.md
    P0.1), sorted for determinism. Forward and reverse edge types canonicalize to the same triple, so
    they share a role; same-named FK columns in different child tables do **not**."""
    node_type_id = {nt: i for i, nt in enumerate(sorted(node_types))}
    roles = sorted({canonical_edge_role(et) for et in edge_types})
    fk_role_id = {r: i + 1 for i, r in enumerate(roles)}          # 0 reserved (FK_NONE)
    metapath_id = {r: i + 3 for i, r in enumerate(roles)}         # 0,1,2 reserved
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
