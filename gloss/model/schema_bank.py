"""Per-database frozen tables, and binding them into one set of shared weights.

With :class:`~gloss.model.schema_free_encoder.SchemaFreeCellEncoder` the model's **parameters** are
already identical across schemas — a rel-f1 model and a rel-trial model have the same state_dict,
key for key and shape for shape. What is still per-database is the frozen *data* those parameters
read: the column-name table, the category-label table, per-column mean/std, the table/role name
tables, and the column layout of each node type.

So multi-database training does not need several models, or adapters, or any parameter surgery. It
needs one model and a **swap of buffer references** before each batch. That is what
:class:`SchemaBank` does.

Why a reference swap is safe here, when in general it would not be:

* every tensor involved is a **non-persistent buffer** — none is in `state_dict`, none is optimized,
  and swapping one cannot desynchronize an optimizer or a checkpoint;
* batches are **homogeneous by construction** — the pretrain stream draws each batch's seeds from one
  table of one database (`data/pretrain_loader.py`), because `to_cell_batch` needs a single
  `entity_table` anyway;
* the schedule is **rank-identical** — derived from `(global_step, seed)`, so under DDP every rank
  binds the same database at the same step and the all-reduce sees the same graph.

Binding is O(number of buffers) pointer assignments, no copies, and costs microseconds against a
batch that takes hundreds of milliseconds to collate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class SchemaTables:
    """Everything about one database that the shared weights read but do not learn."""

    dataset: str
    # cell encoder
    name_emb: Tensor                       # [C, d_text]   column-name table
    cat_emb: Tensor                        # [1+n_cat, d_text]  category-LABEL table
    num_mean: Tensor                       # [C]
    num_std: Tensor                        # [C]
    cat_base: Tensor                       # [C]
    modality_id: Tensor                    # [C]
    year_stats: Tensor                     # [2] global year mean/std
    layout: dict = field(default_factory=dict)     # node_type -> {stype: [column slots]}
    gids: dict = field(default_factory=dict)       # node_type -> [C_nt] global column ids
    # row level
    table_name_emb: Tensor | None = None   # [n_tables, d_text]
    role_name_emb: Tensor | None = None    # [K+1, d_text]

    def to(self, device) -> "SchemaTables":
        f = lambda t: None if t is None else t.to(device)
        return SchemaTables(
            dataset=self.dataset, name_emb=f(self.name_emb), cat_emb=f(self.cat_emb),
            num_mean=f(self.num_mean), num_std=f(self.num_std), cat_base=f(self.cat_base),
            modality_id=f(self.modality_id), year_stats=f(self.year_stats),
            layout=self.layout, gids={k: v.to(device) for k, v in self.gids.items()},
            table_name_emb=f(self.table_name_emb), role_name_emb=f(self.role_name_emb),
        )

    @classmethod
    def from_model(cls, model, dataset: str) -> "SchemaTables":
        """Snapshot the tables a freshly built model already holds for its own database."""
        enc, sub = model.encoder, model.substrate
        return cls(
            dataset=dataset, name_emb=enc.name_emb, cat_emb=enc.cat_emb, num_mean=enc.num_mean,
            num_std=enc.num_std, cat_base=enc.cat_base, modality_id=enc.modality_id,
            year_stats=enc.year_stats, layout=enc._layout, gids=enc._gids,
            table_name_emb=sub.row_sig.table_name_emb, role_name_emb=sub.row_sig.role_name_emb,
        )


def _set_buffer(module: nn.Module, name: str, value: Tensor | None) -> None:
    """Rebind a non-persistent buffer in place. Asserts it *is* one, so a parameter can never be hit."""
    if value is None:
        return
    assert name in module._buffers, f"{type(module).__name__}.{name} is not a buffer"
    module._buffers[name] = value


def bind(model, tables: SchemaTables) -> None:
    """Point ``model``'s frozen slots at ``tables``. No copies, no gradients, no state_dict change."""
    enc = model.encoder
    for attr in ("name_emb", "cat_emb", "num_mean", "num_std", "cat_base", "modality_id",
                 "year_stats"):
        _set_buffer(enc, attr, getattr(tables, attr))
    enc._layout, enc._gids = tables.layout, tables.gids

    _set_buffer(model.signature, "name_emb", tables.name_emb)
    _set_buffer(model.signature, "modality_id", tables.modality_id)

    sub = model.substrate
    _set_buffer(sub.row_sig, "table_name_emb", tables.table_name_emb)
    _set_buffer(sub.row_sig, "role_name_emb", tables.role_name_emb)
    for blk in sub.blocks:
        _set_buffer(blk.pool, "col_name_emb", tables.name_emb)
        _set_buffer(blk.row_attn, "role_name_emb", tables.role_name_emb)


class SchemaBank(nn.Module):
    """Holds every database's :class:`SchemaTables` and binds one at a time into a shared model.

    An ``nn.Module`` only so that ``.to(device)`` moves the tables with the model; it registers
    nothing persistent, so it adds no keys to any checkpoint.
    """

    def __init__(self, tables: dict[str, SchemaTables]):
        super().__init__()
        if not tables:
            raise ValueError("SchemaBank needs at least one database")
        self._tables = dict(tables)
        self.current: str | None = None
        widths = {t.name_emb.shape[-1] for t in tables.values()}
        if len(widths) > 1:
            raise ValueError(
                f"column-name tables disagree on d_text: {widths}. `schema_proj` and `name_proj` are "
                "shared weights of that width, so every database must use the same name encoder.")
        cat_widths = {t.cat_emb.shape[-1] for t in tables.values() if t.cat_emb.numel()}
        if len(cat_widths) > 1:
            raise ValueError(f"category-label tables disagree on d_text: {cat_widths}")

    @property
    def datasets(self) -> list[str]:
        return sorted(self._tables)

    def __getitem__(self, dataset: str) -> SchemaTables:
        return self._tables[dataset]

    def _apply(self, fn, *args, **kwargs):                            # type: ignore[override]
        """Move the tables with the model.

        Overriding ``_apply`` rather than ``to``: ``nn.Module.to`` recurses into children via
        ``_apply``, so a ``to()`` override on a child is never called when the *parent* is moved —
        which is exactly what Lightning does. The symptom is a CPU/CUDA mismatch on the first bound
        batch, and only for the non-anchor databases, because the anchor's tables are real buffers
        that moved with the model.
        """
        out = super()._apply(fn, *args, **kwargs)
        self._tables = {
            k: SchemaTables(
                dataset=t.dataset, name_emb=fn(t.name_emb), cat_emb=fn(t.cat_emb),
                num_mean=fn(t.num_mean), num_std=fn(t.num_std), cat_base=fn(t.cat_base),
                modality_id=fn(t.modality_id), year_stats=fn(t.year_stats), layout=t.layout,
                gids={g: fn(v) for g, v in t.gids.items()},
                table_name_emb=None if t.table_name_emb is None else fn(t.table_name_emb),
                role_name_emb=None if t.role_name_emb is None else fn(t.role_name_emb),
            )
            for k, t in self._tables.items()
        }
        self.current = None                    # force a rebind; the old references are stale
        return out

    def bind(self, model, dataset: str) -> None:
        """Bind ``dataset``'s tables unless they are already bound (the common case within a batch)."""
        if dataset != self.current:
            bind(model, self._tables[dataset])
            self.current = dataset
