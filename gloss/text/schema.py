"""Per-column **schema-name** embeddings — the frozen table MoRE's router routes on.

RT's cell token already carries a frozen-LM embedding of the column name (the *schema component*). MoRE
reuses exactly that signal: we embed each ``(table, column)`` once with a frozen text encoder and gather
it by the collate's global ``col_idx``. The table is built offline (cheap; |columns| is tens–hundreds),
cached, and frozen — there are **no LM forward passes in training**.

The text per column mirrors the phrasing used elsewhere in the stack (``"table <t>, column <c>"``) so
two same-named columns in different tables land at different points and semantically-similar columns
across tables land near each other — which is what makes signature routing schema-transferable.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..data.collate import column_vocab
from ..data.graph import GraphBundle, role_name


def column_name_text(node_type: str, column: str) -> str:
    """The natural-language string we embed for a column (table-qualified so it is unambiguous)."""
    return f"table {node_type}, column {column}"


def column_name_strings(bundle: GraphBundle) -> list[str]:
    """The per-column texts in global ``col_idx`` order (index i == ``column_vocab`` id i)."""
    vocab = column_vocab(bundle)
    texts: list[str] = [""] * len(vocab)
    for (nt, col), i in vocab.items():
        texts[i] = column_name_text(nt, col)
    return texts


def build_column_name_embeddings(bundle: GraphBundle, encode, *, kind: str = "document") -> Tensor:
    """Embed every column name with ``encode(texts, kind) -> [N, d_text]`` (L2-normalized rows).

    Returns a frozen ``[C, d_text]`` table indexed by the global column vocabulary (so it is gathered
    directly by a :class:`CellBatch`'s ``col_idxs``). ``encode`` is any cache/encoder from
    :mod:`gloss.text.cache` (or a plain callable); using an :class:`EmbeddingCache` makes the (single)
    encoder pass idempotent across runs.
    """
    texts = column_name_strings(bundle)
    emb = encode(texts, kind=kind)
    if not torch.is_tensor(emb):
        emb = torch.as_tensor(emb)
    return emb.detach().to(torch.float32).cpu()


# ---------------------------------------------------------------------------------------------
# P0.4 — name tables for TABLES and FK ROLES, the row level's counterpart to the column table.
#
# Both are name-DERIVED, which is the whole point: on a schema never seen in training, `n_tables`
# and `K` are new, so anything *indexed* by them is undefined. Embedding the NAME instead means an
# unseen table or role is representable on day one — you regenerate the table from the new schema's
# strings, with no gradients. Neither `n_tables` nor `K` may ever become a weight shape (§0).
# ---------------------------------------------------------------------------------------------


def table_name_text(node_type: str) -> str:
    """The string embedded for a table (changes.md P0.4)."""
    return f"table {node_type}"


def table_name_strings(bundle: GraphBundle) -> list[str]:
    """Per-table texts in ``bundle.node_type_id`` order (index i == table id i)."""
    texts: list[str] = [""] * len(bundle.node_type_id)
    for nt, i in bundle.node_type_id.items():
        texts[i] = table_name_text(nt)
    return texts


def role_name_strings(bundle: GraphBundle) -> list[str]:
    """Per-role texts in role-id order, index i == role id ``i + 1`` (``FK_NONE`` = 0 has no name).

    Delegates the string to ``graph.role_name`` so P0.1's vocabulary stays the single source of
    truth — the roles are keyed on the ``(child_table, fk_column, parent_table)`` triple, so
    same-named FK columns in different child tables are distinct (13 / 15 / 7 roles on
    rel-f1 / rel-trial / rel-event; they collapsed to 4 / 6 / 4 before P0.1).
    """
    return [role_name(t) for t in bundle.role_triples]


def build_table_name_embeddings(bundle: GraphBundle, encode, *, kind: str = "document") -> Tensor:
    """Frozen ``[n_tables, d_text]`` table-name embeddings, gathered by ``row_table``."""
    return _embed(table_name_strings(bundle), encode, kind)


def build_role_name_embeddings(bundle: GraphBundle, encode, *, kind: str = "document") -> Tensor:
    """Frozen ``[K, d_text]`` role-name embeddings, gathered by ``row_in_role`` / ``adj_role``.

    Row ``i`` is role id ``i + 1``; callers indexing with a raw role id must subtract 1, or index a
    ``[K+1, d_text]`` table built by prepending a zero row for ``FK_NONE``. See
    :func:`role_name_embeddings_with_none`.
    """
    return _embed(role_name_strings(bundle), encode, kind)


def role_name_embeddings_with_none(bundle: GraphBundle, encode, *, kind: str = "document") -> Tensor:
    """``[K+1, d_text]`` role table whose row 0 is an all-zero ``FK_NONE`` slot.

    Convenient for `adj_role`, whose 0 means "no edge" — gather directly without offsetting. The
    zero row is *data*, not a parameter, so it does not make `K` a weight shape.
    """
    emb = build_role_name_embeddings(bundle, encode, kind=kind)
    return torch.cat([torch.zeros(1, emb.shape[-1], dtype=emb.dtype), emb], dim=0)


def _embed(texts: list[str], encode, kind: str) -> Tensor:
    emb = encode(texts, kind=kind)
    if not torch.is_tensor(emb):
        emb = torch.as_tensor(emb)
    return emb.detach().to(torch.float32).cpu()


def assert_name_emb_width(emb: Tensor, *, encoder: str, expected: int | None = None) -> None:
    """Fail loudly if a cached name table has the wrong width for its encoder (P0.4).

    Encoder-DRIVEN rather than a hardcoded 2560, so the `hash` encoder still works in tests while a
    silently-truncated or wrong-encoder qwen cache still fails at startup rather than mid-run.
    """
    expected = expected if expected is not None else NAME_EMB_WIDTH.get(encoder)
    if expected is None:
        return
    got = int(emb.shape[-1])
    if got != expected:
        raise ValueError(
            f"name embedding width {got} != {expected} expected for encoder {encoder!r}. "
            f"A stale or wrong-encoder cache is the usual cause — rebuild with "
            f"scripts/build_schema_cache.py --encoder {encoder}."
        )


#: Known frozen-encoder widths. `hash` is dev/test only and is deliberately absent (any width ok).
NAME_EMB_WIDTH: dict[str, int] = {"qwen": 2560, "harrier": 5376}


# ---------------------------------------------------------------------------------------------
# P0.5 — the modality (stype) id space is PINNED to a fixed, declared enum.
#
# It used to be "the set of stypes actually present in this bundle, sorted by name", which made both
# the id MEANINGS and ``stype_emb``'s SHAPE dataset-dependent: ``numerical`` was id 1 on one bundle
# and id 2 on another, so a checkpoint trained on one silently mis-indexed on the next. That is the
# "embedding sized to the stypes that happen to occur in one bundle" the changes.md §0
# no-dataset-artifact rule names as illegal.
#
# Order is declared here ONCE and must never be reordered — appending is safe, reordering silently
# invalidates every existing checkpoint. Id 0 is reserved for "unknown / not present" so a stype this
# list has never seen (a future pytorch-frame release) degrades to a single shared row instead of
# raising or colliding with a real modality.
# ---------------------------------------------------------------------------------------------
STYPE_ORDER: tuple[str, ...] = (
    "numerical",
    "categorical",
    "multicategorical",
    "timestamp",
    "text_embedded",
    "text_tokenized",
    "embedding",
    "image_embedded",
    "sequence_numerical",
)

MODALITY_UNKNOWN = 0
#: Constant, bundle-independent. ``stype_emb`` is sized by this and by nothing else.
N_STYPES = 1 + len(STYPE_ORDER)

_STYPE_ID: dict[str, int] = {name: i + 1 for i, name in enumerate(STYPE_ORDER)}


def stype_id(st) -> int:
    """Fixed id for a pytorch-frame stype. Unknown stypes -> ``MODALITY_UNKNOWN`` (0).

    Accepts a ``torch_frame.stype`` or its string name. Keyed on ``.value``/``str`` rather than on
    enum identity so it survives a pytorch-frame version bump that re-creates the enum object.
    """
    name = getattr(st, "value", None) or str(st).rsplit(".", 1)[-1]
    return _STYPE_ID.get(name, MODALITY_UNKNOWN)


def build_column_modality_ids(bundle: GraphBundle) -> tuple[Tensor, int]:
    """Per-column **modality** id (the pytorch-frame stype) + the pinned stype count.

    Returns ``(modality_id [C] long, N_STYPES)`` indexed by the global column vocabulary. The id
    space is the **fixed** :data:`STYPE_ORDER` enum, identical across every dataset, so
    ``n_stypes`` is a constant and a given stype has the same id everywhere — see P0.5. Stypes
    absent from a bundle simply leave their row unused; stypes absent from ``STYPE_ORDER`` map to
    :data:`MODALITY_UNKNOWN`.

    The router routes on this as the modality axis of the signature.
    """
    vocab = column_vocab(bundle)
    modality = torch.full((len(vocab),), MODALITY_UNKNOWN, dtype=torch.long)
    for nt in bundle.node_types:
        for st, cols in bundle.data[nt].tf.col_names_dict.items():
            sid = stype_id(st)
            for c in cols:
                if (nt, c) in vocab:
                    modality[vocab[(nt, c)]] = sid
    return modality, N_STYPES
