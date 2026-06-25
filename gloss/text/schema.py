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
from ..data.graph import GraphBundle


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
