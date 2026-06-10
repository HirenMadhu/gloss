"""DocCard schema, template renderer, and the audit regimes (implementation.md §4.1).

A DocCard is a *structured* per-column passage — not RT's bare ``"<col> of <table>"`` string. Each card is
rendered to text by a fixed template, then encoded **once** by the frozen Qwen encoder (``text_cache``) and
gathered by ``col_global_id``.

Three regimes power the audit:
  * ``full``      — the whole structured card (the headline modality).
  * ``name_only`` — RT-style ``"<column> of <table>"`` (the baseline; what name-shuffle perturbs).
  * ``placebo``   — length-matched, **semantically null** filler (the capacity/leakage control; stress-test #2).
Plus a ``blind`` flag recording that the card was authored without sight of labels/task.

Synthetic cards are rendered **programmatically** from ``PlantedTruth`` (Claude never sees the sign);
real-DB cards are authored by Claude Code (``doccard_authoring``) and loaded here.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Literal

Regime = Literal["full", "name_only", "placebo"]

# deterministic, semantically-null vocabulary for placebo cards (length-matched, content-free)
_PLACEBO_VOCAB = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore "
    "et dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris nisi aliquip "
    "ex ea commodo consequat duis aute irure reprehenderit voluptate velit esse cillum eu fugiat nulla"
).split()


@dataclass
class DocCard:
    """Structured per-column documentation (implementation.md §4.1)."""

    table: str
    column: str
    dtype: str                              # numeric|categorical|text|datetime|bool|id
    table_desc: str = ""
    column_desc: str = ""
    unit: str | None = None                 # "USD", "days", "count"
    null_semantics: str | None = None       # "NULL = not yet shipped"
    coded_values: dict | None = None        # {0:"active", 1:"churned"}
    fk_role: str | None = None              # "buyer_id -> users.id: the user who PLACED this order"
    fk_target: str | None = None            # "users.id"
    blind: bool = False                     # authored without seeing labels/task
    provenance: dict = field(default_factory=dict)  # what the author saw (audit trail)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "DocCard":
        known = {k: d.get(k) for k in cls.__dataclass_fields__}
        return cls(**known)  # type: ignore[arg-type]


def _render_full(card: DocCard) -> str:
    parts = [f"Table {card.table}: {card.table_desc}".strip(),
             f"Column {card.column} ({card.dtype}): {card.column_desc}".strip()]
    if card.unit:
        parts.append(f"Unit: {card.unit}.")
    if card.null_semantics:
        parts.append(f"Null semantics: {card.null_semantics}.")
    if card.coded_values:
        coded = "; ".join(f"{k} = {v}" for k, v in card.coded_values.items())
        parts.append(f"Coded values: {coded}.")
    if card.fk_role:
        parts.append(f"Foreign-key role: {card.fk_role}.")
    return " ".join(p for p in parts if p)


def _render_name_only(card: DocCard) -> str:
    return f"{card.column} of {card.table}"


def _render_placebo(card: DocCard, target_words: int, salt: int = 0) -> str:
    """Length-matched, content-free filler. Deterministic given (table, column, salt)."""
    h = hashlib.blake2b(f"{card.table}.{card.column}.{salt}".encode(), digest_size=8).digest()
    rng_state = int.from_bytes(h, "big")
    words = []
    for i in range(max(target_words, 1)):
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        words.append(_PLACEBO_VOCAB[rng_state % len(_PLACEBO_VOCAB)])
    return " ".join(words)


def render(card: DocCard, regime: Regime = "full") -> str:
    """Render a card to text under the chosen audit regime."""
    if regime == "name_only":
        return _render_name_only(card)
    if regime == "full":
        return _render_full(card)
    if regime == "placebo":
        # match the *full* rendering's word count so placebo is length-matched but semantically null
        target = len(_render_full(card).split())
        return _render_placebo(card, target_words=target)
    raise ValueError(f"unknown regime: {regime!r}")


# --------------------------------------------------------------------------------------------------
# Default / programmatic cards
# --------------------------------------------------------------------------------------------------
def default_card(table: str, column: str, dtype: str, fk_target: str | None = None) -> DocCard:
    """A minimal card from schema alone (used before Claude authoring / for columns without a card)."""
    fk_role = f"{column} -> {fk_target}" if fk_target else None
    return DocCard(table=table, column=column, dtype=dtype, fk_target=fk_target, fk_role=fk_role,
                   column_desc=f"the {column} field of {table}")


def render_synthetic_card(table: str, column: str, planted) -> DocCard:
    """Programmatic card for the synthetic generator — encodes the planted sign WITHOUT any LLM.

    ``planted`` is a ``gloss.data.synthetic.PlantedTruth``. The causal twin's card documents its coded
    sign ('higher -> outcome more likely / less likely'); the decoy's card documents that it is inert.
    """
    if column == planted.causal_col:
        direction = "more likely" if planted.sign > 0 else "less likely"
        desc = (f"Measured signal that drives the outcome. Higher recent average of this column makes "
                f"the positive outcome {direction}.")
        coded = {"sign": "+1 (higher = outcome more likely)" if planted.sign > 0
                 else "-1 (higher = outcome less likely)"}
        return DocCard(table=table, column=column, dtype="numeric", column_desc=desc,
                       unit="z-score", coded_values=coded,
                       table_desc="event log of measurements per entity over time")
    if column == planted.decoy_col:
        return DocCard(table=table, column=column, dtype="numeric",
                       column_desc="A measured signal with no documented effect on the outcome (inert).",
                       unit="z-score", table_desc="event log of measurements per entity over time")
    return default_card(table, column, dtype="numeric")
