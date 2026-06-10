"""Claude-authored DocCards for real DBs, with provenance (implementation.md §10; plan stress-test #1).

We have no external documented database, so real-DB DocCards are **authored by Claude Code** (not the
Anthropic API). This module makes that reproducible and auditable. It does NOT call an LLM — the workflow
is:

  1. ``dump_schema_context(bundle, arm=...)`` writes a JSON the author (Claude Code) reads: per-table /
     per-column schema + sampled values, plus — only in the ``informed`` arm — the task description.
  2. Claude Code authors one DocCard per column and writes them back as JSON.
  3. ``load_authored_cards(path)`` validates them and attaches **provenance** (which arm, what the author
     was allowed to see) + the ``blind`` flag, for the audit's blind-authoring control.

Two arms (the control that turns 'you wrote the docs yourself' into a measurable arm):
  * ``informed`` — author may see the task/target.
  * ``blind``    — author sees ONLY schema + sample values; the task and dataset identity are withheld.

**Known limitation (stress-test #1):** column names themselves can reveal a public domain (``driverId`` ->
F1), so an LLM author can leak pretraining knowledge even when blind. This is *why* the synthetic
planted-truth existence proof (where the sign is random and unknowable) carries the falsifiability, and why
we also report inter-author agreement. Provenance records the leakage surface honestly rather than
pretending it is zero.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gloss.data.doccards import DocCard

ARMS = ("informed", "blind")


@dataclass
class Provenance:
    arm: str                       # "informed" | "blind"
    saw_task: bool
    saw_labels: bool
    dataset_revealed: bool
    authored_by: str = "claude-code"
    note: str = ""

    @classmethod
    def for_arm(cls, arm: str) -> "Provenance":
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        informed = arm == "informed"
        return cls(arm=arm, saw_task=informed, saw_labels=False, dataset_revealed=informed,
                   note="blind: task + dataset identity withheld; values/structure only"
                   if not informed else "informed: task target visible; labels never shown")

    def as_dict(self) -> dict:
        return {**self.__dict__}


def _sample_values(series, k: int = 8) -> list:
    vals = series.dropna().unique()[:k]
    return [v.item() if hasattr(v, "item") else (str(v) if not isinstance(v, (int, float, str, bool)) else v)
            for v in vals]


def dump_schema_context(bundle, arm: str = "informed", n_samples: int = 8) -> dict:
    """Build the context the author (Claude Code) reads. Honors the arm's information policy."""
    prov = Provenance.for_arm(arm)
    db = bundle.db
    schema = {}
    for tname, tbl in db.table_dict.items():
        cols = []
        for c in tbl.df.columns:
            cols.append({
                "column": c,
                "is_pkey": c == tbl.pkey_col,
                "fk_target": tbl.fkey_col_to_pkey_table.get(c),
                "is_time": c == tbl.time_col,
                "samples": _sample_values(tbl.df[c], n_samples),
            })
        schema[tname] = {"n_rows": int(len(tbl.df)), "columns": cols}
    ctx = {
        "dataset": bundle.dataset if prov.dataset_revealed else "ANONYMIZED_DB",
        "provenance": prov.as_dict(),
        "schema": schema,
    }
    if prov.saw_task:
        ctx["task"] = {"entity_table": bundle.entity_table, "target_col": bundle.target_col,
                       "task_type": bundle.task_type, "name": bundle.task_name}
    return ctx


def authoring_cache_dir(dataset: str, arm: str) -> Path:
    d = Path(__file__).resolve().parents[2] / "gloss" / "data" / "doccards_cache" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"cards_{arm}.json"


def validate_authored(obj: dict) -> None:
    """Minimal schema check on an authored-cards JSON blob."""
    if "cards" not in obj or not isinstance(obj["cards"], list):
        raise ValueError("authored JSON must have a 'cards' list")
    for c in obj["cards"]:
        for req in ("table", "column", "dtype"):
            if req not in c:
                raise ValueError(f"card missing required field {req!r}: {c}")


def save_authored_cards(cards: list[dict], dataset: str, arm: str, provenance: Provenance) -> Path:
    path = authoring_cache_dir(dataset, arm)
    blob = {"dataset": dataset, "arm": arm, "provenance": provenance.as_dict(), "cards": cards}
    validate_authored(blob)
    path.write_text(json.dumps(blob, indent=2))
    return path


def load_authored_cards(dataset: str, arm: str) -> dict[tuple[str, str], DocCard]:
    """Load Claude-authored cards -> ``{(table, column): DocCard}`` with provenance + blind flag attached."""
    path = authoring_cache_dir(dataset, arm)
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    validate_authored(blob)
    prov = blob.get("provenance", Provenance.for_arm(arm).as_dict())
    out: dict[tuple[str, str], DocCard] = {}
    for c in blob["cards"]:
        card = DocCard.from_json(c)
        card.blind = arm == "blind"
        card.provenance = prov
        out[(card.table, card.column)] = card
    return out
