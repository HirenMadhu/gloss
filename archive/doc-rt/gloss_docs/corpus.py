"""Phase 1 — load + validate the prose doc corpus, chunk it into spans, enumerate schema elements.

A corpus is one ``docs.md`` (senior-dev prose) + ``meta.yaml`` (tier/author/blind attestation) per DB.
Grounding (``grounding.py``) retrieves spans for each *schema element* — a table, a column, or an FK
role — using a short descriptor query. Coverage = fraction of elements that ground above threshold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_REQUIRED_META = ("tier", "author", "blind")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class SchemaElement:
    """A groundable schema element. ``key`` is stable and used to gather embeddings by id."""

    key: str
    kind: str            # 'table' | 'column' | 'fk_role'
    table: str
    column: str | None
    query: str           # short descriptor used as the retrieval query


@dataclass
class DocCorpus:
    dataset: str
    markdown: str
    meta: dict
    path: Path

    @classmethod
    def load(cls, corpus_root: str | Path, dataset: str) -> "DocCorpus":
        root = Path(corpus_root) / dataset
        md = (root / "docs.md").read_text(encoding="utf-8")
        meta = yaml.safe_load((root / "meta.yaml").read_text(encoding="utf-8")) or {}
        missing = [k for k in _REQUIRED_META if k not in meta]
        if missing:
            raise ValueError(f"{root/'meta.yaml'} missing required keys: {missing}")
        if meta.get("blind") is True and not str(meta.get("attestation", "")).strip():
            raise ValueError(f"{root/'meta.yaml'}: blind=true requires a non-empty 'attestation'")
        return cls(dataset=dataset, markdown=md, meta=meta, path=root)

    def spans(self, max_sentences: int = 3) -> list[str]:
        """Chunk the prose into ~``max_sentences``-sentence spans (headings/comments stripped)."""
        text = _HTML_COMMENT.sub(" ", self.markdown)
        # drop markdown heading lines but keep their prose paragraphs
        lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        paragraphs = re.split(r"\n\s*\n", "\n".join(lines))
        spans: list[str] = []
        for para in paragraphs:
            para = " ".join(para.split())
            if not para:
                continue
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip()]
            for i in range(0, len(sentences), max_sentences):
                chunk = " ".join(sentences[i : i + max_sentences])
                if chunk:
                    spans.append(chunk)
        return spans


def schema_elements_from_tables(tables: dict[str, dict]) -> list[SchemaElement]:
    """Build groundable elements from a lightweight schema spec (also used by tests).

    ``tables`` maps table -> {"columns": [...], "pkey": str|None, "fkeys": {col: dst_table}, ...}.
    """
    elements: list[SchemaElement] = []
    for t, spec in tables.items():
        elements.append(SchemaElement(f"table::{t}", "table", t, None, f"table {t}"))
        pkey = spec.get("pkey")
        fkeys = spec.get("fkeys", {}) or {}
        for c in spec.get("columns", []):
            if c == pkey:
                continue
            elements.append(SchemaElement(f"col::{t}::{c}", "column", t, c, f"table {t}, column {c}"))
            if c in fkeys:
                dst = fkeys[c]
                elements.append(
                    SchemaElement(
                        f"fk::{t}::{c}", "fk_role", t, c,
                        f"table {t}, foreign key {c} referencing {dst}",
                    )
                )
    return elements


def schema_elements_from_db(db) -> list[SchemaElement]:
    """Build elements from a relbench ``Database`` object."""
    tables = {}
    for name, tbl in db.table_dict.items():
        tables[name] = {
            "columns": list(tbl.df.columns),
            "pkey": tbl.pkey_col,
            "fkeys": dict(tbl.fkey_col_to_pkey_table),
        }
    return schema_elements_from_tables(tables)


def coverage_report(grounded_by_key: dict[str, bool], elements: list[SchemaElement]) -> dict:
    """Fraction of elements grounded above threshold, overall and by kind."""
    by_kind: dict[str, list[bool]] = {}
    for e in elements:
        by_kind.setdefault(e.kind, []).append(bool(grounded_by_key.get(e.key, False)))
    report = {"overall": _frac([v for vs in by_kind.values() for v in vs])}
    for kind, vs in by_kind.items():
        report[kind] = _frac(vs)
    report["counts"] = {k: (sum(vs), len(vs)) for k, vs in by_kind.items()}
    return report


def _frac(bools: list[bool]) -> float:
    return float(sum(bools)) / len(bools) if bools else 0.0
