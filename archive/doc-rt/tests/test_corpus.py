"""Phase 1 — corpus loading/validation, chunking, schema elements, coverage math."""
from __future__ import annotations

from pathlib import Path

import pytest

from gloss.docs.corpus import (
    DocCorpus,
    SchemaElement,
    coverage_report,
    schema_elements_from_tables,
)

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "doc_corpus"


def test_load_rel_f1_corpus_and_attestation():
    c = DocCorpus.load(CORPUS_ROOT, "rel-f1")
    assert c.meta["tier"] == 2 and c.meta["blind"] is True
    assert c.meta["author"] and c.meta["attestation"].strip()
    assert "Formula" in c.markdown


def test_spans_are_chunked_and_clean():
    c = DocCorpus.load(CORPUS_ROOT, "rel-f1")
    spans = c.spans(max_sentences=3)
    assert len(spans) > 10
    assert all(not s.lstrip().startswith("#") for s in spans)      # headings stripped
    assert all("<!--" not in s for s in spans)                     # comments stripped
    assert all(s.strip() for s in spans)


def test_schema_elements_from_tables():
    spec = {
        "orders": {"columns": ["id", "buyer_id", "amount"], "pkey": "id",
                   "fkeys": {"buyer_id": "users"}},
        "users": {"columns": ["id", "name"], "pkey": "id", "fkeys": {}},
    }
    els = schema_elements_from_tables(spec)
    kinds = {(e.kind, e.table, e.column) for e in els}
    assert ("table", "orders", None) in kinds
    assert ("column", "orders", "amount") in kinds
    assert ("fk_role", "orders", "buyer_id") in kinds      # fk role element for the FK column
    assert ("column", "orders", "id") not in kinds         # pkey excluded
    # the fk column also appears as a plain column
    assert ("column", "orders", "buyer_id") in kinds


def test_meta_validation_rejects_missing_keys(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "docs.md").write_text("hello.")
    (tmp_path / "x" / "meta.yaml").write_text("tier: 2\n")  # missing author, blind
    with pytest.raises(ValueError):
        DocCorpus.load(tmp_path, "x")


def test_coverage_report_math():
    els = [
        SchemaElement("table::t", "table", "t", None, "table t"),
        SchemaElement("col::t::a", "column", "t", "a", "q"),
        SchemaElement("col::t::b", "column", "t", "b", "q"),
    ]
    grounded = {"table::t": True, "col::t::a": True, "col::t::b": False}
    rep = coverage_report(grounded, els)
    assert rep["overall"] == pytest.approx(2 / 3)
    assert rep["table"] == 1.0 and rep["column"] == 0.5
    assert rep["counts"]["column"] == (1, 2)
