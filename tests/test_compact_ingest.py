"""Tests for compact ingest response + lazy map tool (token efficiency)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext.service import INGEST_OUTLINE_LIMIT


def _big_doc(n_sections: int) -> str:
    parts = []
    for i in range(n_sections):
        parts.append(
            f"Section {i + 1}\nClause {i + 1}: party shall pay ${1000 + i} "
            f"within {10 + i} days under article {i + 1}.\n"
        )
    return "\n\n".join(parts)


def test_ingest_outline_capped(service):
    doc = _big_doc(80)
    res = service.ingest(doc, label="big", source_type="contract")
    assert res["block_count"] >= 70
    assert len(res["block_map"]) <= INGEST_OUTLINE_LIMIT
    assert res["block_map_truncated"] is True
    # Compact entries carry no preview.
    assert "preview" not in res["block_map"][0]
    assert "navigation_hint" in res


def test_ingest_atoms_summary_has_no_examples(service, contract_text):
    res = service.ingest(contract_text, label="acme", source_type="contract")
    summ = res["protected_atoms_summary"]
    assert "money" in summ  # type keys + counts still present
    assert summ["money"]["count"] >= 1
    assert "examples" not in summ["money"]  # examples dropped for compactness


def test_small_doc_not_truncated(service, contract_text):
    res = service.ingest(contract_text, label="acme", source_type="contract")
    assert res["block_map_truncated"] is False
    assert len(res["block_map"]) == res["block_count"]


def test_map_returns_full_with_previews(service):
    doc = _big_doc(80)
    ing = service.ingest(doc, label="big", source_type="contract")
    full = service.map(ing["context_id"])
    assert full["block_count"] == ing["block_count"]
    assert full["returned"] == full["block_count"]
    assert "preview" in full["block_map"][0]
    assert "atom_count" in full["block_map"][0]
    # Atom examples available on demand here (unlike compact ingest).
    assert any("examples" in v for v in full["protected_atoms_summary"].values())


def test_map_pagination(service):
    doc = _big_doc(80)
    ing = service.ingest(doc, label="big", source_type="contract")
    page = service.map(ing["context_id"], with_previews=False, offset=10, limit=5)
    assert page["offset"] == 10
    assert page["returned"] == 5
    assert "preview" not in page["block_map"][0]
    assert page["block_map"][0]["block_id"] == "b0010"
