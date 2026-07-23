"""Tests for preservation verification surfaced in audit."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_audit_verbatim_integrity_ok(service, contract_text):
    ing = service.ingest(contract_text, label="acme", source_type="contract")
    res = service.audit(ing["context_id"])
    vi = res["verbatim_integrity"]
    assert vi["ok"] is True
    assert vi["atoms_checked"] > 0
    assert vi["atoms_verbatim_recoverable"] == vi["atoms_checked"]
    assert vi["block_mismatches"] == []
    assert vi["atom_mismatches"] == []


def test_audit_recall_estimate_present(service, contract_text):
    ing = service.ingest(contract_text, label="acme", source_type="contract")
    res = service.audit(ing["context_id"])
    rec = res["extraction_recall_estimate"]
    # money and dates exist in the contract; recall should be high.
    assert "money" in rec
    assert rec["money"]["recall"] >= 0.5


def test_roundtrip_detects_corruption(service, contract_text):
    from savecontext import verify
    from savecontext.extraction import extract_atoms
    from savecontext.chunking import split_blocks

    blocks = split_blocks(contract_text)
    spans = [(b.block_id, b.start_char, b.end_char) for b in blocks]
    atoms = extract_atoms(contract_text, spans)
    bspans = [(b.block_id, b.start_char, b.end_char, b.text) for b in blocks]

    good = verify.roundtrip(contract_text, bspans, atoms)
    assert good["ok"] is True

    # Corrupt the source: atom offsets no longer match.
    corrupted = "X" + contract_text[1:]
    bad = verify.roundtrip(corrupted, bspans, atoms)
    assert bad["ok"] is False
