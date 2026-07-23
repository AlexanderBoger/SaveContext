"""Tests for agent-authored briefs (ingest agent_brief + set_brief)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from savecontext import cli


def test_ingest_with_agent_brief(service, contract_text):
    res = service.ingest(
        contract_text, label="acme", source_type="contract",
        agent_brief="Acme/Globex MSA: $50k/mo fee, liability capped at $500k, 24-month term.",
    )
    assert res["brief_mode"] == "agent"
    assert "500k" in res["semantic_brief"]
    # extractive-only hint should NOT be present when agent supplied a brief
    assert "brief_upgrade_hint" not in res


def test_extractive_ingest_offers_upgrade_hint(service, contract_text):
    res = service.ingest(contract_text, label="acme2", source_type="contract")
    assert res["brief_mode"] == "extractive"
    assert "set_brief" in res["brief_upgrade_hint"]


def test_set_brief_replaces_and_persists(service, contract_text):
    ing = service.ingest(contract_text, label="acme", source_type="contract")
    cid = ing["context_id"]
    new_brief = "Concise agent brief: fees, $500k liability cap, indemnity, 24 months."
    out = service.set_brief(cid, new_brief)
    assert out["brief_mode"] == "agent"
    assert out["semantic_brief"] == new_brief
    assert out["compression_ratio"] > 0

    # Persisted: audit reflects the new mode + ratio.
    aud = service.audit(cid)
    assert aud["brief_mode"] == "agent"
    assert aud["compression_ratio"] == out["compression_ratio"]


def test_set_brief_unknown_context(service):
    with pytest.raises(ValueError):
        service.set_brief("ctx://nope@v1", "x")


def test_set_brief_rejects_empty(service, contract_text):
    ing = service.ingest(contract_text, label="acme")
    with pytest.raises(ValueError):
        service.set_brief(ing["context_id"], "   ")


def test_atoms_unchanged_after_set_brief(service, contract_text):
    ing = service.ingest(contract_text, label="acme", source_type="contract")
    cid = ing["context_id"]
    before = service.audit(cid)["protected_atom_counts"]
    service.set_brief(cid, "totally different short brief")
    after = service.audit(cid)
    assert after["protected_atom_counts"] == before
    assert after["verbatim_integrity"]["ok"] is True  # raw/atoms untouched


def test_cli_set_brief_via_text(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.setenv("SAVECONTEXT_DB", str(tmp_path / "cv.db"))
    f = tmp_path / "d.md"
    f.write_text("Customer shall pay $50,000 on 2024-01-01. " * 8)
    cli.main(["ingest", str(f), "--label", "acme"])
    capsys.readouterr()
    rc = cli.main(["set-brief", "ctx://acme@v1", "--text", "Agent brief here."])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["brief_mode"] == "agent"
    assert out["semantic_brief"] == "Agent brief here."
