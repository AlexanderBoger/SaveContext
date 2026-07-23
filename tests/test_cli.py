"""CLI smoke tests: ingest a file, list, brief, expand, quote, audit, diff."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from savecontext import cli


DOC = """1. Fees
Customer shall pay $50,000 per month, due within 30 days.

2. Liability
Provider's total aggregate liability shall not exceed $500,000.
Provider shall indemnify Customer.

3. Confidentiality
Confidential information must not be disclosed to any third party.
"""


@pytest.fixture
def vault_db(tmp_path, monkeypatch):
    db = tmp_path / "cv.db"
    monkeypatch.setenv("SAVECONTEXT_DB", str(db))
    return str(db)


def run(capsys, *argv):
    rc = cli.main(list(argv))
    out = capsys.readouterr().out
    return rc, out


def test_cli_ingest_and_query_flow(tmp_path, vault_db, capsys):
    doc_file = tmp_path / "contract.md"
    doc_file.write_text(DOC)

    rc, out = run(capsys, "ingest", str(doc_file), "--label", "acme", "--type", "contract")
    assert rc == 0
    res = json.loads(out)
    cid = res["context_id"]
    assert cid == "ctx://acme@v1"

    # list
    rc, out = run(capsys, "list")
    assert rc == 0
    listing = json.loads(out)
    assert any(c["context_id"] == cid for c in listing["contexts"])

    # brief
    rc, out = run(capsys, "brief", cid, "--task", "liability cap")
    assert rc == 0
    assert "liability" in out.lower()

    # expand facts
    rc, out = run(capsys, "expand", cid, "--selector", "Liability", "--fidelity", "facts")
    assert rc == 0
    assert "money" in out.lower()

    # quote
    rc, out = run(capsys, "quote", cid, "--search", "shall not exceed")
    assert rc == 0
    assert "shall not exceed" in json.loads(out)["exact_source_quote"]

    # audit
    rc, out = run(capsys, "audit", cid)
    assert rc == 0
    audit = json.loads(out)
    assert audit["compression_ratio"] > 0  # tiny docs may not compress (<1)
    assert audit["estimated_preservation"]["money_values"] >= 1


def test_cli_brief_only(tmp_path, vault_db, capsys):
    doc_file = tmp_path / "d.txt"
    doc_file.write_text(DOC)
    rc, out = run(capsys, "ingest", str(doc_file), "--label", "x", "--brief-only")
    assert rc == 0
    assert "[contract]" in out or "blocks" in out


def test_cli_unknown_context_returns_error(vault_db, capsys):
    rc, out = run(capsys, "audit", "ctx://nope@v1")
    assert rc == 2
    assert json.loads(out)["error_type"] == "ValueError"


def test_cli_diff_via_files(tmp_path, vault_db, capsys):
    f1 = tmp_path / "v1.md"
    f1.write_text(DOC)
    run(capsys, "ingest", str(f1), "--label", "acme", "--type", "contract")

    f2 = tmp_path / "v2.md"
    f2.write_text(DOC.replace("$500,000", "$1,000,000"))
    rc, out = run(capsys, "diff", "ctx://acme@v1", str(f2))
    assert rc == 0
    res = json.loads(out)
    assert res["new_context_id"] == "ctx://acme@v2"
    assert "money" in " ".join(res["risk_or_meaning_impact"]).lower()
