"""End-to-end tests for the eight MCP tools via VaultService."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext.service import detect_source_type


def test_ingest_basic(service, contract_text):
    res = service.ingest(contract_text, label="Acme Contract", source_type="auto")
    assert res["context_id"] == "ctx://acme-contract@v1"
    assert res["source_type"] == "contract"
    assert res["token_estimate_original"] > res["token_estimate_brief"]
    assert res["compression_ratio"] > 1.0
    assert res["block_map"]
    assert "money" in res["protected_atoms_summary"]
    assert res["safety_notes"]
    # brief must not contain the full source
    assert len(res["semantic_brief"]) < len(contract_text)


def test_ingest_versioning(service, contract_text):
    v1 = service.ingest(contract_text, label="acme", source_type="contract")
    v2 = service.ingest(contract_text + "\nExtra clause.", label="acme", source_type="contract")
    assert v1["context_id"] == "ctx://acme@v1"
    assert v2["context_id"] == "ctx://acme@v2"
    assert v2["version"] == 2


def test_ingest_rejects_empty(service):
    import pytest

    with pytest.raises(ValueError):
        service.ingest("   ", label="x")


def test_brief_task_focus(service, contract_text):
    ing = service.ingest(contract_text, label="acme", source_type="contract")
    res = service.brief(ing["context_id"], task="liability risks")
    assert res["relevant_blocks"]
    # a liability-related block should rank in
    headings = " ".join(b["heading"] for b in res["relevant_blocks"]).lower()
    assert "liability" in headings or any(
        "liability" in a["value"].lower() or "indemnif" in a["value"].lower()
        for a in res["relevant_atoms"]
    )
    assert res["uncertainty_notes"]


def test_expand_fidelity_levels(service, contract_text):
    ing = service.ingest(contract_text, label="acme", source_type="contract")
    cid = ing["context_id"]

    summ = service.expand(cid, selector="Liability", fidelity="summary")
    facts = service.expand(cid, selector="Liability", fidelity="facts")
    quotes = service.expand(cid, selector="Liability", fidelity="quotes")
    full = service.expand(cid, selector="Liability", fidelity="full")

    assert summ["expanded_content"]
    assert "[" in facts["expanded_content"]  # has atom annotations
    assert quotes["exact_quotes"]
    # full fidelity returns verbatim text present in source
    assert full["expanded_content"] in contract_text
    # quotes must be verbatim substrings of source
    for q in quotes["exact_quotes"]:
        assert q["quote"] in contract_text


def test_expand_by_block_id(service, contract_text):
    ing = service.ingest(contract_text, label="acme")
    first_block = ing["block_map"][0]["block_id"]
    res = service.expand(ing["context_id"], selector=first_block, fidelity="full")
    assert res["source_refs"][0]["block_id"] == first_block


def test_quote_by_atom_and_search(service, contract_text):
    ing = service.ingest(contract_text, label="acme")
    cid = ing["context_id"]

    # find a money atom id
    atoms = service.store.get_atoms(cid)
    money_atom = next(a for a in atoms if a["type"] == "money")
    q = service.quote(cid, atom_id=money_atom["atom_id"])
    assert q["exact_source_quote"] == money_atom["value"]
    assert q["surrounding_context"]
    assert q["source_ref"]["char_range"]

    qs = service.quote(cid, search_query="indemnify")
    assert qs["exact_source_quote"].lower() == "indemnify"

    missing = service.quote(cid, search_query="zzz-not-present")
    assert missing["exact_source_quote"] == ""


def test_diff_detects_money_change(service, contract_text):
    ing = service.ingest(contract_text, label="acme", source_type="contract")
    changed = contract_text.replace("$50,000", "$75,000").replace(
        "$500,000", "$250,000"
    )
    res = service.diff(ing["context_id"], changed)
    assert res["new_context_id"] == "ctx://acme@v2"
    # money changes should appear as added/removed/changed and in risk impact
    impact = " ".join(res["risk_or_meaning_impact"]).lower()
    assert "money" in impact
    assert "75,000" in str(res) and "50,000" in str(res)
    assert res["compact_patch_summary"]


def test_zip_and_expand_output(service):
    filler = "This section contains a great deal of supporting narrative detail. " * 12
    content = (
        "# Report\nThis is the intro. " + filler + "\n\n"
        "## Findings\nWe found that revenue grew 12% to $3,000,000. " + filler + "\n\n"
        "## Recommendations\nHire 5 engineers and ship by 2025-01-01. " + filler + "\n"
    )
    z = service.zip_output(content, label="q4 report", structure="markdown")
    assert z["output_id"] == "out://q4-report@v1"
    assert z["section_map"]
    # On representative-sized output the preview is materially smaller.
    assert z["token_estimate_original"] > z["token_estimate_preview"]
    assert z["compression_ratio"] > 1.0

    sec = service.expand_output(z["output_id"], selector="Findings", fidelity="section")
    assert "revenue" in sec["expanded_output_content"].lower()

    full = service.expand_output(z["output_id"], selector="*", fidelity="full")
    assert full["expanded_output_content"] == content

    prev = service.expand_output(z["output_id"], selector="*", fidelity="preview")
    assert prev["expanded_output_content"]


def test_audit(service, contract_text):
    ing = service.ingest(contract_text, label="acme", source_type="contract")
    res = service.audit(ing["context_id"])
    assert res["compression_ratio"] > 1.0
    assert res["protected_atom_counts"]
    pres = res["estimated_preservation"]
    assert set(pres) >= {
        "names", "dates", "numbers", "money_values", "negations",
        "conditions", "obligations", "code_identifiers",
    }
    assert pres["money_values"] >= 1
    assert pres["dates"] >= 1
    assert res["safe_for"] and res["unsafe_for"]


def test_unknown_context_errors(service):
    import pytest

    with pytest.raises(ValueError):
        service.brief("ctx://nope@v1", task="x")
    with pytest.raises(ValueError):
        service.audit("ctx://nope@v1")


def test_source_type_detection(code_text, contract_text):
    assert detect_source_type(code_text) == "code"
    assert detect_source_type(contract_text) == "contract"
    assert detect_source_type("2024-01-01T10:00:00 ERROR boom\n" * 5) == "logs"
    assert detect_source_type("just some plain words here") == "generic"
