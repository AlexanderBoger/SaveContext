"""Tests for the batched multi-query lookup tool.

lookup() answers N retrieval queries in ONE call: for each query it returns
the top-ranked block(s) with summary, the protected atoms inside them, and a
verbatim best-matching sentence with exact offsets. The point is round-trip
elimination — an agent answering several questions should need one call, not
one brief plus one expand/quote per question.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _ingest(service, contract_text):
    return service.ingest(contract_text, label="acme", source_type="contract")["context_id"]


def test_lookup_batch_answers_each_query(service, contract_text):
    cid = _ingest(service, contract_text)
    res = service.lookup(cid, [
        "total liability cap dollar amount",
        "monthly fee payment",
        "termination notice period",
    ])
    assert res["context_id"] == cid
    assert len(res["results"]) == 3
    by_query = {r["query"]: r for r in res["results"]}

    liab = by_query["total liability cap dollar amount"]
    assert any("liability" in m["heading"].lower() for m in liab["matches"])
    liab_values = [a["value"] for m in liab["matches"] for a in m["facts"]]
    assert "$500,000" in liab_values

    fee = by_query["monthly fee payment"]
    fee_values = [a["value"] for m in fee["matches"] for a in m["facts"]]
    assert "$50,000" in fee_values


def test_lookup_verbatim_sentence_is_exact(service, contract_text):
    cid = _ingest(service, contract_text)
    res = service.lookup(cid, ["liability cap"])
    match = res["results"][0]["matches"][0]
    v = match["verbatim"]
    start, end = v["char_range"]
    # Loss-aware guarantee: the quoted sentence reconstructs byte-for-byte.
    assert contract_text[start:end] == v["quote"]
    assert "liability" in v["quote"].lower()


def test_lookup_no_hit_reports_empty_matches(service, contract_text):
    cid = _ingest(service, contract_text)
    res = service.lookup(cid, ["quantum entanglement warranty"])
    r = res["results"][0]
    assert r["matches"] == []
    assert "note" in r


def test_lookup_token_estimate_and_caps(service, contract_text):
    cid = _ingest(service, contract_text)
    res = service.lookup(cid, ["liability", "fees"], top_blocks=1)
    assert res["token_estimate"] > 0
    for r in res["results"]:
        assert len(r["matches"]) <= 1


def test_lookup_rejects_empty_queries(service, contract_text):
    cid = _ingest(service, contract_text)
    with pytest.raises(ValueError):
        service.lookup(cid, [])


def test_lookup_unknown_context_raises(service):
    with pytest.raises(Exception):
        service.lookup("ctx://nope@v1", ["anything"])


# --- semantic upgrades: intent boost, concept expansion, slim payloads ---


def test_lookup_paraphrase_queries_hit(service, contract_text):
    """Queries with no lexical overlap with the clause must still rank it #1."""
    cid = _ingest(service, contract_text)
    res = service.lookup(cid, [
        "how much money is at risk at most if things go wrong",
        "what does the service cost each month",
    ])
    liab, fee = res["results"]
    assert liab["matches"], "paraphrase liability query found nothing"
    assert "liability" in liab["matches"][0]["heading"].lower()
    assert fee["matches"], "paraphrase fee query found nothing"
    assert "fee" in fee["matches"][0]["heading"].lower()


def test_lookup_facts_filtered_to_query_intent(service, contract_text):
    """A money question should not drag every number/section_ref atom along."""
    cid = _ingest(service, contract_text)
    res = service.lookup(cid, ["total liability cap dollar amount"])
    match = res["results"][0]["matches"][0]
    values = [a["value"] for a in match["facts"]]
    assert "$500,000" in values
    # intent filtering: facts stay small and money-relevant
    assert len(match["facts"]) <= 8
    assert any(a["type"] == "money" for a in match["facts"])


def test_lookup_adaptive_second_block(service, contract_text):
    """A decisive top block should not drag a weak runner-up into the payload."""
    cid = _ingest(service, contract_text)
    res = service.lookup(cid, ["recurring monthly service fee dollar amount"])
    matches = res["results"][0]["matches"]
    if len(matches) > 1:
        # runner-up only allowed when it is genuinely competitive
        assert matches[1]["score"] >= 0.55 * matches[0]["score"]


# --- abstention signal: weak matches must be flagged, not dressed up ---


def test_lookup_flags_weak_matches(service, contract_text):
    """A query about something absent must say so, not present filler
    confidently — agents working from compressed context must be able to
    abstain instead of guessing."""
    cid = _ingest(service, contract_text)
    res = service.lookup(cid, ["data breach notification deadline hours"])
    r = res["results"][0]
    if r["matches"]:
        assert r["confidence"] == "weak"
        assert "note" in r
    else:
        assert "note" in r


def test_lookup_strong_match_is_marked_strong(service, contract_text):
    cid = _ingest(service, contract_text)
    res = service.lookup(cid, ["total liability cap dollar amount"])
    r = res["results"][0]
    assert r["matches"]
    assert r["confidence"] == "strong"


# --- held-out-eval regressions: unpunctuated text and flat score fields ---


def test_lookup_verbatim_bounded_for_unpunctuated_logs(service):
    """Log lines have no sentence punctuation; the verbatim quote must be a
    line, not the entire block (regression: 1.4M-token payloads)."""
    log = "\n".join(
        f"2026-03-14T00:{i//60:02d}:{i%60:02d}Z INFO svc-{i%7} req latency_ms={i}"
        for i in range(2000)
    )
    log += "\n2026-03-14T09:00:00Z ERROR payments circuit breaker OPEN after 47 consecutive failures"
    cid = service.ingest(log, label="log", source_type="logs")["context_id"]
    res = service.lookup(cid, ["circuit breaker consecutive failures"])
    m = res["results"][0]["matches"][0]
    assert "47" in m["verbatim"]["quote"]
    assert len(m["verbatim"]["quote"]) < 500
    assert len(m["summary"]) < 700
    import json as _json
    assert len(_json.dumps(res)) < 20000


def test_lookup_flat_score_field_is_weak(service):
    """A query whose terms smear evenly across every block (no block stands
    out) must be flagged weak even when absolute scores are nonzero."""
    doc = "\n\n".join(
        f"## Section {i}\nThe server must support clients. Servers support "
        f"connections and versions. Item {i} applies to the server."
        for i in range(30)
    )
    cid = service.ingest(doc, label="flat", source_type="generic")["context_id"]
    res = service.lookup(cid, ["which cipher suites must servers support"])
    assert res["results"][0]["confidence"] == "weak"
