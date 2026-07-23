"""Tests for BM25 retrieval and its effect on brief/expand relevance."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext.retrieval import BM25Index, tokenize


DOCS = [
    ("b0", "The cat sat on the warm mat in the sun."),
    ("b1", "Provider total aggregate liability shall not exceed five hundred thousand dollars."),
    ("b2", "Payment of monthly fees is due within thirty days of the invoice date."),
    ("b3", "Confidential information must not be disclosed to any third party."),
]


def test_tokenize_drops_stopwords():
    toks = tokenize("The cat is on the mat")
    assert "the" not in toks and "is" not in toks and "on" not in toks
    assert "cat" in toks and "mat" in toks


def test_bm25_ranks_relevant_doc_first():
    idx = BM25Index(DOCS)
    ranked = idx.rank("liability cap maximum")
    assert ranked, "expected at least one match"
    assert ranked[0][0] == "b1"  # the liability clause ranks first


def test_bm25_rare_term_outranks_common():
    idx = BM25Index(DOCS)
    # "disclosed" is rare (1 doc) and should pull b3 to the top.
    ranked = idx.rank("disclosed information")
    assert ranked[0][0] == "b3"


def test_bm25_empty_query_scores_zero():
    idx = BM25Index(DOCS)
    scores = idx.score("")
    assert set(scores.values()) == {0.0}


def test_bm25_no_overlap_returns_empty_rank():
    idx = BM25Index(DOCS)
    assert idx.rank("zebra helicopter quantum") == []


def test_brief_uses_bm25_for_synonym_free_query(service):
    # Query uses "cap" / "maximum exposure" wording not verbatim in the source,
    # but BM25 on shared terms ("liability") still surfaces the right block.
    doc = (
        "1. Fees\nCustomer shall pay $50,000 monthly within 30 days.\n\n"
        "2. Liability\nProvider's total aggregate liability shall not exceed "
        "$500,000 under this agreement.\n\n"
        "3. Confidentiality\nConfidential information must not be disclosed.\n"
    )
    ing = service.ingest(doc, label="bm25-brief", source_type="contract")
    res = service.brief(ing["context_id"], task="what is the liability cap")
    headings = " ".join(b["heading"] for b in res["relevant_blocks"]).lower()
    assert "liability" in headings
