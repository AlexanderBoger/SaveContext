"""Tests for the optional embedding backend and its fallback behaviour.

These verify the *default* path (embeddings off -> BM25) deterministically.
The HybridRanker itself is only exercised when sentence-transformers is
installed and SAVECONTEXT_EMBEDDINGS is set, so it is not asserted here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext import embeddings
from savecontext.retrieval import BM25Index, build_ranker

DOCS = [("b0", "the cat on the mat"), ("b1", "liability cap and indemnification")]


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SAVECONTEXT_EMBEDDINGS", raising=False)
    assert embeddings.enabled() is False


def test_enabled_flag_parsing(monkeypatch):
    for val in ("1", "true", "YES", "on"):
        monkeypatch.setenv("SAVECONTEXT_EMBEDDINGS", val)
        assert embeddings.enabled() is True
    monkeypatch.setenv("SAVECONTEXT_EMBEDDINGS", "0")
    assert embeddings.enabled() is False


def test_build_ranker_falls_back_to_bm25(monkeypatch):
    # Even if enabled, when deps are unavailable it must return BM25.
    monkeypatch.setenv("SAVECONTEXT_EMBEDDINGS", "1")
    monkeypatch.setattr(embeddings, "available", lambda: False)
    ranker = build_ranker(DOCS)
    assert isinstance(ranker, BM25Index)
    # And it still ranks correctly.
    assert ranker.rank("liability cap")[0][0] == "b1"


def test_build_ranker_default_is_bm25(monkeypatch):
    monkeypatch.delenv("SAVECONTEXT_EMBEDDINGS", raising=False)
    assert isinstance(build_ranker(DOCS), BM25Index)
