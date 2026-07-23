"""Tests for the optional local-LLM summarization backend.

Default path (disabled -> extractive) is asserted deterministically. The
network call is verified to fail closed against an unreachable endpoint.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext import llm


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SAVECONTEXT_LLM_SUMMARY", raising=False)
    assert llm.enabled() is False
    # refine_brief returns None when disabled, regardless of inputs.
    assert llm.refine_brief("draft", "source text", "generic") is None


def test_flag_parsing(monkeypatch):
    for v in ("1", "true", "On", "YES"):
        monkeypatch.setenv("SAVECONTEXT_LLM_SUMMARY", v)
        assert llm.enabled() is True
    monkeypatch.setenv("SAVECONTEXT_LLM_SUMMARY", "no")
    assert llm.enabled() is False


def test_fails_closed_on_unreachable(monkeypatch):
    # Enabled but pointed at a closed port -> must return None, not raise.
    monkeypatch.setenv("SAVECONTEXT_LLM_SUMMARY", "1")
    monkeypatch.setattr(llm, "_DEFAULT_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(llm, "_TIMEOUT", 1.0)
    assert llm.refine_brief("draft", "source", "generic") is None


def test_ingest_reports_brief_mode(service, contract_text, monkeypatch):
    monkeypatch.delenv("SAVECONTEXT_LLM_SUMMARY", raising=False)
    res = service.ingest(contract_text, label="acme", source_type="contract")
    assert res["brief_mode"] == "extractive"
