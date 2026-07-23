"""Unit tests for the building blocks: tokenizer, handles, chunking, extraction."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext import handles
from savecontext.chunking import split_blocks
from savecontext.extraction import extract_atoms
from savecontext.tokenizer import compression_ratio, estimate_tokens


def test_estimate_tokens_monotonic():
    assert estimate_tokens("") == 0
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world " * 100)
    assert 0 < short < long


def test_compression_ratio():
    assert compression_ratio(1000, 100) == 10.0
    assert compression_ratio(0, 0) == 1.0
    assert compression_ratio(100, 0) == 100.0


def test_handles_roundtrip():
    h = handles.make_handle("Acme Corp Contract!", 3)
    assert h == "ctx://acme-corp-contract@v3"
    parsed = handles.parse_handle(h)
    assert parsed.label == "acme-corp-contract"
    assert parsed.version == 3
    assert parsed.scheme == "ctx"


def test_handle_slug_fallback():
    assert handles.slugify("!!!") == "untitled"
    assert handles.make_handle("Report", 1, scheme="out") == "out://report@v1"


def test_invalid_handle():
    import pytest

    with pytest.raises(ValueError):
        handles.parse_handle("not-a-handle")


def test_chunking_offsets_exact(contract_text):
    blocks = split_blocks(contract_text)
    assert blocks
    for b in blocks:
        # offsets must reconstruct text byte-for-byte
        assert contract_text[b.start_char:b.end_char] == b.text


def test_chunking_detects_headings(contract_text):
    blocks = split_blocks(contract_text)
    headings = [b.heading for b in blocks if b.heading]
    assert any("Liability" in h or "Fees" in h or "AGREEMENT" in h for h in headings)


def test_extract_money_and_dates(contract_text):
    blocks = split_blocks(contract_text)
    spans = [(b.block_id, b.start_char, b.end_char) for b in blocks]
    atoms = extract_atoms(contract_text, spans)
    by_type = {}
    for a in atoms:
        by_type.setdefault(a.type, []).append(a.value)

    money_norm = {a.replace(",", "").lower() for a in by_type.get("money", [])}
    assert any("50000" in m for m in money_norm)
    assert any("500000" in m for m in money_norm)
    assert by_type.get("date")
    assert "2024-02-01" in by_type["date"] or "January 15, 2024" in by_type["date"]
    assert by_type.get("percentage")
    assert by_type.get("obligation")
    assert by_type.get("email")
    assert by_type.get("url")


def test_atoms_have_exact_spans(contract_text):
    blocks = split_blocks(contract_text)
    spans = [(b.block_id, b.start_char, b.end_char) for b in blocks]
    atoms = extract_atoms(contract_text, spans)
    for a in atoms:
        assert contract_text[a.start_char:a.end_char] == a.value


def test_negation_extracted(contract_text):
    blocks = split_blocks(contract_text)
    spans = [(b.block_id, b.start_char, b.end_char) for b in blocks]
    atoms = extract_atoms(contract_text, spans)
    assert any(a.type == "negation" for a in atoms)


def test_code_identifiers(code_text):
    blocks = split_blocks(code_text)
    spans = [(b.block_id, b.start_char, b.end_char) for b in blocks]
    atoms = extract_atoms(code_text, spans)
    types = {a.type for a in atoms}
    assert "code_identifier" in types
    assert "file_path" in types
    paths = {a.value for a in atoms if a.type == "file_path"}
    assert any("processor.py" in p or "config.yaml" in p for p in paths)
