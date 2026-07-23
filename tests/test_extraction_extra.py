"""Tests for the expanded atom types: duration, version, section_ref."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext.extraction import extract_atoms


def _by_type(text):
    out = {}
    for a in extract_atoms(text):
        out.setdefault(a.type, []).append(a.value)
    return out


def test_duration_extracted():
    t = "Either party may terminate with 90 days notice; the term is 24 months."
    bt = _by_type(t)
    vals = [v.lower() for v in bt.get("duration", [])]
    assert any("90 day" in v for v in vals)
    assert any("24 month" in v for v in vals)


def test_version_extracted_not_as_date():
    t = "Upgrade to v2.3.1 before the release of 1.0.0."
    bt = _by_type(t)
    vers = bt.get("version", [])
    assert "v2.3.1" in vers
    assert "1.0.0" in vers


def test_section_ref_extracted():
    t = "As described in Section 4.2 and Article 7, the terms apply."
    bt = _by_type(t)
    refs = [v.lower() for v in bt.get("section_ref", [])]
    assert any("section 4.2" in r for r in refs)
    assert any("article 7" in r for r in refs)


def test_duration_not_double_counted_as_number():
    # "30 days" should be a duration, and 30 should not also appear as a bare number.
    t = "Payment is due within 30 days."
    atoms = extract_atoms(t)
    durations = [a for a in atoms if a.type == "duration"]
    numbers = [a for a in atoms if a.type == "number" and a.value == "30"]
    assert durations
    assert not numbers  # claimed by duration


def test_existing_atoms_still_work():
    t = "Pay $50,000 by 2024-01-01 (1.5%). Contact a@b.com or https://x.io."
    bt = _by_type(t)
    assert bt.get("money") and bt.get("date") and bt.get("percentage")
    assert bt.get("email") and bt.get("url")
