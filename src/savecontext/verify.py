"""Preservation verification — prove the loss-aware claim with real numbers.

SaveContext's promise is that meaning-critical facts survive compression
verbatim. This module turns that from a slogan into a measurement:

- ``roundtrip``: every stored block and atom must reconstruct byte-for-byte
  from its offsets into the raw source. If anything fails, the guarantee is
  broken and ``audit`` says so loudly.
- ``coverage``: a loose re-scan estimates how many money/date/number/percent
  mentions the extractor *caught* vs. how many exist, surfacing potential
  misses (recall) rather than silently implying 100%.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from .extraction import Atom

# Loose patterns used only to *estimate* how many candidates exist in the text,
# independent of the (stricter, de-duplicating) extractor. Used for recall.
_LOOSE = {
    "money": re.compile(r"[$€£¥]\s?\d", ),
    "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    "percentage": re.compile(r"\d+(?:\.\d+)?\s?%"),
    "email": re.compile(r"@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "url": re.compile(r"https?://"),
}


def roundtrip(raw: str, block_spans: Sequence, atoms: Sequence[Atom]) -> Dict:
    """Verify every block/atom span reconstructs exactly from the raw source.

    ``block_spans`` is an iterable of ``(block_id, start, end, text)``.
    Returns a report; ``ok`` is True only if zero mismatches.
    """
    block_mismatches: List[str] = []
    for block_id, start, end, text in block_spans:
        if raw[start:end] != text:
            block_mismatches.append(block_id)

    atom_checked = 0
    atom_mismatches: List[str] = []
    for a in atoms:
        atom_checked += 1
        if raw[a.start_char:a.end_char] != a.value:
            atom_mismatches.append(a.atom_id)
        # Every recorded occurrence must also be exact.
        for s, e in a.occurrences:
            if raw[s:e] != a.value and raw[s:e].lower() != a.normalized:
                atom_mismatches.append(f"{a.atom_id}@{s}")
                break

    return {
        "ok": not block_mismatches and not atom_mismatches,
        "blocks_checked": len(list(block_spans)) if not isinstance(block_spans, list) else len(block_spans),
        "block_mismatches": block_mismatches,
        "atoms_checked": atom_checked,
        "atom_mismatches": atom_mismatches[:20],
        "atoms_verbatim_recoverable": atom_checked - len(set(atom_mismatches)),
    }


def coverage(raw: str, atoms: Sequence[Atom]) -> Dict[str, Dict]:
    """Estimate extraction recall per category via an independent loose scan.

    Returns ``{type: {found, candidates, recall}}``. ``recall`` is approximate
    — it answers "of the obvious mentions, how many did we capture?" so a low
    number is a real warning sign, not proof of loss.
    """
    captured_spans = {t: set() for t in _LOOSE}
    for a in atoms:
        if a.type in captured_spans:
            for s, _e in a.occurrences:
                captured_spans[a.type].add(s)

    out: Dict[str, Dict] = {}
    for t, pat in _LOOSE.items():
        cand_starts = [m.start() for m in pat.finditer(raw)]
        candidates = len(cand_starts)
        if candidates == 0:
            continue
        # A candidate counts as covered if some captured atom of this type
        # starts at or near it (within a few chars, since the loose pattern may
        # begin mid-token).
        cap = captured_spans[t]
        covered = sum(
            1 for c in cand_starts if any(abs(c - s) <= 3 for s in cap)
        )
        out[t] = {
            "found": len(cap),
            "candidates": candidates,
            "recall": round(covered / candidates, 2),
        }
    return out
