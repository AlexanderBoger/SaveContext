"""Loss-aware semantic diff between two versions of a source.

Rather than a raw text diff, SaveContext diffs the *atoms* — the meaning-
bearing facts — so the report highlights changes that matter (a money value
moved, an obligation removed, a date changed) and de-emphasizes cosmetic
edits. A coarse block-level text diff is included as supporting context.
"""

from __future__ import annotations

import difflib
from typing import Dict, List

from .extraction import Atom

# Types where a value change is meaning-critical and worth surfacing loudly.
_RISK_TYPES = {"money", "obligation", "negation", "date", "percentage"}


def _atom_key(a: Atom) -> str:
    return f"{a.type}::{a.normalized}"


def diff_atoms(old_atoms: List[Atom], new_atoms: List[Atom]) -> dict:
    old_map: Dict[str, Atom] = {_atom_key(a): a for a in old_atoms}
    new_map: Dict[str, Atom] = {_atom_key(a): a for a in new_atoms}

    old_keys = set(old_map)
    new_keys = set(new_map)

    added = [new_map[k] for k in new_keys - old_keys]
    removed = [old_map[k] for k in old_keys - new_keys]

    # "Changed" = same type with a near-match normalized value that differs.
    # Detect by pairing removed/added within a type via difflib ratio.
    changed: List[dict] = []
    used_added = set()
    by_type_added: Dict[str, List[Atom]] = {}
    for a in added:
        by_type_added.setdefault(a.type, []).append(a)

    still_removed = []
    for r in removed:
        candidates = by_type_added.get(r.type, [])
        best = None
        best_ratio = 0.0
        for cand in candidates:
            if id(cand) in used_added:
                continue
            ratio = difflib.SequenceMatcher(
                None, r.normalized, cand.normalized
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = cand
        if best is not None and best_ratio >= 0.6:
            used_added.add(id(best))
            changed.append(
                {
                    "type": r.type,
                    "from": r.value,
                    "to": best.value,
                    "similarity": round(best_ratio, 2),
                }
            )
        else:
            still_removed.append(r)

    added = [a for a in added if id(a) not in used_added]
    removed = still_removed

    def fmt(atoms: List[Atom]) -> List[dict]:
        return [
            {"atom_id": a.atom_id, "type": a.type, "value": a.value}
            for a in sorted(atoms, key=lambda x: x.start_char)
        ]

    return {
        "added_atoms": fmt(added),
        "removed_atoms": fmt(removed),
        "changed_atoms": sorted(changed, key=lambda c: c["type"]),
    }


def block_text_diff(old_text: str, new_text: str, max_hunks: int = 12) -> List[str]:
    """Coarse unified-style summary of changed lines (capped)."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    hunks: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            hunks.append(
                f"~ lines {i1 + 1}-{i2}: replaced "
                f"{_snip(old_lines[i1:i2])} → {_snip(new_lines[j1:j2])}"
            )
        elif tag == "delete":
            hunks.append(f"- lines {i1 + 1}-{i2}: removed {_snip(old_lines[i1:i2])}")
        elif tag == "insert":
            hunks.append(f"+ at line {i1 + 1}: added {_snip(new_lines[j1:j2])}")
        if len(hunks) >= max_hunks:
            hunks.append("… (additional changes truncated)")
            break
    return hunks


def _snip(lines: List[str], width: int = 60) -> str:
    flat = " ".join(" ".join(lines).split())
    return (flat[:width] + "…") if len(flat) > width else flat


def risk_impact(atom_diff: dict) -> List[str]:
    """Flag meaning-critical changes for the ``risk_or_meaning_impact`` field."""
    notes: List[str] = []
    for c in atom_diff["changed_atoms"]:
        if c["type"] in _RISK_TYPES:
            notes.append(f"{c['type']} changed: '{c['from']}' → '{c['to']}'")
    for a in atom_diff["removed_atoms"]:
        if a["type"] in _RISK_TYPES:
            notes.append(f"{a['type']} removed: '{a['value']}'")
    for a in atom_diff["added_atoms"]:
        if a["type"] in _RISK_TYPES:
            notes.append(f"{a['type']} added: '{a['value']}'")
    if not notes:
        notes.append("No changes to money, dates, obligations, or negations detected.")
    return notes


def compact_patch_summary(atom_diff: dict, block_hunks: List[str]) -> str:
    a = len(atom_diff["added_atoms"])
    r = len(atom_diff["removed_atoms"])
    c = len(atom_diff["changed_atoms"])
    return (
        f"{a} atoms added, {r} removed, {c} changed across "
        f"{len(block_hunks)} text region(s)."
    )
