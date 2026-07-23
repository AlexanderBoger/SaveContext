"""Rule-based atom extraction.

An *atom* is a small, high-value, exactly-preserved fact extracted from the
source: a date, a money value, an obligation, a code identifier, etc. Atoms
are the loss-aware core of SaveContext — even when the prose is summarized
away, the atoms keep their verbatim spans and exact source offsets.

This module is pure regex/heuristics (no model calls) so it is deterministic
and fast. The ``ATOM_TYPES`` registry and ``extract_atoms`` return shape are
designed so an LLM-based extractor can later be added as another producer
without changing downstream code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Atom categories. The ordering also defines a rough priority for summaries.
ATOM_TYPES = [
    "date",
    "duration",
    "money",
    "percentage",
    "version",
    "section_ref",
    "number",
    "email",
    "url",
    "code_identifier",
    "file_path",
    "obligation",
    "negation",
    "entity",
]

# Categories that map onto the audit "estimated_preservation" buckets.
PRESERVATION_BUCKETS = {
    "names": ["entity"],
    "dates": ["date"],
    "numbers": ["number", "percentage"],
    "money_values": ["money"],
    "negations": ["negation"],
    "conditions": ["obligation"],  # conditional/obligation language
    "obligations": ["obligation"],
    "code_identifiers": ["code_identifier", "file_path"],
}


@dataclass
class Atom:
    atom_id: str
    type: str
    value: str  # exact verbatim span as it appears in the source
    normalized: str  # lowercased / canonical form for dedup & matching
    start_char: int
    end_char: int
    block_id: Optional[str] = None
    count: int = 1
    occurrences: List[tuple[int, int]] = field(default_factory=list)


# --- Regex library -------------------------------------------------------

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)

_PATTERNS: Dict[str, re.Pattern] = {
    "url": re.compile(r"\bhttps?://[^\s<>\")\]]+", re.I),
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "money": re.compile(
        # Magnitude suffixes need a trailing \b so "m"/"k" don't eat the start of
        # an adjacent word (e.g. "$50,000 monthly" must not become "$50,000 m").
        r"(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand|k|m|bn)\b)?"
        r"|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|JPY|dollars?|euros?|pounds?)\b)",
        re.I,
    ),
    "percentage": re.compile(r"\b\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?percent\b", re.I),
    # Time periods — notice windows, terms, SLAs are meaning-critical.
    "duration": re.compile(
        r"\b\d+\s?(?:business\s+)?(?:second|minute|hour|day|week|month|year|quarter)s?\b",
        re.I,
    ),
    # Semantic versions and v-prefixed versions (kept distinct from dates).
    "version": re.compile(r"\bv\d+(?:\.\d+)+\b|\b\d+\.\d+\.\d+\b", re.I),
    # Cross-references to numbered sections/clauses.
    "section_ref": re.compile(
        r"\b(?:Section|Article|Clause|Appendix|Exhibit|Schedule|Paragraph)\s+\d+(?:\.\d+)*\b",
        re.I,
    ),
    # ISO dates, D Month Y / Month D, Y, and slashed dates.
    "date": re.compile(
        r"\b(?:\d{4}-\d{2}-\d{2}"
        rf"|\d{{1,2}}\s+{_MONTHS}\s+\d{{2,4}}"
        rf"|{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}"
        r"|\d{1,2}/\d{1,2}/\d{2,4})\b",
        re.I,
    ),
    "file_path": re.compile(
        r"(?:(?:\.{0,2}/)?(?:[\w.\-]+/)+[\w.\-]+"  # has at least one slash
        r"|\b[\w\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|c|cpp|h|hpp|rb|php|"
        r"json|yaml|yml|toml|md|txt|sql|sh|cfg|ini)\b)"
    ),
    # snake_case, camelCase, dotted.calls, CONSTANT_CASE, func()
    "code_identifier": re.compile(
        r"`[^`\n]+`"
        r"|\b[a-z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b"  # snake_case
        r"|\b[a-z]+[A-Z][a-zA-Z0-9]*\b"  # camelCase
        r"|\b[A-Z][A-Z0-9]+_[A-Z0-9_]+\b"  # CONSTANT_CASE
        r"|\b[a-zA-Z_][\w]*\([^)\n]{0,40}\)"  # call(...)
    ),
    "number": re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w%])"),
    # Entities stay on one line: inter-word gaps are spaces/tabs, never newlines,
    # so a name can't bleed across a paragraph break.
    "entity": re.compile(r"\b[A-Z][a-zA-Z0-9&.\-]+(?:[ \t]+[A-Z][a-zA-Z0-9&.\-]+){0,4}\b"),
}

# Obligation / modal language (contracts, specs, policies).
_OBLIGATION_RE = re.compile(
    r"\b(?:shall(?:\s+not)?|must(?:\s+not)?|may\s+not|will\s+not|"
    r"required\s+to|obligated\s+to|responsible\s+for|agree(?:s|d)?\s+to|"
    r"shall\s+be\s+entitled|is\s+entitled\s+to|prohibited\s+from|"
    r"is\s+liable\s+for|indemnif(?:y|ies|ication)|warrant(?:s|y|ies)?)\b",
    re.I,
)

# Negation language (often meaning-critical).
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|without|cannot|can't|won't|shall\s+not|"
    r"must\s+not|may\s+not|excluding|except|unless|neither|nor)\b",
    re.I,
)

# Words that are capitalized only because they start a sentence; weak entities.
_STOP_ENTITIES = {
    "The", "This", "That", "These", "Those", "It", "If", "When", "While",
    "However", "Therefore", "Thus", "Hence", "Such", "Any", "All", "Each",
    "Section", "Article", "Clause", "Note", "We", "You", "They", "He", "She",
    "In", "On", "At", "As", "For", "And", "But", "Or", "A", "An",
}


def _normalize(text: str, atom_type: str) -> str:
    t = text.strip().strip("`").lower()
    t = re.sub(r"\s+", " ", t)
    if atom_type == "money":
        t = t.replace(",", "")
    return t


def _block_for(offset: int, block_spans) -> Optional[str]:
    for block_id, start, end in block_spans:
        if start <= offset < end:
            return block_id
    return None


def extract_atoms(text: str, block_spans=None) -> List[Atom]:
    """Extract and de-duplicate atoms from ``text``.

    ``block_spans`` is an optional iterable of ``(block_id, start, end)`` used
    to tag each atom with the block it falls in. Atoms are de-duplicated by
    ``(type, normalized)``; repeated mentions increment ``count`` and append to
    ``occurrences`` so :func:`savecontext.tools` can still quote any instance.

    To avoid double-counting overlapping spans (e.g. a money value also
    matching ``number``), higher-priority types claim their character ranges
    first and lower-priority matches that overlap a claimed range are dropped.
    """
    block_spans = list(block_spans or [])
    claimed: List[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        for cs, ce in claimed:
            if s < ce and cs < e:
                return True
        return False

    # type -> { normalized: Atom }
    dedup: Dict[str, Dict[str, Atom]] = {}
    ordered: List[Atom] = []
    counter = 0

    def add(atom_type: str, value: str, start: int, end: int):
        nonlocal counter
        norm = _normalize(value, atom_type)
        if not norm:
            return
        bucket = dedup.setdefault(atom_type, {})
        existing = bucket.get(norm)
        if existing is not None:
            existing.count += 1
            existing.occurrences.append((start, end))
            return
        atom = Atom(
            atom_id=f"a{counter:04d}",
            type=atom_type,
            value=value,
            normalized=norm,
            start_char=start,
            end_char=end,
            block_id=_block_for(start, block_spans),
            count=1,
            occurrences=[(start, end)],
        )
        bucket[norm] = atom
        ordered.append(atom)
        counter += 1

    # Process types in priority order so high-value atoms claim spans first.
    priority = [
        "url", "email", "money", "percentage", "version", "date", "duration",
        "section_ref", "file_path", "code_identifier", "number", "entity",
    ]
    for atom_type in priority:
        pat = _PATTERNS[atom_type]
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            val = m.group(0).strip()
            if not val:
                continue
            if atom_type == "entity":
                if val in _STOP_ENTITIES or all(
                    w in _STOP_ENTITIES for w in val.split()
                ):
                    continue
                if len(val) < 3:
                    continue
            if overlaps(s, e):
                # Still record repeat occurrences for already-claimed values.
                norm = _normalize(val, atom_type)
                ex = dedup.get(atom_type, {}).get(norm)
                if ex is not None:
                    ex.count += 1
                    ex.occurrences.append((s, e))
                continue
            claimed.append((s, e))
            add(atom_type, val, s, e)

    # Obligations & negations are phrase-level; they may overlap other atoms,
    # so they are extracted independently (no span claiming).
    for atom_type, pat in (("obligation", _OBLIGATION_RE), ("negation", _NEGATION_RE)):
        for m in pat.finditer(text):
            add(atom_type, m.group(0).strip(), m.start(), m.end())

    # Re-id atoms in stable document order for readable, ordered ids.
    ordered.sort(key=lambda a: a.start_char)
    for i, atom in enumerate(ordered):
        atom.atom_id = f"a{i:04d}"
    return ordered


def summarize_atoms(atoms: List[Atom], include_examples: bool = True) -> Dict[str, dict]:
    """Build a compact ``protected_atoms_summary`` keyed by type.

    ``include_examples=False`` drops the example values to keep the payload tiny
    (used in the compact ingest response); examples are retrievable via map/quote.
    """
    out: Dict[str, dict] = {}
    for atom in atoms:
        slot = out.setdefault(atom.type, {"count": 0, "unique": 0, "examples": []})
        slot["count"] += atom.count
        slot["unique"] += 1
        if include_examples and len(slot["examples"]) < 5:
            slot["examples"].append(atom.value)
    if not include_examples:
        for slot in out.values():
            slot.pop("examples", None)
    return out
