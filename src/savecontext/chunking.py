"""Deterministic block splitting.

The MVP splits source text into logical *blocks* without any model calls:

1. Detect heading lines (markdown headings, numbered sections, ALL-CAPS lines).
2. Split the body into paragraphs on blank lines.
3. Greedily pack paragraphs into blocks up to a target token budget so that
   blocks stay small enough to expand cheaply but large enough to be coherent.

Every block records its exact character offsets into the *original* source so
the raw text can be reconstructed byte-for-byte on expansion. This is the
foundation of the loss-aware guarantee: blocks are pointers, not copies.

The architecture is intentionally simple; ``split_blocks`` can later be
swapped for semantic/embedding-based chunking behind the same return shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .tokenizer import estimate_tokens

# Target block size in tokens. Blocks may exceed this if a single paragraph is
# larger than the budget (we never split inside a paragraph in the MVP).
DEFAULT_BLOCK_TOKENS = 350

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_NUMBERED = re.compile(r"^\s{0,3}(\d+(\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.)\s+\S")
_SECTION_KW = re.compile(r"^\s{0,3}(section|article|clause|appendix|exhibit|schedule)\b", re.I)


@dataclass
class Block:
    block_id: str
    index: int
    heading: str
    text: str
    start_char: int
    end_char: int
    token_estimate: int = 0
    atom_ids: List[str] = field(default_factory=list)

    def preview(self, words: int = 14) -> str:
        flat = " ".join(self.text.split())
        parts = flat.split(" ")
        if len(parts) <= words:
            return flat
        return " ".join(parts[:words]) + " …"


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _MD_HEADING.match(line):
        return True
    if _SECTION_KW.match(line):
        return True
    if _NUMBERED.match(line) and len(stripped) <= 90:
        return True
    # Short, title-like line with no terminal punctuation.
    if len(stripped) <= 70 and not stripped.endswith((".", ",", ";", ":")):
        words = stripped.split()
        if 1 <= len(words) <= 10:
            # ALL CAPS or Title Case heuristics.
            if stripped.isupper():
                return True
            capish = sum(1 for w in words if w[:1].isupper())
            if capish >= max(1, len(words) - 1):
                return True
    return False


def _clean_heading(line: str) -> str:
    return re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip()


def split_blocks(text: str, target_tokens: int = DEFAULT_BLOCK_TOKENS) -> List[Block]:
    """Split ``text`` into blocks with exact offsets into ``text``.

    Paragraphs are delimited by blank lines. The first line of a paragraph (or
    a standalone heading line) becomes the block heading when it looks like one.
    """
    if not text:
        return []

    # Split into paragraphs while preserving offsets.
    paragraphs: List[tuple[int, int, str]] = []
    for m in re.finditer(r"[^\n]*(?:\n[^\n]+)*", text):
        seg = m.group(0)
        if seg.strip() == "":
            continue
        paragraphs.append((m.start(), m.end(), seg))

    if not paragraphs:
        paragraphs = [(0, len(text), text)]

    blocks: List[Block] = []
    cur_start = None
    cur_end = None
    cur_tokens = 0
    cur_heading = ""
    idx = 0

    def flush():
        nonlocal cur_start, cur_end, cur_tokens, cur_heading, idx
        if cur_start is None:
            return
        chunk = text[cur_start:cur_end]
        blocks.append(
            Block(
                block_id=f"b{idx:04d}",
                index=idx,
                heading=cur_heading,
                text=chunk,
                start_char=cur_start,
                end_char=cur_end,
                token_estimate=estimate_tokens(chunk),
            )
        )
        idx += 1
        cur_start = cur_end = None
        cur_tokens = 0
        cur_heading = ""

    for start, end, seg in paragraphs:
        first_line = seg.lstrip("\n").split("\n", 1)[0]
        seg_tokens = estimate_tokens(seg)
        is_heading_para = _looks_like_heading(first_line)

        # A heading line that is its own paragraph starts a new block.
        if is_heading_para and cur_start is not None:
            flush()

        if cur_start is None:
            cur_start = start
            cur_end = end
            cur_tokens = seg_tokens
            cur_heading = _clean_heading(first_line) if is_heading_para else ""
        else:
            cur_end = end
            cur_tokens += seg_tokens

        if cur_tokens >= target_tokens:
            flush()

    flush()

    # Edge case: a single heading-only block followed by content gets a heading.
    for b in blocks:
        if not b.heading:
            fl = b.text.lstrip("\n").split("\n", 1)[0]
            if _looks_like_heading(fl):
                b.heading = _clean_heading(fl)
    return blocks
