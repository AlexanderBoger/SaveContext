"""Extractive summarization and task-specific briefs.

No model calls in the MVP. A *semantic brief* is built by scoring blocks by
information density (headings, atom richness, position) and emitting their
headings + lead sentences, plus the highest-value atom lines. A
*task-specific brief* re-scores blocks against the task keywords.

These functions return plain strings/dicts and are deliberately swappable for
local-LLM summarization later (same inputs, same outputs).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .chunking import Block
from .extraction import Atom
from .retrieval import build_ranker, tokenize
from .tokenizer import estimate_tokens

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Atom types that signal a "must-keep" line in a brief.
_HIGH_VALUE = {"obligation", "money", "date", "percentage", "negation", "duration"}


def _first_sentences(text: str, n: int = 1) -> str:
    flat = " ".join(text.split())
    if not flat:
        return ""
    sents = _SENT_SPLIT.split(flat)
    return " ".join(sents[:n]).strip()


def _keywords(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _block_score(block: Block, atoms_by_block: Dict[str, List[Atom]]) -> float:
    score = 0.0
    if block.heading:
        score += 2.0
    if block.index == 0:
        score += 1.5  # intros tend to matter
    block_atoms = atoms_by_block.get(block.block_id, [])
    score += min(5.0, 0.4 * len(block_atoms))
    score += sum(1.0 for a in block_atoms if a.type in _HIGH_VALUE)
    return score


def _atoms_by_block(atoms: List[Atom]) -> Dict[str, List[Atom]]:
    out: Dict[str, List[Atom]] = {}
    for a in atoms:
        if a.block_id:
            out.setdefault(a.block_id, []).append(a)
    return out


def build_semantic_brief(
    blocks: List[Block],
    atoms: List[Atom],
    source_type: str,
    max_tokens: int = 700,
) -> str:
    """Compose a compact extractive brief of the whole source."""
    if not blocks:
        return ""
    abb = _atoms_by_block(atoms)
    scored = sorted(blocks, key=lambda b: _block_score(b, abb), reverse=True)

    lines: List[str] = []
    used = 0
    budget = max_tokens

    header = f"[{source_type}] {len(blocks)} blocks, {len(atoms)} protected atoms."
    lines.append(header)
    used += estimate_tokens(header)

    # Emit lead sentences for top blocks in document order for readability.
    chosen = sorted(scored[: max(3, len(scored) // 2 + 1)], key=lambda b: b.index)
    for b in chosen:
        head = f"• {b.heading}: " if b.heading else "• "
        lead = _first_sentences(b.text, 2)
        line = (head + lead).strip()
        if not line or line == "•":
            continue
        cost = estimate_tokens(line)
        if used + cost > budget:
            break
        lines.append(line)
        used += cost

    # Append a tight list of the most critical atoms not already implied.
    crit = [a for a in atoms if a.type in _HIGH_VALUE][:12]
    if crit and used < budget:
        key_line = "Key facts: " + "; ".join(
            f"{a.value}" for a in crit
        )
        if used + estimate_tokens(key_line) <= budget + 40:
            lines.append(key_line)

    return "\n".join(lines).strip()


def build_task_brief(
    blocks: List[Block],
    atoms: List[Atom],
    task: str,
    max_tokens: int = 500,
) -> dict:
    """Score blocks/atoms against ``task`` and return a focused brief."""
    abb = _atoms_by_block(atoms)
    task_kw = set(_keywords(task))

    # BM25 over block text (heading folded in so heading terms count) gives the
    # primary relevance signal; heading hits and atom density are added as
    # boosts on top so a short but dense clause still surfaces.
    index = build_ranker([(b.block_id, f"{b.heading} {b.text}") for b in blocks])
    bm25 = index.score(task)
    max_bm = max(bm25.values()) if bm25 else 0.0

    def relevance(block: Block) -> float:
        norm_bm = (bm25.get(block.block_id, 0.0) / max_bm) if max_bm else 0.0
        base = _block_score(block, abb)
        heading_hit = 1.0 if task_kw & set(_keywords(block.heading)) else 0.0
        return norm_bm * 5.0 + heading_hit + 0.15 * base

    ranked = sorted(blocks, key=relevance, reverse=True)
    # Keep blocks with real lexical signal; fall back to global importance.
    relevant_blocks = [b for b in ranked if bm25.get(b.block_id, 0.0) > 0][:6]
    if not relevant_blocks:
        relevant_blocks = ranked[:2]

    # Relevant atoms: those inside relevant blocks or whose value matches task.
    rel_block_ids = {b.block_id for b in relevant_blocks}
    relevant_atoms: List[Atom] = []
    seen = set()
    for a in atoms:
        match = a.block_id in rel_block_ids or bool(task_kw & set(_keywords(a.value)))
        if match and a.atom_id not in seen:
            relevant_atoms.append(a)
            seen.add(a.atom_id)

    # Compose prose brief from relevant blocks in document order.
    lines: List[str] = []
    used = 0
    for b in sorted(relevant_blocks, key=lambda x: x.index):
        head = f"• {b.heading}: " if b.heading else "• "
        lead = _first_sentences(b.text, 2)
        line = (head + lead).strip()
        cost = estimate_tokens(line)
        if used + cost > max_tokens:
            break
        lines.append(line)
        used += cost

    prose = "\n".join(lines).strip() or "(no strongly matching content; see block_map)"

    uncertainty: List[str] = []
    if not task_kw:
        uncertainty.append("Empty/short task; ranking fell back to global importance.")
    if max_bm <= 0:
        uncertainty.append("No BM25 overlap with task; results are best-effort.")
    if len(relevant_blocks) >= len(blocks):
        uncertainty.append("Task matched most of the document; consider narrowing.")
    uncertainty.append(
        "Brief is extractive and lossy — call expand()/quote() before relying on exact wording."
    )

    return {
        "task_specific_brief": prose,
        "relevant_blocks": [
            {
                "block_id": b.block_id,
                "heading": b.heading,
                "token_estimate": b.token_estimate,
                "preview": b.preview(),
            }
            for b in sorted(relevant_blocks, key=lambda x: x.index)
        ],
        "relevant_atoms": [
            {
                "atom_id": a.atom_id,
                "type": a.type,
                "value": a.value,
                "block_id": a.block_id,
            }
            for a in relevant_atoms[:40]
        ],
        "uncertainty_notes": uncertainty,
    }


def extractive_block_summary(block: Block, max_sentences: int = 3) -> str:
    return _first_sentences(block.text, max_sentences)
