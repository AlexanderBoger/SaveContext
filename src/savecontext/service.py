"""SaveContext service layer.

Pure business logic for the eight MCP tools, decoupled from the MCP transport
so it can be unit-tested directly. Every method returns a plain JSON-able
dict. The MCP server in :mod:`savecontext.server` is a thin wrapper.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from . import handles
from .chunking import Block, split_blocks
from .diffing import (
    block_text_diff,
    compact_patch_summary,
    diff_atoms,
    risk_impact,
)
from .extraction import (
    Atom,
    PRESERVATION_BUCKETS,
    extract_atoms,
    summarize_atoms,
)
from .retrieval import build_ranker, hybrid_scores, intent_atom_types, tokenize
from .storage import Store
from .summarize import (
    build_semantic_brief,
    build_task_brief,
    extractive_block_summary,
)
from .tokenizer import compression_ratio, estimate_tokens, tokenizer_name
from . import llm, verify

VALID_SOURCE_TYPES = {
    "auto", "contract", "code", "logs", "meeting", "research", "chat", "generic",
}

# Max blocks listed in the compact ingest response; full map via the map tool.
INGEST_OUTLINE_LIMIT = 40

# Max blocks a single expand() returns, so a broad selector can't dump the doc.
EXPAND_BLOCK_LIMIT = 5


# --- source-type auto detection ----------------------------------------

def detect_source_type(text: str) -> str:
    sample = text[:8000]
    low = sample.lower()

    def has(*words):
        return sum(1 for w in words if w in low)

    # logs: many timestamped / level-tagged lines
    log_lines = len(re.findall(r"^\s*[\[\d].*(ERROR|WARN|INFO|DEBUG|TRACE)\b", sample, re.M))
    if log_lines >= 3 or len(re.findall(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", sample)) >= 3:
        return "logs"

    # code: code fences, defs, imports, braces density
    code_hits = has("def ", "function ", "import ", "class ", "return ", "const ", "#include")
    if code_hits >= 3 or sample.count("{") + sample.count("}") > 12 or "```" in sample:
        return "code"

    # contract / legal
    if has("shall", "hereby", "agreement", "liability", "indemnif", "party", "warrant") >= 3:
        return "contract"

    # meeting notes
    if has("attendees", "action items", "agenda", "minutes", "follow-up", "next steps") >= 2:
        return "meeting"

    # research / academic
    if has("abstract", "references", "et al", "hypothesis", "methodology", "conclusion") >= 2:
        return "research"

    # chat transcript
    if len(re.findall(r"^\s*(user|assistant|system|me|you)\s*:", sample, re.I | re.M)) >= 3:
        return "chat"

    return "generic"


def _atoms_to_dataclass(rows) -> List[Atom]:
    out = []
    for r in rows:
        out.append(
            Atom(
                atom_id=r["atom_id"],
                type=r["type"],
                value=r["value"],
                normalized=r["normalized"],
                start_char=r["start_char"],
                end_char=r["end_char"],
                block_id=r["block_id"],
                count=r["count"],
                occurrences=json.loads(r["occurrences"]),
            )
        )
    return out


def _blocks_to_dataclass(rows, raw_text: str) -> List[Block]:
    out = []
    for r in rows:
        text = raw_text[r["start_char"]:r["end_char"]]
        out.append(
            Block(
                block_id=r["block_id"],
                index=r["idx"],
                heading=r["heading"] or "",
                text=text,
                start_char=r["start_char"],
                end_char=r["end_char"],
                token_estimate=r["token_estimate"],
            )
        )
    return out


class VaultService:
    def __init__(self, store: Optional[Store] = None, db_path: Optional[str] = None):
        self.store = store or Store(db_path)

    def close(self):
        self.store.close()

    # === 1. ingest =====================================================

    def ingest(
        self,
        source_text: str,
        label: str,
        source_type: str = "auto",
        task_hint: Optional[str] = None,
        agent_brief: Optional[str] = None,
    ) -> dict:
        if not source_text or not source_text.strip():
            raise ValueError("source_text is empty.")
        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {sorted(VALID_SOURCE_TYPES)}"
            )
        resolved_type = (
            detect_source_type(source_text) if source_type == "auto" else source_type
        )

        slug = handles.slugify(label)
        version = self.store.next_version(slug)
        context_id = handles.make_handle(slug, version)

        blocks = split_blocks(source_text)
        block_spans = [(b.block_id, b.start_char, b.end_char) for b in blocks]
        atoms = extract_atoms(source_text, block_spans)
        for b in blocks:
            b.atom_ids = [a.atom_id for a in atoms if a.block_id == b.block_id]

        # Brief precedence: agent-supplied (best — the calling model already has
        # the doc in context) > optional local-LLM > extractive (always works).
        brief = build_semantic_brief(blocks, atoms, resolved_type)
        brief_mode = "extractive"
        if agent_brief and agent_brief.strip():
            brief = agent_brief.strip()
            brief_mode = "agent"
        else:
            refined = llm.refine_brief(brief, source_text, resolved_type)
            if refined:
                brief = refined
                brief_mode = "abstractive(local-llm)"
        tok_orig = estimate_tokens(source_text)
        tok_brief = estimate_tokens(brief)
        ratio = compression_ratio(tok_orig, tok_brief)

        self.store.save_context(
            context_id=context_id,
            label=slug,
            version=version,
            source_type=resolved_type,
            task_hint=task_hint,
            raw_text=source_text,
            token_estimate_original=tok_orig,
            token_estimate_brief=tok_brief,
            compression_ratio=ratio,
            semantic_brief=brief,
            tokenizer=tokenizer_name(),
            blocks=blocks,
            atoms=atoms,
            brief_mode=brief_mode,
        )

        # COMPACT response: the turn-1 payload must stay small regardless of doc
        # size, else it defeats the whole point. We send a capped outline (no
        # per-block previews) + atom *counts* only. The full block map and atom
        # examples are fetched lazily via the map tool when actually needed.
        outline = [
            {
                "block_id": b.block_id,
                "heading": b.heading,
                "token_estimate": b.token_estimate,
            }
            for b in blocks[:INGEST_OUTLINE_LIMIT]
        ]
        result = {
            "context_id": context_id,
            "version": version,
            "source_type": resolved_type,
            "token_estimate_original": tok_orig,
            "token_estimate_brief": tok_brief,
            "compression_ratio": ratio,
            "semantic_brief": brief,
            "brief_mode": brief_mode,
            "block_count": len(blocks),
            "block_map": outline,
            "block_map_truncated": len(blocks) > INGEST_OUTLINE_LIMIT,
            "protected_atoms_summary": summarize_atoms(atoms, include_examples=False),
            "safety_notes": _safety_notes(resolved_type, len(atoms)),
            "navigation_hint": (
                "block_map is a capped outline. For several questions at once, call "
                "lookup(queries=[…]) — one call answers them all. Otherwise: "
                "brief(task=…) to find relevant blocks, map for the full block "
                "map, expand()/quote() for content."
            ),
        }
        if brief_mode == "extractive":
            # Invite the calling agent to replace the extractive draft with its
            # own higher-quality brief while the doc is still in its context.
            result["brief_upgrade_hint"] = (
                "brief_mode is 'extractive'. If you (the agent) still have this "
                "document in context, you can write a denser, faithful brief and "
                f"store it via set_brief('{context_id}', <your_brief>). "
                "Keep all money/dates/obligations exact."
            )
        return result

    # === set_brief =====================================================

    def set_brief(self, context_id: str, brief: str) -> dict:
        """Replace a context's brief with an agent-authored one.

        This is how the *calling* model (e.g. Claude) becomes the summarizer:
        it reads the source once, writes a faithful brief, and stores it so all
        later turns reuse it — no local LLM, no extra model. The raw source and
        protected atoms are untouched, so exactness is unaffected.
        """
        ctx = self.store.get_context(context_id)
        if ctx is None:
            raise ValueError(f"Unknown context_id {context_id}")
        if not brief or not brief.strip():
            raise ValueError("brief is empty.")
        brief = brief.strip()
        tok_brief = estimate_tokens(brief)
        ratio = compression_ratio(ctx["token_estimate_original"], tok_brief)
        self.store.update_brief(context_id, brief, tok_brief, ratio, "agent")
        return {
            "context_id": context_id,
            "brief_mode": "agent",
            "token_estimate_brief": tok_brief,
            "compression_ratio": ratio,
            "semantic_brief": brief,
        }

    # === map (lazy full block map) =====================================

    def map(self, context_id: str, with_previews: bool = True,
            offset: int = 0, limit: Optional[int] = None,
            include_atom_examples: bool = True) -> dict:
        """Return the full (or paginated) block map — fetched only on demand.

        Kept out of the ingest response so turn-1 cost stays flat with doc size.
        Use offset/limit to page through very large documents.
        """
        raw = self._require_raw(context_id)
        blocks = _blocks_to_dataclass(self.store.get_blocks(context_id), raw)
        atoms = _atoms_to_dataclass(self.store.get_atoms(context_id))
        total = len(blocks)
        window = blocks[offset:(offset + limit) if limit else None]
        block_map = []
        for b in window:
            entry = {
                "block_id": b.block_id,
                "heading": b.heading,
                "token_estimate": b.token_estimate,
                "atom_count": sum(1 for a in atoms if a.block_id == b.block_id),
            }
            if with_previews:
                entry["preview"] = b.preview()
            block_map.append(entry)
        return {
            "context_id": context_id,
            "block_count": total,
            "offset": offset,
            "returned": len(block_map),
            "block_map": block_map,
            "protected_atoms_summary": summarize_atoms(
                atoms, include_examples=include_atom_examples
            ),
        }

    # === 2. brief ======================================================

    def brief(self, context_id: str, task: str, max_tokens: Optional[int] = None) -> dict:
        raw = self._require_raw(context_id)
        blocks = _blocks_to_dataclass(self.store.get_blocks(context_id), raw)
        atoms = _atoms_to_dataclass(self.store.get_atoms(context_id))
        result = build_task_brief(blocks, atoms, task or "", max_tokens or 500)
        result["context_id"] = context_id
        result["token_estimate"] = estimate_tokens(result["task_specific_brief"])
        return result

    # === 3. expand =====================================================

    def expand(self, context_id: str, selector: str, fidelity: str = "summary",
               max_blocks: int = EXPAND_BLOCK_LIMIT) -> dict:
        if fidelity not in {"summary", "facts", "quotes", "full"}:
            raise ValueError("fidelity must be summary|facts|quotes|full")
        raw = self._require_raw(context_id)
        blocks = _blocks_to_dataclass(self.store.get_blocks(context_id), raw)
        atoms = _atoms_to_dataclass(self.store.get_atoms(context_id))

        matched_blocks, matched_atoms = _resolve_selector(selector, blocks, atoms, raw)

        # Cap blocks so a broad selector can't dump the whole document. The
        # caller can raise max_blocks, target a specific block_id, or page.
        total_matched = len(matched_blocks)
        truncated = total_matched > max_blocks
        if truncated:
            matched_blocks = matched_blocks[:max_blocks]
            kept = {b.block_id for b in matched_blocks}
            matched_atoms = [a for a in matched_atoms if a.block_id in kept]

        source_refs = [
            {
                "block_id": b.block_id,
                "heading": b.heading,
                "char_range": [b.start_char, b.end_char],
            }
            for b in matched_blocks
        ]

        content_parts: List[str] = []
        exact_quotes: List[dict] = []

        if fidelity == "full":
            for b in matched_blocks:
                content_parts.append(b.text)
        elif fidelity == "summary":
            for b in matched_blocks:
                head = f"## {b.heading}\n" if b.heading else ""
                content_parts.append(head + extractive_block_summary(b))
        elif fidelity == "facts":
            for b in matched_blocks:
                b_atoms = [a for a in (matched_atoms or atoms) if a.block_id == b.block_id]
                head = f"## {b.heading}\n" if b.heading else f"## {b.block_id}\n"
                facts = "\n".join(f"- [{a.type}] {a.value}" for a in b_atoms) or "- (no atoms)"
                content_parts.append(head + extractive_block_summary(b, 2) + "\n" + facts)
        elif fidelity == "quotes":
            targets = matched_atoms or []
            if targets:
                for a in targets:
                    q = _quote_atom(a, raw)
                    exact_quotes.append(q)
                content_parts.append("\n".join(q["quote"] for q in exact_quotes))
            else:
                for b in matched_blocks:
                    exact_quotes.append(
                        {
                            "quote": b.text,
                            "source_ref": {
                                "block_id": b.block_id,
                                "char_range": [b.start_char, b.end_char],
                            },
                        }
                    )
                content_parts.append("\n\n".join(b.text for b in matched_blocks))

        expanded = "\n\n".join(p for p in content_parts if p).strip()
        result = {
            "context_id": context_id,
            "selector": selector,
            "fidelity": fidelity,
            "expanded_content": expanded,
            "source_refs": source_refs,
            "matched_block_count": total_matched,
            "returned_block_count": len(matched_blocks),
            "truncated": truncated,
            "token_estimate": estimate_tokens(expanded),
        }
        if truncated:
            result["truncation_note"] = (
                f"Selector '{selector}' matched {total_matched} blocks; returned the "
                f"first {len(matched_blocks)}. Narrow the selector, target a specific "
                "block_id from source_refs, or raise max_blocks."
            )
        if fidelity == "quotes":
            result["exact_quotes"] = exact_quotes
        return result

    # === lookup (batched multi-query retrieval) ========================

    def lookup(self, context_id: str, queries: List[str], top_blocks: int = 2) -> dict:
        """Answer several retrieval queries in ONE call.

        Round-trip eliminator: instead of one brief plus one expand/quote per
        question, an agent passes all its questions here and gets, per query,
        the top-ranked block(s) with a short summary, the protected atoms
        inside them, and a verbatim best-matching sentence with exact offsets
        — enough to answer and cite without further calls.
        """
        if not queries:
            raise ValueError("queries must be a non-empty list")
        raw = self._require_raw(context_id)
        blocks = _blocks_to_dataclass(self.store.get_blocks(context_id), raw)
        atoms = _atoms_to_dataclass(self.store.get_atoms(context_id))
        blocks_by_id = {b.block_id: b for b in blocks}
        atoms_by_block: dict = {}
        for a in atoms:
            atoms_by_block.setdefault(a.block_id, []).append(a)

        # One index serves every query — the batch costs one BM25 build.
        index = build_ranker([(b.block_id, f"{b.heading} {b.text}") for b in blocks])
        heading_tokens = {
            b.block_id: frozenset(tokenize(b.heading)) for b in blocks if b.heading
        }

        results: List[dict] = []
        for q in queries:
            intents = intent_atom_types(q)
            intent_counts = {
                bid: sum(1 for a in b_atoms if a.type in intents)
                for bid, b_atoms in atoms_by_block.items()
            } if intents else {}
            field = sorted(
                hybrid_scores(index, q, intent_counts, heading_tokens).items(),
                key=lambda kv: kv[1], reverse=True,
            )
            scored = field[:top_blocks]
            # Adaptive block count: a weak runner-up is dead weight — only
            # ship blocks genuinely competitive with the top hit.
            if scored:
                top_score = scored[0][1]
                scored = [kv for kv in scored if kv[1] >= 0.55 * top_score]

            matches: List[dict] = []
            for block_id, score in scored:
                b = blocks_by_id[block_id]
                verbatim = _best_sentence(b, q)
                facts = _filter_facts(
                    atoms_by_block.get(block_id, []), intents, verbatim["char_range"]
                )
                matches.append({
                    "block_id": b.block_id,
                    "heading": b.heading,
                    "summary": extractive_block_summary(b, 1)[:600],
                    "facts": [
                        {"atom_id": a.atom_id, "type": a.type, "value": a.value}
                        for a in facts
                    ],
                    "verbatim": verbatim,
                    "score": round(score, 3),
                })
            # Confidence must be discriminative, not absolute: on a large
            # document generic terms ("servers", "support") score everywhere.
            # Strong needs the top block to STAND OUT from the field (peak),
            # OR to directly contain most of the query's content terms
            # (coverage) — a flat field is fine when the evidence is direct.
            top = scored[0][1] if scored else 0.0
            rest = [v for _, v in field[1:10]]
            peak = (top / (sum(rest) / len(rest))) if rest else float("inf")
            coverage = index.coverage(q, scored[0][0]) if scored else 0.0
            confidence = (
                "strong"
                if (coverage >= 0.65 and top >= 0.25)
                or (top >= 0.5 and peak >= 1.35)
                else "weak"
            )
            entry: dict = {"query": q, "matches": matches, "confidence": confidence}
            if not matches:
                entry["note"] = (
                    "No block matched this query. The information is likely "
                    "NOT in this document — say so rather than guessing. "
                    "(Or try different terms / call map for the outline.)"
                )
            elif confidence == "weak":
                entry["note"] = (
                    "Only weak matches found — the document may not contain "
                    "this information. Verify with quote before relying on "
                    "these blocks; if nothing confirms, say it is not present "
                    "rather than guessing."
                )
            results.append(entry)

        payload = {"context_id": context_id, "results": results}
        payload["token_estimate"] = estimate_tokens(json.dumps(results))
        return payload

    # === 4. quote ======================================================

    def quote(self, context_id: str, atom_id: Optional[str] = None,
              search_query: Optional[str] = None) -> dict:
        raw = self._require_raw(context_id)
        if atom_id:
            row = self.store.get_atom(context_id, atom_id)
            if row is None:
                raise ValueError(f"No atom {atom_id} in {context_id}")
            atom = _atoms_to_dataclass([row])[0]
            return _quote_atom(atom, raw, with_context=True)
        if search_query:
            idx = raw.lower().find(search_query.lower())
            if idx < 0:
                return {
                    "exact_source_quote": "",
                    "surrounding_context": "",
                    "source_ref": None,
                    "note": f"'{search_query}' not found in source.",
                }
            end = idx + len(search_query)
            return {
                "exact_source_quote": raw[idx:end],
                "surrounding_context": _surrounding(raw, idx, end),
                "source_ref": {"char_range": [idx, end]},
            }
        raise ValueError("Provide either atom_id or search_query.")

    # === 5. diff =======================================================

    def diff(self, context_id: str, new_source_text: str) -> dict:
        old = self.store.get_context(context_id)
        if old is None:
            raise ValueError(f"Unknown context_id {context_id}")
        old_raw = self.store.get_raw_text(context_id)
        old_atoms = _atoms_to_dataclass(self.store.get_atoms(context_id))

        # Ingest the new text as a new version of the same label.
        ingest_res = self.ingest(
            new_source_text,
            label=old["label"],
            source_type=old["source_type"],
            task_hint=old["task_hint"],
        )
        new_context_id = ingest_res["context_id"]
        new_atoms = _atoms_to_dataclass(self.store.get_atoms(new_context_id))

        atom_diff = diff_atoms(old_atoms, new_atoms)
        block_hunks = block_text_diff(old_raw, new_source_text)
        return {
            "new_context_id": new_context_id,
            "semantic_changes": block_hunks,
            "added_atoms": atom_diff["added_atoms"],
            "removed_atoms": atom_diff["removed_atoms"],
            "changed_atoms": atom_diff["changed_atoms"],
            "risk_or_meaning_impact": risk_impact(atom_diff),
            "compact_patch_summary": compact_patch_summary(atom_diff, block_hunks),
        }

    # === 6. zip_output =================================================

    def zip_output(self, content: str, label: str, structure: Optional[str] = None) -> dict:
        if not content or not content.strip():
            raise ValueError("content is empty.")
        slug = handles.slugify(label)
        version = self.store.next_output_version(slug)
        output_id = handles.make_handle(slug, version, scheme=handles.OUT_SCHEME)

        blocks = split_blocks(content, target_tokens=500)
        section_map = [
            {
                "section_id": f"s{b.index:04d}",
                "heading": b.heading or f"Section {b.index + 1}",
                "char_range": [b.start_char, b.end_char],
                "token_estimate": b.token_estimate,
                "preview": b.preview(),
            }
            for b in blocks
        ]
        # Preview = headings + first sentence of each section, capped.
        preview_lines = []
        for b in blocks[:12]:
            head = b.heading or f"Section {b.index + 1}"
            preview_lines.append(f"• {head}: {b.preview(18)}")
        preview = "\n".join(preview_lines)

        tok_orig = estimate_tokens(content)
        tok_prev = estimate_tokens(preview)
        self.store.save_output(
            output_id=output_id,
            label=slug,
            version=version,
            raw_text=content,
            section_map=section_map,
            preview=preview,
            token_estimate_original=tok_orig,
            token_estimate_preview=tok_prev,
        )
        return {
            "output_id": output_id,
            "version": version,
            "structure": structure or "auto",
            "section_map": section_map,
            "preview": preview,
            "token_estimate_original": tok_orig,
            "token_estimate_preview": tok_prev,
            "compression_ratio": compression_ratio(tok_orig, tok_prev),
        }

    # === 7. expand_output ==============================================

    def expand_output(self, output_id: str, selector: str, fidelity: str = "section") -> dict:
        if fidelity not in {"preview", "section", "full"}:
            raise ValueError("fidelity must be preview|section|full")
        row = self.store.get_output(output_id)
        if row is None:
            raise ValueError(f"Unknown output_id {output_id}")
        raw = self.store.get_output_raw(output_id)
        section_map = json.loads(row["section_map"])

        if fidelity == "preview":
            content = row["preview"]
            return {
                "output_id": output_id,
                "selector": selector,
                "fidelity": fidelity,
                "expanded_output_content": content,
                "token_estimate": estimate_tokens(content),
            }
        if fidelity == "full":
            return {
                "output_id": output_id,
                "selector": selector,
                "fidelity": fidelity,
                "expanded_output_content": raw,
                "token_estimate": estimate_tokens(raw),
            }

        # section: resolve selector to one (or more) sections
        matches = _resolve_sections(selector, section_map)
        parts = [raw[m["char_range"][0]:m["char_range"][1]] for m in matches]
        content = "\n\n".join(parts).strip()
        return {
            "output_id": output_id,
            "selector": selector,
            "fidelity": fidelity,
            "matched_sections": [m["section_id"] for m in matches],
            "expanded_output_content": content,
            "token_estimate": estimate_tokens(content),
        }

    # === 8. audit ======================================================

    def audit(self, context_id: str) -> dict:
        ctx = self.store.get_context(context_id)
        if ctx is None:
            raise ValueError(f"Unknown context_id {context_id}")
        raw = self.store.get_raw_text(context_id)
        atoms = _atoms_to_dataclass(self.store.get_atoms(context_id))
        blocks = _blocks_to_dataclass(self.store.get_blocks(context_id), raw)

        counts_by_type: dict = {}
        for a in atoms:
            counts_by_type[a.type] = counts_by_type.get(a.type, 0) + 1

        preservation = {}
        for bucket, types in PRESERVATION_BUCKETS.items():
            preservation[bucket] = sum(counts_by_type.get(t, 0) for t in types)

        # Hard guarantee: prove every span reconstructs byte-for-byte.
        block_spans = [(b.block_id, b.start_char, b.end_char, b.text) for b in blocks]
        rt = verify.roundtrip(raw, block_spans, atoms)
        cov = verify.coverage(raw, atoms)

        safe, unsafe, warnings = _audit_guidance(
            ctx["source_type"], counts_by_type, ctx["compression_ratio"]
        )
        if not rt["ok"]:
            warnings.insert(
                0,
                f"INTEGRITY FAILURE: {len(rt['block_mismatches'])} blocks / "
                f"{len(rt['atom_mismatches'])} atoms did not reconstruct verbatim.",
            )
        low_recall = {t: c["recall"] for t, c in cov.items() if c["recall"] < 0.9}
        if low_recall:
            warnings.append(
                "Possible extraction misses (recall <0.9): "
                + ", ".join(f"{t}={r}" for t, r in low_recall.items())
                + ". Use expand(fidelity='full')/quote() to be safe."
            )

        return {
            "context_id": context_id,
            "source_type": ctx["source_type"],
            "brief_mode": ctx["brief_mode"],
            "tokenizer": ctx["tokenizer"],
            "compression_ratio": ctx["compression_ratio"],
            "token_estimate_original": ctx["token_estimate_original"],
            "token_estimate_brief": ctx["token_estimate_brief"],
            "protected_atom_counts": counts_by_type,
            "estimated_preservation": preservation,
            "verbatim_integrity": {
                "ok": rt["ok"],
                "atoms_checked": rt["atoms_checked"],
                "atoms_verbatim_recoverable": rt["atoms_verbatim_recoverable"],
                "block_mismatches": rt["block_mismatches"],
                "atom_mismatches": rt["atom_mismatches"],
            },
            "extraction_recall_estimate": cov,
            "safe_for": safe,
            "unsafe_for": unsafe,
            "warnings": warnings,
        }

    # === list ==========================================================

    def list_all(self) -> dict:
        contexts = [
            {
                "context_id": r["context_id"],
                "label": r["label"],
                "version": r["version"],
                "source_type": r["source_type"],
                "token_estimate_original": r["token_estimate_original"],
                "token_estimate_brief": r["token_estimate_brief"],
                "compression_ratio": r["compression_ratio"],
            }
            for r in self.store.list_contexts()
        ]
        outputs = [
            {
                "output_id": r["output_id"],
                "label": r["label"],
                "version": r["version"],
                "token_estimate_original": r["token_estimate_original"],
                "token_estimate_preview": r["token_estimate_preview"],
            }
            for r in self.store.list_outputs()
        ]
        return {"contexts": contexts, "outputs": outputs}

    # --- internals ------------------------------------------------------

    def _require_raw(self, context_id: str) -> str:
        raw = self.store.get_raw_text(context_id)
        if raw is None:
            raise ValueError(f"Unknown context_id {context_id}")
        return raw


# --- module helpers -----------------------------------------------------

def _safety_notes(source_type: str, atom_count: int) -> List[str]:
    notes = [
        "Semantic brief is lossy and extractive; do not rely on it for exact "
        "wording. Use expand(fidelity='quotes') or quote() for verbatim text.",
        f"{atom_count} atoms preserved verbatim with exact source offsets.",
        "Full raw source is stored compressed and only returned via "
        "expand(fidelity='full').",
    ]
    if source_type == "contract":
        notes.append(
            "Legal text: always verify obligations, parties, money values and "
            "dates against exact quotes before relying on them."
        )
    if source_type == "code":
        notes.append(
            "Code: identifiers are preserved as atoms but logic/structure is "
            "summarized; expand to 'full' before editing."
        )
    return notes


_SENT_END = re.compile(r"[.!?](?:[ \t]+|(?=\n)|$)|\n+")

# Verbatim quotes are sentences/lines, never blocks; cap defends against
# pathological "sentences" (minified text, giant table rows).
VERBATIM_CAP_CHARS = 400


def _sentence_spans(text: str) -> List[tuple]:
    """Character spans of sentences in ``text``, whitespace-trimmed.

    Boundaries are sentence punctuation OR newlines — unpunctuated text
    (logs, tables) must split into lines, not collapse into one giant
    "sentence" spanning the whole block.
    """
    spans: List[tuple] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.start() + (1 if text[m.start()] in ".!?" else 0)
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    trimmed = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            trimmed.append((s, e))
    return trimmed


def _best_sentence(block: Block, query: str) -> dict:
    """Verbatim sentence in ``block`` best matching ``query``, with offsets.

    Offsets are into the raw source (block offsets are), so the quote
    reconstructs byte-for-byte — same guarantee as quote()/expand(quotes).
    """
    q_terms = set(tokenize(query))
    best_span = None
    best_score = -1.0
    for s, e in _sentence_spans(block.text):
        overlap = len(q_terms & set(tokenize(block.text[s:e])))
        # Prefer higher overlap; among ties, the shorter (denser) sentence.
        score = overlap - (e - s) * 1e-6
        if score > best_score:
            best_score = score
            best_span = (s, e)
    if best_span is None:
        return {"quote": "", "char_range": [block.start_char, block.start_char]}
    s, e = best_span
    e = min(e, s + VERBATIM_CAP_CHARS)
    return {
        "quote": block.text[s:e],
        "char_range": [block.start_char + s, block.start_char + e],
    }


def _filter_facts(b_atoms, intents, verbatim_range, cap: int = 8):
    """Atoms worth shipping for one query: in the verbatim sentence, or of
    the answer type the query implies. Falls back to all atoms (capped) when
    neither signal applies, so unknown intents lose nothing."""
    vs, ve = verbatim_range
    in_sentence = [a for a in b_atoms if a.start_char >= vs and a.end_char <= ve]
    if intents:
        typed = [a for a in b_atoms if a.type in intents and a not in in_sentence]
        keep = in_sentence + typed
    else:
        keep = in_sentence + [a for a in b_atoms if a not in in_sentence]
    if not keep:
        keep = list(b_atoms)
    return keep[:cap]


def _surrounding(raw: str, start: int, end: int, window: int = 160) -> str:
    s = max(0, start - window)
    e = min(len(raw), end + window)
    prefix = "…" if s > 0 else ""
    suffix = "…" if e < len(raw) else ""
    return f"{prefix}{raw[s:e]}{suffix}"


def _quote_atom(atom: Atom, raw: str, with_context: bool = False) -> dict:
    start, end = atom.start_char, atom.end_char
    quote = raw[start:end]
    ref = {
        "atom_id": atom.atom_id,
        "type": atom.type,
        "char_range": [start, end],
        "block_id": atom.block_id,
        "occurrences": atom.count,
    }
    if with_context:
        return {
            "exact_source_quote": quote,
            "surrounding_context": _surrounding(raw, start, end),
            "source_ref": ref,
        }
    return {"quote": quote, "source_ref": ref}


def _resolve_selector(selector: str, blocks: List[Block], atoms: List[Atom], raw: str):
    """Return (matched_blocks, matched_atoms) for an expand selector.

    Selector resolution order: exact block id, exact atom id, heading/section
    name match, then free-text search across atoms and block text.
    """
    sel = (selector or "").strip()
    sel_low = sel.lower()
    block_by_id = {b.block_id: b for b in blocks}
    atom_by_id = {a.atom_id: a for a in atoms}

    if sel in block_by_id:
        return [block_by_id[sel]], [a for a in atoms if a.block_id == sel]
    if sel in atom_by_id:
        a = atom_by_id[sel]
        b = block_by_id.get(a.block_id)
        return ([b] if b else []), [a]

    # Heading / section-name match.
    heading_hits = [b for b in blocks if b.heading and sel_low in b.heading.lower()]
    if heading_hits:
        hit_ids = {b.block_id for b in heading_hits}
        return heading_hits, [a for a in atoms if a.block_id in hit_ids]

    # Free-text search: rank blocks by BM25 relevance to the query (better than
    # raw substring presence), plus atoms whose value matches the query text.
    matched_atoms = [a for a in atoms if sel_low in a.value.lower()]
    index = build_ranker([(b.block_id, f"{b.heading} {b.text}") for b in blocks])
    ranked = index.rank(sel, top_k=8)
    by_id = {b.block_id: b for b in blocks}
    text_blocks = [by_id[bid] for bid, _ in ranked if bid in by_id]

    # Substring fallback catches exact phrases BM25 stopword-filters away.
    if not text_blocks:
        text_blocks = [b for b in blocks if sel_low and sel_low in b.text.lower()][:8]
    if not text_blocks and matched_atoms:
        block_ids = {a.block_id for a in matched_atoms if a.block_id}
        text_blocks = [b for b in blocks if b.block_id in block_ids]
    if not text_blocks and not matched_atoms:
        # Nothing matched; surface top-of-document block so caller isn't empty-handed.
        return (blocks[:1], [])
    return text_blocks[:8], matched_atoms[:40]


def _resolve_sections(selector: str, section_map: List[dict]) -> List[dict]:
    sel = (selector or "").strip().lower()
    by_id = {s["section_id"]: s for s in section_map}
    if selector in by_id:
        return [by_id[selector]]
    hits = [s for s in section_map if sel and sel in s["heading"].lower()]
    if hits:
        return hits
    hits = [s for s in section_map if sel and sel in s["preview"].lower()]
    return hits or section_map[:1]


def _audit_guidance(source_type: str, counts: dict, ratio: float):
    safe = ["high-level overview", "navigation to relevant sections", "locating facts/atoms"]
    unsafe = [
        "verbatim reproduction without expand()/quote()",
        "any exact-wording-dependent decision made from the brief alone",
    ]
    warnings = []
    if ratio < 1.0:
        warnings.append(
            f"Source is small: brief ({ratio}x) is not smaller than the original. "
            "SaveContext adds little value below ~1,000 tokens; pass the text directly."
        )
    if ratio >= 20:
        warnings.append(
            f"High compression ratio ({ratio}x): brief omits most prose; rely on "
            "expand/quote for detail."
        )
    if counts.get("number", 0) > 50:
        warnings.append(
            "Many numeric atoms; verify any arithmetic against exact quotes."
        )

    if source_type == "contract":
        safe.append("identifying clauses, parties, obligations to review")
        unsafe.append("legally binding interpretation without counsel review of exact text")
    elif source_type == "code":
        safe.append("locating identifiers, files, and call sites")
        unsafe.append("editing code from the brief; expand to full first")
    elif source_type == "logs":
        safe.append("spotting error patterns and timestamps")
        unsafe.append("incident root-cause from brief alone; expand the relevant window")
    elif source_type == "research":
        safe.append("understanding scope, claims, and key numbers")
        unsafe.append("citing exact figures/quotes without quote()")
    return safe, unsafe, warnings
