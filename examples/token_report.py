"""Token-tracking harness: how many tokens does each mode actually cost?

Measures the real token payloads that enter a model's context window for a
representative workflow (ingest -> brief -> expand) under each *summary* mode,
versus the baseline of pasting the raw document. This is the precise
compression metric: deterministic, reproducible, and isolated from any agent
reasoning overhead.

Modes compared:
  - extractive   : rule-based default (no model)
  - agent        : a concise brief authored by the calling model (simulated)
  - local-llm    : abstractive brief from a local model (simulated)

Retrieval backend (BM25 vs embeddings) is NOT a variable here: it changes which
blocks are selected, not the size of the payloads, so token cost is identical.

Run:  python examples/token_report.py
      python examples/token_report.py 60 400
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from big_demo import build_big_contract  # noqa: E402
from savecontext import llm  # noqa: E402
from savecontext.service import VaultService  # noqa: E402
from savecontext.storage import Store  # noqa: E402
from savecontext.tokenizer import estimate_tokens  # noqa: E402

TASK = "what is the limitation of liability cap and the indemnification obligations"
SELECTOR = "Limitation of Liability"


def _toks(obj) -> int:
    """Tokens of a payload as the model would receive it (JSON-serialized)."""
    if isinstance(obj, str):
        return estimate_tokens(obj)
    return estimate_tokens(json.dumps(obj, ensure_ascii=False))


def _agent_brief(doc: str) -> str:
    # A realistic, dense brief such as the calling model would write having read
    # the document — concrete facts, no filler.
    return (
        "Master Services & Licensing Agreement (Acme/Globex, eff. 2024-01-15). "
        "Recurring fees from $25k/mo rising per section, due in 15-60 days, "
        "late interest ~1.5%/mo. Liability capped per-section ($250k+), no "
        "indirect damages, carve-out for gross negligence. Provider indemnifies "
        "Customer (third-party IP), claim notice required. Confidentiality 2-5 "
        "yrs; data residency + breach-notice windows. Terms 2-5 yrs, "
        "termination for convenience on 15-90 days notice. Core engine in "
        "src/engine/core_engine.py; no reverse engineering."
    )


def measure(doc: str, mode: str) -> dict:
    svc = VaultService(store=Store(db_path=":memory:"))

    agent_brief = None
    restore = None
    if mode == "agent":
        agent_brief = _agent_brief(doc)
    elif mode == "local-llm":
        os.environ["SAVECONTEXT_LLM_SUMMARY"] = "1"
        restore = llm._generate
        llm._generate = lambda prompt, num_predict: _agent_brief(doc)

    try:
        ing = svc.ingest(doc, label=f"tok-{mode}", source_type="contract",
                         agent_brief=agent_brief)
        cid = ing["context_id"]
        br = svc.brief(cid, task=TASK)
        ex = svc.expand(cid, selector=SELECTOR, fidelity="facts")
    finally:
        if restore is not None:
            llm._generate = restore
            os.environ.pop("SAVECONTEXT_LLM_SUMMARY", None)
        svc.close()

    ingest_total = _toks(ing)
    q_brief = _toks(br)
    q_expand = _toks(ex)
    return {
        "mode": mode,
        "brief_mode": ing["brief_mode"],
        "raw": ing["token_estimate_original"],
        "brief_only": ing["token_estimate_brief"],
        "ingest_resp": ingest_total,      # brief + block_map + atoms (turn 1)
        "q_brief": q_brief,               # task brief response (turn 2)
        "q_expand": q_expand,             # targeted expand response (turn 3)
        "first_answer": ingest_total + q_brief + q_expand,
        "followup": q_brief + q_expand,   # brief reused, no re-ingest
    }


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [60, 400]
    for sections in sizes:
        doc = build_big_contract(num_sections=sections)
        raw = estimate_tokens(doc)
        print(f"\n=========== document: {sections} sections, {raw:,} raw tokens ===========")
        rows = [measure(doc, m) for m in ("extractive", "agent", "local-llm")]

        hdr = f"{'mode':<11}{'brief':>7}{'ingest':>8}{'+brief':>8}{'+expand':>9}{'=1st ans':>9}{'followup':>9}{'1st x':>7}{'foll x':>7}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            first_x = round(raw / max(1, r["first_answer"]), 1)
            foll_x = round(raw / max(1, r["followup"]), 1)
            print(
                f"{r['mode']:<11}{r['brief_only']:>7}{r['ingest_resp']:>8}"
                f"{r['q_brief']:>8}{r['q_expand']:>9}{r['first_answer']:>9}"
                f"{r['followup']:>9}{first_x:>6}x{foll_x:>6}x"
            )

    print(
        "\nReading the table:\n"
        "  brief    = the stored semantic brief alone (token_estimate_brief)\n"
        "  ingest   = full turn-1 response the model reads (brief + block_map + atoms)\n"
        "  +brief   = turn-2 task brief response;  +expand = turn-3 facts expansion\n"
        "  1st ans  = total tokens to answer the FIRST question (ingest+brief+expand)\n"
        "  followup = tokens for each LATER question (brief reused, no re-ingest)\n"
        "  Nx       = compression vs pasting the raw doc (raw / cost). Baseline = raw tokens.\n"
        "Note: block_map scales with #blocks, so 'ingest' grows with doc size even though\n"
        "'brief' stays flat — the per-question 'followup' cost is the steady-state win."
    )


if __name__ == "__main__":
    main()
