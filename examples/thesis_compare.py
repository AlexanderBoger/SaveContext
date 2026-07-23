#!/usr/bin/env python3
"""
Compare ContextVault vs baseline (raw document in context) for a PDF.

Measures:
  - Input token count for each approach (exact, via count_tokens API)
  - Actual response quality by calling claude-opus-4-8 for both
  - Cost at Opus 4.8 rates ($5/1M input, $25/1M output)
  - Side-by-side answers + neutral judge verdict

Usage:
  python thesis_compare.py /path/to/doc.pdf
  python thesis_compare.py  # defaults to thesis PDF
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from contextvault.loaders import load_path
from contextvault.service import VaultService
from contextvault.tokenizer import estimate_tokens

PDF_PATH = "/Users/alex.boger/Downloads/Thesis_in_lukas_group.pdf"
PRICE_IN = 5.00   # $ per 1M input tokens
PRICE_OUT = 25.00  # $ per 1M output tokens

# ---------------------------------------------------------------------------
# CLI args parsed early so they can override env vars.
# Usage:
#   Direct Anthropic key:
#     python thesis_compare.py --api-key sk-ant-...
#
#   LiteLLM / any OpenAI-compatible proxy:
#     python thesis_compare.py --api-base https://litellm.labs.jb.gg --api-key <key> [--model claude-opus-4-8]
# ---------------------------------------------------------------------------

import argparse

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("pdf", nargs="?", default=None)
_ap.add_argument("--api-base", default=os.environ.get("CV_API_BASE", ""))
_ap.add_argument("--api-key",  default=os.environ.get("CV_API_KEY",
                                        os.environ.get("ANTHROPIC_API_KEY", "")))
_ap.add_argument("--model",    default=os.environ.get("CV_MODEL", "claude-opus-4-8"))
_ap.add_argument("--config",   default=None,
                 help="Path to a JSON file with label/pdf/questions/fact_questions")
_ARGS, _ = _ap.parse_known_args()

MODEL    = _ARGS.model
API_BASE = _ARGS.api_base
API_KEY  = _ARGS.api_key
PROVIDER = "openai_compat" if API_BASE else "anthropic"


def _load_config():
    if not _ARGS.config:
        return None
    with open(_ARGS.config) as fh:
        return json.load(fh)


def _make_client():
    if PROVIDER == "openai_compat":
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai not installed. Run: pip3 install openai")
        base = API_BASE.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        return OpenAI(base_url=base, api_key=API_KEY or "placeholder")
    else:
        import anthropic
        kwargs = {"api_key": API_KEY} if API_KEY else {}
        if API_BASE:
            kwargs["base_url"] = API_BASE
        return anthropic.Anthropic(**kwargs)

QUESTIONS = [
    {
        "id": "research_question",
        "q": "What is the main research question or central hypothesis of this thesis? Summarise it in 2-3 sentences.",
        "heading": "introduction",
    },
    {
        "id": "methods",
        "q": "What experimental or analytical methods were used? List the key techniques and tools.",
        "heading": "methods",
    },
    {
        "id": "conclusions",
        "q": "What are the main conclusions and their implications for future work?",
        "heading": "conclusion",
    },
]

# Fact questions: each has an exact string that must appear verbatim in a correct answer.
FACT_QUESTIONS = [
    {
        "id": "rmse_spq",
        "q": "What is the exact position RMSE (in mm) of the single-particle quantizer?",
        "must_contain": "0.2374",
        "heading": "results",
    },
    {
        "id": "rmse_psq",
        "q": "What is the exact position RMSE (in mm) of the particle-set quantizer?",
        "must_contain": "4.7009",
        "heading": "results",
    },
    {
        "id": "codebook_size",
        "q": "How many codebooks are used in residual vector quantization, and what is the size (number of entries K) of each codebook?",
        "must_contain": "512",
        "heading": "methods",
    },
    {
        "id": "energy_range",
        "q": "What energy range (in MeV) was used to sample incoming particle energies in the Geant4 simulation?",
        "must_contain": "300",
        "heading": "methods",
    },
]

SYSTEM_BASE = (
    "You are a research assistant. Answer questions about the provided document "
    "accurately and concisely. Be specific — cite relevant details."
)
SYSTEM_CV = (
    "You are a research assistant working from a ContextVault semantic brief and "
    "targeted content expansions. The brief gives a structured overview; the expand "
    "sections contain the most relevant raw content. Answer accurately and specifically."
)


# ---------------------------------------------------------------------------
# API helpers — dispatch to Anthropic SDK or litellm depending on PROVIDER
# ---------------------------------------------------------------------------

def call_model(client, messages, system, max_tokens=800):
    if PROVIDER == "openai_compat":
        oai_messages = [{"role": "system", "content": system}] + messages
        resp = client.chat.completions.create(
            model=MODEL, messages=oai_messages, max_tokens=max_tokens
        )
        text    = resp.choices[0].message.content or ""
        tok_in  = resp.usage.prompt_tokens
        tok_out = resp.usage.completion_tokens
        return text, tok_in, tok_out
    else:
        with client.messages.stream(
            model=MODEL, max_tokens=max_tokens, system=system, messages=messages
        ) as stream:
            msg = stream.get_final_message()
        text = "\n".join(b.text for b in msg.content if hasattr(b, "text"))
        return text, msg.usage.input_tokens, msg.usage.output_tokens


# ---------------------------------------------------------------------------
# Mode 3: Model-driven retrieval via tool use
#   The model gets only the handle + question. It decides when to call
#   brief / expand / quote, and we accumulate all token costs across the loop.
# ---------------------------------------------------------------------------

CV_TOOLS = [
    {
        "name": "cv_brief",
        "description": (
            "Get a semantic overview of a ContextVault document. "
            "Always call this first to orient yourself before deciding what to expand."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {"type": "string", "description": "The ctx://… handle"},
                "task": {"type": "string", "description": "What you want to answer (free text)"},
            },
            "required": ["context_id"],
        },
    },
    {
        "name": "cv_expand",
        "description": (
            "Retrieve raw content for a specific section of a ContextVault document. "
            "Use the block IDs or section headings from the brief as the selector."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {"type": "string"},
                "selector": {"type": "string", "description": "Block ID (b0012) or heading text"},
                "fidelity": {
                    "type": "string",
                    "enum": ["summary", "facts", "quotes", "full"],
                    "description": "How much raw content to return. 'facts' is a good default.",
                },
                "max_blocks": {"type": "integer", "description": "Cap on blocks returned (default 3)"},
            },
            "required": ["context_id", "selector"],
        },
    },
    {
        "name": "cv_quote",
        "description": "Return an exact verbatim excerpt from the document for a specific claim or phrase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "context_id": {"type": "string"},
                "search_query": {"type": "string", "description": "The claim or phrase to find verbatim"},
            },
            "required": ["context_id", "search_query"],
        },
    },
]

_CV_TOOL_SYSTEM = (
    "You are a research assistant. A document has been pre-indexed in ContextVault. "
    "You do NOT have the document in your context — use the provided tools to retrieve "
    "what you need. Start with cv_brief to orient yourself, then cv_expand for relevant "
    "sections. Only call cv_quote when you need an exact verbatim passage."
)


def _dispatch_cv_tool(svc, name, args):
    if name == "cv_brief":
        return svc.brief(args["context_id"], task=args.get("task", ""), max_tokens=600)
    if name == "cv_expand":
        return svc.expand(
            args["context_id"],
            selector=args["selector"],
            fidelity=args.get("fidelity", "facts"),
            max_blocks=args.get("max_blocks", 3),
        )
    if name == "cv_quote":
        return svc.quote(args["context_id"], search_query=args["search_query"])
    return {"error": f"unknown tool {name}"}


def _run_cv_agentic_one(client, svc, cid, question):
    """Agentic tool-use loop for one question. Returns (answer, total_tok_in, total_tok_out)."""
    if PROVIDER == "openai_compat":
        # OpenAI tool-use format
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in CV_TOOLS
        ]
        messages = [
            {"role": "system", "content": _CV_TOOL_SYSTEM},
            {"role": "user", "content": f"Document handle: {cid}\n\nQuestion: {question}"},
        ]
        tok_in = tok_out = 0
        while True:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=oai_tools,
                tool_choice="auto", max_tokens=1200,
            )
            tok_in  += resp.usage.prompt_tokens
            tok_out += resp.usage.completion_tokens
            msg = resp.choices[0].message
            messages.append(msg)
            if not msg.tool_calls:
                return msg.content or "", tok_in, tok_out
            for tc in msg.tool_calls:
                result = _dispatch_cv_tool(svc, tc.function.name,
                                           json.loads(tc.function.arguments))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
    else:
        # Anthropic tool-use format
        messages = [{"role": "user", "content": f"Document handle: {cid}\n\nQuestion: {question}"}]
        tok_in = tok_out = 0
        while True:
            resp = client.messages.create(
                model=MODEL, system=_CV_TOOL_SYSTEM,
                tools=CV_TOOLS, max_tokens=1200, messages=messages,
            )
            tok_in  += resp.usage.input_tokens
            tok_out += resp.usage.output_tokens
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                text = "\n".join(b.text for b in resp.content if hasattr(b, "text"))
                return text, tok_in, tok_out
            tool_results = []
            for blk in resp.content:
                if blk.type == "tool_use":
                    result = _dispatch_cv_tool(svc, blk.name, blk.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": blk.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
            messages.append({"role": "user", "content": tool_results})


def run_cv_agentic(client, cid):
    print("── ContextVault model-driven (tool use) ──")
    svc = VaultService()
    results = []
    for q in QUESTIONS:
        print(f"  [{q['id']}] calling...", end=" ", flush=True)
        answer, tok_in, tok_out = _run_cv_agentic_one(client, svc, cid, q["q"])
        print(f"{tok_in:,} in / {tok_out:,} out")
        results.append({"id": q["id"], "question": q["q"],
                        "answer": answer, "tok_in": tok_in, "tok_out": tok_out})
    svc.close()
    return results


# ---------------------------------------------------------------------------
# Fact test — run all three conditions on FACT_QUESTIONS, auto-score by
# whether the must_contain string appears verbatim in the answer.
# ---------------------------------------------------------------------------

def run_fact_test(client, doc_text, cid):
    print("── Fact test (exact numbers) ──")
    svc = VaultService()
    rows = []
    for fq in FACT_QUESTIONS:
        print(f"  [{fq['id']}]", end="  ", flush=True)

        # baseline
        msg_b = [{"role": "user", "content": f"DOCUMENT:\n{doc_text}\n\nQUESTION: {fq['q']}"}]
        ans_b, ti_b, _ = call_model(client, msg_b, SYSTEM_BASE)
        print(f"base={ti_b:,}", end="  ", flush=True)

        # cv pre-fetch
        ex = svc.expand(cid, selector=fq["heading"], fidelity="quotes", max_blocks=3)
        msg_c = [{"role": "user", "content":
                  f"EXPAND [{fq['heading']}]:\n{json.dumps(ex, ensure_ascii=False)}\n\nQUESTION: {fq['q']}"}]
        ans_c, ti_c, _ = call_model(client, msg_c, SYSTEM_CV)
        print(f"cv-pre={ti_c:,}", end="  ", flush=True)

        # cv agentic
        ans_a, ti_a, _ = _run_cv_agentic_one(client, svc, cid, fq["q"])
        print(f"cv-agent={ti_a:,}")

        needle = fq["must_contain"]
        rows.append({
            "id": fq["id"],
            "q": fq["q"],
            "must_contain": needle,
            "baseline":   {"answer": ans_b, "tok_in": ti_b, "correct": needle in ans_b},
            "cv_prefetch": {"answer": ans_c, "tok_in": ti_c, "correct": needle in ans_c},
            "cv_agentic":  {"answer": ans_a, "tok_in": ti_a, "correct": needle in ans_a},
        })
    svc.close()
    return rows


def report_facts(rows):
    W = 82
    print("\n" + "═"*W)
    print(" FACT ACCURACY TEST  (must contain exact value)")
    print("═"*W)
    print(f"  {'Question':<30} {'Baseline':^20} {'CV pre-fetch':^20} {'CV agentic':^20}")
    print("─"*W)
    for r in rows:
        def cell(d):
            mark = "✓" if d["correct"] else "✗"
            return f"{mark} {d['tok_in']:,} tok"
        print(f"  {r['id']:<30} {cell(r['baseline']):^20} {cell(r['cv_prefetch']):^20} {cell(r['cv_agentic']):^20}")

    correct_b = sum(1 for r in rows if r["baseline"]["correct"])
    correct_p = sum(1 for r in rows if r["cv_prefetch"]["correct"])
    correct_a = sum(1 for r in rows if r["cv_agentic"]["correct"])
    n = len(rows)
    print("─"*W)
    print(f"  {'SCORE':<30} {correct_b}/{n} correct{' ':^13} {correct_p}/{n} correct{' ':^13} {correct_a}/{n} correct")
    print("═"*W)

    print("\n  Detail:")
    for r in rows:
        print(f"\n  [{r['id']}]  must contain: '{r['must_contain']}'")
        for label, d in [("Baseline", r["baseline"]), ("CV pre-fetch", r["cv_prefetch"]), ("CV agentic", r["cv_agentic"])]:
            mark = "✓" if d["correct"] else "✗"
            print(f"    {mark} {label}: {d['answer'][:200].replace(chr(10),' ')}")


# ---------------------------------------------------------------------------
# Baseline: agentic tool-use loop over the raw document.
# Model gets search_doc and read_section tools — navigates raw text
# the same way the CV agent navigates the compressed index.
# No full document dump; the model fetches only what it needs.
# ---------------------------------------------------------------------------

RAW_TOOLS = [
    {
        "name": "search_doc",
        "description": (
            "Search the document for a keyword or phrase. "
            "Returns up to 20 matching lines with surrounding context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for (case-insensitive)"},
                "context_lines": {"type": "integer", "description": "Lines of context around each match (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_section",
        "description": "Read a contiguous slice of the document by line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_line": {"type": "integer"},
                "end_line":   {"type": "integer"},
            },
            "required": ["start_line", "end_line"],
        },
    },
]

_RAW_SYSTEM = (
    "You are a research assistant. You have search and read tools to navigate a document. "
    "Use search_doc to find relevant passages, read_section to read them in full. "
    "Fetch only what you need — do not request the entire document at once."
)


def _dispatch_raw_tool(doc_lines, name, args):
    if name == "search_doc":
        q = args["query"].lower()
        ctx = args.get("context_lines", 5)
        hits, seen = [], set()
        for i, line in enumerate(doc_lines):
            if q in line.lower():
                lo, hi = max(0, i - ctx), min(len(doc_lines), i + ctx + 1)
                for j in range(lo, hi):
                    if j not in seen:
                        hits.append(f"{j+1:>6}: {doc_lines[j]}")
                        seen.add(j)
                if len(seen) >= 200:
                    break
        return "\n".join(hits) if hits else f"No matches for '{args['query']}'"
    if name == "read_section":
        lo = max(0, args["start_line"] - 1)
        hi = min(len(doc_lines), args["end_line"])
        return "\n".join(f"{i+1:>6}: {doc_lines[i]}" for i in range(lo, hi))
    return {"error": f"unknown tool {name}"}


def _run_baseline_one(client, doc_lines, question):
    """Agentic loop for one question over raw document tools."""
    if PROVIDER == "openai_compat":
        oai_tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["input_schema"],
        }} for t in RAW_TOOLS]
        messages = [
            {"role": "system", "content": _RAW_SYSTEM},
            {"role": "user",   "content": f"Question: {question}"},
        ]
        tok_in = tok_out = 0
        while True:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=oai_tools,
                tool_choice="auto", max_tokens=1200,
            )
            tok_in  += resp.usage.prompt_tokens
            tok_out += resp.usage.completion_tokens
            msg = resp.choices[0].message
            messages.append(msg)
            if not msg.tool_calls:
                return msg.content or "", tok_in, tok_out
            for tc in msg.tool_calls:
                result = _dispatch_raw_tool(doc_lines, tc.function.name,
                                            json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                  "content": str(result)})
    else:
        messages = [{"role": "user", "content": f"Question: {question}"}]
        tok_in = tok_out = 0
        while True:
            resp = client.messages.create(
                model=MODEL, system=_RAW_SYSTEM,
                tools=RAW_TOOLS, max_tokens=1200, messages=messages,
            )
            tok_in  += resp.usage.input_tokens
            tok_out += resp.usage.output_tokens
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return "\n".join(b.text for b in resp.content if hasattr(b, "text")), tok_in, tok_out
            tool_results = []
            for blk in resp.content:
                if blk.type == "tool_use":
                    result = _dispatch_raw_tool(doc_lines, blk.name, blk.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": blk.id,
                                         "content": str(result)})
            messages.append({"role": "user", "content": tool_results})


def run_baseline(client, doc_text):
    print("── Baseline (agentic: search + read tools over raw text) ──")
    doc_lines = doc_text.splitlines()
    results = []
    for q in QUESTIONS:
        print(f"  [{q['id']}] calling...", end=" ", flush=True)
        answer, tok_in, tok_out = _run_baseline_one(client, doc_lines, q["q"])
        print(f"{tok_in:,} in / {tok_out:,} out")
        results.append({"id": q["id"], "question": q["q"],
                        "answer": answer, "tok_in": tok_in, "tok_out": tok_out})
    return results


# ---------------------------------------------------------------------------
# ContextVault: ingest once, brief + expand per question
# ---------------------------------------------------------------------------

def run_cv(client, doc_text, label):
    print("── ContextVault (brief + targeted expand) ──")
    svc = VaultService()

    print(f"  ingesting...", end=" ", flush=True)
    ing = svc.ingest(doc_text, label=label, source_type="auto")
    cid = ing["context_id"]
    orig = ing.get("token_estimate_original", "?")
    ratio = ing.get("compression_ratio", "?")
    print(f"handle={cid}  {orig} tok → {ratio}x compressed")

    combined_task = " | ".join(q["q"] for q in QUESTIONS)
    br = svc.brief(cid, task=combined_task, max_tokens=600)
    brief_text = json.dumps(br, ensure_ascii=False)

    # Fetch all expands up front, then ask everything in one call —
    # mirrors the single-call baseline so the structures are comparable.
    sections = []
    for q in QUESTIONS:
        ex = svc.expand(cid, selector=q["heading"], fidelity="facts", max_blocks=3)
        sections.append(f"SECTION [{q['heading']}]:\n{json.dumps(ex, ensure_ascii=False)}")

    numbered = "\n".join(f"{i+1}. {q['q']}" for i, q in enumerate(QUESTIONS))
    content = (
        f"DOCUMENT BRIEF:\n{brief_text}\n\n"
        + "\n\n".join(sections)
        + f"\n\nAnswer each question below. Label each answer Q1, Q2, Q3, etc.\n\n{numbered}"
    )
    messages = [{"role": "user", "content": content}]
    print(f"  calling (1 request, {len(QUESTIONS)} questions)...", end=" ", flush=True)
    combined, tok_in, tok_out = call_model(client, messages, SYSTEM_CV, max_tokens=2000)
    print(f"{tok_in:,} in / {tok_out:,} out")

    results = []
    for i, q in enumerate(QUESTIONS):
        marker = f"Q{i+1}"
        next_marker = f"Q{i+2}"
        start = combined.find(marker)
        end   = combined.find(next_marker) if i + 1 < len(QUESTIONS) else len(combined)
        answer = combined[start:end].strip() if start != -1 else combined
        results.append({
            "id": q["id"], "question": q["q"], "answer": answer,
            "tok_in": tok_in if i == 0 else 0,
            "tok_out": tok_out if i == 0 else 0,
        })

    svc.close()
    return results, {"cid": cid, "orig": orig, "ratio": ratio}


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge(client, question, ans_base, ans_cv):
    messages = [{"role": "user", "content": (
        f"You are a neutral evaluator comparing two answers to the same academic question.\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER A (Baseline – full document in context):\n{ans_base}\n\n"
        f"ANSWER B (ContextVault – brief + targeted expand):\n{ans_cv}\n\n"
        f"Rate both on accuracy, completeness, and clarity (1-5 each). "
        f"Then give a one-sentence verdict: which is better, why, or 'equivalent'."
    )}]
    system = "You are a neutral academic evaluator."
    text, _, _ = call_model(client, messages, system)
    return text


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(baseline, cv_prefetch, cv_agentic, cv_meta, verdicts_pre, verdicts_agent):
    def totals(rows):
        return sum(r["tok_in"] for r in rows), sum(r["tok_out"] for r in rows)

    def cost(ti, to):
        return (ti * PRICE_IN + to * PRICE_OUT) / 1_000_000

    def pct(a, b):
        return "—" if a == 0 else f"{(1 - b/a)*100:.1f}%"

    b_in,  b_out  = totals(baseline)
    cp_in, cp_out = totals(cv_prefetch)
    ca_in, ca_out = totals(cv_agentic)

    b_cost  = cost(b_in,  b_out)
    cp_cost = cost(cp_in, cp_out)
    ca_cost = cost(ca_in, ca_out)

    W = 82
    print("\n" + "═"*W)
    print(" TOKEN & COST COMPARISON  (3 questions)")
    print("═"*W)
    print(f"  Handle: {cv_meta['cid']}  |  {cv_meta['orig']} tokens → {cv_meta['ratio']}x compressed\n")

    print(f"{'Metric':<26} {'Baseline':>13} {'CV pre-fetch':>13} {'CV agentic':>13}  {'savings (agent)':>15}")
    print("─"*W)
    print(f"  note: baseline is a single call; CV makes one call per question")
    print(f"{'Input tokens':<26} {b_in:>13,} {cp_in:>13,} {ca_in:>13,}  {pct(b_in, ca_in):>15}")
    print(f"{'Output tokens':<26} {b_out:>13,} {cp_out:>13,} {ca_out:>13,}")
    print(f"{'Total cost (Opus 4.8)':<26} ${b_cost:>12.4f} ${cp_cost:>12.4f} ${ca_cost:>12.4f}  {pct(b_cost, ca_cost):>15}")

    print("\n" + "═"*W)
    print(" ANSWERS & QUALITY JUDGMENT")
    print("═"*W)

    for i, q in enumerate(QUESTIONS):
        b  = baseline[i];    cp = cv_prefetch[i];  ca = cv_agentic[i]
        vp = verdicts_pre[i]; va = verdicts_agent[i]
        print(f"\n{'─'*W}")
        print(f"Q: {q['q']}")
        print(f"\n[BASELINE — {b['tok_in']:,} in]\n{b['answer']}")
        print(f"\n[CV PRE-FETCH — {cp['tok_in']:,} in]\n{cp['answer']}")
        print(f"\n[CV AGENTIC — {ca['tok_in']:,} in]\n{ca['answer']}")
        print(f"\n[JUDGE: baseline vs pre-fetch]\n{vp}")
        print(f"\n[JUDGE: baseline vs agentic]\n{va}")

    print("\n" + "═"*W)
    print(f"  Input savings (agentic vs baseline) : {pct(b_in, ca_in)}  ({b_in:,} → {ca_in:,})")
    print(f"  Cost savings  (agentic vs baseline) : {pct(b_cost, ca_cost)}  (${b_cost:.4f} → ${ca_cost:.4f})")
    print("═"*W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global QUESTIONS, FACT_QUESTIONS

    cfg = _load_config()
    if cfg:
        QUESTIONS      = cfg.get("questions", QUESTIONS)
        FACT_QUESTIONS = cfg.get("fact_questions", FACT_QUESTIONS)

    pdf   = (_ARGS.pdf or (cfg and cfg.get("pdf")) or PDF_PATH)
    label = (cfg and cfg.get("label")) or "doc"

    if not os.path.isfile(pdf):
        sys.exit(f"File not found: {pdf}")

    client = _make_client()
    print(f"Provider : {PROVIDER}  model : {MODEL}")
    if API_BASE:
        print(f"API base : {API_BASE}")

    print(f"\nLoading {pdf}…")
    doc_text = load_path(pdf)
    doc_tokens = estimate_tokens(doc_text)
    print(f"  {len(doc_text):,} chars  ~{doc_tokens:,} tokens\n")

    n = len(QUESTIONS)
    est_base_cost = doc_tokens * n * PRICE_IN / 1_000_000
    print(f"  Estimated baseline cost (input only): ~${est_base_cost:.3f}")
    print(f"  (ContextVault will be significantly cheaper)\n")

    print()
    cv_prefetch, cv_meta = run_cv(client, doc_text, label)
    cid = cv_meta["cid"]
    print()
    cv_agentic = run_cv_agentic(client, cid)
    print()
    baseline = run_baseline(client, doc_text)

    print("\n── Judging quality ──")
    verdicts_pre = []
    verdicts_agent = []
    for b, cp, ca in zip(baseline, cv_prefetch, cv_agentic):
        print(f"  [{b['id']}] pre-fetch…", end=" ", flush=True)
        verdicts_pre.append(judge(client, b["question"], b["answer"], cp["answer"]))
        print("done  agentic…", end=" ", flush=True)
        verdicts_agent.append(judge(client, b["question"], b["answer"], ca["answer"]))
        print("done")

    report(baseline, cv_prefetch, cv_agentic, cv_meta, verdicts_pre, verdicts_agent)

    print()
    fact_rows = run_fact_test(client, doc_text, cid)
    report_facts(fact_rows)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_compare_results.json")
    with open(out, "w") as fh:
        json.dump(
            {"pdf": pdf, "model": MODEL,
             "baseline": baseline, "cv_prefetch": cv_prefetch, "cv_agentic": cv_agentic,
             "cv_meta": cv_meta, "verdicts_pre": verdicts_pre, "verdicts_agent": verdicts_agent,
             "fact_test": fact_rows},
            fh, indent=2, ensure_ascii=False,
        )
    print(f"\n  Full results → {out}")


if __name__ == "__main__":
    main()
