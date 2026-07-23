"""Measure token cost and $ cost: baseline (whole doc) vs SaveContext.

Deterministic — no model calls. For each generated contract:
  - baseline_tokens = the whole document the model must read to answer
  - cv_tokens       = the SaveContext workflow payload (compact ingest +
                      task brief + targeted expands for the 6 facts)
Then price both at Opus 4.8 and Sonnet 4.6 INPUT rates.

Quality % is filled in separately from real agent runs (see run_agents notes
in README); this script writes results.json with quality left null, and a
results.md table. plot.py reads results.json.

Run:  python measure.py <suite_dir>
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from savecontext.service import VaultService  # noqa: E402
from savecontext.storage import Store  # noqa: E402
from savecontext.tokenizer import estimate_tokens  # noqa: E402

# Per-million-token INPUT prices (USD). Output is small and equal across
# conditions, so the context-cost comparison prices input.
PRICES = {"opus-4.8": 5.00, "sonnet-4.6": 3.00}

COMBINED_TASK = (
    "liability cap; termination for convenience notice period; confidentiality "
    "duration; recurring monthly fee; governing law; initial term length"
)


def _toks(obj) -> int:
    return estimate_tokens(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False))


def measure_doc(path: str, gt: dict) -> dict:
    text = open(path).read()
    doc_tokens = estimate_tokens(text)

    # --- Baseline: whole doc + the six questions in one prompt ---
    questions = " ".join(q["question"] for q in gt["questions"])
    baseline_tokens = doc_tokens + estimate_tokens(questions)

    # --- SaveContext workflow for the same six questions ---
    svc = VaultService(store=Store(db_path=":memory:"))
    ing = svc.ingest(text, label=f"bench-{doc_tokens}", source_type="contract")
    cid = ing["context_id"]
    cv_tokens = _toks(ing)  # compact ingest response (turn 1)

    br = svc.brief(cid, task=COMBINED_TASK, max_tokens=500)
    cv_tokens += _toks(br)  # task brief (turn 2)

    # Targeted expand per fact heading at quotes fidelity (exact, smallest).
    for q in gt["questions"]:
        ex = svc.expand(cid, selector=q["heading"], fidelity="quotes", max_blocks=2)
        cv_tokens += _toks(ex)
    svc.close()

    row = {
        "doc_tokens": doc_tokens,
        "baseline_tokens": baseline_tokens,
        "cv_tokens": cv_tokens,
        "token_reduction_x": round(baseline_tokens / max(1, cv_tokens), 1),
        "quality": {"baseline_pct": None, "savecontext_pct": None},  # filled by agents
    }
    for model, rate in PRICES.items():
        row[f"baseline_usd_{model}"] = round(baseline_tokens / 1_000_000 * rate, 5)
        row[f"cv_usd_{model}"] = round(cv_tokens / 1_000_000 * rate, 5)
    return row


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    gt = json.load(open(os.path.join(suite, "ground_truth.json")))

    rows = []
    for d in sorted(gt["docs"], key=lambda x: x["tokens"]):
        rows.append(measure_doc(d["path"], gt))
        print(f"measured {d['tokens']:>6} tok -> baseline {rows[-1]['baseline_tokens']:>6}"
              f" / cv {rows[-1]['cv_tokens']:>5}  ({rows[-1]['token_reduction_x']}x)")

    results = {"prices_per_1m_input": PRICES, "rows": rows}
    with open(os.path.join(suite, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _write_md(os.path.join(suite, "results.md"), results)
    print(f"\nwrote results.json and results.md in {suite}")


def _write_md(path, results):
    rows = results["rows"]
    lines = ["# SaveContext benchmark — tokens, cost, quality\n",
             "INPUT-token cost to answer 6 fact questions about one contract.\n",
             "| doc tokens | baseline tok | CV tok | reduction | baseline $ (Opus) | CV $ (Opus) | baseline $ (Sonnet) | CV $ (Sonnet) | quality baseline | quality CV |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|"]
    for r in rows:
        qb = r["quality"]["baseline_pct"]
        qc = r["quality"]["savecontext_pct"]
        lines.append(
            f"| {r['doc_tokens']:,} | {r['baseline_tokens']:,} | {r['cv_tokens']:,} | "
            f"{r['token_reduction_x']}x | ${r['baseline_usd_opus-4.8']:.4f} | "
            f"${r['cv_usd_opus-4.8']:.4f} | ${r['baseline_usd_sonnet-4.6']:.4f} | "
            f"${r['cv_usd_sonnet-4.6']:.4f} | "
            f"{'—' if qb is None else str(qb)+'%'} | {'—' if qc is None else str(qc)+'%'} |"
        )
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
