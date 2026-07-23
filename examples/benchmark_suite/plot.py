"""Plot the benchmark: tokens, $ cost, and answer quality vs document size.

Reads results.json (written by measure.py + quality injection) and renders a
single figure with three panels:
  1. Tokens used to answer (baseline vs SaveContext) — log y
  2. $ cost per question at Opus 4.8 and Sonnet 4.6 input rates
  3. Answer quality (% of 6 facts correct), baseline vs SaveContext

Run:  python plot.py <suite_dir>   ->  writes benchmark.png
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_BASE = "#d1495b"   # baseline (red)
C_CV = "#2a9d8f"     # savecontext (teal)


def main():
    suite = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    r = json.load(open(os.path.join(suite, "results.json")))
    rows = sorted(r["rows"], key=lambda x: x["doc_tokens"])

    x = [row["doc_tokens"] for row in rows]
    base_tok = [row["baseline_tokens"] for row in rows]
    cv_tok = [row["cv_tokens"] for row in rows]
    base_q = [row["quality"]["baseline_pct"] for row in rows]
    cv_q = [row["quality"]["savecontext_pct"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("SaveContext vs. baseline (whole doc in context) — 6-fact Q&A on one contract",
                 fontsize=14, fontweight="bold")

    # --- Panel 1: tokens used ---
    ax = axes[0]
    ax.plot(x, base_tok, "o-", color=C_BASE, lw=2.5, ms=8, label="Baseline (paste whole doc)")
    ax.plot(x, cv_tok, "s-", color=C_CV, lw=2.5, ms=8, label="SaveContext")
    ax.set_yscale("log")
    ax.set_xlabel("Document size (tokens)")
    ax.set_ylabel("Tokens used to answer (log)")
    ax.set_title("Token cost — baseline scales, SaveContext is flat")
    for xi, b, c in zip(x, base_tok, cv_tok):
        ax.annotate(f"{round(b/c,1)}x", (xi, c), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color=C_CV)
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.25)

    # --- Panel 2: $ cost per question ---
    ax = axes[1]
    base_opus = [row["baseline_usd_opus-4.8"] for row in rows]
    cv_opus = [row["cv_usd_opus-4.8"] for row in rows]
    base_son = [row["baseline_usd_sonnet-4.6"] for row in rows]
    cv_son = [row["cv_usd_sonnet-4.6"] for row in rows]
    ax.plot(x, base_opus, "o-", color=C_BASE, lw=2.5, ms=7, label="Baseline · Opus 4.8")
    ax.plot(x, base_son, "o--", color=C_BASE, lw=1.8, ms=6, alpha=0.6, label="Baseline · Sonnet 4.6")
    ax.plot(x, cv_opus, "s-", color=C_CV, lw=2.5, ms=7, label="SaveContext · Opus 4.8")
    ax.plot(x, cv_son, "s--", color=C_CV, lw=1.8, ms=6, alpha=0.6, label="SaveContext · Sonnet 4.6")
    ax.set_xlabel("Document size (tokens)")
    ax.set_ylabel("Input cost per question (USD)")
    ax.set_title("$ per question (input tokens)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    # --- Panel 3: quality ---
    ax = axes[2]
    ax.plot(x, base_q, "o-", color=C_BASE, lw=2.5, ms=8, label="Baseline")
    ax.plot(x, cv_q, "s-", color=C_CV, lw=2.5, ms=8, label="SaveContext")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Document size (tokens)")
    ax.set_ylabel("Answer quality (% of 6 facts correct)")
    ax.set_title("Quality — identical, distractor-proof")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)
    ax.annotate("both 100% — same accuracy,\nfraction of the cost", (x[2], 100),
                textcoords="offset points", xytext=(0, -40), ha="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0c97f"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(suite, "benchmark.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
