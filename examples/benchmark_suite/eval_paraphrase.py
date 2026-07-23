"""Paraphrase-robustness eval for lookup retrieval.

Three phrasing tiers per fact — lexical (shares clause vocabulary), mixed,
and paraphrase (no overlap with the clause heading or its key terms) — run
against the generated benchmark contracts. Reports, per tier:

  hit@1     the governing block is the top-ranked match
  answer@1  the ground-truth answer value is present in the top match's
            facts or verbatim sentence (what an agent actually needs)
  payload   mean lookup response tokens per 6-query batch

Usage:  python eval_paraphrase.py /tmp/cv-bench [--db PATH]
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from savecontext.service import VaultService  # noqa: E402
from savecontext.tokenizer import estimate_tokens  # noqa: E402

# fact key -> (governing heading fragment, answer substring, [lexical, mixed, paraphrase])
CASES = {
    "liability_cap": (
        "limitation of liability", "$750,000",
        [
            "total aggregate limitation of liability cap",
            "maximum liability dollar amount",
            "how much money is at risk at most if things go wrong",
        ],
    ),
    "termination_notice_days": (
        "termination for convenience", "60",
        [
            "days written notice to terminate for convenience",
            "notice period to end the agreement early",
            "how long in advance must we say we are walking away",
        ],
    ),
    "confidentiality_years": (
        "confidentiality term", "7",
        [
            "years confidential information must be protected",
            "duration of the secrecy obligation",
            "how long do we have to keep quiet about private material",
        ],
    ),
    "monthly_fee": (
        "recurring fees", "$25,000",
        [
            "recurring monthly service fee",
            "what does the service cost each month",
            "regular amount charged on an ongoing basis",
        ],
    ),
    "governing_law": (
        "governing law", "Delaware",
        [
            "governing law state",
            "which state's courts and rules apply",
            "under whose legal system are disputes decided",
        ],
    ),
    "initial_term_months": (
        "initial term", "36",
        [
            "initial term of the agreement in months",
            "how long does the contract run at first",
            "length of the starting commitment period",
        ],
    ),
}

TIERS = ["lexical", "mixed", "paraphrase"]


def run(doc_dir: str, db_path: str, sizes=(15065, 45053, 150065)) -> dict:
    svc = VaultService(db_path=db_path)
    stats = {t: {"hit": 0, "ans": 0, "n": 0} for t in TIERS}
    payload_tokens = []
    for size in sizes:
        cid = f"ctx://contract-{size}@v1"
        for ti, tier in enumerate(TIERS):
            queries = [CASES[k][2][ti] for k in CASES]
            res = svc.lookup(cid, queries)
            payload_tokens.append(estimate_tokens(json.dumps(res["results"])))
            for key, r in zip(CASES, res["results"]):
                heading_frag, answer, _ = CASES[key]
                stats[tier]["n"] += 1
                if not r["matches"]:
                    continue
                top = r["matches"][0]
                if heading_frag in top["heading"].lower():
                    stats[tier]["hit"] += 1
                blob = top["verbatim"]["quote"] + " " + " ".join(
                    a["value"] for a in top["facts"]
                )
                if answer in blob:
                    stats[tier]["ans"] += 1
    svc.close()
    return {
        "tiers": {
            t: {
                "hit@1": round(s["hit"] / s["n"], 3),
                "answer@1": round(s["ans"] / s["n"], 3),
                "n": s["n"],
            }
            for t, s in stats.items()
        },
        "mean_batch_payload_tokens": round(sum(payload_tokens) / len(payload_tokens)),
    }


if __name__ == "__main__":
    doc_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cv-bench"
    db = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else os.path.join(doc_dir, "quality.db")
    print(json.dumps(run(doc_dir, db), indent=2))
