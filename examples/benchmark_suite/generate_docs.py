"""Generate 5 benchmark contracts of increasing size with known answers.

Each document embeds the SAME six ground-truth facts (with nearby distractors)
at varied positions, padded with realistic filler clauses to hit a target token
size. This lets us measure, at each size: token cost, $ cost, and answer
quality — for the baseline (whole doc in context) vs SaveContext.

Run:  python generate_docs.py [output_dir]
Writes contract_<tokens>.txt files + ground_truth.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from savecontext.tokenizer import estimate_tokens  # noqa: E402

# Five target sizes (approx tokens). The largest is bounded so the baseline
# agent run (whole doc in context) stays affordable.
TARGET_TOKENS = [1500, 5000, 15000, 35000, 45000, 65000, 100000, 150000]

# The six ground-truth facts, each with distractors placed nearby so a careless
# reader can pick the wrong value. The agent must cite the correct clause.
KEY_CLAUSES = [
    ("Limitation of Liability",
     "Notwithstanding anything to the contrary, Provider's total aggregate "
     "liability under this Agreement shall not exceed $750,000. For clarity, the "
     "$250,000 indemnification sub-limit in the Indemnification section and the "
     "$1,000,000 insurance coverage requirement are separate and do not raise this cap.",
     "liability_cap", "$750,000"),
    ("Termination for Convenience",
     "Either party may terminate this Agreement for convenience upon 60 days "
     "prior written notice. This is distinct from the 30 day cure period for "
     "material breach and the 45 day window to settle outstanding invoices after termination.",
     "termination_notice_days", "60"),
    ("Confidentiality Term",
     "Each party shall protect the other's Confidential Information and shall not "
     "disclose it for a period of 7 years following termination. Note that "
     "operational telemetry data is retained for only 3 years, which does not "
     "govern the confidentiality obligation.",
     "confidentiality_years", "7"),
    ("Recurring Fees",
     "The Customer shall pay a recurring service fee of $25,000 per month. A "
     "one-time onboarding processing fee of $1,200 applies in month one, and late "
     "payments accrue interest of 2.5% per month; neither alters the recurring fee.",
     "monthly_fee", "$25,000"),
    ("Governing Law",
     "This Agreement shall be governed by and construed in accordance with the "
     "laws of the State of Delaware, notwithstanding that Customer is incorporated "
     "in New York and Provider maintains offices in California.",
     "governing_law", "Delaware"),
    ("Initial Term",
     "The initial term of this Agreement is 36 months from the Effective Date. "
     "The renewal term, if exercised, is 24 months, and the pilot phase referenced "
     "in Schedule A was 6 months; the initial term is 36 months.",
     "initial_term_months", "36"),
]

_FILLER = (
    "{n}. General Provision {n}\n"
    "The parties shall cooperate in good faith to implement the intent of this "
    "Agreement. Each party represents that it has full authority to enter into "
    "this Agreement and that doing so does not conflict with any other obligation. "
    "Routine administrative correspondence shall be directed to the operations "
    "contact of record. This Section {n} is administrative and contains no "
    "financial, temporal, or liability terms material to the parties' core obligations."
)

_HEADER = (
    "MASTER SERVICES AGREEMENT\n\n"
    "This Master Services Agreement (the \"Agreement\") is entered into as of "
    "March 3, 2024 between Initech LLC (\"Provider\") and Hooli Inc (\"Customer\").\n"
)


def build_doc(target_tokens: int):
    # Place the 6 key clauses at ~evenly spaced fractions of the document.
    # First estimate how many filler clauses we need.
    filler_tokens = estimate_tokens(_FILLER.format(n=1))
    approx_fillers = max(6, target_tokens // max(1, filler_tokens))

    key_positions = {
        int(approx_fillers * frac): clause
        for frac, clause in zip(
            (0.10, 0.25, 0.42, 0.58, 0.74, 0.88), KEY_CLAUSES
        )
    }

    parts = [_HEADER]
    section_no = 1
    placed = 0
    i = 0
    while placed < len(KEY_CLAUSES) or estimate_tokens("\n\n".join(parts)) < target_tokens:
        if i in key_positions:
            heading, body, _, _ = key_positions[i]
            parts.append(f"{section_no}. {heading}\n{body}")
            placed += 1
        else:
            parts.append(_FILLER.format(n=section_no))
        section_no += 1
        i += 1
        # Safety: once all keys placed and size reached, stop.
        if placed >= len(KEY_CLAUSES) and estimate_tokens("\n\n".join(parts)) >= target_tokens:
            break
    return "\n\n".join(parts)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    for target in TARGET_TOKENS:
        doc = build_doc(target)
        toks = estimate_tokens(doc)
        path = os.path.join(out_dir, f"contract_{toks}.txt")
        with open(path, "w") as fh:
            fh.write(doc)
        manifest.append({"target": target, "tokens": toks, "path": path})
        print(f"wrote {path}  ({toks} tokens, {len(doc)} chars)")

    ground_truth = {
        "questions": [
            {"key": k, "heading": h, "answer": a, "question": _q(h)}
            for (h, _b, k, a) in KEY_CLAUSES
        ],
        "distractors_present": True,
        "docs": manifest,
    }
    gt_path = os.path.join(out_dir, "ground_truth.json")
    with open(gt_path, "w") as fh:
        json.dump(ground_truth, fh, indent=2)
    print(f"wrote {gt_path}")


def _q(heading: str) -> str:
    return {
        "Limitation of Liability": "What is the total aggregate limitation of liability cap (dollar amount)?",
        "Termination for Convenience": "How many days written notice are required to terminate for convenience?",
        "Confidentiality Term": "For how many years must Confidential Information be protected?",
        "Recurring Fees": "What is the recurring monthly service fee (dollar amount)?",
        "Governing Law": "Which US state's law governs the Agreement?",
        "Initial Term": "What is the initial term of the Agreement, in months?",
    }[heading]


if __name__ == "__main__":
    main()
