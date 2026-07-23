"""Large-document demo: generate a long synthetic contract and run the workflow.

Builds a realistic multi-section master services agreement (~tens of thousands
of tokens), ingests it, and shows the real compression ratio plus task brief,
lazy expansion, exact quote, and audit.

Run:  python examples/big_demo.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext.service import VaultService
from savecontext.storage import Store


def build_big_contract(num_sections: int = 60) -> str:
    """Generate a long, varied contract with real atoms scattered throughout."""
    header = (
        "MASTER SERVICES AND LICENSING AGREEMENT\n\n"
        "This Master Services and Licensing Agreement (the \"Agreement\") is "
        "entered into and made effective as of January 15, 2024 (the "
        "\"Effective Date\") by and between Acme Corporation, a Delaware "
        "corporation (\"Provider\"), and Globex International Inc., a New York "
        "corporation (\"Customer\"). Provider and Customer are each a \"Party\" "
        "and collectively the \"Parties\".\n\n"
        "Contact for notices: legal@acme.example.com and counsel@globex.example.com. "
        "Portal: https://portal.acme.example.com/contracts/2024.\n\n"
    )

    topics = [
        ("Fees and Payment",
         "Customer shall pay Provider a recurring fee of ${fee:,} per month, "
         "due within {days} days of the invoice date. Late payments shall accrue "
         "interest at {rate}% per month. Provider may suspend service if amounts "
         "remain unpaid for more than {grace} days."),
        ("Limitation of Liability",
         "Provider's total aggregate liability under this Agreement shall not "
         "exceed ${cap:,}. In no event shall either Party be liable for indirect, "
         "incidental, or consequential damages. This limitation must not be "
         "construed to exclude liability for gross negligence."),
        ("Indemnification",
         "Provider shall indemnify and hold harmless Customer against any "
         "third-party claims arising on or after {date} up to a maximum of "
         "${cap:,}. Customer shall notify Provider within {days} days of any "
         "claim. The indemnified Party may not settle any claim without consent."),
        ("Confidentiality",
         "Each Party shall protect the other's Confidential Information and must "
         "not disclose it to any third party for a period of {years} years. "
         "Confidential Information does not include information that is publicly "
         "available. Breach may result in damages of up to ${cap:,}."),
        ("Term and Termination",
         "This Section is effective from {date} and continues for {years} years. "
         "Either Party may terminate for convenience with {days} days written "
         "notice. Upon termination Customer shall pay all outstanding amounts "
         "within {grace} days."),
        ("Service Levels",
         "Provider warrants {rate}% monthly uptime measured at the load balancer. "
         "If uptime falls below {rate}%, Customer is entitled to a service credit "
         "of ${fee:,}. Credits must be requested within {days} days."),
        ("Data Protection",
         "Provider shall process personal data only as instructed and must not "
         "transfer data outside the approved region after {date}. A breach must "
         "be reported within {grace} hours. Penalties may reach ${cap:,}."),
        ("Intellectual Property",
         "All pre-existing IP remains with its owner. Deliverables created after "
         "{date} are licensed, not assigned. Customer may not reverse engineer "
         "the software identified as core_engine.process() in module "
         "src/engine/core_engine.py."),
    ]

    body = [header]
    for i in range(num_sections):
        title_base, template = topics[i % len(topics)]
        section_no = i + 1
        params = {
            "fee": 25000 + (i * 1500),
            "cap": 250000 + (i * 50000),
            "rate": round(99.0 + (i % 9) * 0.1, 1),
            "days": 15 + (i % 4) * 15,
            "grace": 30 + (i % 3) * 15,
            "years": 2 + (i % 4),
            "date": f"202{4 + (i % 5)}-{(i % 12) + 1:02d}-01",
        }
        para = template.format(**params)
        # Add some filler prose so sections read like real clauses.
        filler = (
            f" The parties acknowledge that the provisions of this Section "
            f"{section_no} are material to the bargain. Each obligation herein "
            f"shall survive termination to the extent necessary. Nothing in this "
            f"Section limits the rights expressly granted elsewhere in the "
            f"Agreement. Capitalized terms not defined here have the meaning "
            f"given in Section 1."
        )
        body.append(f"{section_no}. {title_base}\n{para}{filler}\n")

    return "\n".join(body)


def main():
    num_sections = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    doc = build_big_contract(num_sections=num_sections)
    svc = VaultService(store=Store(db_path=":memory:"))

    print(f"Source document: {len(doc):,} characters")

    ing = svc.ingest(doc, label="bigco-msa", source_type="auto")
    print(
        f"\nINGEST  {ing['context_id']}  (source_type={ing['source_type']})\n"
        f"  original : {ing['token_estimate_original']:,} tokens\n"
        f"  brief    : {ing['token_estimate_brief']:,} tokens\n"
        f"  ratio    : {ing['compression_ratio']}x compression\n"
        f"  blocks   : {len(ing['block_map'])}\n"
        f"  atom types: "
        + ", ".join(
            f"{t}={d['unique']}u/{d['count']}x"
            for t, d in ing["protected_atoms_summary"].items()
        )
    )

    cid = ing["context_id"]

    print("\nSEMANTIC BRIEF (first 600 chars):")
    print("  " + ing["semantic_brief"][:600].replace("\n", "\n  "))

    # Task brief — far smaller than the source.
    br = svc.brief(cid, task="liability cap and indemnification limits", max_tokens=400)
    print(
        f"\nBRIEF(task='liability cap and indemnification limits')\n"
        f"  brief tokens: {br['token_estimate']}\n"
        f"  relevant blocks: "
        + ", ".join(b["heading"] for b in br["relevant_blocks"][:5])
    )
    print(
        "  relevant atoms (sample): "
        + ", ".join(f"[{a['type']}]{a['value']}" for a in br["relevant_atoms"][:6])
    )

    # Lazy expand just the liability facts.
    ex = svc.expand(cid, selector="Limitation of Liability", fidelity="facts")
    print(
        f"\nEXPAND('Limitation of Liability', fidelity='facts')\n"
        f"  expanded tokens: {ex['token_estimate']} "
        f"(vs {ing['token_estimate_original']:,} for full source)\n"
        f"  matched blocks: {len(ex['source_refs'])}"
    )

    # Exact quote.
    q = svc.quote(cid, search_query="shall not exceed")
    print(
        f"\nQUOTE(search_query='shall not exceed')\n"
        f"  exact: \"{q['exact_source_quote']}\"  @ {q['source_ref']['char_range']}"
    )

    aud = svc.audit(cid)
    print(
        f"\nAUDIT\n"
        f"  compression_ratio: {aud['compression_ratio']}x\n"
        f"  preservation: {json.dumps(aud['estimated_preservation'])}\n"
        f"  warnings: {aud['warnings'] or '(none)'}"
    )

    # Token math: what does the model actually pay to work with this doc?
    workflow_tokens = (
        ing["token_estimate_brief"] + br["token_estimate"] + ex["token_estimate"]
    )
    print(
        f"\nNET EFFECT\n"
        f"  Naive approach: re-read {ing['token_estimate_original']:,} tokens "
        f"every turn.\n"
        f"  SaveContext : ~{workflow_tokens:,} tokens for brief + task brief + "
        f"targeted expand\n"
        f"  => ~{round(ing['token_estimate_original'] / max(1, workflow_tokens), 1)}x "
        f"fewer tokens to answer a liability question."
    )
    svc.close()


if __name__ == "__main__":
    main()
