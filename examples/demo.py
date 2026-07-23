"""Runnable demo of the SaveContext workflow (no MCP client required).

Drives the VaultService directly the same way an LLM would drive the MCP tools,
following the contract example from the README:

    ingest -> brief -> expand(facts) -> quote -> audit -> diff

Run:  python examples/demo.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext.service import VaultService
from savecontext.storage import Store

CONTRACT = """MASTER SERVICES AGREEMENT

This Agreement is entered into on January 15, 2024 between Acme Corporation
("Provider") and Globex Inc ("Customer").

1. Fees
The Customer shall pay $50,000 per month. Late payments accrue interest at 1.5%
per month. Payment is due within 30 days of each invoice.

2. Liability
Provider's total aggregate liability shall not exceed $500,000. Provider shall
indemnify Customer against third-party intellectual property claims. The Customer
may not assign this Agreement without Provider's prior written consent.

3. Term and Termination
This Agreement is effective from 2024-02-01 and continues for 24 months. Either
party may terminate for convenience with 90 days written notice. Confidential
information must not be disclosed to any third party.

Contact: legal@acme.example.com or https://acme.example.com/legal
"""


def show(title, obj):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2)[:1400])


def main():
    svc = VaultService(store=Store(db_path=":memory:"))

    ing = svc.ingest(CONTRACT, label="acme-contract", source_type="auto")
    print(
        f"ingest: {ing['context_id']}  "
        f"{ing['token_estimate_original']} tokens -> brief "
        f"{ing['token_estimate_brief']} tokens "
        f"({ing['compression_ratio']}x), source_type={ing['source_type']}"
    )
    print("\nsemantic_brief:\n" + ing["semantic_brief"])
    show("block_map", ing["block_map"])
    show("protected_atoms_summary", ing["protected_atoms_summary"])

    cid = ing["context_id"]

    show("brief(task='liability risks')", svc.brief(cid, task="liability risks"))
    show("expand('liability', fidelity='facts')",
         svc.expand(cid, selector="liability", fidelity="facts"))
    show("quote(search_query='indemnify')",
         svc.quote(cid, search_query="indemnify"))
    show("audit", svc.audit(cid))

    changed = CONTRACT.replace("$500,000", "$1,000,000").replace(
        "90 days", "30 days"
    )
    show("diff (liability cap raised, notice shortened)", svc.diff(cid, changed))

    svc.close()


if __name__ == "__main__":
    main()
