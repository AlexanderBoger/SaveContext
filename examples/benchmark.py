"""Benchmark SaveContext: compression, speed, retrieval, and integrity.

Produces publishable numbers across document sizes — what compression you get,
how fast ingest/brief are, whether targeted retrieval finds the right block,
and whether every atom still reconstructs verbatim.

Run:  python examples/benchmark.py
      python examples/benchmark.py 50 400 1500   # custom section counts
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext.service import VaultService
from savecontext.storage import Store

# Reuse the realistic contract generator from the big demo.
sys.path.insert(0, os.path.dirname(__file__))
from big_demo import build_big_contract  # noqa: E402


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def run_one(sections: int):
    doc = build_big_contract(num_sections=sections)
    svc = VaultService(store=Store(db_path=":memory:"))

    ing, ingest_ms = _timed(lambda: svc.ingest(doc, label=f"bench-{sections}", source_type="contract"))
    cid = ing["context_id"]

    # Retrieval quality: does the liability question surface a liability block?
    br, brief_ms = _timed(lambda: svc.brief(cid, task="limitation of liability cap"))
    headings = " ".join(b["heading"] for b in br["relevant_blocks"]).lower()
    retrieval_hit = "liability" in headings

    # Integrity: every atom must reconstruct verbatim.
    aud = svc.audit(cid)
    integrity_ok = aud["verbatim_integrity"]["ok"]
    atoms = aud["verbatim_integrity"]["atoms_checked"]

    svc.close()
    return {
        "sections": sections,
        "chars": len(doc),
        "orig_tokens": ing["token_estimate_original"],
        "brief_tokens": ing["token_estimate_brief"],
        "ratio": ing["compression_ratio"],
        "blocks": len(ing["block_map"]),
        "atoms": atoms,
        "ingest_ms": round(ingest_ms, 1),
        "brief_ms": round(brief_ms, 1),
        "retrieval_hit": retrieval_hit,
        "integrity_ok": integrity_ok,
    }


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [20, 60, 200, 600]
    rows = [run_one(n) for n in sizes]

    cols = [
        ("orig_tokens", "orig tok", "{:>9,}"),
        ("brief_tokens", "brief tok", "{:>9,}"),
        ("ratio", "ratio", "{:>7}x"),
        ("blocks", "blocks", "{:>6}"),
        ("atoms", "atoms", "{:>6}"),
        ("ingest_ms", "ingest ms", "{:>9}"),
        ("brief_ms", "brief ms", "{:>8}"),
        ("retrieval_hit", "retr ok", "{:>7}"),
        ("integrity_ok", "intact", "{:>6}"),
    ]
    header = "  ".join(f"{label:>9}" for _key, label, _fmt in cols)
    print("SaveContext benchmark\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(
            (fmt.format(r[key]) if not isinstance(r[key], bool) else f"{str(r[key]):>9}")
            for key, _label, fmt in cols
        )
        print(line)

    print(
        "\nNote: brief stays roughly constant as the source grows, so the "
        "compression ratio rises with document size. Integrity must always be "
        "True — any False is a loss-aware guarantee violation."
    )


if __name__ == "__main__":
    main()
