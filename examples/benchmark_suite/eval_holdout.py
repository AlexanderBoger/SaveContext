"""Held-out evaluation: documents and questions the code was never tuned on.

Two domains beyond the contract benchmark:
  * RFC 9110 (real technical prose, ~117k tokens) — fetched on first run
  * a synthetic production incident log (~118k tokens, no headings, no
    sentence punctuation) — generated deterministically (seed 42)

Per domain: 6 answerable questions (graded answer-in-top2-payload) and 3
unanswerable ones (graded: lookup must flag weak/empty, the abstention
signal). This suite caught the 1.4M-token verbatim blowup on logs and the
false-confidence miscalibration on large prose docs.

Usage:  python eval_holdout.py [workdir]     (default /tmp/cv-holdout)
"""

import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from savecontext.service import VaultService  # noqa: E402
from savecontext.tokenizer import estimate_tokens  # noqa: E402

RFC_URL = "https://www.rfc-editor.org/rfc/rfc9110.txt"

QUESTIONS = {
    "rfc9110": {
        "answerable": [
            ("Which status code means the client should pay before proceeding?", ["402"]),
            ("What TCP port does a plain http URL use when none is given?", ["80"]),
            ("How many octets should implementations at minimum support for URI length?", ["8000"]),
            ("Which redirect-era status code is now unused and reserved?", ["306"]),
            ("The Retry-After field can be either of which two formats?", ["HTTP-date", "delay-seconds"]),
            ("When a resource is permanently gone, which status is preferred over 404?", ["410"]),
        ],
        "unanswerable": [
            "What minimum TLS version does HTTP require?",
            "What is the mandated maximum number of concurrent connections per client?",
            "Which cipher suites must servers support?",
        ],
    },
    "incident": {
        "answerable": [
            ("After how many consecutive failures did the payments-router circuit breaker open?", ["47"]),
            ("What was the memory limit of the pod that got OOMKilled?", ["2048"]),
            ("What did the cache hit ratio drop to in eu-west-1?", ["61"]),
            ("How many invoices were left unprocessed when the batch aborted?", ["342"]),
            ("How long did INCIDENT-4471 take to resolve?", ["96"]),
            ("What was the root cause of INCIDENT-4471?", ["TLS cert"]),
        ],
        "unanswerable": [
            "What database engine does billing-core run on?",
            "How many customers were impacted by INCIDENT-4471?",
            "Who was the on-call engineer during the incident?",
        ],
    },
}


def build_incident_log(path: str) -> None:
    random.seed(42)
    services = ["auth-gateway", "billing-core", "search-indexer", "mail-relay",
                "cdn-edge", "user-profile", "payments-router", "audit-log"]
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    lines, t = [], 0

    def stamp(t):
        h, rem = divmod(t, 3600)
        m, sec = divmod(rem, 60)
        return f"2026-03-14T{h:02d}:{m:02d}:{sec:02d}Z"

    for _ in range(4000):
        t += random.randint(1, 25)
        svc = random.choice(services)
        lvl = random.choices(levels, weights=[70, 15, 5, 10])[0]
        lines.append(
            f"{stamp(t)} {lvl} {svc} req_id={random.randint(100000, 999999)} "
            f"latency_ms={random.randint(3, 900)} status={random.choice([200, 200, 200, 301, 404, 500])}"
        )
    # Ground truth with nearby distractors:
    lines[850] = "2026-03-14T04:12:33Z ERROR payments-router circuit breaker OPEN after 47 consecutive failures upstream=stripe-proxy"
    lines[852] = "2026-03-14T04:13:02Z WARN payments-router retry budget exhausted, 12 consecutive failures on fallback path (not the trigger)"
    lines[1400] = "2026-03-14T07:02:10Z ERROR auth-gateway OOMKilled pod=auth-gateway-7d9f memory_limit=2048Mi"
    lines[1402] = "2026-03-14T07:02:41Z INFO auth-gateway sibling pod auth-gateway-3c1a healthy memory_limit=4096Mi (canary, not killed)"
    lines[2600] = "2026-03-14T13:44:55Z WARN cdn-edge cache hit ratio dropped to 61% (baseline 94%) region=eu-west-1"
    lines[2602] = "2026-03-14T13:45:20Z INFO cdn-edge us-east-1 cache hit ratio steady at 93%"
    lines[3300] = "2026-03-14T17:20:08Z ERROR billing-core invoice batch 88123 aborted: 342 invoices unprocessed, rollback complete"
    lines[3302] = "2026-03-14T17:21:00Z INFO billing-core nightly batch 88124 scheduled, expected 5100 invoices"
    lines[3700] = "2026-03-14T19:55:12Z INFO incident-bot INCIDENT-4471 resolved after 96 minutes, root cause: expired TLS cert on stripe-proxy"
    with open(path, "w") as f:
        f.write("\n".join(lines))


def run(workdir: str) -> dict:
    os.makedirs(workdir, exist_ok=True)
    rfc_path = os.path.join(workdir, "rfc9110.txt")
    log_path = os.path.join(workdir, "incident.log")
    if not os.path.exists(rfc_path):
        urllib.request.urlretrieve(RFC_URL, rfc_path)
    if not os.path.exists(log_path):
        build_incident_log(log_path)

    svc = VaultService(db_path=":memory:")
    handles = {
        "rfc9110": svc.ingest(open(rfc_path).read(), label="rfc9110")["context_id"],
        "incident": svc.ingest(open(log_path).read(), label="incident")["context_id"],
    }

    report = {}
    for name, qs in QUESTIONS.items():
        cid = handles[name]
        res = svc.lookup(cid, [q for q, _ in qs["answerable"]])
        ans = sum(
            1
            for (q, needles), r in zip(qs["answerable"], res["results"])
            if all(n in json.dumps(r["matches"][:2]) for n in needles)
        )
        weak_ans = sum(1 for r in res["results"] if r["confidence"] == "weak")
        ures = svc.lookup(cid, qs["unanswerable"])
        abstain = sum(
            1 for r in ures["results"] if r["confidence"] == "weak" or not r["matches"]
        )
        report[name] = {
            "answer_in_top2": f"{ans}/{len(qs['answerable'])}",
            "abstention_flagged": f"{abstain}/{len(qs['unanswerable'])}",
            "weak_flag_on_answerable": f"{weak_ans}/{len(qs['answerable'])}",
            "batch_payload_tokens": estimate_tokens(json.dumps(res["results"])),
        }
    svc.close()
    return report


if __name__ == "__main__":
    workdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cv-holdout"
    print(json.dumps(run(workdir), indent=2))
