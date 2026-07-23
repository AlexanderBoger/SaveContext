"""BM25 ranking for relevance — zero-dependency, zero-LLM, deterministic.

The MVP's first retrieval pass used substring/keyword overlap, which misses
relevant blocks whenever the query wording differs from the source and ranks
poorly when a term is common. BM25 fixes both: it weights rare query terms
more (IDF) and normalizes for block length, so ``brief`` and ``expand`` pick
genuinely relevant blocks without needing embeddings or a model.

This keeps SaveContext's core advantage — instant, free, offline ingest —
while closing the relevance gap against embedding/graph-based competitors.
An embedding backend can later implement the same ``rank`` interface.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Light stopword list: kept short so BM25's IDF does most of the work.
_STOP = frozenset(
    """a an the of to in on at for and or but is are was were be been being
    this that these those it its as by with from into over under than then
    will shall may can could would should not no nor""".split()
)

_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25Index:
    """In-memory BM25 over a small set of documents (blocks of one context).

    Built fresh per query in practice — cheap for the block counts SaveContext
    handles (hundreds to low thousands), and avoids any persistence coupling.
    """

    def __init__(self, docs: Sequence[Tuple[str, str]]):
        # docs: sequence of (doc_id, text)
        self.doc_ids: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.doc_len: List[int] = []
        df: Counter = Counter()

        for doc_id, text in docs:
            toks = tokenize(text)
            self.doc_ids.append(doc_id)
            self.doc_tokens.append(toks)
            freqs = Counter(toks)
            self.doc_freqs.append(freqs)
            self.doc_len.append(len(toks))
            for term in freqs:
                df[term] += 1

        self.n = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        # BM25 idf with +1 smoothing (always positive).
        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str) -> Dict[str, float]:
        """Return ``{doc_id: bm25_score}`` for every doc (0.0 when no overlap)."""
        q_terms = tokenize(query)
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}
        if not q_terms or self.n == 0:
            return scores
        for i, doc_id in enumerate(self.doc_ids):
            freqs = self.doc_freqs[i]
            dl = self.doc_len[i] or 1
            s = 0.0
            for term in q_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = tf + _K1 * (1 - _B + _B * dl / (self.avgdl or 1))
                s += idf * (tf * (_K1 + 1)) / denom
            scores[doc_id] = s
        return scores

    def match_counts(self, query: str) -> Dict[str, int]:
        """Distinct query terms present per doc — evidence breadth, not depth."""
        q_terms = set(tokenize(query))
        return {
            doc_id: sum(1 for t in q_terms if self.doc_freqs[i].get(t))
            for i, doc_id in enumerate(self.doc_ids)
        }

    def coverage(self, query: str, doc_id: str) -> float:
        """Fraction of the query's content terms present in ``doc_id``.

        Prefix-tolerant (min 4 chars) so morphology doesn't break evidence:
        'resolve' matches 'resolved', 'drop' matches 'dropped'.
        """
        q_terms = set(content_terms(query))
        if not q_terms or doc_id not in self.doc_ids:
            return 0.0
        doc_terms = set(self.doc_freqs[self.doc_ids.index(doc_id)])
        hit = 0
        for t in q_terms:
            if t in doc_terms:
                hit += 1
                continue
            if len(t) >= 4 and any(
                d.startswith(t) or t.startswith(d)
                for d in doc_terms
                if len(d) >= 4
            ):
                hit += 1
        return hit / len(q_terms)

    def rank(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        """Return ``[(doc_id, score), …]`` sorted by score desc (>0 only)."""
        scored = [(d, s) for d, s in self.score(query).items() if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k] if top_k else scored


def build_ranker(docs: Sequence[Tuple[str, str]]):
    """Return a ranker exposing ``score``/``rank`` for ``docs``.

    Uses the optional embedding backend only when explicitly enabled and
    importable; otherwise (the default) returns the free/offline BM25 index.
    Any failure setting up embeddings falls back to BM25 silently — the tool
    must never break because an optional dependency misbehaves.
    """
    try:
        from . import embeddings

        if embeddings.enabled() and embeddings.available():
            return embeddings.HybridRanker(docs)
    except Exception:
        pass
    return BM25Index(docs)


# --- concept expansion & answer-type intents ----------------------------
#
# BM25 is lexical: a paraphrased query ("how much money is at risk") shares
# no terms with the governing clause ("Limitation of Liability") and scores
# zero everywhere. Two cheap, deterministic signals close most of that gap:
#
#  * CONCEPT_GROUPS — small groups of near-synonymous document vocabulary.
#    A query token matching a group recruits the whole group as a weighted
#    expansion query, scored per group (max, not a diluted bag).
#  * INTENT — the *kind* of answer a query wants (money, duration, place…),
#    detected from question vocabulary and mapped to atom types, so blocks
#    holding atoms of that type get a boost and responses can drop atoms
#    the question cannot be about.
#
# Both are domain-generic office/legal/ops vocabulary, not tuned to any one
# document; an embedding backend can later replace them behind rank().

CONCEPT_GROUPS: Dict[str, frozenset] = {
    "payment": frozenset(
        "fee fees cost costs price pricing charge charged charges pay payment "
        "payments paid invoice invoiced billing billed compensation".split()
    ),
    "termination": frozenset(
        "terminate terminates terminated termination end ending cancel "
        "cancellation exit leave leaving walk walking away discontinue".split()
    ),
    "confidentiality": frozenset(
        "confidential confidentiality secret secrecy quiet private "
        "nondisclosure disclose disclosure disclosed".split()
    ),
    "liability": frozenset(
        "liability liable risk exposure damages losses indemnify "
        "indemnification harm".split()
    ),
    "law": frozenset(
        "governing law laws legal courts court jurisdiction disputes dispute "
        "venue litigation".split()
    ),
    "term": frozenset(
        "term duration length period commitment run runs running initial "
        "renewal renew expires expiry".split()
    ),
    "notice": frozenset(
        "notice notify notification notified advance warning inform".split()
    ),
    "warranty": frozenset(
        "warranty warranties warrant guarantee guarantees guaranteed".split()
    ),
}

_QUERY_NOISE = frozenset(
    """how what when where which who whose why much many most more less
    thing things go goes going gone do does doing done did have has having
    had keep keeps keeping say says said get gets getting want wants need
    needs if we us our you your they them their there here about out up
    down off wrong right way ways make makes got take takes took give
    gives given before after should would could means""".split()
)


_INTENT_KEYWORDS: Dict[str, frozenset] = {
    "money": frozenset(
        "money dollar dollars amount cost costs price fee fees charged pay "
        "paid much worth".split()
    ),
    "duration": frozenset(
        "days day months month years year long duration period advance "
        "deadline when until often".split()
    ),
    "place": frozenset(
        "state country where jurisdiction law legal courts venue".split()
    ),
    "obligation": frozenset(
        "must required obligation obligations duty duties responsible "
        "forbidden prohibited allowed".split()
    ),
}

INTENT_ATOM_TYPES: Dict[str, frozenset] = {
    "money": frozenset({"money", "percentage"}),
    "duration": frozenset({"duration", "number", "date"}),
    "place": frozenset({"entity"}),
    "obligation": frozenset({"obligation", "negation"}),
}


def content_terms(query: str) -> List[str]:
    """Query tokens minus conversational noise — the terms that carry meaning."""
    return [t for t in tokenize(query) if t not in _QUERY_NOISE]


def matched_concept_groups(query: str) -> List[frozenset]:
    """Concept groups recruited by any token of ``query``."""
    q = set(tokenize(query))
    return [terms for terms in CONCEPT_GROUPS.values() if q & terms]


def intent_atom_types(query: str) -> frozenset:
    """Atom types the query's answer is likely to be (empty = unknown)."""
    q = set(tokenize(query))
    out: set = set()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if q & keywords:
            out |= INTENT_ATOM_TYPES[intent]
    return frozenset(out)


_SAT_K = 2.0  # BM25 saturation constant: s/(s+k) maps raw scores to [0,1)


def _sat(s: float) -> float:
    """Saturating normalization. Unlike divide-by-max, a weak absolute score
    stays weak — one stray query word matching filler text must not outvote
    a dense concept match elsewhere."""
    return s / (s + _SAT_K) if s > 0 else 0.0


def hybrid_scores(
    index: BM25Index,
    query: str,
    intent_counts: Dict[str, int],
    headings: Dict[str, frozenset] | None = None,
) -> Dict[str, float]:
    """Combine lexical, concept-expansion, intent, and heading signals.

    score = sat(bm25(query)) + 0.45 * max_g sat(bm25(group g))
          + 0.2 * min(1, intent_atoms/3) + 0.3 * heading_coverage

    All signals live in [0, 1]. Saturation (not divide-by-max) keeps weak
    lexical hits weak. Group expansions are scored separately and combined
    with max so one group's vocabulary can't dilute another's.
    heading_coverage is the fraction of a doc's heading tokens covered by
    the query or its recruited concept vocabulary.
    """
    # Conversational words ("how much ... things go wrong") match filler text
    # and poison lexical scoring; strip them from the BM25 query only. Intent
    # detection keeps the full query ("how long" carries duration intent).
    content_q = " ".join(t for t in tokenize(query) if t not in _QUERY_NOISE)
    orig = index.score(content_q) if content_q else {}
    matched = index.match_counts(content_q) if content_q else {}
    groups = matched_concept_groups(query)

    exp_norm: Dict[str, float] = {}
    for terms in groups:
        g = index.score(" ".join(sorted(terms)))
        for doc_id, s in g.items():
            if s > 0:
                exp_norm[doc_id] = max(exp_norm.get(doc_id, 0.0), _sat(s))

    q_vocab = {t for t in tokenize(query) if t not in _QUERY_NOISE}
    for terms in groups:
        q_vocab |= terms

    combined: Dict[str, float] = {}
    for doc_id in index.doc_ids:
        n_orig = _sat(orig.get(doc_id, 0.0))
        if matched.get(doc_id, 0) == 1:
            # A single matched word is fragile evidence ("period" appearing
            # in an unrelated clause); it may support but not dominate.
            n_orig *= 0.5
        n_exp = exp_norm.get(doc_id, 0.0)
        if n_orig <= 0 and n_exp <= 0:
            # Intent and headings boost grounded blocks; they never create a
            # match on their own (else every number-bearing block would
            # surface for any "how many ..." question).
            continue
        n_int = min(1.0, intent_counts.get(doc_id, 0) / 3.0)
        h_tokens = (headings or {}).get(doc_id) or frozenset()
        h_alpha = {t for t in h_tokens if not t.isdigit()}
        coverage = (len(h_alpha & q_vocab) / len(h_alpha)) if h_alpha else 0.0
        total = n_orig + 0.6 * n_exp + 0.2 * n_int + 0.3 * coverage
        if total > 0.05:
            combined[doc_id] = total
    return combined
