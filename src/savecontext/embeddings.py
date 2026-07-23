"""Optional embedding-based ranking — strictly opt-in, BM25 otherwise.

SaveContext's default stays free/offline: BM25 with no model. But for queries
where wording diverges from the source ("what's the cap on damages?" vs.
"liability shall not exceed"), dense embeddings help. This backend is enabled
only when BOTH:

  * env ``SAVECONTEXT_EMBEDDINGS`` is truthy, and
  * ``sentence-transformers`` + ``numpy`` are importable,

so it never changes behaviour or adds latency unless the user asks for it.
``HybridRanker`` blends normalized BM25 and cosine similarity, and exposes the
same ``score``/``rank`` interface as :class:`savecontext.retrieval.BM25Index`,
so call sites are backend-agnostic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from .retrieval import BM25Index

_DEFAULT_MODEL = os.environ.get(
    "SAVECONTEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def enabled() -> bool:
    return os.environ.get("SAVECONTEXT_EMBEDDINGS", "").lower() in {"1", "true", "yes", "on"}


def available() -> bool:
    try:
        import numpy  # noqa: F401
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=2)
def _model(name: str):  # pragma: no cover - requires the optional dependency
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class HybridRanker:  # pragma: no cover - exercised only when deps installed
    """Blend BM25 and embedding cosine similarity over a set of docs."""

    def __init__(self, docs: Sequence[Tuple[str, str]], alpha: float = 0.5,
                 model_name: str = _DEFAULT_MODEL):
        import numpy as np

        self.alpha = alpha
        self.bm25 = BM25Index(docs)
        self.doc_ids = [d for d, _ in docs]
        texts = [t for _, t in docs]
        model = _model(model_name)
        self._np = np
        embs = model.encode(texts, normalize_embeddings=True)
        self.embs = np.asarray(embs, dtype="float32")
        self._model_name = model_name

    def score(self, query: str) -> Dict[str, float]:
        np = self._np
        bm = self.bm25.score(query)
        max_bm = max(bm.values()) if bm else 0.0
        q = _model(self._model_name).encode([query], normalize_embeddings=True)
        sims = (self.embs @ np.asarray(q, dtype="float32").T).ravel()
        out: Dict[str, float] = {}
        for i, doc_id in enumerate(self.doc_ids):
            nb = (bm.get(doc_id, 0.0) / max_bm) if max_bm else 0.0
            cos = float((sims[i] + 1.0) / 2.0)  # map [-1,1] -> [0,1]
            out[doc_id] = (1 - self.alpha) * nb + self.alpha * cos
        return out

    def rank(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        scored = sorted(self.score(query).items(), key=lambda x: x[1], reverse=True)
        scored = [(d, s) for d, s in scored if s > 0]
        return scored[:top_k] if top_k else scored
