"""Token estimation.

Uses ``tiktoken`` (cl100k_base) when available; otherwise falls back to the
classic ``len(text) / 4`` approximation. The encoder is loaded lazily and
cached so the first call pays the import cost and subsequent calls are cheap.
"""

from __future__ import annotations

from functools import lru_cache

_APPROX_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def _encoder():
    """Return a cached tiktoken encoder, or ``None`` if tiktoken is absent."""
    try:
        import tiktoken
    except Exception:  # pragma: no cover - exercised only without tiktoken
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - network/cache failure
        return None


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in ``text``.

    Deterministic for a given environment: the same input always returns the
    same count, which keeps compression ratios stable across calls.
    """
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:  # pragma: no cover
            pass
    # Fallback heuristic.
    return max(1, round(len(text) / _APPROX_CHARS_PER_TOKEN))


def tokenizer_name() -> str:
    """Human-readable name of the active tokenizer (for audit/transparency)."""
    return "tiktoken:cl100k_base" if _encoder() is not None else "approx:chars/4"


def compression_ratio(original_tokens: int, compressed_tokens: int) -> float:
    """Ratio of original to compressed tokens, rounded to 2 dp.

    A ratio of 10.0 means the compressed form is ~1/10th the size. Returns
    ``1.0`` when the original is empty to avoid division by zero.
    """
    if original_tokens <= 0:
        return 1.0
    if compressed_tokens <= 0:
        compressed_tokens = 1
    return round(original_tokens / compressed_tokens, 2)
