"""Optional local-LLM summarization via Ollama — strictly opt-in.

By default SaveContext's briefs are *extractive* (real sentences from the
source), which is fast, free, and keeps the loss-aware guarantee simple. When
a user wants genuinely abstractive briefs, this module calls a local model
(Ollama) over HTTP using only the standard library.

It activates only when ``SAVECONTEXT_LLM_SUMMARY`` is truthy, and every call
fails *closed*: any timeout, connection error, or bad response returns ``None``
so the caller transparently keeps the extractive brief. No network is touched
unless the feature is explicitly enabled.

Env:
  SAVECONTEXT_LLM_SUMMARY   enable (1/true/yes/on)
  SAVECONTEXT_OLLAMA_URL    default http://localhost:11434
  SAVECONTEXT_LLM_MODEL     default llama3.2
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

_DEFAULT_URL = os.environ.get("SAVECONTEXT_OLLAMA_URL", "http://localhost:11434")
_DEFAULT_MODEL = os.environ.get("SAVECONTEXT_LLM_MODEL", "llama3.2")
_TIMEOUT = float(os.environ.get("SAVECONTEXT_LLM_TIMEOUT", "30"))
_MAX_INPUT_CHARS = 24000  # keep the prompt bounded regardless of source size


def enabled() -> bool:
    return os.environ.get("SAVECONTEXT_LLM_SUMMARY", "").lower() in {"1", "true", "yes", "on"}


def _generate(prompt: str, num_predict: int) -> Optional[str]:
    """Call Ollama /api/generate; return text or None on any failure."""
    payload = json.dumps(
        {
            "model": _DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": num_predict},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{_DEFAULT_URL.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = (data.get("response") or "").strip()
        return text or None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def refine_brief(extractive_brief: str, source_text: str, source_type: str,
                 max_tokens: int = 400) -> Optional[str]:
    """Produce an abstractive brief, grounded in the extractive draft.

    The model is instructed to stay faithful and not invent facts; the
    extractive draft is provided as scaffolding. Returns None if disabled or
    the call fails, so the caller falls back to the extractive brief.
    """
    if not enabled():
        return None
    prompt = (
        f"You are compressing a {source_type} document for later retrieval.\n"
        "Write a faithful, dense brief. Do NOT invent facts, numbers, or names; "
        "use only what appears in the material. Prefer concrete facts "
        "(amounts, dates, obligations) over generalities.\n\n"
        f"Extractive key points:\n{extractive_brief}\n\n"
        f"Source excerpt:\n{source_text[:_MAX_INPUT_CHARS]}\n\n"
        "Brief:"
    )
    return _generate(prompt, num_predict=max_tokens)
