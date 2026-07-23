#!/usr/bin/env python3
"""UserPromptSubmit hook: auto-ingest large files referenced in the prompt.

When the prompt mentions a file path and that file is large (> threshold),
this hook ingests it into SaveContext before Claude sees the message.
Claude receives the handle + brief in its context and never needs to read
the raw file — no 20-page permission loops.

Files inside the session's working directory are skipped: those are usually
edit targets that Claude must read raw. Only external documents are vaulted.

For large pasted text (no file path), falls back to the original nudge.

Exit code 0 always (never block). Stdout is appended to Claude's context.
Threshold configurable via SAVECONTEXT_HINT_TOKENS (default 2000 tokens).
"""

import json
import os
import re
import sys

THRESHOLD_TOKENS = int(os.environ.get("SAVECONTEXT_HINT_TOKENS", "2000"))
FILE_THRESHOLD_BYTES = THRESHOLD_TOKENS * 4  # rough: 1 token ≈ 4 chars

_EXTENSIONS = (
    "pdf", "txt", "md", "docx", "html", "htm",
    "py", "js", "ts", "java", "kt", "go", "rs",
    "csv", "json", "yaml", "yml", "xml", "log",
)
_EXT_PAT = "|".join(_EXTENSIONS)
# Match absolute or relative paths ending in a known extension.
_PATH_RE = re.compile(
    rf'(?:^|[\s"\'(])(/[^\s"\'()]+\.(?:{_EXT_PAT}))'
    rf'|(?:^|[\s"\'(])(\.{{0,2}}/[^\s"\'()]+\.(?:{_EXT_PAT}))',
    re.IGNORECASE | re.MULTILINE,
)


def _find_large_files(text: str, cwd: str) -> list[str]:
    seen: set[str] = set()
    result = []
    cwd_real = os.path.realpath(cwd) if cwd else None
    for m in _PATH_RE.finditer(text):
        raw = (m.group(1) or m.group(2)).rstrip(".,;)")
        path = os.path.expanduser(raw)
        if path in seen:
            continue
        seen.add(path)
        if not (os.path.isfile(path) and os.path.getsize(path) > FILE_THRESHOLD_BYTES):
            continue
        # Files inside the working directory are likely edit targets — Claude
        # must read those raw. Only auto-vault external documents.
        if cwd_real and os.path.realpath(path).startswith(cwd_real + os.sep):
            continue
        result.append(path)
    return result


def _ingest(path: str) -> str:
    try:
        # Works whether savecontext is pip-installed or run from the repo.
        repo_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
        if os.path.isdir(repo_src):
            sys.path.insert(0, os.path.abspath(repo_src))

        from savecontext.loaders import load_path
        from savecontext.service import VaultService

        text = load_path(path)
        label = re.sub(r"[^a-z0-9-]", "-", os.path.splitext(os.path.basename(path))[0].lower())
        svc = VaultService()
        result = svc.ingest(text, label=label, source_type="auto")
        svc.close()

        cid = result["context_id"]
        orig = result.get("token_estimate_original", "?")
        ratio = result.get("compression_ratio", "?")
        brief = result.get("semantic_brief", "").strip()

        return (
            f"[SaveContext] Auto-ingested {path}\n"
            f"  handle : {cid}  ({orig} tokens → {ratio}x compressed)\n"
            f"  brief  : {brief[:400]}\n"
            f"Work from handle {cid}. Use brief / expand / quote "
            f"for details. Do NOT read the raw file."
        )
    except Exception as exc:
        return f"[SaveContext] Could not auto-ingest {path}: {exc}"


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    prompt = payload.get("prompt") or payload.get("user_prompt") or ""
    cwd = payload.get("cwd") or ""

    large_files = _find_large_files(prompt, cwd)
    if large_files:
        for path in large_files:
            print(_ingest(path))
        return 0

    # Fallback: warn about large pasted text.
    approx_tokens = len(prompt) // 4
    if approx_tokens >= THRESHOLD_TOKENS:
        print(
            f"[SaveContext] This input is ~{approx_tokens} tokens. Before reasoning "
            "over it, store it with ingest (or zip_output for generated "
            "content) and work from the returned handle + brief. Use expand/quote for "
            "detail and exact wording rather than the raw text."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
