"""Stable, readable context/output handles.

A context handle looks like ``ctx://acme-contract@v1`` and an output handle
looks like ``out://summary@v1``. Handles are the public identity of a stored
object: they are stable (the same label+version always maps to the same
handle) and human-readable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CTX_SCHEME = "ctx"
OUT_SCHEME = "out"

_HANDLE_RE = re.compile(r"^(?P<scheme>ctx|out)://(?P<label>[^@]+)@v(?P<version>\d+)$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(label: str) -> str:
    """Normalize an arbitrary label into a stable, readable slug.

    ``"Acme Corp Contract!"`` -> ``"acme-corp-contract"``. Empty or
    symbol-only labels fall back to ``"untitled"``.
    """
    slug = _SLUG_RE.sub("-", (label or "").strip().lower()).strip("-")
    return slug or "untitled"


def make_handle(label: str, version: int, scheme: str = CTX_SCHEME) -> str:
    """Build a handle string from a (already-slugged or raw) label."""
    return f"{scheme}://{slugify(label)}@v{int(version)}"


@dataclass(frozen=True)
class Handle:
    scheme: str
    label: str
    version: int

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.scheme}://{self.label}@v{self.version}"


def parse_handle(handle: str) -> Handle:
    """Parse a handle string, raising ``ValueError`` on malformed input."""
    m = _HANDLE_RE.match((handle or "").strip())
    if not m:
        raise ValueError(
            f"Invalid handle {handle!r}; expected e.g. 'ctx://label@v1'"
        )
    return Handle(
        scheme=m.group("scheme"),
        label=m.group("label"),
        version=int(m.group("version")),
    )


def is_handle(value: str) -> bool:
    return bool(_HANDLE_RE.match((value or "").strip()))
