"""MCP server exposing SaveContext's eleven tools over stdio.

Thin wrapper around :class:`savecontext.service.VaultService`. Each tool is a
FastMCP-registered function that validates inputs, delegates to the service,
and returns a JSON-able dict. Errors are returned as ``{"error": ...}`` so the
calling model always gets a clean JSON response.

Run with:  ``python -m savecontext``  (stdio transport)
"""

from __future__ import annotations

import functools
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .service import VaultService

mcp = FastMCP("savecontext")

# A single service/connection for the life of the process.
_service: Optional[VaultService] = None


def service() -> VaultService:
    global _service
    if _service is None:
        _service = VaultService()
    return _service


def _safe(fn):
    """Wrap a tool so exceptions become structured JSON instead of crashes."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface as JSON to the model
            return {"error": str(exc), "error_type": type(exc).__name__}

    return wrapper


@mcp.tool(name="ingest")
@_safe
def ingest(
    source_text: str,
    label: str,
    source_type: str = "auto",
    task_hint: Optional[str] = None,
    agent_brief: Optional[str] = None,
) -> dict:
    """Ingest long text, store it compressed, and return a compact context handle.

    Returns a stable handle (ctx://label@vN), a semantic brief, a block map,
    and a summary of protected atoms. The full raw source is NOT returned.

    agent_brief (optional): if YOU (the calling model) already have this
    document in context, pass your own faithful, dense summary here — it
    becomes the stored brief (brief_mode='agent'), which is higher quality
    than the rule-based extractive default. Keep all money/dates/obligations
    exact. You can also set it later with the set_brief tool.
    """
    return service().ingest(source_text, label, source_type, task_hint, agent_brief)


@mcp.tool(name="set_brief")
@_safe
def set_brief(context_id: str, brief: str) -> dict:
    """Store an agent-authored brief for a context (brief_mode='agent').

    Use this to make Claude itself the summarizer: after ingest, write a
    faithful, compact brief from the source you already have in context and
    save it here so every later turn reuses it. Raw source and protected atoms
    are unchanged, so exactness is preserved.
    """
    return service().set_brief(context_id, brief)


@mcp.tool(name="brief")
@_safe
def brief(context_id: str, task: str, max_tokens: Optional[int] = None) -> dict:
    """Return a task-focused brief: relevant blocks + atoms for a specific task."""
    return service().brief(context_id, task, max_tokens)


@mcp.tool(name="lookup")
@_safe
def lookup(context_id: str, queries: list[str], top_blocks: int = 2) -> dict:
    """Batched retrieval: answer SEVERAL queries in ONE call.

    When you have more than one question about a context, pass them ALL here
    at once instead of making separate brief/expand/quote calls. For each
    query you get the top-ranked block(s) with a short summary, the exact
    protected atoms (money, dates, durations, obligations…) inside them, and
    a verbatim best-matching sentence with exact source offsets — enough to
    answer and cite without further round trips.
    """
    return service().lookup(context_id, queries, top_blocks)


@mcp.tool(name="map")
@_safe
def map_(
    context_id: str,
    with_previews: bool = True,
    offset: int = 0,
    limit: Optional[int] = None,
    include_atom_examples: bool = True,
) -> dict:
    """Full (or paginated) block map + atom summary, fetched lazily.

    The ingest response only includes a small capped outline to keep token cost
    flat with document size; call this when you need the complete structure.
    Use offset/limit to page through very large documents.
    """
    return service().map(context_id, with_previews, offset, limit, include_atom_examples)


@mcp.tool(name="expand")
@_safe
def expand(context_id: str, selector: str, fidelity: str = "summary",
           max_blocks: int = 5) -> dict:
    """Lazily expand part of a context.

    selector: a block id, atom id, section/heading name, or search query.
    fidelity: summary | facts | quotes | full.
    max_blocks: cap on blocks returned (default 5) so a broad selector can't
    dump the whole document; raise it or target a specific block_id to get more.
    """
    return service().expand(context_id, selector, fidelity, max_blocks)


@mcp.tool(name="quote")
@_safe
def quote(
    context_id: str,
    atom_id: Optional[str] = None,
    search_query: Optional[str] = None,
) -> dict:
    """Return an exact verbatim quote (by atom id or search query) with context."""
    return service().quote(context_id, atom_id, search_query)


@mcp.tool(name="diff")
@_safe
def diff(context_id: str, new_source_text: str) -> dict:
    """Diff a new source against a stored context; report semantic/atom changes.

    Creates a new version and returns its handle plus added/removed/changed
    atoms and a meaning-impact assessment.
    """
    return service().diff(context_id, new_source_text)


@mcp.tool(name="zip_output")
@_safe
def zip_output(content: str, label: str, structure: Optional[str] = None) -> dict:
    """Store a large generated OUTPUT and return a section map + preview."""
    return service().zip_output(content, label, structure)


@mcp.tool(name="expand_output")
@_safe
def expand_output(output_id: str, selector: str, fidelity: str = "section") -> dict:
    """Expand part of a stored output. fidelity: preview | section | full."""
    return service().expand_output(output_id, selector, fidelity)


@mcp.tool(name="audit")
@_safe
def audit(context_id: str) -> dict:
    """Report compression ratio, atom counts, preservation estimates, and safety."""
    return service().audit(context_id)


def _configure_net() -> None:
    """Apply host/port from env to FastMCP settings, defensively."""
    import os

    host = os.environ.get("SAVECONTEXT_HOST") or os.environ.get("CONTEXTSAVER_HOST") or os.environ.get("CONTEXTVAULT_HOST")
    port = os.environ.get("SAVECONTEXT_PORT") or os.environ.get("CONTEXTSAVER_PORT") or os.environ.get("CONTEXTVAULT_PORT")
    settings = getattr(mcp, "settings", None)
    if settings is not None:
        if host:
            try:
                settings.host = host
            except Exception:  # pragma: no cover
                pass
        if port:
            try:
                settings.port = int(port)
            except Exception:  # pragma: no cover
                pass


def main() -> None:
    """Run the MCP server. Transport selected by SAVECONTEXT_TRANSPORT.

    stdio (default) | http (streamable-http) | sse. For http/sse, set
    SAVECONTEXT_HOST / SAVECONTEXT_PORT (defaults from FastMCP).
    """
    import os

    transport = (os.environ.get("SAVECONTEXT_TRANSPORT")
                 or os.environ.get("CONTEXTSAVER_TRANSPORT")
                 or os.environ.get("CONTEXTVAULT_TRANSPORT", "stdio")).lower()
    if transport in ("http", "streamable-http", "streamable_http"):
        _configure_net()
        mcp.run(transport="streamable-http")
    elif transport == "sse":
        _configure_net()
        mcp.run(transport="sse")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
