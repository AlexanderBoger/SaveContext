"""Command-line interface for SaveContext.

Lets people use the vault directly — ingest files, inspect, expand, quote,
diff — without an MCP client, and run the MCP server. Sharing the same SQLite
DB (``SAVECONTEXT_DB``) means you can ingest a file on the CLI and immediately
query it from Claude over MCP, or vice versa.

Usage:
    savecontext serve                         # run the MCP server (stdio)
    savecontext ingest report.md --label q4
    savecontext ingest - --label pasted       # read from stdin
    savecontext list
    savecontext brief ctx://q4@v1 --task "risks"
    savecontext expand ctx://q4@v1 --selector liability --fidelity facts
    savecontext quote ctx://q4@v1 --search "shall not exceed"
    savecontext audit ctx://q4@v1
    savecontext diff ctx://q4@v1 report_v2.md

Bare ``savecontext`` (no args) runs the MCP server, so existing MCP configs
that call ``python -m savecontext`` keep working unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def _read_source(path: str) -> str:
    """Read one source: stdin ('-'), or any supported file via loaders."""
    if path == "-":
        return sys.stdin.read()
    from .loaders import load_path

    return load_path(path)


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _service():
    # Imported lazily so `savecontext serve` doesn't pay for it unnecessarily.
    from .service import VaultService

    return VaultService()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="savecontext",
        description="Git LFS for LLM context — versioned, loss-aware compression.",
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("serve", help="Run the MCP server over stdio.")

    ing = sub.add_parser(
        "ingest",
        help="Ingest a file, a directory (recursively), or '-' for stdin. "
             "Supports .txt/.md/code, .html, .docx, and .pdf (with pypdf).",
    )
    ing.add_argument("path")
    ing.add_argument("--label", required=True)
    ing.add_argument("--type", default="auto", dest="source_type")
    ing.add_argument("--task-hint", default=None)
    ing.add_argument("--brief-only", action="store_true",
                     help="Print only the semantic brief, not the full JSON.")

    sub.add_parser("list", help="List stored contexts and outputs.")

    br = sub.add_parser("brief", help="Task-specific brief for a context.")
    br.add_argument("context_id")
    br.add_argument("--task", required=True)
    br.add_argument("--max-tokens", type=int, default=None)

    lk = sub.add_parser(
        "lookup",
        help="Batched retrieval: answer several queries in one call "
             "(repeat --query).",
    )
    lk.add_argument("context_id")
    lk.add_argument("--query", action="append", required=True, dest="queries",
                    help="A retrieval query; repeat for a batch.")
    lk.add_argument("--top-blocks", type=int, default=2)

    ex = sub.add_parser("expand", help="Expand part of a context.")
    ex.add_argument("context_id")
    ex.add_argument("--selector", required=True)
    ex.add_argument("--fidelity", default="summary",
                    choices=["summary", "facts", "quotes", "full"])

    qu = sub.add_parser("quote", help="Exact quote by atom id or search query.")
    qu.add_argument("context_id")
    qu.add_argument("--atom-id", default=None)
    qu.add_argument("--search", default=None)

    au = sub.add_parser("audit", help="Audit a context's preservation/safety.")
    au.add_argument("context_id")

    sb = sub.add_parser(
        "set-brief", help="Store an agent-authored brief (file, '-' for stdin, or --text)."
    )
    sb.add_argument("context_id")
    sb.add_argument("path", nargs="?", default="-", help="File with the brief, or '-' for stdin.")
    sb.add_argument("--text", default=None, help="Inline brief text (overrides path).")

    df = sub.add_parser("diff", help="Diff new source (file or '-') vs a context.")
    df.add_argument("context_id")
    df.add_argument("path")

    return p


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Bare invocation or `serve` -> run the MCP server (keeps MCP configs working).
    if not argv or argv[0] == "serve":
        import os

        rest = argv[1:] if argv else []
        sp = argparse.ArgumentParser(prog="savecontext serve")
        sp.add_argument("--http", action="store_true", help="Serve over streamable-HTTP.")
        sp.add_argument("--sse", action="store_true", help="Serve over SSE.")
        sp.add_argument("--host", default=None)
        sp.add_argument("--port", type=int, default=None)
        sa = sp.parse_args(rest)
        if sa.http:
            os.environ["SAVECONTEXT_TRANSPORT"] = "http"
        elif sa.sse:
            os.environ["SAVECONTEXT_TRANSPORT"] = "sse"
        if sa.host:
            os.environ["SAVECONTEXT_HOST"] = sa.host
        if sa.port:
            os.environ["SAVECONTEXT_PORT"] = str(sa.port)

        from .server import main as serve_main

        serve_main()
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    svc = _service()
    try:
        if args.command == "ingest":
            import os

            from .loaders import iter_text_files

            if os.path.isdir(args.path):
                # Folder ingest: one context per file, labelled base/relpath.
                results = []
                for relpath, abspath in iter_text_files(args.path):
                    from .loaders import load_path

                    sub_label = f"{args.label}/{relpath}"
                    try:
                        r = svc.ingest(
                            load_path(abspath), sub_label, args.source_type, args.task_hint
                        )
                        results.append({
                            "file": relpath,
                            "context_id": r["context_id"],
                            "tokens": r["token_estimate_original"],
                            "ratio": r["compression_ratio"],
                        })
                    except Exception as exc:  # noqa: BLE001
                        results.append({"file": relpath, "error": str(exc)})
                _emit({"ingested": len([r for r in results if "context_id" in r]),
                       "results": results})
            else:
                res = svc.ingest(
                    _read_source(args.path), args.label, args.source_type, args.task_hint
                )
                if args.brief_only:
                    print(res["semantic_brief"])
                else:
                    _emit(res)

        elif args.command == "list":
            _emit(svc.list_all())

        elif args.command == "brief":
            _emit(svc.brief(args.context_id, args.task, args.max_tokens))

        elif args.command == "lookup":
            _emit(svc.lookup(args.context_id, args.queries, args.top_blocks))

        elif args.command == "expand":
            _emit(svc.expand(args.context_id, args.selector, args.fidelity))

        elif args.command == "quote":
            _emit(svc.quote(args.context_id, args.atom_id, args.search))

        elif args.command == "audit":
            _emit(svc.audit(args.context_id))

        elif args.command == "set-brief":
            brief_text = args.text if args.text is not None else _read_source(args.path)
            _emit(svc.set_brief(args.context_id, brief_text))

        elif args.command == "diff":
            _emit(svc.diff(args.context_id, _read_source(args.path)))

        else:  # pragma: no cover - argparse guards this
            parser.print_help()
            return 1
    except Exception as exc:  # noqa: BLE001 - CLI surfaces errors cleanly
        _emit({"error": str(exc), "error_type": type(exc).__name__})
        return 2
    finally:
        svc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
