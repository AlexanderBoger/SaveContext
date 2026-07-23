"""File loaders: turn real documents into plain text for ingestion.

HTML and DOCX are handled with the standard library only (no new
dependencies) — DOCX is just a zip of XML, and HTML is stripped with
``html.parser``. PDF uses ``pypdf`` if installed and otherwise raises a clear,
actionable error rather than silently producing garbage.

``load_path`` dispatches by extension; ``iter_text_files`` walks a directory
for ingestible files so a whole codebase or doc set can be vaulted at once.
"""

from __future__ import annotations

import os
import re
import zipfile
from html.parser import HTMLParser
from typing import Iterator, List, Tuple

# Extensions read as-is (text/code/config/markup-ish).
TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".sh", ".sql", ".xml",
}
HTML_EXTS = {".html", ".htm"}
DOCX_EXTS = {".docx"}
PDF_EXTS = {".pdf"}

# Directories skipped during folder ingest.
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
_MAX_BYTES = 25 * 1024 * 1024  # skip absurdly large files during folder walks


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of blank lines/spaces produced by tag boundaries.
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _read_html(path: str) -> str:
    parser = _TextExtractor()
    parser.feed(_read_text(path))
    return parser.text()


def _read_docx(path: str) -> str:
    with zipfile.ZipFile(path) as zf:
        try:
            xml = zf.read("word/document.xml").decode("utf-8", "replace")
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"{path}: not a valid .docx (no word/document.xml)") from exc
    # Paragraph breaks <w:p> -> newline; drop all other tags.
    xml = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:  # pragma: no cover - depends on env
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as exc:
            raise ValueError(
                f"{path}: PDF support needs 'pypdf' (pip install pypdf)."
            ) from exc
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n\n".join(pages).strip()


def load_path(path: str) -> str:
    """Load a single file to plain text, dispatching on its extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in HTML_EXTS:
        return _read_html(path)
    if ext in DOCX_EXTS:
        return _read_docx(path)
    if ext in PDF_EXTS:
        return _read_pdf(path)
    # Default: treat as text (covers TEXT_EXTS and unknown text-ish files).
    return _read_text(path)


def is_ingestible(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in TEXT_EXTS | HTML_EXTS | DOCX_EXTS | PDF_EXTS


def iter_text_files(root: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(relpath, abspath)`` for ingestible files under ``root``."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in sorted(filenames):
            ap = os.path.join(dirpath, name)
            if not is_ingestible(ap):
                continue
            try:
                if os.path.getsize(ap) > _MAX_BYTES:
                    continue
            except OSError:  # pragma: no cover
                continue
            yield os.path.relpath(ap, root), ap
