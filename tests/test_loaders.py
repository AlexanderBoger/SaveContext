"""Tests for document loaders and folder ingestion."""

import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savecontext import cli
from savecontext.loaders import is_ingestible, iter_text_files, load_path


def test_load_text(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# Title\nhello world")
    assert "hello world" in load_path(str(f))


def test_load_html_strips_tags(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><head><style>.x{}</style><script>var a=1;</script></head>"
        "<body><h1>Heading</h1><p>Body text here.</p></body></html>"
    )
    out = load_path(str(f))
    assert "Heading" in out and "Body text here." in out
    assert "var a=1" not in out and ".x{}" not in out
    assert "<" not in out


def test_load_docx_dependency_free(tmp_path):
    # Build a minimal valid .docx (zip with word/document.xml).
    f = tmp_path / "doc.docx"
    document_xml = (
        '<?xml version="1.0"?><w:document xmlns:w="x"><w:body>'
        "<w:p><w:r><w:t>First paragraph.</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Second paragraph with $500.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(f, "w") as zf:
        zf.writestr("word/document.xml", document_xml)
    out = load_path(str(f))
    assert "First paragraph." in out
    assert "Second paragraph with $500." in out
    assert "<w:" not in out


def test_is_ingestible():
    assert is_ingestible("a.py") and is_ingestible("b.pdf") and is_ingestible("c.html")
    assert not is_ingestible("image.png") and not is_ingestible("video.mp4")


def test_iter_text_files_skips_junk(tmp_path):
    (tmp_path / "keep.md").write_text("x")
    (tmp_path / "img.png").write_bytes(b"\x89PNG")
    junk = tmp_path / "node_modules"
    junk.mkdir()
    (junk / "dep.js").write_text("y")
    found = {rel for rel, _ in iter_text_files(str(tmp_path))}
    assert "keep.md" in found
    assert "img.png" not in found
    assert not any("node_modules" in f for f in found)


def test_cli_folder_ingest(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SAVECONTEXT_DB", str(tmp_path / "cv.db"))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.md").write_text("Customer shall pay $50,000 on 2024-01-01. " * 6)
    (docs / "two.txt").write_text("Provider liability shall not exceed $500,000. " * 6)
    rc = cli.main(["ingest", str(docs), "--label", "proj"])
    assert rc == 0
    import json

    res = json.loads(capsys.readouterr().out)
    assert res["ingested"] == 2
    cids = {r["context_id"] for r in res["results"]}
    assert "ctx://proj-one-md@v1" in cids
