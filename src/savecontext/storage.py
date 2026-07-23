"""Persistence layer: SQLite metadata + zstd-compressed raw source.

The raw source is the source of truth and is stored compressed on disk so it
never has to enter the model context window unless explicitly expanded at
``fidelity="full"``. Blocks and atoms are stored as offsets/values so any span
can be reconstructed byte-for-byte from the raw blob.

A single :class:`Store` wraps the connection. It is intentionally synchronous
and dependency-light (stdlib ``sqlite3`` + ``zstandard``).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import zstandard as zstd

from .chunking import Block
from .extraction import Atom

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contexts (
    context_id      TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    version         INTEGER NOT NULL,
    source_type     TEXT NOT NULL,
    task_hint       TEXT,
    created_at      REAL NOT NULL,
    char_len        INTEGER NOT NULL,
    raw_sha256      TEXT NOT NULL,
    raw_blob        BLOB NOT NULL,
    token_estimate_original INTEGER NOT NULL,
    token_estimate_brief    INTEGER NOT NULL,
    compression_ratio       REAL NOT NULL,
    semantic_brief  TEXT NOT NULL,
    tokenizer       TEXT NOT NULL,
    brief_mode      TEXT NOT NULL DEFAULT 'extractive',
    UNIQUE(label, version)
);
CREATE TABLE IF NOT EXISTS blocks (
    context_id  TEXT NOT NULL,
    block_id    TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    heading     TEXT,
    start_char  INTEGER NOT NULL,
    end_char    INTEGER NOT NULL,
    token_estimate INTEGER NOT NULL,
    PRIMARY KEY (context_id, block_id)
);
CREATE TABLE IF NOT EXISTS atoms (
    context_id  TEXT NOT NULL,
    atom_id     TEXT NOT NULL,
    type        TEXT NOT NULL,
    value       TEXT NOT NULL,
    normalized  TEXT NOT NULL,
    start_char  INTEGER NOT NULL,
    end_char    INTEGER NOT NULL,
    block_id    TEXT,
    count       INTEGER NOT NULL,
    occurrences TEXT NOT NULL,
    PRIMARY KEY (context_id, atom_id)
);
CREATE TABLE IF NOT EXISTS outputs (
    output_id   TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    version     INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    raw_blob    BLOB NOT NULL,
    raw_sha256  TEXT NOT NULL,
    section_map TEXT NOT NULL,
    preview     TEXT NOT NULL,
    token_estimate_original INTEGER NOT NULL,
    token_estimate_preview  INTEGER NOT NULL,
    UNIQUE(label, version)
);
"""

_ZSTD_LEVEL = 10


def default_db_path() -> str:
    # Accept the current env var, then the pre-rename one, for continuity.
    env = (os.environ.get("SAVECONTEXT_DB") or os.environ.get("CONTEXTSAVER_DB")
           or os.environ.get("CONTEXTVAULT_DB"))
    if env:
        return env
    home = os.path.expanduser("~")
    # If a pre-rename store exists and no new one does, keep using it.
    base = os.path.join(home, ".savecontext")
    current = os.path.join(base, "savecontext.db")
    for legacy in (os.path.join(home, ".contextsaver", "contextsaver.db"),
                   os.path.join(home, ".contextvault", "contextvault.db")):
        if os.path.exists(legacy) and not os.path.exists(current):
            return legacy
    os.makedirs(base, exist_ok=True)
    return current


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or default_db_path()
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()
        self._cctx = zstd.ZstdCompressor(level=_ZSTD_LEVEL)
        self._dctx = zstd.ZstdDecompressor()

    def close(self):
        self.conn.close()

    def _migrate(self):
        """Additive migrations for DBs created by older versions."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(contexts)")}
        if "brief_mode" not in cols:
            self.conn.execute(
                "ALTER TABLE contexts ADD COLUMN brief_mode TEXT NOT NULL DEFAULT 'extractive'"
            )

    # --- versioning -----------------------------------------------------

    def next_version(self, label: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(version) AS v FROM contexts WHERE label = ?", (label,)
        ).fetchone()
        return (row["v"] or 0) + 1

    def latest_version(self, label: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT MAX(version) AS v FROM contexts WHERE label = ?", (label,)
        ).fetchone()
        return row["v"]

    # --- context write/read --------------------------------------------

    def save_context(
        self,
        *,
        context_id: str,
        label: str,
        version: int,
        source_type: str,
        task_hint: Optional[str],
        raw_text: str,
        token_estimate_original: int,
        token_estimate_brief: int,
        compression_ratio: float,
        semantic_brief: str,
        tokenizer: str,
        blocks: List[Block],
        atoms: List[Atom],
        brief_mode: str = "extractive",
    ) -> None:
        blob = self._cctx.compress(raw_text.encode("utf-8"))
        self.conn.execute(
            """INSERT INTO contexts
               (context_id, label, version, source_type, task_hint, created_at,
                char_len, raw_sha256, raw_blob, token_estimate_original,
                token_estimate_brief, compression_ratio, semantic_brief, tokenizer,
                brief_mode)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                context_id, label, version, source_type, task_hint, time.time(),
                len(raw_text), _sha256(raw_text), blob, token_estimate_original,
                token_estimate_brief, compression_ratio, semantic_brief, tokenizer,
                brief_mode,
            ),
        )
        self.conn.executemany(
            """INSERT INTO blocks
               (context_id, block_id, idx, heading, start_char, end_char, token_estimate)
               VALUES (?,?,?,?,?,?,?)""",
            [
                (context_id, b.block_id, b.index, b.heading, b.start_char,
                 b.end_char, b.token_estimate)
                for b in blocks
            ],
        )
        self.conn.executemany(
            """INSERT INTO atoms
               (context_id, atom_id, type, value, normalized, start_char,
                end_char, block_id, count, occurrences)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [
                (context_id, a.atom_id, a.type, a.value, a.normalized,
                 a.start_char, a.end_char, a.block_id, a.count,
                 json.dumps(a.occurrences))
                for a in atoms
            ],
        )
        self.conn.commit()

    def list_contexts(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """SELECT context_id, label, version, source_type, created_at,
                      token_estimate_original, token_estimate_brief,
                      compression_ratio
               FROM contexts ORDER BY created_at DESC"""
        ).fetchall()

    def list_outputs(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            """SELECT output_id, label, version, created_at,
                      token_estimate_original, token_estimate_preview
               FROM outputs ORDER BY created_at DESC"""
        ).fetchall()

    def update_brief(self, context_id: str, brief: str, token_estimate_brief: int,
                     compression_ratio: float, brief_mode: str) -> None:
        self.conn.execute(
            """UPDATE contexts
               SET semantic_brief = ?, token_estimate_brief = ?,
                   compression_ratio = ?, brief_mode = ?
               WHERE context_id = ?""",
            (brief, token_estimate_brief, compression_ratio, brief_mode, context_id),
        )
        self.conn.commit()

    def context_exists(self, context_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM contexts WHERE context_id = ?", (context_id,)
        ).fetchone()
        return row is not None

    def get_context(self, context_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM contexts WHERE context_id = ?", (context_id,)
        ).fetchone()

    def get_raw_text(self, context_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT raw_blob FROM contexts WHERE context_id = ?", (context_id,)
        ).fetchone()
        if row is None:
            return None
        return self._dctx.decompress(row["raw_blob"]).decode("utf-8")

    def get_blocks(self, context_id: str) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM blocks WHERE context_id = ? ORDER BY idx", (context_id,)
        ).fetchall()

    def get_atoms(self, context_id: str) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM atoms WHERE context_id = ? ORDER BY start_char",
            (context_id,),
        ).fetchall()

    def get_atom(self, context_id: str, atom_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM atoms WHERE context_id = ? AND atom_id = ?",
            (context_id, atom_id),
        ).fetchone()

    # --- outputs --------------------------------------------------------

    def next_output_version(self, label: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(version) AS v FROM outputs WHERE label = ?", (label,)
        ).fetchone()
        return (row["v"] or 0) + 1

    def save_output(
        self,
        *,
        output_id: str,
        label: str,
        version: int,
        raw_text: str,
        section_map: list,
        preview: str,
        token_estimate_original: int,
        token_estimate_preview: int,
    ) -> None:
        blob = self._cctx.compress(raw_text.encode("utf-8"))
        self.conn.execute(
            """INSERT INTO outputs
               (output_id, label, version, created_at, raw_blob, raw_sha256,
                section_map, preview, token_estimate_original, token_estimate_preview)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                output_id, label, version, time.time(), blob, _sha256(raw_text),
                json.dumps(section_map), preview, token_estimate_original,
                token_estimate_preview,
            ),
        )
        self.conn.commit()

    def get_output(self, output_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM outputs WHERE output_id = ?", (output_id,)
        ).fetchone()

    def get_output_raw(self, output_id: str) -> Optional[str]:
        row = self.get_output(output_id)
        if row is None:
            return None
        return self._dctx.decompress(row["raw_blob"]).decode("utf-8")
