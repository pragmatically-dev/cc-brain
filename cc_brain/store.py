from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from .config import BrainPaths, ensure_dirs, paths


def connect(p: BrainPaths | None = None) -> sqlite3.Connection:
    p = ensure_dirs(p or paths())
    con = sqlite3.connect(p.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS files(
            path TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            project TEXT,
            mtime REAL,
            size INTEGER
        );
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            source TEXT NOT NULL,
            project TEXT,
            loc TEXT,
            title TEXT,
            text TEXT,
            mtime REAL,
            trust REAL DEFAULT 1.0,
            uses INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
        CREATE INDEX IF NOT EXISTS chunks_project ON chunks(project);
        CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
        CREATE TABLE IF NOT EXISTS postings(
            term TEXT NOT NULL,
            chunk INTEGER NOT NULL,
            tf INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS postings_term ON postings(term);
        CREATE INDEX IF NOT EXISTS postings_chunk ON postings(chunk);
        CREATE TABLE IF NOT EXISTS embeddings(
            chunk INTEGER PRIMARY KEY,
            vec BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta(
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS edges(
            a INTEGER NOT NULL,
            b INTEGER NOT NULL,
            w REAL NOT NULL,
            PRIMARY KEY(a, b)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_b ON edges(b);
        """
    )
    # Pre-v0.4 DBs have a chunks table without `uses`; CREATE TABLE IF NOT
    # EXISTS above is a no-op for them, so migrate it in tolerantly.
    try:
        con.execute("ALTER TABLE chunks ADD COLUMN uses INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    return con


def mark_dirty(reason: str, path: str | Path = "", p: BrainPaths | None = None) -> None:
    p = ensure_dirs(p or paths())
    payload = {"ts": time.time(), "reason": reason, "path": str(path)}
    p.dirty.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def dirty_info(p: BrainPaths | None = None) -> dict | None:
    p = ensure_dirs(p or paths())
    if not p.dirty.exists():
        return None
    try:
        return json.loads(p.dirty.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ts": 0, "reason": "unreadable", "path": str(p.dirty)}


def clear_dirty(p: BrainPaths | None = None) -> None:
    p = ensure_dirs(p or paths())
    try:
        p.dirty.unlink()
    except FileNotFoundError:
        pass


def meta_get(con, key: str, default: str = "") -> str:
    row = con.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    return row[0] if row else default


def meta_set(con, key: str, value: str) -> None:
    con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES(?,?)", (key, str(value)))


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
