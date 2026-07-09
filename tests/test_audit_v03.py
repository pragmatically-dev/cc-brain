"""Regression tests for the v0.3 audit findings.

1. Legacy DBs (embeddings without embed_model metadata) must trigger a full
   re-embed instead of silently mixing vector dimensions.
2. Repo/private-ingest chunks must be scrubbed before persisting.
3. Background refresh failures must be recorded and surfaced, not swallowed.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import cc_brain.indexer as indexer
import cc_brain.mcp_server as mcp_server
from cc_brain.store import connect, mark_dirty, meta_get
from cc_brain.text import scrub


class FakeEmbedder:
    def __init__(self, dim):
        self.dim = dim

    def embed(self, texts):
        return [np.ones(self.dim, dtype=np.float32) for _ in texts]


@pytest.fixture()
def embed_env(brain_home, monkeypatch):
    """Deterministic 4-dim embedder, no turbovec, no fastembed."""
    monkeypatch.setattr(indexer, "embed_model", lambda: "fake-model")
    monkeypatch.setattr(indexer, "embed_dim", lambda: 4)
    monkeypatch.setattr(indexer, "_embedder", lambda: FakeEmbedder(4))
    monkeypatch.setattr(indexer, "_rebuild_turbovec", lambda con, p: None)
    return brain_home


def test_legacy_db_without_embed_metadata_forces_reembed(embed_env):
    """A v0.1 DB has embeddings but no meta.embed_model: index() must treat it
    as a model mismatch and re-embed everything at the current dimension."""
    p = embed_env
    con = connect(p)
    con.execute(
        "INSERT INTO chunks(id, path, source, project, loc, title, text, mtime, trust)"
        " VALUES(1, 'old.md', 's', '', 'L1', 't', 'legacy body', 0, 1.0)"
    )
    legacy_vec = np.ones(768, dtype=np.float32).tobytes()  # old 768-dim model
    con.execute("INSERT INTO embeddings(chunk, vec) VALUES(1, ?)", (legacy_vec,))
    con.commit()

    indexer.index(p=p)

    con = connect(p)
    vecs = [row[0] for row in con.execute("SELECT vec FROM embeddings")]
    assert vecs, "chunk must still be embedded"
    assert all(len(v) == 4 * 4 for v in vecs), "legacy 768-dim vectors must be re-embedded at current dim"
    assert meta_get(con, "embed_model") == "fake-model"


def test_metadata_lies_dim_mismatch_forces_reembed(embed_env):
    """Even with matching metadata (e.g. crashed half-migration), a stored
    vector whose byte length contradicts embed_dim() must force a re-embed."""
    p = embed_env
    con = connect(p)
    con.execute(
        "INSERT INTO chunks(id, path, source, project, loc, title, text, mtime, trust)"
        " VALUES(1, 'old.md', 's', '', 'L1', 't', 'body', 0, 1.0)"
    )
    con.execute("INSERT INTO embeddings(chunk, vec) VALUES(1, ?)", (np.ones(768, dtype=np.float32).tobytes(),))
    con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('embed_model', 'fake-model')")
    con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('embed_dim', '4')")
    con.commit()

    indexer.index(p=p)

    con = connect(p)
    vecs = [row[0] for row in con.execute("SELECT vec FROM embeddings")]
    assert all(len(v) == 4 * 4 for v in vecs)


def test_repo_ingest_scrubs_secrets(embed_env, tmp_path):
    """Secrets inside indexed source files must not be persisted verbatim."""
    repo = tmp_path / "somerepo"
    repo.mkdir()
    secret = "sk-" + "a1B2" * 8
    (repo / "config.py").write_text(
        f'API_KEY = "{secret}"\npassword = "hunter2secret99"\n', encoding="utf-8"
    )
    indexer.register_repo(str(repo), name="somerepo")

    indexer.index(p=embed_env)

    con = connect(embed_env)
    bodies = [row[0] for row in con.execute("SELECT text FROM chunks")]
    assert bodies, "the repo file must be chunked"
    joined = "\n".join(bodies)
    assert secret not in joined
    assert "hunter2secret99" not in joined
    assert "[REDACTED]" in joined
    terms = {row[0] for row in con.execute("SELECT term FROM postings")}
    assert secret.lower() not in terms, "secret must not be searchable via BM25 either"


def test_scrub_is_idempotent_on_clean_code():
    src = "def add(a, b):\n    return a + b\n"
    assert scrub(src) == src


def _wait_refresh_done(timeout=5.0):
    deadline = time.time() + timeout
    while mcp_server._REFRESHING and time.time() < deadline:
        time.sleep(0.02)
    assert not mcp_server._REFRESHING, "background refresh did not finish"


def test_background_refresh_failure_is_recorded_and_surfaced(brain_home, monkeypatch):
    p = brain_home
    monkeypatch.setattr(mcp_server, "_LAST_INDEX", 0.0)
    monkeypatch.setattr(mcp_server, "_REFRESHING", False)
    # Keep doctor() away from the real embedder (slow model init) in unit tests.
    monkeypatch.setattr(indexer, "vector_available", lambda: False)

    def boom():
        raise RuntimeError("model download exploded")

    monkeypatch.setattr(mcp_server, "run_index", boom)
    mark_dirty("test", p=p)

    note = mcp_server._index_if_dirty()
    assert "background" in note
    _wait_refresh_done()

    con = connect(p)
    err = meta_get(con, "last_refresh_error")
    assert "RuntimeError" in err and "model download exploded" in err

    # The next tool call must warn even though the vault is still dirty.
    warning = mcp_server._refresh_warning()
    assert "model download exploded" in warning

    # doctor() must surface it too.
    report = indexer.doctor(p=p)
    assert "model download exploded" in report.get("last_refresh_error", "")
    assert any("refresh" in w for w in report["warnings"])


def test_background_refresh_success_clears_error(brain_home, monkeypatch):
    p = brain_home
    monkeypatch.setattr(mcp_server, "_LAST_INDEX", 0.0)
    monkeypatch.setattr(mcp_server, "_REFRESHING", False)
    con = connect(p)
    con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('last_refresh_error', 'stale error')")
    con.commit()

    monkeypatch.setattr(mcp_server, "run_index", lambda: "ok")
    mark_dirty("test", p=p)

    mcp_server._index_if_dirty()
    _wait_refresh_done()

    con = connect(p)
    assert meta_get(con, "last_refresh_error") == ""
    assert mcp_server._refresh_warning() == ""
