"""Tests for plan v0.4 phase 1: MMR diversity + per-source-kind recency half-life.

Real SourceSpec.kind values in this codebase are only "md" (notes, commits,
sessions, web) and "code" (private-ingest, register_repo) -- see config.py's
_default_sources() and contracts.py's SourceSpec default. The spec's example
kinds ("session"/"web"/"memory") don't exist as SourceSpec.kind values (those
are source *names*, a finer axis than kind), so RECENCY_HALF_LIFE_DAYS is keyed
on the two real kinds plus a default for any future custom kind.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from conftest import make_chunk

import cc_brain.indexer as indexer
from cc_brain.config import upsert_source
from cc_brain.contracts import SourceSpec
from cc_brain.store import connect


def _embed(con, cid, vec):
    con.execute(
        "INSERT INTO embeddings(chunk, vec) VALUES(?,?)",
        (cid, np.asarray(vec, dtype=np.float32).tobytes()),
    )


# ---------------------------------------------------------------------------
# 1a. MMR diversity
# ---------------------------------------------------------------------------

def test_mmr_promotes_orthogonal_chunk_over_near_duplicate(brain_home, monkeypatch):
    """3 chunks share an identical embedding, 1 is orthogonal. Pure RRF score
    order would pick the 3 duplicates for k=3; MMR must swap the weakest
    duplicate for the orthogonal chunk instead."""
    p = brain_home
    con = connect(p)
    now = time.time()
    make_chunk(con, 1, "notes", "", "dup one", trust=1.0, mtime=now, path="p1.md")
    make_chunk(con, 2, "notes", "", "dup two", trust=1.0, mtime=now, path="p2.md")
    make_chunk(con, 3, "notes", "", "dup three", trust=1.0, mtime=now, path="p3.md")
    make_chunk(con, 4, "notes", "", "orthogonal", trust=1.0, mtime=now, path="p4.md")
    _embed(con, 1, [1, 0, 0, 0])
    _embed(con, 2, [1, 0, 0, 0])
    _embed(con, 3, [1, 0, 0, 0])
    _embed(con, 4, [0, 1, 0, 0])
    con.commit()

    # Rank order 1,2,3,4 (best to worst) via RRF; empty BM25 side.
    monkeypatch.setattr(indexer, "_vec_search", lambda query, limit, p: [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0)])
    monkeypatch.setattr(indexer, "_bm25", lambda con, tokens, limit, project="", source="": [])

    hits = indexer.search("q", k=3, p=p)
    ids = [h.id for h in hits]

    assert len(ids) == 3
    assert 4 in ids, f"orthogonal chunk must survive MMR diversification, got {ids}"
    # The strongest duplicate (highest raw relevance) must still be first.
    assert ids[0] == 1


def test_mmr_pool_of_one_returns_the_single_candidate(brain_home, monkeypatch):
    p = brain_home
    con = connect(p)
    make_chunk(con, 1, "notes", "", "solo chunk", trust=1.0, mtime=time.time(), path="solo.md")
    _embed(con, 1, [1, 0, 0, 0])
    con.commit()

    monkeypatch.setattr(indexer, "_vec_search", lambda query, limit, p: [(1, 0.0)])
    monkeypatch.setattr(indexer, "_bm25", lambda con, tokens, limit, project="", source="": [])

    hits = indexer.search("q", k=3, p=p)
    assert [h.id for h in hits] == [1]


def test_lex_true_skips_mmr_and_vec_search_entirely(brain_home, monkeypatch):
    """lex=True must keep today's behavior: no vector search, no MMR reorder."""
    p = brain_home
    con = connect(p)
    now = time.time()
    make_chunk(con, 1, "notes", "", "alpha term", trust=1.0, mtime=now, path="p1.md")
    make_chunk(con, 2, "notes", "", "alpha term second", trust=1.0, mtime=now, path="p2.md")
    con.commit()

    def _boom(*a, **k):
        raise AssertionError("_vec_search must not be called when lex=True")

    monkeypatch.setattr(indexer, "_vec_search", _boom)
    monkeypatch.setattr(indexer, "_bm25", lambda con, tokens, limit, project="", source="": [(1, 2.0), (2, 1.0)])

    hits = indexer.search("alpha", k=2, lex=True, p=p)
    assert [h.id for h in hits] == [1, 2]


# ---------------------------------------------------------------------------
# 1b. Per-source-kind recency half-life
# ---------------------------------------------------------------------------

def test_recency_half_life_differs_by_source_kind(brain_home, monkeypatch):
    """Two chunks, equal fused RRF score, equal age. The one whose source kind
    has the longer half-life ("code", 90d) must outrank the shorter one
    ("md", 21d) once the recency term is no longer a single fixed constant."""
    p = brain_home
    con = connect(p)
    now = time.time()
    thirty_days_ago = now - 30 * 86400

    upsert_source(SourceSpec("code-src", p.home / "code-src", kind="code"), p=p)
    upsert_source(SourceSpec("md-src", p.home / "md-src", kind="md"), p=p)

    make_chunk(con, 1, "code-src", "", "alpha", trust=1.0, mtime=thirty_days_ago, path="a.py")
    make_chunk(con, 2, "md-src", "", "beta", trust=1.0, mtime=thirty_days_ago, path="b.md")
    con.commit()

    # Give each chunk an equal-sized RRF contribution from a *different* leg
    # (vec vs bm25) so the fused score is identical without any tie-breaking.
    # cid 2 (md) is inserted into the fused dict first (via the vec leg) so
    # that under the OLD fixed-decay behavior a stable sort on a tied score
    # would list it first -- this must flip to [1, 2] only once per-kind
    # half-life makes the code chunk's recency term genuinely larger.
    monkeypatch.setattr(indexer, "_vec_search", lambda query, limit, p: [(2, 0.0)])
    monkeypatch.setattr(indexer, "_bm25", lambda con, tokens, limit, project="", source="": [(1, 0.0)])

    hits = indexer.search("q", k=2, p=p)
    ids = [h.id for h in hits]

    assert ids == [1, 2], f"kind=code (90d half-life) should outrank kind=md (21d) at equal age, got {ids}"


def test_recency_half_life_dict_uses_real_kinds():
    """Guard against re-introducing the spec's example kinds ('session',
    'web', 'memory') which are not real SourceSpec.kind values in this repo."""
    assert set(indexer.RECENCY_HALF_LIFE_DAYS) <= {"md", "code"}
    assert indexer.RECENCY_HALF_LIFE_DAYS.get("code") == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# 1b-bis. Per-source-name recency half-life override
# ---------------------------------------------------------------------------

def test_recency_half_life_source_name_override_beats_kind(brain_home, monkeypatch):
    """'notes' (365d via RECENCY_HALF_LIFE_BY_SOURCE) must outrank 'sessions'
    (14d) even though both share kind='md' and thus the same kind-level
    half-life -- the source-name override is the intended fast path for
    "curated note vs session transcript" that kind alone can't express."""
    p = brain_home
    con = connect(p)
    thirty_days_ago = time.time() - 30 * 86400

    make_chunk(con, 1, "sessions", "", "alpha", trust=1.0, mtime=thirty_days_ago, path="s.md")
    make_chunk(con, 2, "notes", "", "beta", trust=1.0, mtime=thirty_days_ago, path="n.md")
    con.commit()

    # cid 1 (sessions) inserted into the fused dict first (vec leg) so that,
    # pre-fix (kind-only resolution, both "md" => tied), a stable sort on the
    # tie would list it first -- must flip to [2, 1] only once the per-source
    # override makes notes' recency term genuinely larger.
    monkeypatch.setattr(indexer, "_vec_search", lambda query, limit, p: [(1, 0.0)])
    monkeypatch.setattr(indexer, "_bm25", lambda con, tokens, limit, project="", source="": [(2, 0.0)])

    hits = indexer.search("q", k=2, p=p)
    ids = [h.id for h in hits]

    assert ids == [2, 1], f"source=notes (365d) should outrank source=sessions (14d) at equal age/kind, got {ids}"


def test_half_life_days_resolution_order():
    """_half_life_days(source_name, kind) must resolve source-name override
    first, then kind, then the 30.0 default -- in that order."""
    # 1. source-name match wins even against a mismatched/unrelated kind.
    assert indexer._half_life_days("notes", "md") == pytest.approx(365.0)
    assert indexer._half_life_days("sessions", "code") == pytest.approx(14.0)
    # 2. no source-name match -> fall back to kind.
    assert indexer._half_life_days("some-repo", "code") == pytest.approx(90.0)
    assert indexer._half_life_days("random-md-source", "md") == pytest.approx(21.0)
    # 3. no source-name and no kind match -> default.
    assert indexer._half_life_days("random-md-source", "unknown-kind") == pytest.approx(30.0)
