"""Tests for plan v0.4 phase 1: MMR diversity + per-source-kind recency half-life.

Real SourceSpec.kind values in this codebase are only "md" (notes, commits,
sessions, web) and "code" (private-ingest, register_repo) -- see config.py's
_default_sources() and contracts.py's SourceSpec default. The spec's example
kinds ("session"/"web"/"memory") don't exist as SourceSpec.kind values (those
are source *names*, a finer axis than kind), so RECENCY_HALF_LIFE_DAYS is keyed
on the two real kinds plus a default for any future custom kind.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Phase 2 -- implicit relevance feedback via get() usage counts
# ---------------------------------------------------------------------------

def test_get_increments_uses(brain_home):
    p = brain_home
    con = connect(p)
    make_chunk(con, 1, "notes", "", "alpha", trust=1.0, mtime=time.time(), path="a.md")
    make_chunk(con, 2, "notes", "", "beta", trust=1.0, mtime=time.time(), path="b.md")
    con.commit()

    indexer.get([1], p=p)
    indexer.get([1, 2], p=p)

    con = connect(p)
    uses = {r[0]: r[1] for r in con.execute("SELECT id, uses FROM chunks")}
    assert uses[1] == 2, "id 1 was requested in two separate get() calls"
    assert uses[2] == 1, "id 2 was requested in one get() call"


def test_uses_bonus_ranks_higher_uses_above_at_equal_score(brain_home, monkeypatch):
    """Two chunks, equal fused RRF score, equal age/kind (both 'notes').
    Pre-fix there is no uses bonus, so a tie stable-sorts by fused-dict
    insertion order; post-fix the uses=20 chunk must win regardless of that
    insertion order."""
    p = brain_home
    con = connect(p)
    now = time.time()
    make_chunk(con, 1, "notes", "", "alpha", trust=1.0, mtime=now, path="a.md")
    make_chunk(con, 2, "notes", "", "beta", trust=1.0, mtime=now, path="b.md")
    con.commit()
    con.execute("UPDATE chunks SET uses=20 WHERE id=2")
    con.commit()

    # id=1 (uses=0) inserted first via the vec leg -- pre-fix tie would list
    # it first; must flip to [2, 1] once the log1p(uses) bonus exists.
    monkeypatch.setattr(indexer, "_vec_search", lambda query, limit, p: [(1, 0.0)])
    monkeypatch.setattr(indexer, "_bm25", lambda con, tokens, limit, project="", source="": [(2, 0.0)])

    hits = indexer.search("q", k=2, p=p)
    ids = [h.id for h in hits]
    assert ids == [2, 1], f"uses=20 chunk should outrank uses=0 at equal score/age, got {ids}"


def test_uses_bonus_is_capped():
    assert indexer._uses_bonus(0) == 0.0
    assert indexer._uses_bonus(10**6) == pytest.approx(0.04)
    assert indexer._uses_bonus(5) < indexer._uses_bonus(20) <= 0.04


def test_stats_reports_top_used(brain_home):
    p = brain_home
    con = connect(p)
    make_chunk(con, 1, "notes", "", "a", path="a.md")
    make_chunk(con, 2, "notes", "", "b", path="b.md")
    make_chunk(con, 3, "notes", "", "c", path="c.md")
    con.commit()
    con.execute("UPDATE chunks SET uses=5 WHERE id=1")
    con.execute("UPDATE chunks SET uses=9 WHERE id=2")
    # id=3 stays at uses=0 and must be excluded from top_used.
    con.commit()

    s = indexer.stats(p=p)
    top = s["top_used"]
    assert [t["id"] for t in top] == [2, 1]
    assert all(t["uses"] > 0 for t in top)
    assert top[0]["path"] == "b.md"


def test_legacy_db_without_uses_column_migrates_on_connect(brain_home):
    """A pre-v0.4 DB has a chunks table with no uses column. connect() must
    add it via a tolerant ALTER TABLE, defaulting existing rows to 0."""
    p = brain_home
    p.data.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(p.db)
    legacy.execute(
        """
        CREATE TABLE chunks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            source TEXT NOT NULL,
            project TEXT,
            loc TEXT,
            title TEXT,
            text TEXT,
            mtime REAL,
            trust REAL DEFAULT 1.0
        )
        """
    )
    legacy.execute(
        "INSERT INTO chunks(id, path, source, project, loc, title, text, mtime, trust)"
        " VALUES(1, 'old.md', 's', '', 'L1', 't', 'legacy body', 0, 1.0)"
    )
    legacy.commit()
    legacy.close()

    con = connect(p)
    row = con.execute("SELECT uses FROM chunks WHERE id=1").fetchone()
    assert row == (0,), "legacy chunks row must get uses=0 via the tolerant ALTER TABLE migration"


# ---------------------------------------------------------------------------
# Phase 3 -- memory graph (edges) + personalized PageRank blend
#
# Design deviations from the original spec, per supervisor direction:
#  1. Edges are ALWAYS fully rebuilt at the end of index() when anything
#     changed (changed/removed/rebuild/model_changed), never incrementally.
#     A changed file's chunks are deleted and re-inserted with brand-new
#     autoincrement ids, so "delete just this file's edges" would leave
#     dangling edges at dead cids and stale inter-file temporal links. Full
#     rebuild is cheap and trivially correct at this scale.
#  2. Edge construction is one ordered SELECT (source, path, id) producing
#     three rule sets: intra-file chain (w=1.0), inter-file temporal chain
#     per source within 1h (w=0.5, between each file's first chunk), and
#     notes-only [[wikilinks]] (w=2.0, chunk -> target file's first chunk).
# ---------------------------------------------------------------------------

class _FakeEmbedder:
    def __init__(self, dim):
        self.dim = dim

    def embed(self, texts):
        return [np.ones(self.dim, dtype=np.float32) for _ in texts]


@pytest.fixture()
def edge_index_env(brain_home, monkeypatch):
    """Deterministic embedder, no turbovec/fastembed -- lets real index() run
    (to exercise the edges-rebuild trigger, incl. chunk delete/reinsert on
    file change) without touching the real embedding stack."""
    monkeypatch.setattr(indexer, "embed_model", lambda: "fake-model")
    monkeypatch.setattr(indexer, "embed_dim", lambda: 4)
    monkeypatch.setattr(indexer, "_embedder", lambda: _FakeEmbedder(4))
    monkeypatch.setattr(indexer, "_rebuild_turbovec", lambda con, p: None)
    return brain_home


def _write_md(path: Path, sections: list[tuple[str, str]]) -> None:
    body = "\n\n".join(f"## {title}\n{text}" for title, text in sections)
    path.write_text(body, encoding="utf-8")


def test_index_builds_intra_file_chain_and_no_dangling_edges_after_reindex(edge_index_env):
    p = edge_index_env
    p.notes.mkdir(parents=True, exist_ok=True)
    f1 = p.notes / "file1.md"
    f2 = p.notes / "file2.md"
    _write_md(f1, [("A", "alpha one"), ("B", "alpha two"), ("C", "alpha three")])
    _write_md(f2, [("D", "beta one"), ("E", "beta two"), ("F", "beta three")])

    indexer.index(p=p)

    con = connect(p)
    ids_f1 = [r[0] for r in con.execute("SELECT id FROM chunks WHERE path=? ORDER BY id", (str(f1),))]
    assert len(ids_f1) == 3
    a, b, c = ids_f1
    assert con.execute("SELECT w FROM edges WHERE a=? AND b=?", (a, b)).fetchone() == (1.0,)
    assert con.execute("SELECT w FROM edges WHERE a=? AND b=?", (b, c)).fetchone() == (1.0,)

    # Modify file1 so its chunks are deleted and reinserted with brand-new ids.
    _write_md(f1, [("A2", "alpha ONE changed"), ("B2", "alpha two"), ("C2", "alpha three")])
    indexer.index(p=p)

    con = connect(p)
    dangling = con.execute(
        "SELECT COUNT(*) FROM edges WHERE a NOT IN (SELECT id FROM chunks) OR b NOT IN (SELECT id FROM chunks)"
    ).fetchone()[0]
    assert dangling == 0, "no edge may reference a chunk id that no longer exists after reindex"
    assert a not in {r[0] for r in con.execute("SELECT id FROM chunks")}, "old file1 chunk ids must be gone"


def test_wikilink_creates_edge_to_target_first_chunk(edge_index_env):
    p = edge_index_env
    p.notes.mkdir(parents=True, exist_ok=True)
    target = p.notes / "target.md"
    linker = p.notes / "linker.md"
    _write_md(target, [("Target Note", "This is the target body.")])
    _write_md(linker, [("Linker", "See [[target]] for details."), ("More", "second section")])

    indexer.index(p=p)

    con = connect(p)
    target_first_id = con.execute(
        "SELECT id FROM chunks WHERE path=? ORDER BY id LIMIT 1", (str(target),)
    ).fetchone()[0]
    linker_chunk_id = con.execute(
        "SELECT id FROM chunks WHERE path=? AND text LIKE '%[[target]]%'", (str(linker),)
    ).fetchone()[0]

    a, b = sorted((linker_chunk_id, target_first_id))
    row = con.execute("SELECT w FROM edges WHERE a=? AND b=?", (a, b)).fetchone()
    assert row == (2.0,), f"expected a w=2.0 wikilink edge between {linker_chunk_id} and {target_first_id}, got {row}"


def test_ppr_blend_boosts_node_with_reinforcing_neighbor(brain_home, monkeypatch):
    """Synthetic graph: A(1)-B(2) strong edge, B is never a search candidate
    itself; C(3) is isolated. A and C start with an equal fused RRF score
    (one RRF leg each). cid 3 is inserted into the fused dict first (vec leg)
    so a pre-fix tie would stable-sort it first -- must flip to [1, 3] once
    the PPR blend is active, since B continuously feeds mass back to A while
    isolated C only ever gets flat teleport mass."""
    p = brain_home
    con = connect(p)
    now = time.time()
    make_chunk(con, 1, "notes", "", "node A", trust=1.0, mtime=now, path="a.md")
    make_chunk(con, 2, "notes", "", "node B", trust=1.0, mtime=now, path="b.md")
    make_chunk(con, 3, "notes", "", "node C", trust=1.0, mtime=now, path="c.md")
    con.commit()
    con.execute("INSERT INTO edges(a, b, w) VALUES(1, 2, 5.0)")
    con.commit()

    monkeypatch.setattr(indexer, "_vec_search", lambda query, limit, p: [(3, 0.0)])
    monkeypatch.setattr(indexer, "_bm25", lambda con, tokens, limit, project="", source="": [(1, 0.0)])

    hits = indexer.search("q", k=2, p=p)
    ids = [h.id for h in hits]

    assert ids == [1, 3], f"A (reinforced via neighbor B) should outrank isolated C, got {ids}"
    assert 2 not in ids, "B was never a fused candidate and must not appear as a result"


def test_no_regression_when_edges_present_but_unrelated_to_candidates(brain_home, monkeypatch):
    """With the edges table non-empty but containing no edge touching any
    fused candidate, the PPR blend must be a strict no-op: ranking must match
    the edges-table-empty case exactly, for the same seed data."""
    p = brain_home
    con = connect(p)
    now = time.time()
    make_chunk(con, 1, "notes", "", "alpha", trust=1.0, mtime=now, path="a.md")
    make_chunk(con, 2, "notes", "", "beta", trust=1.2, mtime=now - 5 * 86400, path="b.md")
    make_chunk(con, 3, "sessions", "", "gamma", trust=1.0, mtime=now - 40 * 86400, path="c.md")
    con.commit()

    monkeypatch.setattr(indexer, "_vec_search", lambda query, limit, p: [(1, 0.0), (2, 0.0)])
    monkeypatch.setattr(indexer, "_bm25", lambda con, tokens, limit, project="", source="": [(3, 0.0)])

    assert con.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
    baseline = [h.id for h in indexer.search("q", k=3, p=p)]

    # Neither endpoint of this edge is a fused candidate (98/99 don't even
    # exist as chunks) -- the neighbor lookup for {1,2,3} must return nothing.
    con.execute("INSERT INTO edges(a, b, w) VALUES(98, 99, 1.0)")
    con.commit()

    with_unrelated_edges = [h.id for h in indexer.search("q", k=3, p=p)]
    assert with_unrelated_edges == baseline


def test_personalized_pagerank_no_edges_converges_to_teleport():
    teleport = {1: 0.5, 2: 0.0, 3: 0.5}
    result = indexer._personalized_pagerank([1, 2, 3], [], teleport, iters=8, damping=0.85)
    assert result[1] == pytest.approx(0.5)
    assert result[2] == pytest.approx(0.0)
    assert result[3] == pytest.approx(0.5)


def test_personalized_pagerank_hand_calculable_one_iteration():
    # nodes 1=A, 2=B, 3=C; edge A-B w=1.0; C isolated; teleport split A/C.
    # By hand: W_norm = [[0,1,0],[1,0,0],[0,0,0]] (idx order A,B,C); r0=t=[.5,0,.5].
    # Wr = [0, .5, 0]; dangling_mass = r0[C] = .5.
    # r1 = 0.85*([0,.5,0] + .5*[.5,0,.5]) + 0.15*[.5,0,.5] = [.2875, .425, .2875]
    result = indexer._personalized_pagerank([1, 2, 3], [(1, 2, 1.0)], {1: 0.5, 3: 0.5}, iters=1, damping=0.85)
    assert result[1] == pytest.approx(0.2875)
    assert result[2] == pytest.approx(0.425)
    assert result[3] == pytest.approx(0.2875)


# ---------------------------------------------------------------------------
# Phase 4a -- near-duplicate detection (find_near_duplicates / cc-brain dedupe)
# ---------------------------------------------------------------------------

def test_find_near_duplicates_finds_planted_pair_cross_path(brain_home):
    p = brain_home
    con = connect(p)
    make_chunk(con, 1, "notes", "", "alpha", path="a.md")
    make_chunk(con, 2, "notes", "", "beta", path="b.md")
    make_chunk(con, 3, "notes", "", "gamma", path="c.md")
    con.commit()
    _embed(con, 1, [1, 0, 0, 0])
    _embed(con, 2, [1, 0, 0, 0])  # near-identical to 1, different path
    _embed(con, 3, [0, 1, 0, 0])  # orthogonal, no match
    con.commit()

    pairs = indexer.find_near_duplicates(threshold=0.95, limit=50, p=p)
    assert len(pairs) == 1
    cid_a, path_a, cid_b, path_b, cos = pairs[0]
    assert {cid_a, cid_b} == {1, 2}
    assert cos == pytest.approx(1.0)


def test_find_near_duplicates_ignores_same_path_and_self_pairs(brain_home):
    p = brain_home
    con = connect(p)
    make_chunk(con, 1, "notes", "", "alpha one", path="same.md")
    make_chunk(con, 2, "notes", "", "alpha two", path="same.md")  # same path as 1
    con.commit()
    _embed(con, 1, [1, 0, 0, 0])
    _embed(con, 2, [1, 0, 0, 0])
    con.commit()

    pairs = indexer.find_near_duplicates(threshold=0.95, limit=50, p=p)
    assert pairs == [], "same-path chunks must never be reported, even if their embeddings are identical"


def test_find_near_duplicates_respects_threshold(brain_home):
    p = brain_home
    con = connect(p)
    make_chunk(con, 1, "notes", "", "a", path="a.md")
    make_chunk(con, 2, "notes", "", "b", path="b.md")
    con.commit()
    _embed(con, 1, [1.0, 0.0, 0.0, 0.0])
    _embed(con, 2, [0.9, (1 - 0.81) ** 0.5, 0.0, 0.0])  # cos(1, 2) == 0.9 exactly
    con.commit()

    assert indexer.find_near_duplicates(threshold=0.95, limit=50, p=p) == []
    pairs = indexer.find_near_duplicates(threshold=0.85, limit=50, p=p)
    assert len(pairs) == 1
    assert pairs[0][4] == pytest.approx(0.9)


def test_find_near_duplicates_respects_limit(brain_home):
    p = brain_home
    con = connect(p)
    # 4 chunks, all pairwise-identical embeddings, distinct paths -> C(4,2)=6 candidate pairs.
    for i in range(1, 5):
        make_chunk(con, i, "notes", "", f"chunk {i}", path=f"f{i}.md")
    con.commit()
    for i in range(1, 5):
        _embed(con, i, [1, 0, 0, 0])
    con.commit()

    pairs = indexer.find_near_duplicates(threshold=0.95, limit=3, p=p)
    assert len(pairs) == 3


def test_doctor_warns_on_many_duplicates_in_sample(brain_home, monkeypatch):
    """This compares vectors already on disk (find_near_duplicates never
    touches _embedder -- see its source: only SELECTs from `embeddings`).
    The sample is injected via _random_embedded_cids (monkeypatched here)
    instead of relying on real ORDER BY RANDOM()."""
    p = brain_home
    con = connect(p)
    n = 7  # C(7,2) = 21 pairs > the 20-pair warning threshold, all identical.
    for i in range(1, n + 1):
        make_chunk(con, i, "notes", "", f"chunk {i}", path=f"f{i}.md")
    con.commit()
    for i in range(1, n + 1):
        _embed(con, i, [1, 0, 0, 0])
    con.commit()

    sample = list(range(1, n + 1))
    monkeypatch.setattr(indexer, "_random_embedded_cids", lambda con, limit=256: sample)

    report = indexer.doctor(p=p)
    assert any("duplicate" in w for w in report["warnings"])


def test_doctor_no_warning_for_clean_corpus(brain_home, monkeypatch):
    p = brain_home
    con = connect(p)
    make_chunk(con, 1, "notes", "", "alpha", path="a.md")
    make_chunk(con, 2, "notes", "", "beta", path="b.md")
    con.commit()
    _embed(con, 1, [1, 0, 0, 0])
    _embed(con, 2, [0, 1, 0, 0])
    con.commit()

    monkeypatch.setattr(indexer, "_random_embedded_cids", lambda con, limit=256: [1, 2])

    report = indexer.doctor(p=p)
    assert not any("duplicate" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# Phase 4b -- submodular packing of project_snapshot's related chunks
# ---------------------------------------------------------------------------

def test_pack_related_chunks_prefers_diverse_over_duplicate_when_budget_forces_a_cut(brain_home):
    """Same shape as the Phase 1 MMR test: 3 near-duplicate embeddings + 1
    orthogonal. Padded to ~1200 chars each so the 4000-char budget can only
    fit 3 of the 4 -- the diversity-aware reorder must keep the orthogonal
    chunk and drop the weakest (3rd) duplicate instead of a flat top-3 cut."""
    p = brain_home
    con = connect(p)
    _embed(con, 1, [1, 0, 0, 0])
    _embed(con, 2, [1, 0, 0, 0])
    _embed(con, 3, [1, 0, 0, 0])
    _embed(con, 4, [0, 1, 0, 0])
    con.commit()

    pad = "x" * 1200
    hits = [
        indexer.SearchHit(1, 0.04, "notes", "p1.md", "L1", "t", pad, ""),
        indexer.SearchHit(2, 0.03, "notes", "p2.md", "L1", "t", pad, ""),
        indexer.SearchHit(3, 0.02, "notes", "p3.md", "L1", "t", pad, ""),
        indexer.SearchHit(4, 0.01, "notes", "p4.md", "L1", "t", pad, ""),
    ]

    selected = indexer._pack_related_chunks(con, hits, k=4)
    ids = [h.id for h in selected]
    assert 4 in ids, f"budget-constrained packing must keep the diverse chunk, got {ids}"
    assert 3 not in ids, f"the weakest near-duplicate should be dropped first, got {ids}"
    assert sum(len(h.text) for h in selected) <= 4000


def test_pack_related_chunks_respects_budget(brain_home):
    p = brain_home
    con = connect(p)
    for i in range(1, 4):
        _embed(con, i, [float(i), 0, 0, 0])
    con.commit()

    big = "y" * 2500
    hits = [
        indexer.SearchHit(1, 0.03, "notes", "a.md", "L1", "t", big, ""),
        indexer.SearchHit(2, 0.02, "notes", "b.md", "L1", "t", big, ""),
        indexer.SearchHit(3, 0.01, "notes", "c.md", "L1", "t", big, ""),
    ]
    selected = indexer._pack_related_chunks(con, hits, k=3)
    assert len(selected) < len(hits), "large chunks must not all fit under the 4000-char budget"
    assert sum(len(h.text) for h in selected) <= 4000


def test_pack_related_chunks_falls_back_without_embeddings(brain_home):
    p = brain_home
    con = connect(p)
    # No embeddings inserted at all for these ids.
    hits = [
        indexer.SearchHit(1, 0.04, "notes", "a.md", "L1", "t", "x" * 3000, ""),
        indexer.SearchHit(2, 0.03, "notes", "b.md", "L1", "t", "y" * 3000, ""),
        indexer.SearchHit(3, 0.02, "notes", "c.md", "L1", "t", "z" * 3000, ""),
    ]
    selected = indexer._pack_related_chunks(con, hits, k=2)
    assert [h.id for h in selected] == [1, 2], "no embeddings -> fall back to the flat top-k, budget ignored"
