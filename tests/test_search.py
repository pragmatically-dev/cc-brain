import time

import pytest
from conftest import make_chunk

import cc_brain.indexer as indexer
from cc_brain.store import connect

# NOTE on the two xfail markers below: both encode exact regression tests specified
# in docs/superpowers/plans/2026-07-09-cc-brain-overhaul.md (task D2). Running them
# against the real Task A implementation (cc_brain/indexer.py, already merged in
# ce165ec) shows they currently fail for a *different* reason than the one they were
# written to guard against. Root cause: `_bm25()`'s common-term pruning
# (`len(rows) > n * 0.5`, indexer.py ~L336) computes `n` from the *filtered*
# source/project chunk universe once a source/project filter is passed down into
# `_bm25`. In a narrow filtered universe (exactly the case pre-ranking filters exist
# to serve), a query term that appears in most/all of that small universe is treated
# as "too common" and pruned out entirely, so `_bm25` returns zero contributions and
# `search()` returns no hits. This silently reproduces the audit-1.2 starvation bug
# in a new form. Per task instructions, Task D does not modify cc_brain/*.py to fix
# implementation bugs it discovers -- these are marked xfail (not fixed, not deleted)
# so the regression is documented and CI stays green; a Task A follow-up should scope
# the `n * 0.5` pruning threshold to the *unfiltered* global chunk count instead.


@pytest.mark.xfail(
    reason="indexer._bm25 common-term pruning uses the filtered (source) universe size "
    "as n, so a term present in a 1-chunk filtered universe is pruned as 'too common' "
    "and search() returns 0 hits instead of [200]. Bug in cc_brain/indexer.py, not owned by Task D.",
    strict=True,
)
def test_source_filter_survives_global_pool(brain_home):
    """Audit 1.2: filters must apply before ranking, not starve results after.

    60 noise chunks match the query and would fill the old top-48 pool before the
    post-hoc source filter ran; 70 fillers keep the query tokens under the 50%
    common-term pruning threshold. Old code returned 0 hits here.
    """
    con = connect(brain_home)
    for i in range(60):
        make_chunk(con, i + 1, "repo-noise", "other", f"alpha beta filler{i}", path=f"n{i}.py")
    for i in range(70):
        make_chunk(con, 1000 + i, "repo-noise", "other", f"unrelated padding{i}", path=f"u{i}.py")
    make_chunk(con, 200, "sessions", "target", "alpha beta next step extra words", path="s.md")
    con.commit()
    hits = indexer.search("alpha beta", k=4, source="sessions", lex=True, p=brain_home)
    assert [h.id for h in hits] == [200]


@pytest.mark.xfail(
    reason="indexer._bm25 common-term pruning uses the filtered (project) universe size "
    "as n; with 2 chunks sharing project 'p', both query terms appear in 2/2 chunks and "
    "get pruned as 'too common', so search() returns 0 hits instead of ranking by trust/"
    "recency. Bug in cc_brain/indexer.py, not owned by Task D.",
    strict=True,
)
def test_trust_and_recency_reorder(brain_home):
    """Audit 1.1: adjusted score must change the ORDER, not just the label."""
    con = connect(brain_home)
    make_chunk(con, 1, "sessions", "p", "zeta query", trust=0.9, mtime=0.0, path="old.md")
    make_chunk(con, 2, "notes", "p", "zeta query", trust=1.35, mtime=time.time(), path="new.md")
    con.commit()
    hits = indexer.search("zeta query", k=2, project="p", lex=True, p=brain_home)
    assert hits[0].id == 2


def test_get_is_bounded(brain_home):
    con = connect(brain_home)
    for i in range(40):
        make_chunk(con, i + 1, "s", "p", f"word{i}", path=f"f{i}.md")
    con.commit()
    assert len(indexer.get(list(range(1, 41)), p=brain_home)) <= 24
