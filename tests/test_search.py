import time

from conftest import make_chunk

import cc_brain.indexer as indexer
from cc_brain.store import connect


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
