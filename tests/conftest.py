import pytest

import cc_brain.indexer as indexer
from cc_brain.config import paths


@pytest.fixture()
def brain_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CC_BRAIN_HOME", str(tmp_path / "brain"))
    monkeypatch.setattr(indexer, "require_vector", lambda: None)
    monkeypatch.setattr(indexer, "_vec_search", lambda query, limit, p: [])
    return paths()


def make_chunk(con, cid, source, project, text, trust=1.0, mtime=0.0, path="f.md"):
    con.execute(
        "INSERT INTO chunks(id, path, source, project, loc, title, text, mtime, trust) VALUES(?,?,?,?,?,?,?,?,?)",
        (cid, path, source, project, "L1", text.split()[0], text, mtime, trust),
    )
    for tok in set(t.lower() for t in text.split()):
        con.execute("INSERT INTO postings(term, chunk, tf) VALUES(?,?,1)", (tok, cid))
