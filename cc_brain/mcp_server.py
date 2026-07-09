from __future__ import annotations

import json
import threading
import time

from mcp.server.fastmcp import FastMCP

from . import __version__
from .config import load_sources, paths
from .config import paths as brain_paths
from .indexer import doctor as run_doctor
from .indexer import get as get_chunks
from .indexer import index as run_index
from .indexer import project_snapshot, register_repo
from .indexer import recent as run_recent
from .indexer import remove_source as run_remove_source
from .indexer import search as run_search
from .indexer import stats as run_stats
from .memory import write_note
from .store import connect, dirty_info, meta_get, meta_set
from .web import capture_web

mcp = FastMCP("cc-brain")
_LOCK = threading.Lock()
_LAST_INDEX = 0.0
_REFRESHING = False


def _record_refresh_error(message: str) -> None:
    """Persist (or clear, with "") the last background refresh failure so it
    survives process restarts and shows up in doctor()/health()."""
    try:
        con = connect(brain_paths())
        meta_set(con, "last_refresh_error", message)
        con.commit()
    except Exception:
        pass  # never let error bookkeeping take down the server


def _refresh_warning() -> str:
    try:
        err = meta_get(connect(brain_paths()), "last_refresh_error")
    except Exception:
        return ""
    if not err:
        return ""
    return f"(warning: last background index refresh FAILED: {err} — results may be stale; see doctor())"


def _index_if_dirty() -> str:
    global _LAST_INDEX, _REFRESHING
    warning = _refresh_warning()
    info = dirty_info()
    if not info or _REFRESHING or time.time() - _LAST_INDEX < 30:
        return warning
    def _run():
        global _LAST_INDEX, _REFRESHING
        try:
            with _LOCK:
                run_index()
                _LAST_INDEX = time.time()
                _record_refresh_error("")
        except Exception as exc:
            _record_refresh_error(f"{type(exc).__name__}: {exc}")
        finally:
            _REFRESHING = False
    _REFRESHING = True
    threading.Thread(target=_run, daemon=True).start()
    note = "(vault dirty: index refresh started in background; results may be a few seconds stale)"
    return f"{warning}\n{note}" if warning else note


@mcp.tool()
def ping() -> str:
    """Fast liveness probe. Does not touch the embedder or index."""
    return f"pong cc-brain {__version__}"


@mcp.tool()
def health() -> str:
    """Return cc-brain health, source count and turbovec readiness."""
    return json.dumps(run_doctor(), indent=2, ensure_ascii=False)


@mcp.tool()
def index(rebuild: bool = False) -> str:
    """Index all registered sources into SQLite BM25 plus mandatory turbovec."""
    global _LAST_INDEX
    with _LOCK:
        out = run_index(rebuild=rebuild)
        _LAST_INDEX = time.time()
        return out


@mcp.tool()
def search(query: str, k: int = 6, source: str = "", project: str = "", lex: bool = False) -> str:
    """Search the brain. Uses turbovec+BM25 hybrid unless lex=True."""
    note = _index_if_dirty()
    hits = run_search(query, k=k, source=source, project=project, lex=lex)
    if not hits:
        return (note + "\n" if note else "") + "(no hits)"
    lines = [note] if note else []
    for h in hits:
        snippet = " ".join(h.text.split())[:180]
        proj = f" project={h.project}" if h.project else ""
        lines.append(f"[{h.id}] {h.score:.3f} {h.source}:{h.path}#{h.loc}{proj}\n    {snippet}")
    return "\n".join(lines)


@mcp.tool()
def get(ids: list[int]) -> str:
    """Expand search result ids to full chunk text."""
    rows = get_chunks(ids)
    if not rows:
        return "(no chunks)"
    return "\n\n".join(f"===== [{h.id}] {h.source}:{h.path}#{h.loc} =====\n{h.text}" for h in rows)


@mcp.tool()
def note(name: str, content: str) -> str:
    """Save curated knowledge as a brain note and mark the vault dirty."""
    path = write_note(name, content)
    return f"saved {path}"


@mcp.tool()
def add_repo(path: str, name: str = "", project: str = "") -> str:
    """Register a local repository/source for indexing."""
    spec = register_repo(path, name=name, project=project)
    return f"registered {spec.name}: {spec.root} project={spec.project}"


@mcp.tool()
def add_web(url: str, project: str = "") -> str:
    """Fetch and persist a web page into the ingest/web source."""
    try:
        path = capture_web(url, project=project)
    except Exception as exc:
        return f"failed to capture {url}: {exc}"
    return f"captured {url} -> {path}"


@mcp.tool()
def sources() -> str:
    """List registered brain sources."""
    out = []
    for src in load_sources(paths()):
        out.append(f"{src.name}: {src.root} kind={src.kind} project={src.project or '-'}")
    return "\n".join(out)


@mcp.tool()
def project_state(project: str, k: int = 8) -> str:
    """Freshest project memory: newest session atom + recent commits + related chunks."""
    note = _index_if_dirty()
    body = project_snapshot(project, k=k)
    return (note + "\n" if note else "") + body


@mcp.tool()
def doctor() -> str:
    """Check source roots, chunk counts, embeddings and turbovec availability."""
    return json.dumps(run_doctor(), indent=2, ensure_ascii=False)


@mcp.tool()
def stats() -> str:
    """Brain size and freshness: chunks per source/project, model, last index time."""
    return json.dumps(run_stats(), indent=2, ensure_ascii=False)


@mcp.tool()
def recent(project: str = "", limit: int = 10) -> str:
    """Most recently indexed files, newest first. Good freshness probe."""
    rows = run_recent(project, limit)
    return "\n".join(f"{r['source']}:{r['path']} project={r['project'] or '-'}" for r in rows) or "(empty)"


@mcp.tool()
def remove_source(name: str) -> str:
    """Unregister a source and delete its chunks from the brain."""
    return run_remove_source(name)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
