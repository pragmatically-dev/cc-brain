from __future__ import annotations

import json
import threading
import time

from mcp.server.fastmcp import FastMCP

from . import __version__
from .config import load_sources, paths, slug
from .indexer import doctor as run_doctor
from .indexer import get as get_chunks
from .indexer import index as run_index
from .indexer import register_repo, search as run_search
from .memory import write_note
from .store import dirty_info
from .web import capture_web


mcp = FastMCP("cc-brain")
_LOCK = threading.Lock()
_LAST_INDEX = 0.0


def _index_if_dirty() -> str:
    global _LAST_INDEX
    info = dirty_info()
    if not info:
        return ""
    with _LOCK:
        if time.time() - _LAST_INDEX < 2:
            return ""
        out = run_index()
        _LAST_INDEX = time.time()
        return out


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
    path = capture_web(url, project=project)
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
    """Return freshest/current chunks for a project."""
    q = f"{slug(project)} current status next step handoff checkpoint recent"
    return search(q, k=k, project=project)


@mcp.tool()
def doctor() -> str:
    """Check source roots, chunk counts, embeddings and turbovec availability."""
    return json.dumps(run_doctor(), indent=2, ensure_ascii=False)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
