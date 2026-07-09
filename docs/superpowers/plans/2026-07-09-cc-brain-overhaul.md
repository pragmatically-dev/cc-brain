# cc-brain Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every correctness bug found in the 2026-07-09 audit (dead trust/recency ranking, post-ranking filter starvation, web project poisoning, wrong hook payload field, lost user messages) and land the usefulness upgrades (multilingual embeddings with model-migration guard, deterministic project_state, PreCompact capture, HTML-to-text web ingest, source management commands, chunk overlap, expanded secret scrubbing, tests + CI + LICENSE).

**Architecture:** Same contract-driven layout — all changes stay inside existing modules. Four task groups with disjoint file ownership so they can be implemented by parallel agents: (A) retrieval core `indexer.py`/`store.py`/`text.py`, (B) capture pipeline `hooks.py`/`transcript.py`/`web.py`/`memory.py`/`contracts.py`, (C) wiring `cli.py`/`mcp_server.py`/`installer.py`/`config.py`/`runtime.py`/`recall.py`, (D) `tests/`, CI, LICENSE, docs. C depends on A's new function signatures (specified below, implement exactly). D depends on A+B+C.

**Tech Stack:** Python 3.11+, stdlib only (no new runtime deps). sqlite3, fastembed, turbovec, mcp (FastMCP). pytest for tests. GitHub Actions for CI.

## Global Constraints

- Python `>=3.11`, no new runtime dependencies. Tests must NOT download embedding models and must pass without `turbovec` installed (skip/monkeypatch).
- All dataclasses stay `frozen=True`. Hooks must stay fail-safe (exit 0 on any error).
- Keep existing code style: no type-annotation churn, `from __future__ import annotations`, compact functions, no docstring bloat.
- Every file written with UTF-8. Windows is the primary platform; everything must also work on Linux (CI runs both).
- Commit at the end of your task with the message given in the task. Do not touch files owned by another task.
- Verify before committing: `python -m compileall cc_brain` and `python -c "import cc_brain.indexer, cc_brain.hooks, cc_brain.cli, cc_brain.mcp_server"` must succeed (mcp import may fail only if `mcp` package is missing in the env — then skip that import).

---

### Task A: Retrieval core — ranking, filters, performance, model guard

**Files:**
- Modify: `cc_brain/indexer.py`
- Modify: `cc_brain/store.py`
- Modify: `cc_brain/text.py`

**Interfaces (Produces — Task C wires these into CLI/MCP exactly as specified):**
- `indexer.search(query, k=6, source="", project="", lex=False, p=None) -> list[SearchHit]` (unchanged signature, fixed behavior)
- `indexer.get(ids, p=None) -> list[SearchHit]` (caps at 24 ids)
- `indexer.stats(p=None) -> dict`
- `indexer.recent(project="", limit=10, p=None) -> list[dict]` with keys `source, path, project, mtime`
- `indexer.project_snapshot(project, k=8, p=None) -> str`
- `indexer.remove_source(name, p=None) -> str`
- `store.meta_get(con, key, default="") -> str`, `store.meta_set(con, key, value) -> None`

#### A1. `store.py`: busy timeout + meta helpers

- [ ] In `connect()`, right after the WAL pragma add `con.execute("PRAGMA busy_timeout=5000")`.
- [ ] Add at module level:

```python
def meta_get(con, key: str, default: str = "") -> str:
    row = con.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    return row[0] if row else default


def meta_set(con, key: str, value: str) -> None:
    con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES(?,?)", (key, str(value)))
```

#### A2. `indexer.py`: embedding model selection + migration guard

- [ ] Replace the `EMBED_MODEL`/`EMBED_DIM` constants with lazy resolution. Multilingual default (the user's content is Spanish): prefer a multilingual model that the installed fastembed supports, fall back to the old default. Dim comes from fastembed metadata, env can still override both:

```python
_PREFERRED_MODELS = (
    "intfloat/multilingual-e5-large",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-base-en-v1.5",
)
_MODEL = None
_DIM = None


def _supported_models() -> dict[str, int]:
    from fastembed import TextEmbedding
    out = {}
    for m in TextEmbedding.list_supported_models():
        name = m.get("model") if isinstance(m, dict) else getattr(m, "model", "")
        dim = m.get("dim") if isinstance(m, dict) else getattr(m, "dim", 0)
        if name:
            out[name] = int(dim or 0)
    return out


def embed_model() -> str:
    global _MODEL
    if _MODEL:
        return _MODEL
    env = os.environ.get("CC_BRAIN_EMBED_MODEL", "").strip()
    if env:
        _MODEL = env
        return _MODEL
    try:
        supported = _supported_models()
        _MODEL = next((m for m in _PREFERRED_MODELS if m in supported), _PREFERRED_MODELS[-1])
    except Exception:
        _MODEL = _PREFERRED_MODELS[-1]
    return _MODEL


def embed_dim() -> int:
    global _DIM
    if _DIM:
        return _DIM
    env = os.environ.get("CC_BRAIN_EMBED_DIM", "").strip()
    if env:
        _DIM = int(env)
        return _DIM
    try:
        _DIM = _supported_models().get(embed_model(), 0) or 768
    except Exception:
        _DIM = 768
    return _DIM
```

  Replace every use of `EMBED_MODEL`/`EMBED_DIM` in the module with `embed_model()`/`embed_dim()` (in `_embedder()` and `_rebuild_turbovec`).
- [ ] In `index()`, before scanning files, add the migration guard — if the stored model/dim differ from the current ones, wipe embeddings and force a full re-embed:

```python
from .store import clear_dirty, connect, meta_get, meta_set  # update import

    stored_model = meta_get(con, "embed_model")
    stored_dim = meta_get(con, "embed_dim")
    model_changed = bool(stored_model) and (stored_model != embed_model() or stored_dim != str(embed_dim()))
    if model_changed:
        con.execute("DELETE FROM embeddings")
        con.commit()
```

  When `model_changed` is true, ALL existing chunks need re-embedding, not just changed files. After the `new_chunks` list is built from changed files, add the orphans:

```python
    if model_changed:
        have = {r[0] for r in con.execute("SELECT chunk FROM embeddings")}
        for cid, title, body in con.execute("SELECT id, title, text FROM chunks"):
            if cid not in have and all(cid != c for c, _ in new_chunks):
                new_chunks.append((cid, f"{title}\n{body}"))
```

  (Simpler equivalent allowed: since embeddings were wiped, `have` is what got inserted this pass — implement whichever is cleaner, behavior must be "every chunk ends with an embedding of the current model".)
  After a successful pass: `meta_set(con, "embed_model", embed_model())`, `meta_set(con, "embed_dim", str(embed_dim()))`, `meta_set(con, "last_index", str(time.time()))`, then `con.commit()`.

#### A3. `indexer.py`: skip turbovec rebuild when nothing changed

- [ ] Replace the unconditional `_rebuild_turbovec(con, p)` call with:

```python
    if rebuild or changed or removed or model_changed or not p.tv.exists():
        _rebuild_turbovec(con, p)
        vector_note = "[index] vector: turbovec rebuilt"
    else:
        vector_note = "[index] vector: unchanged, rebuild skipped"
```

#### A4. `indexer.py`: versioned ids sidecar

- [ ] In `_rebuild_turbovec`, write the ids file as a dict (readers must accept the legacy plain-list format too):

```python
    Path(ids_tmp).write_text(json.dumps({
        "ids": [r[0] for r in rows],
        "model": embed_model(),
        "dim": embed_dim(),
    }), encoding="utf-8")
```

  Replace `EMBED_DIM` with `embed_dim()` in the reshape and `TurboQuantIndex(dim=...)`.
- [ ] In `_vec_search`, load either format and reject stale sidecars:

```python
        raw = json.loads(p.ids.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if raw.get("model") and raw["model"] != embed_model():
                return []
            ids = raw["ids"]
        else:
            ids = raw
```

#### A5. `indexer.py`: BM25 — source filter + batched doc lengths

- [ ] Change signature to `_bm25(con, tokens, limit, project="", source="")`. Add `AND c.source=?` to both filtered queries when `source` is set (always join `chunks c` when either filter is present; keep the unfiltered fast path when neither is set). The `n`/`avg` statistics queries get the same filters.
- [ ] Kill the per-row `SELECT SUM(tf)` doc-length query. After collecting all `(cid, tf, idf)` contributions, fetch lengths once in batches:

```python
    contribs: list[tuple[int, int, float]] = []  # (cid, tf, idf) accumulated in the token loop
    ...
    cids = list({c for c, _, _ in contribs})
    dl: dict[int, int] = {}
    for i in range(0, len(cids), 500):
        batch = cids[i:i + 500]
        q = ",".join("?" * len(batch))
        for cid, s in con.execute(f"SELECT chunk, SUM(tf) FROM postings WHERE chunk IN ({q}) GROUP BY chunk", batch):
            dl[cid] = s or 1
    scores: dict[int, float] = {}
    for cid, tf, idf in contribs:
        d = dl.get(cid, 1)
        scores[cid] = scores.get(cid, 0.0) + idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * d / avg))
    return sorted(scores.items(), key=lambda x: -x[1])[:limit]
```

#### A6. `indexer.py`: `search()` — pre-ranking filters + trust/recency that actually reorders

- [ ] Replace the body of `search()` with:

```python
def _allowed_ids(con, source: str, project: str) -> set[int] | None:
    if not source and not project:
        return None
    q = "SELECT id FROM chunks WHERE 1=1"
    args: list[str] = []
    if source:
        q += " AND source=?"
        args.append(source)
    if project:
        q += " AND project=?"
        args.append(normalize_project(project))
    return {r[0] for r in con.execute(q, args)}


def search(query: str, k: int = 6, source: str = "", project: str = "", lex: bool = False,
           p: BrainPaths | None = None) -> list[SearchHit]:
    require_vector()
    p = p or paths()
    con = connect(p)
    pool = max(k * 8, 48)
    allowed = _allowed_ids(con, source, project)
    fused: dict[int, float] = {}
    if not lex:
        vec_pool = pool if allowed is None else min(pool * 6, 512)
        rank = 0
        for cid, _ in _vec_search(query, vec_pool, p):
            if allowed is not None and cid not in allowed:
                continue
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
            rank += 1
            if rank >= pool:
                break
    for rank, (cid, _) in enumerate(_bm25(con, tokenize(query), pool, project, source)):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
    now = time.time()
    candidates: list[tuple[float, int, tuple]] = []
    for cid, score in fused.items():
        row = con.execute(
            "SELECT source, path, loc, title, text, project, trust, mtime FROM chunks WHERE id=?", (cid,)
        ).fetchone()
        if not row:
            continue
        if source and row[0] != source:
            continue
        if project and normalize_project(project) != normalize_project(row[5] or ""):
            continue
        age_days = max(0.0, (now - float(row[7] or 0.0)) / 86400.0) if row[7] else 999.0
        adjusted = float(score) + 0.02 * float(row[6] or 1.0) + 0.015 * math.exp(-age_days / 21.0)
        candidates.append((adjusted, cid, row))
    candidates.sort(key=lambda x: -x[0])
    out: list[SearchHit] = []
    per_path: dict[str, int] = {}
    for adjusted, cid, row in candidates:
        src, path, loc, title, text, proj = row[0], row[1], row[2], row[3], row[4], row[5]
        if per_path.get(path, 0) >= 2 and not source:
            continue
        per_path[path] = per_path.get(path, 0) + 1
        try:
            rel = str(Path(path).relative_to(p.home))
        except ValueError:
            rel = path
        out.append(SearchHit(int(cid), adjusted, src, rel, loc or "", title or "", text or "", proj or ""))
        if len(out) >= k:
            break
    return out
```

#### A7. `indexer.py`: bounded `get`, new `stats`/`recent`/`project_snapshot`/`remove_source`

- [ ] `get()`: first line becomes `ids = list(ids)[:24]`.
- [ ] Add:

```python
def stats(p: BrainPaths | None = None) -> dict:
    p = p or paths()
    con = connect(p)
    per_source = {r[0]: r[1] for r in con.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY 2 DESC")}
    per_project = {r[0] or "-": r[1] for r in con.execute("SELECT project, COUNT(*) FROM chunks GROUP BY project ORDER BY 2 DESC LIMIT 30")}
    from .store import meta_get
    return {
        "chunks": con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        "files": con.execute("SELECT COUNT(*) FROM files").fetchone()[0],
        "embeddings": con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0],
        "per_source": per_source,
        "per_project": per_project,
        "db_bytes": p.db.stat().st_size if p.db.exists() else 0,
        "tv_bytes": p.tv.stat().st_size if p.tv.exists() else 0,
        "embed_model": meta_get(con, "embed_model") or embed_model(),
        "last_index": meta_get(con, "last_index"),
    }


def recent(project: str = "", limit: int = 10, p: BrainPaths | None = None) -> list[dict]:
    p = p or paths()
    con = connect(p)
    q = "SELECT source, path, project, mtime FROM files"
    args: list[str] = []
    if project:
        q += " WHERE project=?"
        args.append(normalize_project(project))
    q += " ORDER BY mtime DESC LIMIT ?"
    args.append(int(limit))
    return [{"source": r[0], "path": r[1], "project": r[2] or "", "mtime": r[3]} for r in con.execute(q, args)]


def project_snapshot(project: str, k: int = 8, p: BrainPaths | None = None) -> str:
    p = p or paths()
    con = connect(p)
    proj = normalize_project(project)
    parts: list[str] = []
    row = con.execute(
        "SELECT path FROM files WHERE source='sessions' AND project=? ORDER BY mtime DESC LIMIT 1", (proj,)
    ).fetchone()
    if row:
        atom = Path(row[0])
        if atom.exists():
            parts.append(f"## Last session atom ({atom.name})\n" + atom.read_text(encoding="utf-8", errors="replace")[:3200])
    commit_log = p.commits / f"{slug(project)}.md"
    if commit_log.exists():
        tail = commit_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-12:]
        parts.append("## Recent commits\n" + "\n".join(tail))
    hits = search(f"{proj} current status next step handoff", k=k, project=proj, lex=True)
    if hits:
        lines = [f"- [{h.id}] {h.source}:{h.path}#{h.loc} " + " ".join(h.text.split())[:200] for h in hits]
        parts.append("## Related chunks (use get(ids) to expand)\n" + "\n".join(lines))
    return "\n\n".join(parts) if parts else f"(no memory for project {proj!r} yet — index first?)"


def remove_source(name: str, p: BrainPaths | None = None) -> str:
    from .config import load_sources, save_sources
    p = p or paths()
    sources = load_sources(p)
    keep = [s for s in sources if s.name != name]
    if len(keep) == len(sources):
        return f"source {name!r} not found"
    save_sources(keep, p)
    con = connect(p)
    ids = [r[0] for r in con.execute("SELECT id FROM chunks WHERE source=?", (name,))]
    for i in range(0, len(ids), 500):
        batch = ids[i:i + 500]
        q = ",".join("?" * len(batch))
        con.execute(f"DELETE FROM postings WHERE chunk IN ({q})", batch)
        con.execute(f"DELETE FROM embeddings WHERE chunk IN ({q})", batch)
        con.execute(f"DELETE FROM chunks WHERE id IN ({q})", batch)
    con.execute("DELETE FROM files WHERE source=?", (name,))
    con.commit()
    if ids:
        _rebuild_turbovec(con, p)
    return f"removed source {name!r} ({len(ids)} chunks)"
```

  Add `slug` to the imports from `.config`.
- [ ] `doctor()`: after computing `embedded`, add consistency checks to the returned dict:

```python
    ids_count = 0
    if p.ids.exists():
        try:
            raw = json.loads(p.ids.read_text(encoding="utf-8"))
            ids_count = len(raw["ids"] if isinstance(raw, dict) else raw)
        except (json.JSONDecodeError, KeyError, TypeError):
            ids_count = -1
    warnings = []
    if ids_count >= 0 and ids_count != embedded:
        warnings.append(f"turbovec sidecar has {ids_count} ids but DB has {embedded} embeddings; run index --rebuild")
    stored_model = meta_get(con, "embed_model")
    if stored_model and stored_model != embed_model():
        warnings.append(f"index built with {stored_model} but current model is {embed_model()}; next index() re-embeds everything")
```

  Include `"embed_model": embed_model()`, `"ids_count": ids_count`, `"warnings": warnings` in the result. Import `meta_get` from `.store`.

#### A8. `text.py`: chunk overlap + expanded scrub

- [ ] Code chunking with overlap (keeps chunks anchored, no duplicate tail):

```python
CODE_WINDOW = 80
CODE_OVERLAP = 20


def chunk_code(text: str):
    lines = text.split("\n")
    step = CODE_WINDOW - CODE_OVERLAP
    start = 0
    while start < len(lines):
        end = min(start + CODE_WINDOW, len(lines))
        body = "\n".join(lines[start:end])
        if body.strip():
            yield (f"L{start + 1}-{end}", f"L{start + 1}", body)
        if end >= len(lines):
            break
        start += step
```

- [ ] Extend `SECRET_RE` into a list of patterns applied in order by `scrub` (keep the function signature):

```python
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(
        r"(AQ\.[\w\-]{10,}|nvapi-[\w\-]{10,}|AIza[\w\-]{20,}|sk-[A-Za-z0-9\-_]{20,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[\w\-]{10,}|AKIA[0-9A-Z]{16})"
    ),
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b(\s*[=:]\s*)([\"']?)[^\s\"',;]{8,}\3"),
    re.compile(r"\b([a-z][a-z0-9+.\-]*://[^\s:@/]+):([^\s@/]+)@"),
)


def scrub(text: str) -> str:
    out = text or ""
    out = SECRET_PATTERNS[0].sub("[REDACTED-KEY-BLOCK]", out)
    out = SECRET_PATTERNS[1].sub("[REDACTED]", out)
    out = SECRET_PATTERNS[2].sub("[REDACTED-JWT]", out)
    out = SECRET_PATTERNS[3].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)
    out = SECRET_PATTERNS[4].sub(lambda m: f"{m.group(1)}:[REDACTED]@", out)
    return out
```

  Keep `SECRET_RE` as an alias for `SECRET_PATTERNS[1]` if anything else imports it (grep first).

- [ ] **Verify:** `python -m compileall cc_brain` passes; `python -c "from cc_brain import indexer, text, store"` passes.
- [ ] **Commit:** `git add -A && git commit -m "fix(retrieval): effective trust/recency ranking, pre-ranking filters, model migration guard, batched BM25, chunk overlap, scrub hardening"`

---

### Task B: Capture pipeline — hook payloads, transcripts, web ingest

**Files:**
- Modify: `cc_brain/hooks.py`
- Modify: `cc_brain/transcript.py`
- Modify: `cc_brain/web.py`
- Modify: `cc_brain/contracts.py`
- Modify: `cc_brain/memory.py` (only if needed; likely untouched)

**Interfaces:**
- Consumes: nothing new from Task A.
- Produces: `hooks.main()` now also dispatches `PreCompact`; `web._html_to_text(html) -> str` (Task D tests it); hook behavior below.

#### B1. `hooks.py`: read `tool_response` (real Claude Code field) with fallback

- [ ] Add helper and use it in both `post_tool_use` sites (WebFetch output and git-commit output):

```python
def _tool_output(data: dict):
    return data.get("tool_response", data.get("tool_output"))
```

#### B2. `hooks.py`: only mark dirty when something actually changed

- [ ] `_register_cwd` currently rewrites `sources.json` and marks the vault dirty on EVERY session start and EVERY prompt, which forces a full reindex on the next MCP search. Make it register+dirty only for genuinely new sources:

```python
def _register_cwd(cwd: str) -> None:
    if not cwd:
        return
    root = Path(cwd)
    if not (root.exists() and root.is_dir()):
        return
    from .config import load_sources
    name = slug(f"repo-{root.name}")
    if any(s.name == name for s in load_sources()):
        return
    register_repo(root, name=f"repo-{root.name}", project=root.name)
    mark_dirty("repo-new", root)
```

#### B3. `hooks.py`: PreCompact capture + debug logging

- [ ] In `main()`, dispatch `PreCompact` to the same transcript capture as `SessionEnd`:

```python
    elif event in ("SessionEnd", "PreCompact"):
        session_end(data)
```

  (Replace the existing `elif event == "SessionEnd"` branch.)
- [ ] At the top of `main()` after `data = _payload()`, add opt-in debug logging:

```python
    if os.environ.get("CC_BRAIN_DEBUG"):
        log_dir = paths().home / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "hooks.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
```

#### B4. `transcript.py`: user messages with list content

- [ ] In the `typ == "user"` branch, also accept `text` blocks (Claude Code writes both shapes). Replace the inner loop with:

```python
                for kind, block in _blocks(data.get("message", {})):
                    if kind == "str":
                        text = block.strip()
                    elif kind == "text" and isinstance(block, dict):
                        text = (block.get("text") or "").strip()
                    else:
                        continue
                    norm = " ".join(text.split()).lower()[:240]
                    if len(text) > 2 and not text.startswith(INJECTED_PREFIXES) and norm not in seen_asks:
                        info["user_requests"].append(scrub(text)[:700])
                        info["turns"] += 1
                        seen_asks.add(norm)
```

#### B5. `web.py`: stop project poisoning, add UA, HTML→text, error handling

- [ ] Full replacement of `web.py`:

```python
from __future__ import annotations

import hashlib
import html as _html
import re
import urllib.request
from pathlib import Path

from .config import ensure_dirs, paths, slug
from .store import mark_dirty
from .text import scrub

_TAG_BLOCKS = re.compile(r"<(script|style|noscript|svg|head)\b.*?</\1>", re.S | re.I)
_TAGS = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"\n{3,}")


def _html_to_text(body: str) -> str:
    if "<" not in body[:2000].lower():
        return body
    if not re.search(r"<(html|body|div|p|br|span|h[1-6])\b", body[:4000], re.I):
        return body
    out = _TAG_BLOCKS.sub(" ", body)
    out = re.sub(r"<(br|/p|/div|/h[1-6]|/li|/tr)\b[^>]*>", "\n", out, flags=re.I)
    out = _TAGS.sub(" ", out)
    out = _html.unescape(out)
    out = "\n".join(" ".join(line.split()) for line in out.splitlines())
    return _BLANKS.sub("\n\n", out).strip()


def capture_web(url: str, content: str = "", project: str = "") -> Path:
    p = ensure_dirs(paths())
    digest = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()[:12]
    host = re.sub(r"^https?://", "", url).split("/", 1)[0]
    name = f"{slug(host, 'web')}__{digest}.md"
    path = p.web / name
    body = (content or "").strip()
    if not body:
        req = urllib.request.Request(url, headers={"User-Agent": "cc-brain/0.2 (+local knowledge vault)"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read(700_000)
        except OSError as exc:
            raise RuntimeError(f"fetch failed for {url}: {exc}") from exc
        body = raw.decode("utf-8", errors="replace")
    body = _html_to_text(body)
    md = f"---\nurl: {url}\nproject: {project}\nprovenance: web\n---\n\n# {url}\n\n{scrub(body)[:250000]}\n"
    path.write_text(md, encoding="utf-8", newline="\n")
    mark_dirty("web", path, p)
    return path
```

  Note the removed `upsert_source(...)` call — that line was retagging EVERY web page with the last capture's project. The per-file `project:` frontmatter (already parsed by `indexer._source_project`) is the single source of truth now.

#### B6. `contracts.py`: rename the dead-wrong field

- [ ] In `HookPayload`, rename `tool_output` → `tool_response` (grep confirms nothing constructs it with `tool_output=` today).

- [ ] **Verify:** `python -m compileall cc_brain` passes; run `echo '{}' | python -m cc_brain.hooks` — must exit 0 silently.
- [ ] **Commit:** `git add -A && git commit -m "fix(capture): read tool_response, PreCompact capture, dirty only on change, web HTML-to-text + UA + no project poisoning, list-content user msgs"`

---

### Task C: Wiring — CLI, MCP, installer, config, runtime, recall

**Files:**
- Modify: `cc_brain/cli.py`
- Modify: `cc_brain/mcp_server.py`
- Modify: `cc_brain/installer.py`
- Modify: `cc_brain/config.py`
- Modify: `cc_brain/runtime.py`
- Modify: `cc_brain/recall.py`

**Interfaces:**
- Consumes from Task A (already merged when you start): `indexer.stats()`, `indexer.recent(project, limit)`, `indexer.project_snapshot(project, k)`, `indexer.remove_source(name)`.
- Produces: CLI subcommands `stats`, `recent`, `remove-source`, `notes`, `uninstall`; MCP tools `stats`, `recent`, `remove_source`; `installer.uninstall() -> Path`.

#### C1. `config.py` + `runtime.py`: survive non-editable installs

- [ ] `config.py`: `REPO_ROOT` currently points into site-packages for non-editable installs, so `ensure_dirs` would create `ingest/` inside site-packages. Guard it:

```python
def _repo_root() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    return root if (root / "pyproject.toml").exists() else None
```

  In `paths()`: `private_ingest=(_repo_root() / "ingest") if _repo_root() else home / "private-ingest"`. Keep the module-level `REPO_ROOT` name only if something imports it (grep; `runtime.py` has its own copy — fix that separately below).
- [ ] `runtime.py`: same guard for the vendor dir:

```python
def _default_vendor() -> Path:
    root = Path(__file__).resolve().parents[1]
    if (root / "pyproject.toml").exists():
        return root / "vendor" / "python"
    import os as _os
    home = Path(_os.environ.get("CC_BRAIN_HOME", "~/.cc-brain")).expanduser()
    return home / "vendor" / "python"
```

  `vendor_dir()` uses `_default_vendor()` instead of the module constant.

#### C2. `installer.py`: PreCompact, stale-hook dedupe, uninstall

- [ ] `_add_hook` dedupe should match any existing cc-brain hook (even from another Python path) and update it in place rather than duplicating:

```python
def _add_hook(settings: dict, event: str, command: str, matcher: str = "") -> None:
    hooks = settings.setdefault("hooks", {})
    entries = hooks.setdefault(event, [])
    for entry in entries:
        for hook in entry.get("hooks", []):
            if "cc_brain.hooks" in str(hook.get("command", "")) and entry.get("matcher", "") == matcher:
                hook["command"] = command
                return
    item = {"hooks": [{"type": "command", "command": command}]}
    if matcher:
        item["matcher"] = matcher
    entries.append(item)
```

- [ ] In `install()`, also add: `_add_hook(settings, "PreCompact", hook_cmd)`.
- [ ] Add `uninstall()`:

```python
def uninstall() -> Path:
    sp = _settings_path()
    if not sp.exists():
        return sp
    try:
        settings = json.loads(sp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return sp
    hooks = settings.get("hooks", {})
    for event in list(hooks):
        kept = []
        for entry in hooks[event]:
            inner = [h for h in entry.get("hooks", []) if "cc_brain.hooks" not in str(h.get("command", ""))]
            if inner:
                entry["hooks"] = inner
                kept.append(entry)
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event)
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, sp)
    return sp
```

#### C3. `cli.py`: new subcommands

- [ ] Register and implement (import `project_snapshot, recent, remove_source, stats` from `.indexer`, `uninstall` from `.installer`):

```python
    sub.add_parser("stats")
    sub.add_parser("uninstall")

    p_recent = sub.add_parser("recent")
    p_recent.add_argument("--project", default="")
    p_recent.add_argument("-n", type=int, default=10)

    p_rm = sub.add_parser("remove-source")
    p_rm.add_argument("name")

    sub.add_parser("notes")

    p_state = sub.add_parser("project-state")
    p_state.add_argument("project")
```

```python
    elif args.cmd == "stats":
        print(json.dumps(stats(), indent=2, ensure_ascii=False))
    elif args.cmd == "recent":
        for row in recent(args.project, args.n):
            print(f"{row['mtime']:.0f}  {row['source']:14s} {row['project']:20s} {row['path']}")
    elif args.cmd == "remove-source":
        print(remove_source(args.name))
    elif args.cmd == "notes":
        for f in sorted(paths().notes.glob("*.md")):
            first = f.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            print(f"{f.stem}: {first[0][:100] if first else ''}")
    elif args.cmd == "project-state":
        print(project_snapshot(args.project))
    elif args.cmd == "uninstall":
        print(f"removed cc-brain hooks from {uninstall()}")
```

#### C4. `mcp_server.py`: background dirty-refresh, new tools, deterministic project_state, safe add_web

- [ ] Replace `_index_if_dirty` with a non-blocking background refresh (a search must never pay a multi-minute synchronous index):

```python
_REFRESHING = False


def _index_if_dirty() -> str:
    global _LAST_INDEX, _REFRESHING
    info = dirty_info()
    if not info or _REFRESHING or time.time() - _LAST_INDEX < 30:
        return ""
    def _run():
        global _LAST_INDEX, _REFRESHING
        try:
            with _LOCK:
                run_index()
                _LAST_INDEX = time.time()
        except Exception:
            pass
        finally:
            _REFRESHING = False
    _REFRESHING = True
    threading.Thread(target=_run, daemon=True).start()
    return "(vault dirty: index refresh started in background; results may be a few seconds stale)"
```

- [ ] `project_state` becomes deterministic:

```python
@mcp.tool()
def project_state(project: str, k: int = 8) -> str:
    """Freshest project memory: newest session atom + recent commits + related chunks."""
    note = _index_if_dirty()
    body = project_snapshot(project, k=k)
    return (note + "\n" if note else "") + body
```

  Import `project_snapshot` from `.indexer`.
- [ ] New tools:

```python
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
```

  Import as `from .indexer import stats as run_stats, recent as run_recent, remove_source as run_remove_source`.
- [ ] `add_web` catches failure:

```python
@mcp.tool()
def add_web(url: str, project: str = "") -> str:
    """Fetch and persist a web page into the ingest/web source."""
    try:
        path = capture_web(url, project=project)
    except Exception as exc:
        return f"failed to capture {url}: {exc}"
    return f"captured {url} -> {path}"
```

#### C5. `recall.py`: use the deterministic snapshot at session start

- [ ] `session_start_context` should lead with the newest session atom instead of hoping lexical search finds it:

```python
def session_start_context(cwd: str | Path, cap: int = 4200) -> str:
    project = project_from_path(Path(cwd))
    body = indexer.project_snapshot(project, k=4)
    if body.startswith("(no memory"):
        return ""
    return (
        f"[cc-brain] Persistent memory for {project}. Use the cc-brain MCP search/get tools for drill-down.\n"
        f"{body}"
    )[:cap]
```

- [ ] **Verify:** `python -m compileall cc_brain`; `python -m cc_brain --version`; `python -m cc_brain stats` runs (may show zeros); `echo '{}' | python -m cc_brain.hooks` exits 0.
- [ ] **Commit:** `git add -A && git commit -m "feat(wiring): stats/recent/remove-source/notes/uninstall, deterministic project_state, background dirty refresh, PreCompact install, non-editable install paths"`

---

### Task D: Tests, CI, LICENSE, docs, version bump

**Files:**
- Create: `tests/conftest.py`, `tests/test_text.py`, `tests/test_transcript.py`, `tests/test_hooks.py`, `tests/test_search.py`, `tests/test_web.py`, `tests/test_config.py`, `tests/test_installer.py`
- Create: `.github/workflows/ci.yml`, `LICENSE`
- Modify: `README.md`, `llm.md`, `docs/INSTALL.md`, `docs/ARCHITECTURE.md`, `skills/cc-brain/SKILL.md`, `pyproject.toml`, `cc_brain/__init__.py`

**Interfaces:** Consumes everything from A/B/C. Tests import only `cc_brain.*` public functions named in those tasks.

#### D1. Test scaffolding

Rules: no network, no model downloads, no turbovec requirement. Everything runs against a tmp `CC_BRAIN_HOME`.

- [ ] `tests/conftest.py`:

```python
import json
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
```

#### D2. The regression tests (write all of these; they encode the audit bugs)

- [ ] `tests/test_search.py`:

```python
import time

import cc_brain.indexer as indexer
from cc_brain.store import connect
from conftest import make_chunk


def test_source_filter_survives_global_pool(brain_home):
    """Audit 1.2: filters must apply before ranking, not starve results after.

    60 noise chunks match the query and would fill the old top-48 pool before the
    post-hoc source filter ran; 70 fillers keep the query tokens under the 50%%
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
```

- [ ] `tests/test_web.py`:

```python
from cc_brain.config import load_sources
from cc_brain.web import _html_to_text, capture_web


def test_html_to_text_strips_tags():
    out = _html_to_text("<html><head><style>x{}</style></head><body><h1>Title</h1><p>Hello <b>world</b></p><script>evil()</script></body></html>")
    assert "Title" in out and "Hello world" in out
    assert "<" not in out and "evil" not in out


def test_capture_web_does_not_poison_source_project(brain_home):
    capture_web("https://example.com/a", content="plain text", project="proj-a")
    web = [s for s in load_sources(brain_home) if s.name == "web"]
    assert web and web[0].project == ""


def test_capture_web_frontmatter_has_project(brain_home):
    path = capture_web("https://example.com/b", content="hola", project="proj-b")
    assert "project: proj-b" in path.read_text(encoding="utf-8")
```

- [ ] `tests/test_hooks.py`:

```python
import json

import cc_brain.hooks as hooks


def test_tool_output_prefers_tool_response():
    assert hooks._tool_output({"tool_response": "R", "tool_output": "O"}) == "R"
    assert hooks._tool_output({"tool_output": "O"}) == "O"


def test_precompact_dispatches_capture(monkeypatch, brain_home):
    called = {}
    monkeypatch.setattr(hooks, "capture_transcript", lambda tp, cwd: called.setdefault("tp", tp))
    monkeypatch.setattr(hooks, "_register_cwd", lambda cwd: None)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(
        {"hook_event_name": "PreCompact", "transcript_path": "t.jsonl", "cwd": "."}
    )))
    hooks.main()
    assert called["tp"] == "t.jsonl"


def test_commit_capture_reads_tool_response(monkeypatch, brain_home, tmp_path):
    recorded = {}
    monkeypatch.setattr(hooks, "append_commit", lambda proj, entry: recorded.setdefault("entry", entry))
    monkeypatch.setattr(hooks.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})())
    hooks.post_tool_use({
        "tool_name": "Bash", "cwd": str(tmp_path),
        "tool_input": {"command": "git commit -m 'x'"},
        "tool_response": "[main abc1234] x",
    })
    assert "abc1234" in recorded["entry"]
```

- [ ] `tests/test_transcript.py` — build a small JSONL fixture inline (`tmp_path / "t.jsonl"`) with: one string-content user message, one list-content user message (`{"type":"text","text":"segunda pregunta"}`), one injected message starting with `<system-reminder>`, one assistant message with a `tool_use` Edit block and a Bash `git commit -m "feat: done"` command, one assistant text block > 40 chars. Assert: both real user messages captured, injected skipped, commit extracted, file captured, `memory_name` format `YYYY-MM-DD__project__session8`, and `build_l1_markdown` contains `## What happened`, `## User asked`, `evidence:`.
- [ ] `tests/test_text.py` — assert each scrub pattern redacts (AKIA key, JWT `eyJx.eyJy.zz` style with realistic lengths, `password = "hunter2secret"`, `postgres://user:pass@host`, PEM block) and that plain code (`sk_test` short strings, normal prose) is untouched; `chunk_code` overlap: 200-line input → first chunk `L1-80`, second starts at `L61`; last line of file appears in the final chunk exactly once per chunk containing it.
- [ ] `tests/test_config.py` — round-trip `load_sources`/`save_sources` under `brain_home`; defaults auto-added; `upsert_source` replaces by name.
- [ ] `tests/test_installer.py` — monkeypatch `installer._settings_path` to a tmp file: `install()` is idempotent (run twice → same hook count), updates a stale hook with a different python path instead of duplicating (pre-seed a `{"type":"command","command":"\"C:/old/python.exe\" -m cc_brain.hooks"}` entry), adds `PreCompact`, and `uninstall()` removes only cc-brain entries (pre-seed one unrelated hook and assert it survives).

#### D3. CI, LICENSE, metadata

- [ ] `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push: {branches: [main]}
  pull_request:
jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python: ["3.11", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "${{ matrix.python }}"}
      - run: python -m pip install --upgrade pip
      - run: pip install numpy pytest ruff
      - run: pip install mcp fastembed
        continue-on-error: true
      - run: pip install turbovec
        continue-on-error: true
      - run: pip install -e . --no-deps
      - run: ruff check cc_brain tests
      - run: pytest -q
```

  Tests must therefore not hard-import `mcp` or `fastembed` at collection time (they don't, if you only test the modules listed above — do NOT import `cc_brain.mcp_server` in tests).
- [ ] `LICENSE`: MIT, copyright `2026 pragmatically-dev`.
- [ ] `pyproject.toml`: bump version `0.2.0`, add `license = {text = "MIT"}`, and a minimal ruff config:

```toml
[tool.ruff]
line-length = 140
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = ["E402"]
```

  Run `ruff check cc_brain tests` locally if ruff is available and fix trivial violations (unused imports, import order); if ruff is not installed, `pip install ruff` in the venv.
- [ ] `cc_brain/__init__.py`: `__version__ = "0.2.0"`.

#### D4. Docs refresh

- [ ] `README.md`: update the MCP tools table (add `stats`, `recent`, `remove_source`; describe the new deterministic `project_state`), the CLI list (add `stats`, `recent`, `remove-source`, `notes`, `project-state`, `uninstall`), the hooks table (add `PreCompact`), and replace the "Turbovec Is Required" embedding claim with a short **Embeddings** section: default model is auto-selected preferring multilingual (`intfloat/multilingual-e5-large` when available — important for non-English sessions), override with `CC_BRAIN_EMBED_MODEL`; changing models triggers an automatic full re-embed on next index.
- [ ] `llm.md`: mirror the same tool/command additions in the expected-tools list and hard usage rules; add rule "prefer `project_state` (deterministic) at session resume; use `stats`/`recent` to check freshness before assuming staleness".
- [ ] `docs/INSTALL.md` + `docs/ARCHITECTURE.md`: add PreCompact to the hook list; note the background index refresh on dirty; document `remove-source` and `uninstall`.
- [ ] `skills/cc-brain/SKILL.md`: add `stats`/`recent`/`remove_source` to First Moves/Growth Rule where relevant.
- [ ] **Verify:** `pytest -q` all green locally. `ruff check cc_brain tests` clean.
- [ ] **Commit:** `git add -A && git commit -m "test+ci+docs: regression suite for audit bugs, GitHub Actions matrix, MIT license, docs for v0.2.0"`

---

## Execution order

1. Task A and Task B in parallel (disjoint files).
2. Task C after A+B are committed.
3. Task D after C is committed.
4. Orchestrator: full review of `git diff <baseline>..HEAD`, run `pytest`, fix anything, final verification.
