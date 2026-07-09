from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

from .config import BrainPaths, load_sources, paths, project_from_path, slug, upsert_source
from .contracts import SearchHit, SourceSpec
from .runtime import device_mode, requested_providers
from .runtime import status as runtime_status
from .store import clear_dirty, connect, meta_get, meta_set
from .text import chunk_file, tokenize

_PREFERRED_MODELS = (
    "intfloat/multilingual-e5-large",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-base-en-v1.5",
)
_MODEL = None
_DIM = None
_EMBEDDER = None


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


def normalize_project(value: str) -> str:
    return slug((value or "").replace("_", "-").lower(), "")


def register_repo(path: str | Path, name: str = "", project: str = "") -> SourceSpec:
    root = Path(path).expanduser().resolve()
    source_name = slug(name or f"repo-{root.name}")
    spec = SourceSpec(source_name, root, kind="code", project=project or project_from_path(root), trust=1.0)
    return upsert_source(spec)


def _source_project(src: SourceSpec, path: Path, text: str = "") -> str:
    if src.project:
        return normalize_project(src.project)
    if src.name == "private-ingest":
        try:
            rel = path.relative_to(src.root)
            if rel.parts:
                return normalize_project(rel.parts[0])
        except ValueError:
            pass
    if src.name == "sessions":
        m = re.match(r"\d{4}-\d{2}-\d{2}__(.*?)__", path.name)
        if m:
            return normalize_project(m.group(1))
    m = re.search(r"^project:\s*(.+)$", text[:1200], re.M)
    if m:
        return normalize_project(m.group(1))
    return normalize_project(path.parent.name if src.name in ("notes", "web") else src.root.name)


def _iter_files(src: SourceSpec):
    if not src.root.exists() or not src.root.is_dir():
        return
    inc = re.compile(src.include, re.I) if src.include else None
    exc = re.compile(src.exclude, re.I) if src.exclude else None
    for base, dirs, names in os.walk(src.root):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__", "dist", "build", "target", ".venv", "venv")]
        for name in names:
            full = Path(base) / name
            rel = str(full.relative_to(src.root))
            rel_norm = rel.replace(os.sep, "/")
            if inc and not inc.search(rel_norm):
                continue
            if exc and exc.search(rel_norm):
                continue
            yield full


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    from fastembed import TextEmbedding
    cache = Path(os.environ.get("FASTEMBED_CACHE_PATH", str(paths().home / ".fastembed_cache")))
    providers = requested_providers()
    _EMBEDDER = TextEmbedding(embed_model(), providers=providers, cache_dir=str(cache))
    active = embedder_providers()
    if device_mode() == "gpu" and "CUDAExecutionProvider" not in active:
        raise RuntimeError("CC_BRAIN_DEVICE=gpu requested, but fastembed did not activate CUDAExecutionProvider")
    return _EMBEDDER


def embedder_providers() -> list[str]:
    model = _embedder()
    try:
        return list(model.model.model.get_providers())
    except Exception:
        return []


def require_vector() -> None:
    missing: list[str] = []
    for mod in ("numpy", "turbovec", "fastembed"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        raise RuntimeError(
            "cc-brain requires turbovec search. Missing dependencies: "
            + ", ".join(missing)
            + ". Install with `python -m pip install -e .`."
        )


def vector_available() -> bool:
    try:
        require_vector()
        return True
    except RuntimeError:
        return False


def index(rebuild: bool = False, p: BrainPaths | None = None) -> str:
    require_vector()
    p = p or paths()
    con = connect(p)
    if rebuild:
        for table in ("files", "chunks", "postings", "embeddings"):
            con.execute(f"DELETE FROM {table}")
        con.commit()

    stored_model = meta_get(con, "embed_model")
    stored_dim = meta_get(con, "embed_dim")
    model_changed = bool(stored_model) and (stored_model != embed_model() or stored_dim != str(embed_dim()))
    if model_changed:
        con.execute("DELETE FROM embeddings")
        con.commit()

    sources = load_sources(p)
    known = {Path(row[0]): f"{row[1]}:{row[2]}" for row in con.execute("SELECT path, mtime, size FROM files")}
    seen: set[Path] = set()
    changed: list[tuple[SourceSpec, Path, os.stat_result]] = []
    missing_roots: list[str] = []
    for src in sources:
        if not src.root.exists():
            missing_roots.append(str(src.root))
            continue
        for full in _iter_files(src) or []:
            seen.add(full)
            st = full.stat()
            sig = f"{st.st_mtime}:{st.st_size}"
            if known.get(full) != sig:
                changed.append((src, full, st))
    removed = [x for x in known if x not in seen]

    for full in removed + [x[1] for x in changed]:
        ids = [r[0] for r in con.execute("SELECT id FROM chunks WHERE path=?", (str(full),))]
        if ids:
            q = ",".join("?" * len(ids))
            con.execute(f"DELETE FROM postings WHERE chunk IN ({q})", ids)
            con.execute(f"DELETE FROM embeddings WHERE chunk IN ({q})", ids)
            con.execute(f"DELETE FROM chunks WHERE id IN ({q})", ids)
        con.execute("DELETE FROM files WHERE path=?", (str(full),))
    con.commit()

    new_chunks: list[tuple[int, str]] = []
    for src, full, st in changed:
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > 300_000:
            con.execute("INSERT OR REPLACE INTO files(path, source, project, mtime, size) VALUES(?,?,?,?,?)",
                        (str(full), src.name, src.project, st.st_mtime, st.st_size))
            continue
        project = _source_project(src, full, text)
        for loc, title, body in chunk_file(full, text):
            cur = con.execute(
                "INSERT INTO chunks(path, source, project, loc, title, text, mtime, trust) VALUES(?,?,?,?,?,?,?,?)",
                (str(full), src.name, project, loc, title, body, st.st_mtime, src.trust),
            )
            cid = int(cur.lastrowid)
            counts: dict[str, int] = {}
            for tok in tokenize(title + "\n" + body):
                counts[tok] = counts.get(tok, 0) + 1
            con.executemany("INSERT INTO postings(term, chunk, tf) VALUES(?,?,?)", [(t, cid, n) for t, n in counts.items()])
            new_chunks.append((cid, f"{title}\n{body}"))
        con.execute("INSERT OR REPLACE INTO files(path, source, project, mtime, size) VALUES(?,?,?,?,?)",
                    (str(full), src.name, project, st.st_mtime, st.st_size))
    con.commit()

    if model_changed:
        have = {cid for cid, _ in new_chunks}
        for cid, title, body in con.execute("SELECT id, title, text FROM chunks"):
            if cid not in have:
                new_chunks.append((cid, f"{title}\n{body}"))

    import numpy as np
    model = _embedder()
    if new_chunks:
        for i in range(0, len(new_chunks), 256):
            batch = new_chunks[i:i + 256]
            vecs = list(model.embed([text[:2400] for _, text in batch]))
            con.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk, vec) VALUES(?,?)",
                [(cid, np.asarray(vec, dtype=np.float32).tobytes()) for (cid, _), vec in zip(batch, vecs)],
            )
            con.commit()

    if rebuild or changed or removed or model_changed or not p.tv.exists():
        _rebuild_turbovec(con, p)
        vector_note = "[index] vector: turbovec rebuilt"
    else:
        vector_note = "[index] vector: unchanged, rebuild skipped"

    meta_set(con, "embed_model", embed_model())
    meta_set(con, "embed_dim", str(embed_dim()))
    meta_set(con, "last_index", str(time.time()))
    con.commit()

    clear_dirty(p)
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    lines = [
        f"[index] {len(changed)} changed/new file(s), {len(removed)} removed",
        f"[index] {len(new_chunks)} new chunk(s) stored",
        vector_note,
        f"[index] DONE. {total} chunks total.",
    ]
    if missing_roots:
        lines.append(f"[index] missing roots: {len(missing_roots)}")
    return "\n".join(lines)


def _rebuild_turbovec(con, p: BrainPaths) -> None:
    import numpy as np
    from turbovec import TurboQuantIndex
    rows = con.execute("SELECT chunk, vec FROM embeddings ORDER BY chunk").fetchall()
    if not rows:
        return
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
    mat = mat.reshape(len(rows), embed_dim()).copy()
    idx = TurboQuantIndex(dim=embed_dim(), bit_width=4)
    idx.add(mat)
    tv_tmp = str(p.tv) + ".tmp"
    ids_tmp = str(p.ids) + ".tmp"
    idx.write(tv_tmp)
    Path(ids_tmp).write_text(json.dumps({
        "ids": [r[0] for r in rows],
        "model": embed_model(),
        "dim": embed_dim(),
    }), encoding="utf-8")
    os.replace(tv_tmp, p.tv)
    os.replace(ids_tmp, p.ids)


def _bm25(con, tokens: list[str], limit: int, project: str = "", source: str = "") -> list[tuple[int, float]]:
    project = normalize_project(project)
    filtered = bool(project or source)
    if filtered:
        q = "SELECT COUNT(*) FROM chunks c WHERE 1=1"
        args: list[str] = []
        if project:
            q += " AND c.project=?"
            args.append(project)
        if source:
            q += " AND c.source=?"
            args.append(source)
        n = con.execute(q, args).fetchone()[0] or 1
        avg_q = "SELECT AVG(n) FROM (SELECT SUM(p.tf) n FROM postings p JOIN chunks c ON c.id=p.chunk WHERE 1=1"
        avg_args: list[str] = []
        if project:
            avg_q += " AND c.project=?"
            avg_args.append(project)
        if source:
            avg_q += " AND c.source=?"
            avg_args.append(source)
        avg_q += " GROUP BY p.chunk)"
        avg = con.execute(avg_q, avg_args).fetchone()[0] or 1.0
    else:
        n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 1
        avg = con.execute("SELECT AVG(n) FROM (SELECT SUM(tf) n FROM postings GROUP BY chunk)").fetchone()[0] or 1.0
    contribs: list[tuple[int, int, float]] = []  # (cid, tf, idf)
    for tok in set(tokens):
        if filtered:
            q = "SELECT p.chunk, p.tf FROM postings p JOIN chunks c ON c.id=p.chunk WHERE p.term=?"
            args = [tok]
            if project:
                q += " AND c.project=?"
                args.append(project)
            if source:
                q += " AND c.source=?"
                args.append(source)
            rows = con.execute(q, args).fetchall()
        else:
            rows = con.execute("SELECT chunk, tf FROM postings WHERE term=?", (tok,)).fetchall()
        # Common-term pruning only makes sense at scale: in a small (often
        # filtered) universe a term matching most docs is usually the target,
        # not a stopword, so the 50% rule gets an absolute floor.
        if not rows or len(rows) > max(n * 0.5, 16):
            continue
        idf = math.log(1 + (n - len(rows) + 0.5) / (len(rows) + 0.5))
        for cid, tf in rows:
            contribs.append((cid, tf, idf))
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


def _vec_search(query: str, limit: int, p: BrainPaths) -> list[tuple[int, float]]:
    if not (p.tv.exists() and p.ids.exists() and vector_available()):
        return []
    try:
        import numpy as np
        from turbovec import TurboQuantIndex
        idx = TurboQuantIndex.load(str(p.tv))
        raw = json.loads(p.ids.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if raw.get("model") and raw["model"] != embed_model():
                return []
            ids = raw["ids"]
        else:
            ids = raw
        q = np.asarray(list(_embedder().embed([query]))[0], dtype=np.float32).reshape(1, -1)
        scores, rows = idx.search(q, k=min(limit, len(ids)))
        return [(ids[int(r)], float(s)) for s, r in zip(scores[0], rows[0]) if 0 <= int(r) < len(ids)]
    except Exception:
        return []


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


def get(ids: list[int], p: BrainPaths | None = None) -> list[SearchHit]:
    ids = list(ids)[:24]
    p = p or paths()
    con = connect(p)
    out: list[SearchHit] = []
    for cid in ids:
        row = con.execute("SELECT source, path, loc, title, text, project FROM chunks WHERE id=?", (int(cid),)).fetchone()
        if row:
            out.append(SearchHit(int(cid), 0.0, row[0], row[1], row[2] or "", row[3] or "", row[4] or "", row[5] or ""))
    return out


def stats(p: BrainPaths | None = None) -> dict:
    p = p or paths()
    con = connect(p)
    per_source = {r[0]: r[1] for r in con.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY 2 DESC")}
    per_project = {r[0] or "-": r[1] for r in con.execute("SELECT project, COUNT(*) FROM chunks GROUP BY project ORDER BY 2 DESC LIMIT 30")}
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
    hits = search(f"{proj} current status next step handoff", k=k, project=proj, lex=True, p=p)
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


def doctor(p: BrainPaths | None = None) -> dict:
    p = p or paths()
    con = connect(p)
    sources = load_sources(p)
    missing = [str(s.root) for s in sources if not s.root.exists()]
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    vec_ok = vector_available()
    providers: list[str] = []
    provider_warning = ""
    rt = runtime_status()
    if vec_ok:
        try:
            providers = embedder_providers()
            if providers and "CUDAExecutionProvider" not in providers and device_mode() == "auto" and rt.gpu_detected:
                provider_warning = "CUDAExecutionProvider not active; turbovec works but embedding will be slower"
        except Exception as exc:
            provider_warning = f"embedder init failed: {type(exc).__name__}: {exc}"
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
    return {
        "home": str(p.home),
        "sources": len(sources),
        "missing_roots": missing,
        "chunks": total,
        "embeddings": embedded,
        "vector_available": vec_ok,
        "device_mode": rt.device_mode,
        "gpu_detected": rt.gpu_detected,
        "vendor_dir": rt.vendor_dir,
        "vendor_present": rt.vendor_present,
        "dll_dirs": rt.dll_dirs,
        "embedder_providers": providers,
        "provider_warning": provider_warning,
        "embed_model": embed_model(),
        "ids_count": ids_count,
        "warnings": warnings,
        "recommendation": "install turbovec dependencies" if not vec_ok else ("index" if total == 0 else "ok"),
    }
