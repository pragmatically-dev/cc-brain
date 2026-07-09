from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

from .config import BrainPaths, load_sources, paths, project_from_path, slug, upsert_source
from .contracts import SearchHit, SourceSpec
from .runtime import device_mode, requested_providers, status as runtime_status
from .store import clear_dirty, connect
from .text import chunk_file, tokenize


EMBED_MODEL = os.environ.get("CC_BRAIN_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DIM = int(os.environ.get("CC_BRAIN_EMBED_DIM", "768"))

_EMBEDDER = None


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
    _EMBEDDER = TextEmbedding(EMBED_MODEL, providers=providers, cache_dir=str(cache))
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
    _rebuild_turbovec(con, p)
    vector_note = "[index] vector: turbovec rebuilt"

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
    mat = mat.reshape(len(rows), EMBED_DIM).copy()
    idx = TurboQuantIndex(dim=EMBED_DIM, bit_width=4)
    idx.add(mat)
    tv_tmp = str(p.tv) + ".tmp"
    ids_tmp = str(p.ids) + ".tmp"
    idx.write(tv_tmp)
    Path(ids_tmp).write_text(json.dumps([r[0] for r in rows]), encoding="utf-8")
    os.replace(tv_tmp, p.tv)
    os.replace(ids_tmp, p.ids)


def _bm25(con, tokens: list[str], limit: int, project: str = "") -> list[tuple[int, float]]:
    project = normalize_project(project)
    if project:
        n = con.execute("SELECT COUNT(*) FROM chunks WHERE project=?", (project,)).fetchone()[0] or 1
        avg = con.execute(
            "SELECT AVG(n) FROM (SELECT SUM(p.tf) n FROM postings p JOIN chunks c ON c.id=p.chunk WHERE c.project=? GROUP BY p.chunk)",
            (project,),
        ).fetchone()[0] or 1.0
    else:
        n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 1
        avg = con.execute("SELECT AVG(n) FROM (SELECT SUM(tf) n FROM postings GROUP BY chunk)").fetchone()[0] or 1.0
    scores: dict[int, float] = {}
    for tok in set(tokens):
        if project:
            rows = con.execute(
                "SELECT p.chunk, p.tf FROM postings p JOIN chunks c ON c.id=p.chunk WHERE p.term=? AND c.project=?",
                (tok, project),
            ).fetchall()
        else:
            rows = con.execute("SELECT chunk, tf FROM postings WHERE term=?", (tok,)).fetchall()
        if not rows or len(rows) > n * 0.5:
            continue
        idf = math.log(1 + (n - len(rows) + 0.5) / (len(rows) + 0.5))
        for cid, tf in rows:
            dl = con.execute("SELECT SUM(tf) FROM postings WHERE chunk=?", (cid,)).fetchone()[0] or 1
            scores[cid] = scores.get(cid, 0.0) + idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg))
    return sorted(scores.items(), key=lambda x: -x[1])[:limit]


def _vec_search(query: str, limit: int, p: BrainPaths) -> list[tuple[int, float]]:
    if not (p.tv.exists() and p.ids.exists() and vector_available()):
        return []
    try:
        import numpy as np
        from turbovec import TurboQuantIndex
        idx = TurboQuantIndex.load(str(p.tv))
        ids = json.loads(p.ids.read_text(encoding="utf-8"))
        q = np.asarray(list(_embedder().embed([query]))[0], dtype=np.float32).reshape(1, -1)
        scores, rows = idx.search(q, k=min(limit, len(ids)))
        return [(ids[int(r)], float(s)) for s, r in zip(scores[0], rows[0]) if 0 <= int(r) < len(ids)]
    except Exception:
        return []


def search(query: str, k: int = 6, source: str = "", project: str = "", lex: bool = False,
           p: BrainPaths | None = None) -> list[SearchHit]:
    require_vector()
    p = p or paths()
    con = connect(p)
    pool = max(k * 8, 48)
    bm = _bm25(con, tokenize(query), pool, project)
    ranked: list[tuple[int, float]]
    fused: dict[int, float] = {}
    if not lex:
        for rank, (cid, _) in enumerate(_vec_search(query, pool, p)):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
    for rank, (cid, _) in enumerate(bm):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    out: list[SearchHit] = []
    per_path: dict[str, int] = {}
    for cid, score in ranked:
        row = con.execute(
            "SELECT source, path, loc, title, text, project, trust, mtime FROM chunks WHERE id=?", (cid,)
        ).fetchone()
        if not row:
            continue
        src, path, loc, title, text, proj, trust, mtime = row
        if source and src != source:
            continue
        if project and normalize_project(project) != normalize_project(proj):
            continue
        age_days = max(0.0, (time.time() - float(mtime or 0.0)) / 86400.0) if mtime else 999.0
        adjusted = float(score) + 0.02 * float(trust or 1.0) + 0.015 * math.exp(-age_days / 21.0)
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
    p = p or paths()
    con = connect(p)
    out: list[SearchHit] = []
    for cid in ids:
        row = con.execute("SELECT source, path, loc, title, text, project FROM chunks WHERE id=?", (int(cid),)).fetchone()
        if row:
            out.append(SearchHit(int(cid), 0.0, row[0], row[1], row[2] or "", row[3] or "", row[4] or "", row[5] or ""))
    return out


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
        "recommendation": "install turbovec dependencies" if not vec_ok else ("index" if total == 0 else "ok"),
    }
