from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import SourceSpec


DEFAULT_INCLUDE = SourceSpec("x", Path(".")).include
DEFAULT_EXCLUDE = SourceSpec("x", Path(".")).exclude
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BrainPaths:
    home: Path
    data: Path
    notes: Path
    commits: Path
    ingest: Path
    private_ingest: Path
    web: Path
    memories: Path
    l0: Path
    l1: Path
    db: Path
    tv: Path
    ids: Path
    sources: Path
    dirty: Path
    hook_state: Path


def paths() -> BrainPaths:
    home = Path(os.environ.get("CC_BRAIN_HOME", "~/.cc-brain")).expanduser().resolve()
    data = home / "data"
    memories = home / "memories"
    return BrainPaths(
        home=home,
        data=data,
        notes=home / "notes",
        commits=home / "commits",
        ingest=home / "ingest",
        private_ingest=REPO_ROOT / "ingest",
        web=home / "ingest" / "web",
        memories=memories,
        l0=memories / "l0" / "conversations",
        l1=memories / "l1" / "atoms",
        db=data / "brain.db",
        tv=data / "brain.tv",
        ids=data / "ids.json",
        sources=home / "sources.json",
        dirty=home / "dirty.json",
        hook_state=data / "hook_state.json",
    )


def ensure_dirs(p: BrainPaths | None = None) -> BrainPaths:
    p = p or paths()
    for d in (p.home, p.data, p.notes, p.commits, p.ingest, p.private_ingest, p.web, p.l0, p.l1):
        d.mkdir(parents=True, exist_ok=True)
    return p


def slug(text: str, default: str = "item") -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", (text or "").strip()).strip("-")
    return s or default


def project_from_path(path: Path) -> str:
    try:
        root = Path(path).resolve()
    except OSError:
        root = Path(path)
    return slug(root.name, "project")


def _default_sources(p: BrainPaths) -> list[SourceSpec]:
    return [
        SourceSpec("notes", p.notes, kind="md", include=r"\.md$", exclude="", project="notes", trust=1.35),
        SourceSpec("commits", p.commits, kind="md", include=r"\.md$", exclude="", trust=1.15),
        SourceSpec("sessions", p.l1, kind="md", include=r"\.md$", exclude="", trust=0.90),
        SourceSpec("web", p.web, kind="md", include=r"\.md$", exclude="", trust=1.05),
        SourceSpec("private-ingest", p.private_ingest, kind="code", project="", trust=1.10),
    ]


def load_sources(p: BrainPaths | None = None) -> list[SourceSpec]:
    p = ensure_dirs(p)
    if not p.sources.exists():
        save_sources(_default_sources(p), p)
    raw = json.loads(p.sources.read_text(encoding="utf-8"))
    out: list[SourceSpec] = []
    for item in raw:
        out.append(SourceSpec(
            name=item["name"],
            root=Path(item["root"]).expanduser().resolve(),
            kind=item.get("kind", "code"),
            include=item.get("include", DEFAULT_INCLUDE),
            exclude=item.get("exclude", DEFAULT_EXCLUDE),
            project=item.get("project", ""),
            trust=float(item.get("trust", 1.0)),
        ))
    known = {s.name for s in out}
    changed = False
    for src in _default_sources(p):
        if src.name not in known:
            out.append(src)
            changed = True
    if changed:
        save_sources(out, p)
    return out


def save_sources(sources: list[SourceSpec], p: BrainPaths | None = None) -> None:
    p = ensure_dirs(p)
    payload = []
    seen = set()
    for src in sources:
        key = src.name
        if key in seen:
            continue
        seen.add(key)
        d = asdict(src)
        d["root"] = str(src.root)
        payload.append(d)
    p.sources.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def upsert_source(spec: SourceSpec, p: BrainPaths | None = None) -> SourceSpec:
    p = ensure_dirs(p)
    sources = [s for s in load_sources(p) if s.name != spec.name]
    sources.append(spec)
    save_sources(sources, p)
    return spec
