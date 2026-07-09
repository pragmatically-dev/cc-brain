from __future__ import annotations

import json
from pathlib import Path

from .config import ensure_dirs, paths, slug
from .contracts import CaptureResult
from .store import mark_dirty
from .transcript import build_l1_markdown, memory_name, parse


def capture_transcript(transcript_path: str | Path, cwd: str | Path = "") -> CaptureResult | None:
    p = ensure_dirs(paths())
    tx = Path(transcript_path)
    if not tx.exists():
        return None
    info = parse(tx)
    if cwd and not info.get("cwd"):
        info["cwd"] = str(cwd)
        info["project"] = Path(cwd).name
    if info.get("turns", 0) == 0 and not info.get("tools"):
        return None
    name = memory_name(info)
    l0_path = p.l0 / f"{name}.json"
    l1_path = p.l1 / f"{name}.md"
    l0_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    l0_rel = str(l0_path.relative_to(p.home)).replace("\\", "/")
    l1_path.write_text(build_l1_markdown(info, l0_rel), encoding="utf-8", newline="\n")
    mark_dirty("session-capture", l1_path, p)
    return CaptureResult(l0_path, l1_path, slug(info.get("project") or "misc"), info.get("ai_title") or "session")


def write_note(name: str, content: str) -> Path:
    p = ensure_dirs(paths())
    safe = slug(name)
    if not safe.endswith(".md"):
        safe += ".md"
    path = p.notes / safe
    path.write_text(content.strip() + "\n", encoding="utf-8", newline="\n")
    mark_dirty("note", path, p)
    return path


def append_commit(project: str, entry: str) -> Path:
    p = ensure_dirs(paths())
    path = p.commits / f"{slug(project)}.md"
    if not path.exists():
        path.write_text(f"# Commit log - {project}\n\n", encoding="utf-8", newline="\n")
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(entry.rstrip() + "\n")
    mark_dirty("commit", path, p)
    return path
