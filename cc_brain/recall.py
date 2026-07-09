from __future__ import annotations

import re
from pathlib import Path

from . import indexer
from .config import paths, project_from_path


def format_hits(hits, max_chars: int = 2800) -> str:
    lines: list[str] = []
    used = 0
    for h in hits:
        snippet = " ".join(h.text.split())[:320]
        line = f"- [{h.id}] {h.source}:{h.path}#{h.loc} ({h.project}) {snippet}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def session_start_context(cwd: str | Path, cap: int = 4200) -> str:
    project = project_from_path(Path(cwd))
    body = indexer.project_snapshot(project, k=4)
    if body.startswith("(no memory"):
        return ""
    return (
        f"[cc-brain] Persistent memory for {project}. Use the cc-brain MCP search/get tools for drill-down.\n"
        f"{body}"
    )[:cap]


def prompt_context(prompt: str, project: str = "", cap: int = 2600) -> str:
    toks = re.findall(r"[A-Za-z0-9_]{3,}", prompt or "")
    if len(set(t.lower() for t in toks)) < 2:
        return ""
    hits = indexer.search(prompt, k=3, project=project, lex=True)
    if not hits:
        return ""
    return ("[cc-brain] Possibly relevant brain memory:\n" + format_hits(hits, cap))[:cap]
