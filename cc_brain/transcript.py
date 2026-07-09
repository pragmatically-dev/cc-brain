from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from .config import slug
from .text import scrub

INJECTED_PREFIXES = (
    "<system-reminder>", "Caveat:", "<local-command", "<command-name>",
    "<command-message>", "[Request interrupted", "This session is being continued",
    "<task-notification>", "[SYSTEM NOTIFICATION", "<user-prompt-submit-hook>",
    "<post-tool", "Your questions have been answered", "<bash-",
)


def _blocks(message):
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return [("str", content)]
    out = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                out.append((block.get("type", "?"), block))
    return out


def parse(path: str | Path) -> dict:
    path = Path(path)
    info = {
        "session": "", "project": "", "cwd": "", "started": "", "ended": "",
        "turns": 0, "ai_title": "", "tools": Counter(), "files": [],
        "files_full": [], "commits": [], "user_requests": [], "assistant_text": [],
        "errors": 0,
    }
    files: list[str] = []
    seen_files: set[str] = set()
    seen_asks: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = data.get("type")
            if data.get("sessionId") and not info["session"]:
                info["session"] = data["sessionId"]
            if data.get("cwd") and not info["cwd"]:
                info["cwd"] = data["cwd"]
                info["project"] = os.path.basename(data["cwd"].rstrip("/\\"))
            ts = data.get("timestamp")
            if ts:
                info["started"] = info["started"] or ts
                info["ended"] = ts
            if typ == "ai-title" and data.get("aiTitle"):
                info["ai_title"] = scrub(data["aiTitle"])
            if typ == "user":
                if data.get("isMeta") or data.get("isSidechain") or data.get("isCompactSummary"):
                    continue
                for kind, block in _blocks(data.get("message", {})):
                    if kind != "str":
                        continue
                    text = block.strip()
                    norm = " ".join(text.split()).lower()[:240]
                    if len(text) > 2 and not text.startswith(INJECTED_PREFIXES) and norm not in seen_asks:
                        info["user_requests"].append(scrub(text)[:700])
                        info["turns"] += 1
                        seen_asks.add(norm)
            elif typ == "assistant":
                if data.get("isApiErrorMessage") or data.get("apiErrorStatus") or data.get("error"):
                    continue
                for kind, block in _blocks(data.get("message", {})):
                    if kind == "thinking":
                        continue
                    if kind == "text" and isinstance(block, dict):
                        text = (block.get("text") or "").strip()
                        if len(text) > 40:
                            info["assistant_text"].append(scrub(text))
                    elif kind == "tool_use" and isinstance(block, dict):
                        name = block.get("name", "?")
                        info["tools"][name] += 1
                        tool_input = block.get("input", {}) or {}
                        if name in ("Edit", "Write", "NotebookEdit"):
                            fp = tool_input.get("file_path") or tool_input.get("notebook_path")
                            if fp and fp not in seen_files:
                                seen_files.add(fp)
                                files.append(fp)
                        cmd = tool_input.get("command", "")
                        if isinstance(cmd, str):
                            for m in re.finditer(r'git commit[^\n]*?-m\s+(["\'])(.+?)\1', cmd, re.S):
                                info["commits"].append(scrub(m.group(2).split("\n")[0])[:160])
    info["files_full"] = files[:80]
    info["files"] = [os.path.basename(x) for x in files][:80]
    info["tools"] = dict(info["tools"].most_common())
    seen = set()
    info["commits"] = [c for c in info["commits"] if c and not (c in seen or seen.add(c))][:40]
    return info


def memory_name(info: dict) -> str:
    date = (info.get("started") or info.get("ended") or "0000-00-00")[:10]
    project = slug(info.get("project") or "misc")
    session = (info.get("session") or "unknown")[:8]
    return f"{date}__{project}__{session}"


def build_l1_markdown(info: dict, l0_rel: str) -> str:
    title = info.get("ai_title") or f"{info.get('project') or 'session'} session"
    date = (info.get("started") or info.get("ended") or "")[:10]
    tools = ", ".join(f"{k}:{v}" for k, v in list((info.get("tools") or {}).items())[:12])
    asks = info.get("user_requests") or []
    assistant = [" ".join(x.split()) for x in info.get("assistant_text", []) if len(x) > 80]
    summary_bits = []
    for block in assistant[-6:]:
        first = block.split(". ")[0]
        if len(first) > 30:
            summary_bits.append(first[:240])
    summary = " · ".join(summary_bits) or "(no narrative captured)"
    nxt = assistant[-1][-420:] if assistant else ""
    front = (
        "---\n"
        f"session: {info.get('session','')}\n"
        f"project: {info.get('project','')}\n"
        f"cwd: {info.get('cwd','')}\n"
        f"date: {date}\n"
        f"started: {info.get('started','')}\n"
        f"ended: {info.get('ended','')}\n"
        f"turns: {info.get('turns', 0)}\n"
        "provenance: auto-session\n"
        f"evidence: {l0_rel}\n"
        f"title: {title}\n"
        f"tools: {tools}\n"
        "---\n\n"
    )
    body = [f"# {title}", "", "## What happened", summary, ""]
    if asks:
        body += ["## User asked", "\n".join("- " + a.replace("\n", " ")[:260] for a in asks[:16]), ""]
    if info.get("files"):
        body += ["## Files touched", ", ".join(f"`{f}`" for f in info["files"][:40]), ""]
    if info.get("commits"):
        body += ["## Commits", "\n".join("- " + c for c in info["commits"][:20]), ""]
    if nxt:
        body += ["## Next step", nxt, ""]
    return front + "\n".join(body).strip() + "\n"
