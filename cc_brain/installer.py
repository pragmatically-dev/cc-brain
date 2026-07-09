from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .config import ensure_dirs, paths


def _settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def _command(module: str) -> str:
    return f'"{sys.executable}" -m {module}'


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


def install() -> Path:
    ensure_dirs(paths())
    sp = _settings_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    if sp.exists():
        try:
            settings = json.loads(sp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup = sp.with_suffix(".json.bak")
            backup.write_text(sp.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            settings = {}
    else:
        settings = {}
    hook_cmd = _command("cc_brain.hooks")
    _add_hook(settings, "SessionStart", hook_cmd)
    _add_hook(settings, "UserPromptSubmit", hook_cmd)
    _add_hook(settings, "SessionEnd", hook_cmd)
    _add_hook(settings, "PreCompact", hook_cmd)
    for matcher in ("Bash", "PowerShell", "Edit", "Write", "MultiEdit", "NotebookEdit", "WebFetch"):
        _add_hook(settings, "PostToolUse", hook_cmd, matcher)
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, sp)
    return sp


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


def mcp_command() -> str:
    return f'claude mcp add --scope user cc-brain -- "{sys.executable}" -m cc_brain mcp'
