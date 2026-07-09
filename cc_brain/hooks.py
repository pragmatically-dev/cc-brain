from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .config import ensure_dirs, paths, project_from_path, slug
from .indexer import register_repo
from .memory import append_commit, capture_transcript
from .recall import prompt_context, session_start_context
from .store import mark_dirty
from .web import capture_web


COMMIT_RE = re.compile(r"\[(\S+)\s+(?:\(root-commit\)\s+)?([0-9a-f]{7,40})\]\s*(.*)")


def _emit(event: str, context: str) -> None:
    if context.strip():
        print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": context.strip()}}))


def _payload() -> dict:
    raw = sys.stdin.read()
    return json.loads(raw) if raw.strip() else {}


def _register_cwd(cwd: str) -> None:
    if not cwd:
        return
    root = Path(cwd)
    if root.exists() and root.is_dir():
        register_repo(root, name=f"repo-{root.name}", project=root.name)
        mark_dirty("repo-seen", root)


def session_start(data: dict) -> None:
    cwd = data.get("cwd") or os.getcwd()
    _register_cwd(cwd)
    _emit("SessionStart", session_start_context(cwd))


def user_prompt_submit(data: dict) -> None:
    cwd = data.get("cwd") or os.getcwd()
    _register_cwd(cwd)
    project = project_from_path(Path(cwd))
    _emit("UserPromptSubmit", prompt_context(data.get("prompt", ""), project=project))


def session_end(data: dict) -> None:
    cwd = data.get("cwd") or os.getcwd()
    _register_cwd(cwd)
    capture_transcript(data.get("transcript_path", ""), cwd)


def post_tool_use(data: dict) -> None:
    tool = data.get("tool_name", "")
    tool_input = data.get("tool_input") or {}
    cwd = data.get("cwd") or os.getcwd()
    project = project_from_path(Path(cwd))
    if tool in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        _register_cwd(cwd)
        mark_dirty("repo-edit", cwd)
        return
    if tool in ("WebFetch", "webfetch"):
        url = tool_input.get("url") or ""
        output = data.get("tool_output") or ""
        if isinstance(output, dict):
            output = output.get("stdout") or output.get("content") or json.dumps(output, ensure_ascii=False)
        if url:
            capture_web(url, str(output), project=project)
        return
    if tool not in ("Bash", "PowerShell", "bash"):
        return
    cmd = tool_input.get("command", "") or ""
    if "git commit" in cmd:
        out = data.get("tool_output") or ""
        if isinstance(out, dict):
            out = out.get("stdout", "") or json.dumps(out)
        m = COMMIT_RE.search(str(out))
        if m:
            branch, sha, subject = m.group(1), m.group(2), m.group(3).strip()
            files = ""
            try:
                r = subprocess.run(["git", "-C", cwd, "show", "--stat", "--format=", sha],
                                   capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace")
                files = "\n".join("- " + line.split("|")[0].strip() for line in r.stdout.splitlines() if "|" in line)
            except Exception:
                pass
            append_commit(project, f"## {sha[:8]} ({branch}) - {subject}\n{files}\n")
    if "git clone" in cmd:
        mark_dirty("git-clone", cwd)


def main() -> int:
    ensure_dirs(paths())
    data = _payload()
    event = data.get("hook_event_name", "")
    if event == "UserPromptSubmit":
        user_prompt_submit(data)
    elif event == "SessionEnd":
        session_end(data)
    elif event == "PostToolUse":
        post_tool_use(data)
    else:
        session_start(data)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit(0)
