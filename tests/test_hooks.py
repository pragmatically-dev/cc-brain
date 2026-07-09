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
