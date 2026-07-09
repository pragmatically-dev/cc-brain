import json

from cc_brain.transcript import build_l1_markdown, memory_name, parse

SESSION_ID = "abcdefgh12345678"


def _write_fixture(tmp_path):
    lines = [
        {
            "type": "user",
            "sessionId": SESSION_ID,
            "cwd": "/repo/myproj",
            "timestamp": "2026-07-09T10:00:00Z",
            "message": {"content": "primera pregunta"},
        },
        {
            "type": "user",
            "timestamp": "2026-07-09T10:01:00Z",
            "message": {"content": [{"type": "text", "text": "segunda pregunta"}]},
        },
        {
            "type": "user",
            "timestamp": "2026-07-09T10:01:30Z",
            "message": {"content": "<system-reminder>ignore this injected content</system-reminder>"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-07-09T10:02:00Z",
            "message": {
                "content": [
                    {"type": "text", "text": "I will edit the file now to implement the requested feature properly."},
                    {"type": "tool_use", "name": "Edit", "input": {"file_path": "/repo/myproj/app.py"}},
                ]
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-07-09T10:03:00Z",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {"command": 'git commit -m "feat: done"'}},
                ]
            },
        },
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    return path


def test_parse_captures_both_user_messages_and_skips_injected(tmp_path):
    info = parse(_write_fixture(tmp_path))
    assert info["user_requests"] == ["primera pregunta", "segunda pregunta"]


def test_parse_extracts_commit_and_file(tmp_path):
    info = parse(_write_fixture(tmp_path))
    assert "feat: done" in info["commits"]
    assert "app.py" in info["files"]
    assert "/repo/myproj/app.py" in info["files_full"]


def test_memory_name_format(tmp_path):
    info = parse(_write_fixture(tmp_path))
    name = memory_name(info)
    assert name == "2026-07-09__myproj__abcdefgh"


def test_build_l1_markdown_has_required_sections(tmp_path):
    info = parse(_write_fixture(tmp_path))
    md = build_l1_markdown(info, "memories/l0/conversations/2026-07-09__myproj__abcdefgh.json")
    assert "## What happened" in md
    assert "## User asked" in md
    assert "evidence:" in md
