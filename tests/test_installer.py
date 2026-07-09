import json

import cc_brain.installer as installer


def _hook_count(settings):
    return sum(len(entry.get("hooks", [])) for entries in settings.get("hooks", {}).values() for entry in entries)


def test_install_is_idempotent(tmp_path, monkeypatch, brain_home):
    sp = tmp_path / "settings.json"
    monkeypatch.setattr(installer, "_settings_path", lambda: sp)
    installer.install()
    first = json.loads(sp.read_text(encoding="utf-8"))
    installer.install()
    second = json.loads(sp.read_text(encoding="utf-8"))
    assert _hook_count(first) == _hook_count(second)


def test_install_updates_stale_hook_instead_of_duplicating(tmp_path, monkeypatch, brain_home):
    sp = tmp_path / "settings.json"
    sp.write_text(json.dumps({
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": '"C:/old/python.exe" -m cc_brain.hooks'}]}
            ]
        }
    }), encoding="utf-8")
    monkeypatch.setattr(installer, "_settings_path", lambda: sp)
    installer.install()
    settings = json.loads(sp.read_text(encoding="utf-8"))
    session_start_entries = settings["hooks"]["SessionStart"]
    cmds = [h["command"] for entry in session_start_entries for h in entry["hooks"]]
    assert len(cmds) == 1
    assert "cc_brain.hooks" in cmds[0]
    assert "old/python.exe" not in cmds[0]


def test_install_adds_precompact_hook(tmp_path, monkeypatch, brain_home):
    sp = tmp_path / "settings.json"
    monkeypatch.setattr(installer, "_settings_path", lambda: sp)
    installer.install()
    settings = json.loads(sp.read_text(encoding="utf-8"))
    assert "PreCompact" in settings["hooks"]
    precompact_cmds = [h["command"] for entry in settings["hooks"]["PreCompact"] for h in entry["hooks"]]
    assert any("cc_brain.hooks" in c for c in precompact_cmds)


def test_uninstall_removes_only_cc_brain_entries(tmp_path, monkeypatch, brain_home):
    sp = tmp_path / "settings.json"
    monkeypatch.setattr(installer, "_settings_path", lambda: sp)
    installer.install()
    settings = json.loads(sp.read_text(encoding="utf-8"))
    settings.setdefault("hooks", {}).setdefault("SessionStart", []).append(
        {"hooks": [{"type": "command", "command": "some-other-tool --flag"}]}
    )
    sp.write_text(json.dumps(settings), encoding="utf-8")
    installer.uninstall()
    after = json.loads(sp.read_text(encoding="utf-8"))
    session_start_cmds = [h["command"] for entry in after["hooks"]["SessionStart"] for h in entry["hooks"]]
    assert session_start_cmds == ["some-other-tool --flag"]
    assert "PreCompact" not in after.get("hooks", {})
