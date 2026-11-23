import json

import config
from history import HistoryManager
from styles import CLEAN_NAME, StylesManager


def test_config_defaults_and_roundtrip(isolated_config):
    cfg = config.load()
    assert cfg["transcription_model"] == "gpt-4o-mini-transcribe"
    assert cfg["transcription_models"] == []
    cfg["api_key"] = "sk-тест"
    config.save(cfg)
    assert config.load()["api_key"] == "sk-тест"


def test_config_survives_damage(isolated_config):
    (isolated_config / "config.json").write_text("{broken", encoding="utf-8")
    assert config.load() == config.DEFAULTS
    (isolated_config / "config.json").write_text("[1, 2]", encoding="utf-8")
    assert config.load() == config.DEFAULTS


def test_old_config_gets_new_keys(isolated_config):
    (isolated_config / "config.json").write_text(
        json.dumps({"api_key": "sk-x", "hotkey": "f13"}), encoding="utf-8")
    cfg = config.load()
    assert cfg["api_key"] == "sk-x" and cfg["hotkey"] == "f13"
    assert cfg["transcription_model"] == config.DEFAULT_MODEL


def test_history_is_capped_and_persisted():
    h = HistoryManager(max_size=2)
    for t in ("один", "two", "three"):
        h.add(t)
    assert [e["text"] for e in HistoryManager(2).get_all()] == ["three", "two"]
    h.clear()
    assert HistoryManager().get_all() == []


def test_styles_crud():
    s = StylesManager()
    assert s.active_name == CLEAN_NAME
    name = s.add()
    final = s.update(name, "Формально", "Перепиши формально")
    s.set_active(final)
    again = StylesManager()
    assert again.active_name == "Формально"
    assert again.get_active()["prompt"] == "Перепиши формально"
    again.delete("Формально")
    assert StylesManager().active_name == CLEAN_NAME


def test_mac_legacy_login_item_is_migrated(tmp_path, monkeypatch):
    import sys
    if sys.platform != "darwin":
        import pytest
        pytest.skip("macOS only")
    import plistlib

    import mac_support
    legacy = tmp_path / "com.pressvoice.app.plist"
    with open(legacy, "wb") as f:
        plistlib.dump({"ProgramArguments": ["/Applications/PressVoice.app/Contents/MacOS/PressVoice"]}, f)
    other = tmp_path / "other.plist"
    with open(other, "wb") as f:
        plistlib.dump({"ProgramArguments": ["/usr/bin/something"]}, f)
    monkeypatch.setattr(mac_support, "_LEGACY_LAUNCH_AGENTS", [str(legacy), str(other)])
    assert mac_support.remove_legacy_login_items() is True
    assert not legacy.exists() and other.exists()
    assert mac_support.remove_legacy_login_items() is False
