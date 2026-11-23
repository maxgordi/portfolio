import pytest

import config
from qtapp.settings_dialog import SettingsDialog
from qtapp.styles_dialog import StylesDialog
from styles import CLEAN_NAME, StylesManager
from tests.conftest import qwait

DEVICES = [{"index": 3, "name": "USB Mic"}, {"index": 5, "name": "Микрофон"}]


def make_dialog(fetch=lambda key: ["gpt-transcribe", "whisper-1", "gpt-new-transcribe"],
                mic=lambda index, name: (True, f"✓ works {index} {name}")):
    return SettingsDialog(list_devices=lambda: DEVICES, fetch_models=fetch,
                          test_microphone=mic)


def test_settings_load_and_save(qapp, no_autostart):
    cfg = config.load()
    cfg.update(api_key="sk-old", microphone_name="Микрофон", hotkey="f13",
               recording_mode="toggle")
    config.save(cfg)
    saved = []
    d = make_dialog()
    d.open_settings(on_save=saved.append)
    assert d.api_edit.text() == "sk-old"
    assert d.mic_combo.currentText() == "Микрофон"
    assert d.hotkey_combo.currentData() == "f13"
    assert d.mode_combo.currentData() == "toggle"
    assert d.text_combo.currentData() == "whole"  # the default
    assert d.model_combo.currentText() == "gpt-4o-mini-transcribe"

    d.api_edit.setText("  sk-new  ")
    d.model_combo.setCurrentText("whisper-1")
    d.mic_combo.setCurrentIndex(1)
    d.hotkey_combo.setCurrentIndex(d.hotkey_combo.findData("alt_r"))
    d.text_combo.setCurrentIndex(d.text_combo.findData("phrases"))
    d.autostart_check.setChecked(True)
    d.save()

    cfg = config.load()
    assert cfg["api_key"] == "sk-new"
    assert cfg["transcription_model"] == "whisper-1"
    assert (cfg["microphone_index"], cfg["microphone_name"]) == (3, "USB Mic")
    assert cfg["hotkey"] == "alt_r"
    assert cfg["text_mode"] == "phrases"
    assert saved and saved[0]["hotkey"] == "alt_r"
    assert not d.isVisible()
    if d.autostart_check.isVisibleTo(d):
        assert no_autostart["calls"] == [True]


def test_refresh_models_fills_combo_and_caches(qapp, no_autostart):
    d = make_dialog()
    d.open_settings()
    d.api_edit.setText("sk-test")
    d.refresh_models()
    assert qwait(qapp, lambda: d.refresh_btn.isEnabled(), 5)
    items = [d.model_combo.itemText(i) for i in range(d.model_combo.count())]
    assert "gpt-new-transcribe" in items
    assert d.model_combo.currentText() == "gpt-4o-mini-transcribe"  # selection kept
    assert "Models found: 3." in d.model_status.text()
    assert config.load()["transcription_models"][-1] == "gpt-new-transcribe"
    d.hide()


def test_refresh_without_key(qapp, no_autostart):
    d = make_dialog()
    d.open_settings()
    d.api_edit.setText("")
    d.refresh_models()
    assert "API key" in d.model_status.text()
    d.hide()


def test_refresh_error_is_shown(qapp, no_autostart):
    def boom(key):
        raise ConnectionError("down")
    d = make_dialog(fetch=boom)
    d.open_settings()
    d.api_edit.setText("sk-x")
    d.refresh_models()
    assert qwait(qapp, lambda: d.refresh_btn.isEnabled(), 5)
    assert "ConnectionError" in d.model_status.text()
    d.hide()


def test_styles_dialog_add_edit_activate(qapp):
    mgr = StylesManager()
    changes = []
    d = StylesDialog()
    d.open_styles(mgr, on_change=lambda: changes.append(1))
    d.add_style()
    d.name_edit.setText("Email")
    d.prompt_edit.setPlainText("Make it an email")
    d.active_check.click()
    assert mgr.active_name == "Email"
    d.close()
    assert StylesManager().get_active()["prompt"] == "Make it an email"
    assert changes

    d.open_styles(mgr)
    d.delete_style()
    assert mgr.active_name == CLEAN_NAME
    assert [s["name"] for s in mgr.get_all()] == [CLEAN_NAME]
    d.close()


def test_microphone_test_button(qapp, no_autostart):
    d = make_dialog()
    d.open_settings()
    d.mic_combo.setCurrentIndex(2)  # "Микрофон"
    d.test_microphone()
    assert qwait(qapp, lambda: d.mic_test_btn.isEnabled(), 5)
    assert d.mic_status.text() == "✓ works 5 Микрофон"
    d.hide()


def test_microphone_test_failure_links_to_privacy_settings(qapp, no_autostart):
    import sys
    d = make_dialog(mic=lambda i, n: (False, "Windows is blocking the microphone."))
    d.open_settings()
    d.test_microphone()
    assert qwait(qapp, lambda: d.mic_test_btn.isEnabled(), 5)
    assert d.mic_status.text().startswith("Windows is blocking")
    assert ("ms-settings:privacy-microphone" in d.mic_status.text()) == (sys.platform == "win32")
    d.hide()


def test_language_choice_is_saved_and_dialog_retranslates(qapp, no_autostart):
    import i18n
    d = make_dialog()
    d.open_settings()
    assert d.language_combo.itemData(0) == i18n.AUTO
    assert [d.language_combo.itemData(i) for i in range(1, d.language_combo.count())] \
        == list(i18n.LANGUAGES)
    d.language_combo.setCurrentIndex(d.language_combo.findData("uk"))
    d.hotkey_combo.setCurrentIndex(d.hotkey_combo.findData("alt_r"))
    d.save()
    assert config.load()["language"] == "uk"
    try:
        i18n.set_language("uk")
        d.retranslate()
        assert d.windowTitle() == "PressVoice — Налаштування"
        assert d.buttons.button(d.buttons.StandardButton.Save).text() == "Зберегти"
        assert d.hotkey_combo.currentData() == "alt_r"           # selection kept
        assert d.hotkey_combo.currentText() == "Правий Alt"
        assert d.language_combo.currentData() == "uk"
    finally:
        i18n.set_language("en")


def test_clean_style_is_shown_translated_but_stored_as_is(qapp):
    import i18n
    mgr = StylesManager()
    d = StylesDialog()
    try:
        i18n.set_language("ru")
        d.retranslate()
        d.open_styles(mgr)
        assert d.list.item(0).text().startswith("Без обработки")
        d.add_style()
        assert "Новый стиль" in [s["name"] for s in mgr.get_all()]
        assert mgr.get_all()[0]["name"] == CLEAN_NAME
    finally:
        d.close()
        i18n.set_language("en")
