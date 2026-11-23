"""End-to-end on a real desktop: the real keyboard hook sees an injected
hotkey, the controller records (fake microphone), "transcribes" (fake API)
and pastes into a focused text field with a real Ctrl+V. Runs on Windows and
on Linux under Xvfb; skipped on macOS."""
import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit

import config
from qtapp.app import PressVoiceQt
from qtapp.settings_dialog import SettingsDialog
from tests.conftest import qwait
from tests.test_controller import FakeRecorder

pytestmark = pytest.mark.skipif(sys.platform == "darwin",
                                reason="needs a Windows or X11 desktop")

TEXT = "Привет from PressVoice 123"
_x11_keyboard = None


def press(key: str, up: bool):
    global _x11_keyboard
    if sys.platform == "win32":
        from qtapp import winapi
        vk = {"ctrl_r": 0xA3, "insert": 0x2D, "a": 0x41}[key]
        assert winapi.send_keys([(vk, up)]) == 1
    else:
        from pynput.keyboard import Controller, Key, KeyCode
        # One controller, so it knows Ctrl is held when "a" goes out.
        _x11_keyboard = _x11_keyboard or Controller()
        k = KeyCode.from_char("a") if key == "a" else getattr(Key, key)
        (_x11_keyboard.release if up else _x11_keyboard.press)(k)


class KeyLog(QLineEdit):
    def __init__(self):
        super().__init__()
        self.keys = []

    def keyPressEvent(self, e):
        self.keys.append(e.key())
        super().keyPressEvent(e)


@pytest.fixture
def field(qapp):
    w = KeyLog()
    w.setWindowTitle("PressVoice e2e")
    w.resize(500, 40)
    w.show()
    w.raise_()
    w.activateWindow()
    if sys.platform == "win32":
        from qtapp import winapi
        winapi.force_foreground(int(w.winId()))
    w.setFocus()
    if not qwait(qapp, lambda: w.isActiveWindow() and w.hasFocus(), 5):
        pytest.skip("cannot focus a window on this desktop")
    yield w
    w.close()


def make_app(hotkey, mode, calls):
    cfg = config.load()
    cfg.update(api_key="sk-test", hotkey=hotkey, recording_mode=mode, language="en")
    config.save(cfg)

    def fake_transcribe(api_key, wav, model, **kw):
        calls.append(model)
        return TEXT

    c = PressVoiceQt(recorder=FakeRecorder(), transcribe=fake_transcribe,
                     settings_dialog=SettingsDialog(list_devices=lambda: []))
    c.start()
    return c


@pytest.fixture
def layout(request):
    """Run a test once with the US and once with the Russian layout."""
    if sys.platform != "win32" or request.param == "us":
        yield request.param
        return
    from qtapp import winapi
    prev = winapi.current_keyboard_layout()
    winapi.activate_keyboard_layout("00000419")
    yield request.param
    if prev:
        winapi.user32.ActivateKeyboardLayout(prev, 0)


@pytest.mark.parametrize("layout", ["us", "ru"], indirect=True)
def test_push_to_talk_pastes_into_focused_field(qapp, field, layout, no_autostart):
    calls = []
    c = make_app("ctrl_r", "push_to_talk", calls)
    try:
        qwait(qapp, lambda: False, 0.5)  # hook installs
        press("ctrl_r", up=False)
        assert qwait(qapp, lambda: c._recording, 3), "hook did not see Right Ctrl"
        qwait(qapp, lambda: False, 0.6)
        press("ctrl_r", up=True)
        assert qwait(qapp, lambda: field.text() == TEXT, 6), repr(field.text())
        assert calls == ["gpt-4o-mini-transcribe"]
    finally:
        c.quit()


def test_shortcut_cancels_recording(qapp, field, no_autostart):
    calls = []
    c = make_app("ctrl_r", "push_to_talk", calls)
    try:
        qwait(qapp, lambda: False, 0.5)
        press("ctrl_r", up=False)
        assert qwait(qapp, lambda: c._recording, 3)
        press("a", up=False)
        press("a", up=True)
        press("ctrl_r", up=True)
        assert qwait(qapp, lambda: not c._recording, 3)
        qwait(qapp, lambda: False, 1.0)
        assert calls == []
        assert TEXT not in field.text()  # the shortcut itself reaches the app
    finally:
        c.quit()


@pytest.mark.skipif(sys.platform != "win32", reason="suppression is Windows-only")
def test_toggle_key_is_swallowed(qapp, field, no_autostart):
    calls = []
    c = make_app("insert", "toggle", calls)
    try:
        qwait(qapp, lambda: False, 0.5)
        press("insert", up=False); press("insert", up=True)
        assert qwait(qapp, lambda: c._recording, 3)
        qwait(qapp, lambda: False, 0.6)
        press("insert", up=False); press("insert", up=True)
        assert qwait(qapp, lambda: field.text() == TEXT, 6), repr(field.text())
        assert Qt.Key_Insert not in field.keys
    finally:
        c.quit()


def test_live_mode_pastes_while_the_key_is_held(qapp, field, no_autostart):
    """By phrase: pieces are pasted with Ctrl+V while Right Ctrl is still held;
    those keystrokes must neither cancel the recording nor be lost."""
    from live.session import LiveDictation
    from tests.test_controller import FakeLiveMic
    cfg = config.load()
    cfg.update(api_key="sk-test", hotkey="ctrl_r", recording_mode="push_to_talk",
               language="en", text_mode="phrases")
    config.save(cfg)
    c = PressVoiceQt(recorder=FakeRecorder(),
                     settings_dialog=SettingsDialog(list_devices=lambda: []),
                     live=LiveDictation(mic=FakeLiveMic()), live_fake_text=TEXT)
    c.start()
    try:
        qwait(qapp, lambda: False, 0.5)
        press("ctrl_r", up=False)
        try:
            assert qwait(qapp, lambda: c._recording, 3), "hook did not see Right Ctrl"
            assert qwait(qapp, lambda: field.text().startswith("Привет from"), 6), \
                repr(field.text())
            assert c._recording, "our own Ctrl+V cancelled the recording"
            qwait(qapp, lambda: False, 0.5)
        finally:
            press("ctrl_r", up=True)
        assert qwait(qapp, lambda: field.text() == TEXT and not c._busy, 6), repr(field.text())
    finally:
        c.quit()
