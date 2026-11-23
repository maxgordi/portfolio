import threading

import pytest

import config
from qtapp.app import PressVoiceQt
from qtapp.settings_dialog import SettingsDialog
from tests.conftest import make_wav, qwait


class FakeListener:
    def __init__(self):
        self.cb = {}
        self.resets = 0

    def start(self, hotkey, mode, on_start, on_stop, on_cancel):
        self.cb = dict(start=on_start, stop=on_stop, cancel=on_cancel)
        self.hotkey, self.mode = hotkey, mode

    def stop(self):
        pass

    def reset(self):
        self.resets += 1

    def is_alive(self):
        return True


class FakeRecorder:
    def __init__(self, seconds=1.0, fail=None):
        self.seconds = seconds
        self.fail = fail
        self.log = []
        self.backend = "default"

    def start(self, index=-1, name=""):
        self.log.append("start")
        if self.fail:
            raise self.fail

    def stop(self):
        self.log.append("stop")
        return make_wav(self.seconds)

    def reset(self):
        self.log.append("reset")


@pytest.fixture
def env(qapp, no_autostart):
    cfg = config.load()
    cfg["api_key"] = "sk-test"
    cfg["language"] = "en"  # don't follow the language of the test machine
    config.save(cfg)
    state = {"pasted": [], "calls": [], "notices": []}
    gate = threading.Event()
    gate.set()

    def fake_transcribe(api_key, wav, model, **kw):
        gate.wait(5)
        state["calls"].append(model)
        return "привет world"

    def fake_paste(text, on_done):
        state["pasted"].append(text)
        on_done(True)

    listener = FakeListener()
    recorder = FakeRecorder()
    c = PressVoiceQt(recorder=recorder, transcribe=fake_transcribe,
                     hotkey_listener=listener, paster=fake_paste,
                     settings_dialog=SettingsDialog(list_devices=lambda: []))
    c._notify = lambda t, m, on_click=None: state["notices"].append(t)
    c.start()
    yield c, listener, recorder, state, gate
    c._settings.hide()
    c._tray.hide()


def test_dictation_flow(env, qapp):
    c, listener, rec, state, _ = env
    listener.cb["start"]()
    assert c._rec_action.isVisible()
    listener.cb["stop"]()
    assert qwait(qapp, lambda: state["pasted"] == ["привет world"])
    assert state["calls"] == ["gpt-4o-mini-transcribe"]
    assert rec.log == ["start", "stop"]
    assert c._history.get_all()[0]["text"] == "привет world"
    assert qwait(qapp, lambda: not c._busy and not c._rec_action.isVisible())


def test_cancelled_recording_is_dropped(env, qapp):
    c, listener, rec, state, _ = env
    listener.cb["start"]()
    listener.cb["cancel"]()
    assert qwait(qapp, lambda: rec.log == ["start", "stop"])
    qwait(qapp, lambda: False, 0.3)
    assert state["calls"] == []


def test_new_recording_ignored_while_busy(env, qapp):
    c, listener, rec, state, gate = env
    gate.clear()
    listener.cb["start"]()
    listener.cb["stop"]()
    assert qwait(qapp, lambda: c._busy)
    listener.cb["start"]()   # ignored: still transcribing
    listener.cb["stop"]()
    gate.set()
    assert qwait(qapp, lambda: state["pasted"] == ["привет world"])
    qwait(qapp, lambda: False, 0.3)
    assert state["calls"] == ["gpt-4o-mini-transcribe"]


def test_missing_key_opens_settings(env, qapp):
    c, listener, rec, state, _ = env
    c._cfg["api_key"] = ""
    listener.cb["start"]()
    assert "No API key" in state["notices"]
    assert c._settings.isVisible()
    assert rec.log == []
    assert listener.resets == 1


def test_errors_are_reported(env, qapp):
    c, listener, rec, state, _ = env

    def broken(*a, **kw):
        raise RuntimeError("api down")
    c._transcribe = broken
    listener.cb["start"]()
    listener.cb["stop"]()
    assert qwait(qapp, lambda: "Error" in state["notices"])
    assert qwait(qapp, lambda: not c._busy)


def test_settings_save_restarts_hotkey(env, qapp):
    c, listener, rec, state, _ = env
    cfg = config.load()
    cfg["hotkey"] = "f19"
    cfg["recording_mode"] = "toggle"
    c._on_settings_saved(cfg)
    assert (listener.hotkey, listener.mode) == ("f19", "toggle")
    assert "F19" in c._hint_action.text()


@pytest.mark.parametrize("block", ["", "“Let desktop apps access your microphone” is turned off"])
def test_microphone_failure_is_explained_once(env, qapp, monkeypatch, block):
    from qtapp import platform_utils
    c, listener, rec, state, _ = env
    opened = []
    monkeypatch.setattr(platform_utils, "windows_microphone_block", lambda: block)
    monkeypatch.setattr(c, "_open_mic_privacy", lambda: opened.append(1))
    rec.fail = OSError("Error opening InputStream: Unanticipated host error "
                       "[PaErrorCode -9999]: 'Undefined external error.' [MME error 1]")
    for _ in range(2):  # the second failure must not reopen Windows settings
        listener.cb["start"]()
        listener.cb["stop"]()
        assert qwait(qapp, lambda: not c._busy and not c._recording)
        qwait(qapp, lambda: False, 0.3)
    if block:
        assert state["notices"] == ["Windows is blocking the microphone"] * 2
        assert opened == [1]
    else:
        assert state["notices"] == ["Microphone error"] * 2
        assert opened == []
    assert "Empty recording" not in state["notices"]
    assert listener.resets == 2


def test_switching_language_retranslates_the_tray(env, qapp):
    import i18n
    c, listener, rec, state, _ = env
    assert c._settings_action.text() == "Settings…"
    cfg = config.load()
    cfg["language"] = "de"
    try:
        c._on_settings_saved(cfg)
        assert i18n.current_language() == "de"
        assert c._settings_action.text() == "Einstellungen…"
        assert c._quit_action.text() == "PressVoice beenden"
        assert "Rechte Ctrl-Taste" in c._hint_action.text()
        assert c._settings.windowTitle() == "PressVoice – Einstellungen"
    finally:
        i18n.set_language("en")


# ── Live dictation (text pasted while you speak) ─────────────────────────────

class FakeLiveMic:
    backend = "fake"

    def __init__(self):
        self.log = []

    def start(self, on_chunk, device_index=-1, device_name=""):
        self.log.append("start")
        on_chunk(b"\x01\x00" * 24000)

    def stop(self):
        self.log.append("stop")


@pytest.fixture
def live_env(qapp, no_autostart):
    from live.session import LiveDictation
    cfg = config.load()
    cfg.update(api_key="sk-test", language="en", text_mode="phrases")
    config.save(cfg)
    state = {"pasted": [], "notices": [], "own": 0}

    def fake_paste(text, on_done):
        state["pasted"].append(text)
        on_done(True)

    listener = FakeListener()
    listener.ignore_own_input = lambda: state.__setitem__("own", state["own"] + 1)
    recorder = FakeRecorder()
    mic = FakeLiveMic()
    c = PressVoiceQt(recorder=recorder, transcribe=lambda *a, **kw: "unused",
                     hotkey_listener=listener, paster=fake_paste,
                     settings_dialog=SettingsDialog(list_devices=lambda: []),
                     live=LiveDictation(mic=mic), live_fake_text="Привет from live mode")
    c._notify = lambda t, m, on_click=None: state["notices"].append(t)
    c.start()
    yield c, listener, recorder, mic, state
    c._settings.hide()
    c._tray.hide()


def test_live_dictation_pastes_while_speaking(live_env, qapp):
    c, listener, rec, mic, state = live_env
    listener.cb["start"]()
    # Phrases arrive while the key is still held.
    assert qwait(qapp, lambda: len(state["pasted"]) >= 2, 3)
    assert c._recording
    listener.cb["stop"]()
    assert qwait(qapp, lambda: not c._busy, 5)
    assert "".join(state["pasted"]) == "Привет from live mode"
    assert state["own"] == len(state["pasted"])  # each paste announced to the hotkey
    assert rec.log == []                         # the normal recorder stays unused
    assert mic.log == ["start", "stop"]
    assert c._history.get_all()[0]["text"] == "Привет from live mode"
    from PySide6.QtGui import QGuiApplication
    assert QGuiApplication.clipboard().text() == "Привет from live mode"
    assert state["notices"] == []


def test_live_dictation_cancel_stops_pasting(live_env, qapp):
    c, listener, rec, mic, state = live_env
    listener.cb["start"]()
    assert qwait(qapp, lambda: len(state["pasted"]) >= 1, 3)
    listener.cb["cancel"]()
    assert qwait(qapp, lambda: mic.log == ["start", "stop"], 5)
    pasted = len(state["pasted"])
    qwait(qapp, lambda: False, 0.6)
    assert len(state["pasted"]) == pasted < 4
    assert c._history.get_all() == []


def test_switching_back_to_whole_uses_the_recorder(live_env, qapp):
    c, listener, rec, mic, state = live_env
    c._transcribe = lambda *a, **kw: "whole text"
    cfg = config.load()
    cfg["text_mode"] = "whole"
    c._on_settings_saved(cfg)
    listener.cb["start"]()
    listener.cb["stop"]()
    assert qwait(qapp, lambda: state["pasted"] == ["whole text"], 5)
    assert rec.log == ["start", "stop"] and mic.log == []


def test_live_mode_without_key_falls_back_to_the_key_prompt(live_env, qapp):
    c, listener, rec, mic, state = live_env
    cfg = config.load()
    cfg["api_key"] = ""
    c._on_settings_saved(cfg)
    listener.cb["start"]()
    assert "No API key" in state["notices"]
    assert mic.log == []
