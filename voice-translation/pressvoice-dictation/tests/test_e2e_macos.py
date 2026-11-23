"""End-to-end tests of the packaged macOS app: launch the real
PressVoice.app process, press the hotkey with synthetic key events, and check
that the text lands in another app's text field.

Runs only when PRESSVOICE_MAC_APP points at a built PressVoice.app (CI sets
it). Needs Accessibility for the test runner. Each test uses a throwaway
config directory; the microphone and OpenAI are replaced by the app's test
hooks (PRESSVOICE_FAKE_MIC / PRESSVOICE_FAKE_TRANSCRIPT).
PRESSVOICE_E2E_HOTKEY picks the key (default Right ⌘)."""
import json
import os
import subprocess
import sys
import time

import pytest

APP = os.environ.get("PRESSVOICE_MAC_APP", "")
HOTKEY = os.environ.get("PRESSVOICE_E2E_HOTKEY", "cmd_r")
TEXT = "Привет from PressVoice for macOS 123"
HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH_AGENT = os.path.expanduser("~/Library/LaunchAgents/com.pressvoice.mac.plist")

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not APP,
    reason="set PRESSVOICE_MAC_APP to a built PressVoice.app")

_KEYCODES = {"cmd_r": 54, "f19": 80, "f18": 79}


def _post(key: str, down: bool) -> None:
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventPost,
        CGEventSetFlags,
        CGEventSetType,
        kCGEventFlagMaskCommand,
        kCGEventFlagsChanged,
        kCGHIDEventTap,
    )
    ev = CGEventCreateKeyboardEvent(None, _KEYCODES[key], down)
    if key == "cmd_r":
        CGEventSetType(ev, kCGEventFlagsChanged)
        CGEventSetFlags(ev, (kCGEventFlagMaskCommand | 0x10) if down else 0)
    CGEventPost(kCGHIDEventTap, ev)


def _wait(cond, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return bool(cond())


def _app_windows(pid: int) -> int:
    """On-screen normal windows of a process (window titles would need the
    Screen Recording permission; counting them doesn't)."""
    from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly
    wins = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, 0) or []
    # Layer 0 = normal, 3 = floating utility panel (Settings); the menu bar
    # item lives on layer 25 and is not counted.
    return sum(1 for w in wins
               if w.get("kCGWindowOwnerPID") == pid and 0 <= w.get("kCGWindowLayer", 99) < 20)


def _make_wav(path: str) -> None:
    sys.path.insert(0, os.path.dirname(HERE))
    from tests.conftest import make_wav
    with open(path, "wb") as f:
        f.write(make_wav(1.5))


@pytest.fixture
def launch(tmp_path):
    """Start PressVoice.app with a throwaway config; yields a starter."""
    procs = []
    agent_existed = os.path.exists(LAUNCH_AGENT)

    def start(cfg: dict, extra_env=None):
        cfg_dir = tmp_path / "config"
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        wav = str(tmp_path / "speech.wav")
        _make_wav(wav)
        env = {**os.environ, "PRESSVOICE_CONFIG_DIR": str(cfg_dir),
               "PRESSVOICE_FAKE_MIC": wav, "PRESSVOICE_FAKE_TRANSCRIPT": TEXT,
               **(extra_env or {})}
        p = subprocess.Popen([os.path.join(APP, "Contents/MacOS/PressVoice")], env=env)
        p.log = cfg_dir / "pressvoice.log"
        procs.append(p)
        return p

    yield start
    for p in procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
    # `open` may have started a copy we don't own; never leave one running.
    subprocess.run(["pkill", "-f", os.path.join(APP, "Contents/MacOS/PressVoice")])
    if not agent_existed and os.path.exists(LAUNCH_AGENT):
        os.remove(LAUNCH_AGENT)  # don't leave a test login item behind


@pytest.fixture
def target(tmp_path):
    out = str(tmp_path / "target.json")
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "mac_paste_target.py"), out, "40"])

    def state():
        try:
            with open(out, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    yield state
    p.terminate()


def _log_text(app) -> str:
    try:
        return app.log.read_text(encoding="utf-8")
    except OSError:
        return ""


def _started(app) -> bool:
    """Waits until the hotkey monitor is registered (slow CI machines need
    well over the few seconds a normal Mac takes)."""
    ok = _wait(lambda: "AXIsProcessTrusted=" in _log_text(app), 30)
    time.sleep(0.5)
    return ok


def test_hotkey_records_and_pastes_into_another_app(launch, target):
    app = launch({"api_key": "sk-test", "hotkey": HOTKEY, "language": "en",
                  "recording_mode": "push_to_talk", "login_item_initialized": True})
    assert _started(app), _log_text(app)
    assert app.poll() is None, "PressVoice exited at startup"
    assert _wait(lambda: target().get("active"), 10), "paste target never got focus"

    _post(HOTKEY, True)
    time.sleep(1.0)
    _post(HOTKEY, False)

    assert _wait(lambda: target().get("text") == TEXT, 10), (target(), _log_text(app))
    assert app.poll() is None


def test_first_run_opens_settings_and_adds_login_item(launch):
    app = launch({})  # fresh install: no API key yet
    assert _wait(lambda: _app_windows(app.pid) >= 1, 10), "Settings did not open"
    assert os.path.exists(LAUNCH_AGENT), "login item not created on first run"
    assert app.poll() is None


def test_second_launch_exits_and_reopen_shows_settings(launch):
    first = launch({"api_key": "sk-test", "language": "en", "login_item_initialized": True})
    time.sleep(3)
    assert _app_windows(first.pid) == 0  # has a key: no Settings at start

    second = launch({"api_key": "sk-test", "language": "en", "login_item_initialized": True})
    assert second.wait(15) == 0, "a second copy kept running"

    subprocess.run(["open", APP], check=True)  # Finder double-click
    assert _wait(lambda: _app_windows(first.pid) >= 1, 10), "reopen did not show Settings"
    assert first.poll() is None


def test_launched_like_finder_pastes_or_says_why_not(tmp_path, target):
    """The way users start it: through LaunchServices, so the app holds its
    own Accessibility permission instead of inheriting the test runner's.
    With the permission the text must be pasted; without it the text must
    stay on the clipboard and the user must be told — never silently lost."""
    sys.path.insert(0, os.path.dirname(HERE))
    import mac_support

    cfg_dir = tmp_path / "finder"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps(
        {"api_key": "sk-test", "hotkey": HOTKEY, "language": "en",
         "login_item_initialized": True}),
        encoding="utf-8")
    wav = str(tmp_path / "speech.wav")
    _make_wav(wav)
    log = cfg_dir / "pressvoice.log"
    saved = mac_support.save_clipboard()
    try:
        subprocess.run(["open", "-n",
                        "--env", f"PRESSVOICE_CONFIG_DIR={cfg_dir}",
                        "--env", f"PRESSVOICE_FAKE_MIC={wav}",
                        "--env", f"PRESSVOICE_FAKE_TRANSCRIPT={TEXT}",
                        APP], check=True)
        assert _wait(lambda: log.exists() and "AXIsProcessTrusted=" in log.read_text(), 15)
        time.sleep(1)
        trusted = "AXIsProcessTrusted=True" in log.read_text()
        assert _wait(lambda: target().get("active"), 10), "paste target never got focus"

        _post(HOTKEY, True)
        time.sleep(1.0)
        _post(HOTKEY, False)

        if trusted:
            assert _wait(lambda: target().get("text") == TEXT, 10), target()
        else:
            assert _wait(lambda: "notify: Copied — press ⌘V" in log.read_text(), 10), \
                log.read_text()
            assert mac_support.clipboard_text() == TEXT
            assert target().get("text") != TEXT
    finally:
        subprocess.run(["pkill", "-f", os.path.join(APP, "Contents/MacOS/PressVoice")])
        mac_support.restore_clipboard(saved)


def test_live_mode_types_while_the_key_is_held(launch, target):
    """By phrase: the text goes into the other app while Right ⌘ is still
    held, and the held ⌘ must not turn them into shortcuts."""
    app = launch({"api_key": "sk-test", "hotkey": HOTKEY, "language": "en",
                  "recording_mode": "push_to_talk", "text_mode": "phrases",
                  "login_item_initialized": True})
    assert _started(app), _log_text(app)
    assert app.poll() is None, "PressVoice exited at startup"
    assert _wait(lambda: target().get("active"), 10), "paste target never got focus"

    _post(HOTKEY, True)
    try:
        assert _wait(lambda: target().get("text", "").startswith("Привет from"), 6), \
            (target(), _log_text(app))
    finally:
        time.sleep(0.5)
        _post(HOTKEY, False)

    assert _wait(lambda: target().get("text") == TEXT, 10), (target(), _log_text(app))
    assert app.poll() is None
