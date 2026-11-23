"""`PressVoice --selftest report.json`: checks that a built/installed copy
works on this machine — audio library, keyboard hook, clipboard, paste, TLS
to OpenAI — and writes the results as JSON. Exit code 0 = all critical
checks passed."""
import json
import os
import platform
import sys
import tempfile
import time
import traceback

from PySide6.QtCore import QEventLoop
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QLineEdit, QSystemTrayIcon

import config as cfg_module
from qtapp import platform_utils
from version import __version__

PASTE_SAMPLE = "PressVoice selftest ✓ тест 123"


class Skip(Exception):
    """The environment can't run this check (no desktop, no microphone)."""


class Warn(Exception):
    """Not fatal for the app, but worth knowing."""


def _wait(app: QApplication, cond, timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents(QEventLoop.AllEvents, 50)
        if cond():
            return True
        time.sleep(0.02)
    app.processEvents()
    return bool(cond())


# ── Checks ───────────────────────────────────────────────────────────────────

def check_imports(app):
    import numpy
    import openai
    from pynput import keyboard
    return (f"openai {openai.__version__}, numpy {numpy.__version__}, "
            f"pynput backend {keyboard.Listener.__module__}")


def check_translations(app):
    """Every language file made it into the build and is complete."""
    import i18n
    expected = len(i18n._strings("en"))
    broken = [c for c in i18n.LANGUAGES if len(i18n._strings(c)) != expected]
    if not expected or broken:
        raise RuntimeError(f"missing or incomplete: {broken or ['en']} in {i18n.locales_dir()}")
    system = i18n.detect_system_language()
    return (f"{len(i18n.LANGUAGES)} languages × {expected} strings; "
            f"system language here: {i18n.LANGUAGES[system]}")


def check_config_dir(app):
    for d in {cfg_module.CONFIG_DIR, cfg_module.DATA_DIR}:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".selftest")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    return f"config: {cfg_module.CONFIG_DIR}; data: {cfg_module.DATA_DIR}"


def check_portaudio(app):
    import sounddevice as sd
    from audio import AudioRecorder
    version = sd.get_portaudio_version()[1]
    apis = [a["name"] for a in sd.query_hostapis()]
    default_api = apis[sd.default.hostapi] if sd.default.hostapi >= 0 else "none"
    devices = AudioRecorder.list_devices()
    return (f"{version}; host APIs {apis}; default {default_api}; "
            f"{len(devices)} input device(s): {[d['name'] for d in devices][:5]}")


def check_microphone(app):
    from audio import AudioRecorder
    if not AudioRecorder.list_devices():
        raise Skip("no input devices on this machine")
    rec = AudioRecorder()
    try:
        rec.start()
    except Exception as e:
        raise Warn(f"could not open the microphone: {e}")
    time.sleep(0.6)
    wav = rec.stop()
    if not wav:
        raise Warn("microphone opened but delivered no audio")
    return f"recorded {len(wav)} bytes"


def check_tray(app):
    if not QSystemTrayIcon.isSystemTrayAvailable():
        raise Warn("no system tray on this desktop")
    return "available"


def _inject(key_name: str, up: bool) -> bool:
    if platform_utils.IS_WINDOWS:
        from qtapp import winapi
        vk = {"ctrl_r": winapi.VK_RCONTROL}[key_name]
        return winapi.send_keys([(vk, up)]) == 1
    from pynput.keyboard import Controller, Key
    kb = Controller()
    (kb.release if up else kb.press)(getattr(Key, key_name))
    return True


def check_hotkey(app):
    from qtapp.hotkey import HotkeyListener
    events = []
    listener = HotkeyListener()
    listener.start("ctrl_r", "push_to_talk",
                   on_start=lambda: events.append("start"),
                   on_stop=lambda: events.append("stop"),
                   on_cancel=lambda: events.append("cancel"))
    try:
        _wait(app, lambda: False, 0.5)  # let the hook install
        if not _inject("ctrl_r", up=False):
            raise Skip("keyboard input can't be injected here (no interactive desktop)")
        if not _wait(app, lambda: "start" in events, 3):
            _inject("ctrl_r", up=True)
            raise RuntimeError("the keyboard hook did not see Right Ctrl")
        _wait(app, lambda: False, 0.3)
        _inject("ctrl_r", up=True)
        if not _wait(app, lambda: "stop" in events, 3):
            raise RuntimeError("press seen, release not seen")
        return f"events {events}"
    finally:
        listener.stop()


def check_clipboard(app):
    cb = QGuiApplication.clipboard()
    cb.setText(PASTE_SAMPLE)
    if not _wait(app, lambda: cb.text() == PASTE_SAMPLE, 2):
        raise RuntimeError(f"read back {cb.text()!r}")
    return "round-trip ok"


def _paste_into(app, edit: QLineEdit) -> str:
    from qtapp.paste import paste_text
    edit.clear()
    result = {}
    paste_text(PASTE_SAMPLE, lambda ok: result.setdefault("ok", ok))
    _wait(app, lambda: edit.text() == PASTE_SAMPLE, 3)
    if result.get("ok") is False:
        raise Skip("the system refused the Ctrl+V keystroke")
    return edit.text()


def check_paste(app):
    edit = QLineEdit()
    edit.setWindowTitle("PressVoice selftest")
    edit.resize(420, 40)
    edit.show()
    edit.raise_()
    edit.activateWindow()
    if platform_utils.IS_WINDOWS:
        from qtapp import winapi
        winapi.force_foreground(int(edit.winId()))
    edit.setFocus()
    try:
        if not _wait(app, lambda: edit.isActiveWindow() and edit.hasFocus(), 3):
            raise Skip("could not focus a test window (no interactive desktop)")
        got = _paste_into(app, edit)
        if got != PASTE_SAMPLE:
            raise RuntimeError(f"pasted {got!r}")
        detail = "pasted into a text field"
        if platform_utils.IS_WINDOWS:
            from qtapp import winapi
            prev = winapi.current_keyboard_layout()
            winapi.activate_keyboard_layout("00000419")  # Russian
            try:
                got = _paste_into(app, edit)
            finally:
                if prev:
                    winapi.user32.ActivateKeyboardLayout(prev, 0)
            if got != PASTE_SAMPLE:
                raise RuntimeError(f"with Russian layout pasted {got!r}")
            detail += "; also with the Russian layout"
        return detail
    finally:
        edit.close()


def check_https(app):
    """The same client the app uses, with a fake key: a 401 proves DNS, TLS
    certificates and the HTTP stack all work inside this build."""
    import openai
    from transcribe import make_client
    try:
        make_client("sk-selftest-invalid").models.list()
    except openai.AuthenticationError:
        return "TLS ok (401 for a fake key, as expected)"
    except openai.APITimeoutError as e:
        raise Warn(f"timeout: {e}")
    except openai.APIConnectionError as e:
        cause = repr(e.__cause__ or e)
        if "CERTIFICATE" in cause.upper() or "SSL" in cause.upper():
            raise RuntimeError(f"TLS failure: {cause}")
        raise Warn(f"no network: {cause}")
    raise Warn("OpenAI accepted a fake key?")


def check_autostart(app):
    if not platform_utils.autostart_supported():
        raise Skip("not supported on this platform")
    return f"enabled={platform_utils.is_autostart_enabled()}"


CHECKS = [
    ("imports", check_imports, True),
    ("translations", check_translations, True),
    ("config_dir", check_config_dir, True),
    ("portaudio", check_portaudio, True),
    ("microphone", check_microphone, False),
    ("tray", check_tray, False),
    ("hotkey", check_hotkey, True),
    ("clipboard", check_clipboard, True),
    ("paste", check_paste, True),
    ("https", check_https, True),
    ("autostart", check_autostart, False),
]


def run_selftest(app: QApplication, report_path: str) -> int:
    checks = {}
    failed = False
    for name, fn, critical in CHECKS:
        t0 = time.time()
        try:
            status, detail = "ok", fn(app)
        except Skip as e:
            status, detail = "skip", str(e)
        except Warn as e:
            status, detail = "warn", str(e)
        except Exception as e:
            status = "fail" if critical else "warn"
            detail = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        failed |= status == "fail"
        checks[name] = {"status": status, "detail": detail,
                        "seconds": round(time.time() - t0, 2)}

    report = {
        "app": f"PressVoice {__version__}",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "frozen": platform_utils.is_frozen(),
        "executable": sys.executable,
        "ok": not failed,
        "checks": checks,
    }
    report_path = report_path or os.path.join(tempfile.gettempdir(),
                                              "pressvoice-selftest.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return 0 if not failed else 1
