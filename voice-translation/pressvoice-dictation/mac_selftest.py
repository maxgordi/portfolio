"""`PressVoice.app/Contents/MacOS/PressVoice --selftest report.json`:
checks that a packaged (or source) copy works on this Mac — bundle and
signature, audio library, clipboard, TLS to OpenAI and, when Accessibility is
granted, the real hotkey monitor and Cmd+V paste — and writes JSON.
Exit code 0 = all critical checks passed.

The checks run on a worker thread while the main thread runs the AppKit
event loop, exactly like the app: global key monitors and paste only work
inside a running NSApplication."""
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import config as cfg_module
import mac_support

PASTE_SAMPLE = "PressVoice selftest ✓ тест 123"
_RIGHT_CMD = 54
_NX_DEVICERCMDKEYMASK = 0x10  # "right" bit that accompanies the Command flag


class Skip(Exception):
    """This environment can't run the check (e.g. no permission granted)."""


class Warn(Exception):
    """Not fatal, but worth knowing."""


def _on_main(fn, timeout: float = 10.0):
    """Run fn on the AppKit main thread and return its result."""
    from PyObjCTools import AppHelper
    done = threading.Event()
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: B902 — re-raised on the worker
            box["error"] = e
        finally:
            done.set()

    AppHelper.callAfter(run)
    if not done.wait(timeout):
        raise RuntimeError("the main thread did not respond")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _wait(cond, timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return bool(cond())


# ── Checks ───────────────────────────────────────────────────────────────────

def check_imports():
    import numpy
    import objc
    import openai
    import rumps
    import sounddevice
    return (f"rumps {rumps.__version__}, pyobjc {objc.__version__}, "
            f"openai {openai.__version__}, numpy {numpy.__version__}, "
            f"sounddevice {sounddevice.__version__}")


def check_app_module():
    """Import the menu bar app itself (without starting it): catches a
    broken build or startup-time error before the user ever sees it."""
    import app
    return f"loaded {app.__name__} ({len(app.PressVoiceApp.__dict__)} members)"


def check_translations():
    """Every language file made it into the build and is complete."""
    import i18n
    expected = len(i18n._strings("en"))
    broken = [c for c in i18n.LANGUAGES if len(i18n._strings(c)) != expected]
    if not expected or broken:
        raise RuntimeError(f"missing or incomplete: {broken or ['en']} in {i18n.locales_dir()}")
    system = i18n.detect_system_language()
    return (f"{len(i18n.LANGUAGES)} languages × {expected} strings; "
            f"system language here: {i18n.LANGUAGES[system]}")


def check_bundle():
    from Foundation import NSBundle
    if not mac_support.is_frozen():
        raise Skip("running from source, not from PressVoice.app")
    info = NSBundle.mainBundle().infoDictionary()
    problems = []
    if info.get("CFBundleIdentifier") != mac_support.BUNDLE_ID:
        problems.append(f"bundle id {info.get('CFBundleIdentifier')}")
    if not info.get("LSUIElement"):
        problems.append("LSUIElement missing (would show a Dock icon)")
    if not info.get("NSMicrophoneUsageDescription"):
        problems.append("NSMicrophoneUsageDescription missing (mic would be denied)")
    if problems:
        raise RuntimeError("; ".join(problems))
    return (f"{mac_support.app_bundle_path()} — {info.get('CFBundleIdentifier')} "
            f"{info.get('CFBundleShortVersionString')}")


def check_codesign():
    bundle = mac_support.app_bundle_path()
    if not bundle:
        raise Skip("not running from an app bundle")
    r = subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", bundle],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"signature broken: {r.stderr.strip()}")
    d = subprocess.run(["/usr/bin/codesign", "-dvv", bundle], capture_output=True, text=True)
    lines = d.stderr.splitlines()
    authority = next((ln.split("=", 1)[1] for ln in lines if ln.startswith("Authority=")), "")
    if not authority:
        raise Warn("valid but ad-hoc signed: macOS will ask for Accessibility "
                   "again after every update")
    return f"valid, signed by “{authority}” (permissions survive updates)"


def check_config_dir():
    for d in {cfg_module.CONFIG_DIR, os.path.dirname(_log_path())}:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".selftest")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    return f"config: {cfg_module.CONFIG_DIR}; log: {_log_path()}"


def _log_path() -> str:
    return mac_support.log_path(os.path.dirname(os.path.abspath(__file__)))


def check_portaudio():
    import sounddevice as sd
    from audio import AudioRecorder
    devices = AudioRecorder.list_devices()
    return (f"{sd.get_portaudio_version()[1]}; "
            f"{len(devices)} input device(s): {[d['name'] for d in devices][:5]}")


def check_microphone():
    if os.environ.get("PRESSVOICE_SELFTEST_RECORD") != "1":
        raise Skip("set PRESSVOICE_SELFTEST_RECORD=1 to record (asks for mic permission)")
    from audio import AudioRecorder
    from transcribe import is_digital_silence
    rec = AudioRecorder()
    rec.start()
    time.sleep(0.8)
    wav = rec.stop()
    if not wav:
        raise Warn("no audio captured")
    if is_digital_silence(wav):
        raise Warn("only silence — microphone permission not granted?")
    return f"recorded {len(wav)} bytes of real audio"


def check_https():
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


def check_clipboard():
    def roundtrip():
        saved = mac_support.save_clipboard()
        try:
            mac_support.copy_to_clipboard(PASTE_SAMPLE)
            return mac_support.clipboard_text()
        finally:
            mac_support.restore_clipboard(saved)
    got = _on_main(roundtrip)
    if got != PASTE_SAMPLE:
        raise RuntimeError(f"read back {got!r}")
    return "round-trip ok (Cyrillic included); your clipboard was restored"


def _trusted() -> bool:
    from ApplicationServices import AXIsProcessTrusted
    return bool(AXIsProcessTrusted())


def check_accessibility():
    if not _trusted():
        raise Warn("not granted — System Settings → Privacy & Security → "
                   "Accessibility → PressVoice. Hotkey and paste checks skipped.")
    return "granted"


def _post_right_cmd(down: bool) -> None:
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventPost,
        CGEventSetFlags,
        CGEventSetType,
        kCGEventFlagMaskCommand,
        kCGEventFlagsChanged,
        kCGHIDEventTap,
    )
    ev = CGEventCreateKeyboardEvent(None, _RIGHT_CMD, down)
    CGEventSetType(ev, kCGEventFlagsChanged)
    CGEventSetFlags(ev, (kCGEventFlagMaskCommand | _NX_DEVICERCMDKEYMASK) if down else 0)
    CGEventPost(kCGHIDEventTap, ev)


def check_hotkey():
    if not _trusted():
        raise Skip("needs Accessibility")
    from hotkey import HotkeyListener
    events = []
    listener = HotkeyListener()
    _on_main(lambda: listener.start(
        hotkey="cmd_r", mode="push_to_talk",
        on_start=lambda: events.append("start"),
        on_stop=lambda: events.append("stop"),
        dispatch_to_main=lambda f: f()))
    try:
        time.sleep(0.3)
        _post_right_cmd(True)
        if not _wait(lambda: "start" in events, 3):
            _post_right_cmd(False)
            raise RuntimeError("the key monitor did not see Right ⌘")
        # Under 0.4 s, so a PressVoice already running on this Mac ignores it.
        time.sleep(0.15)
        _post_right_cmd(False)
        if not _wait(lambda: "stop" in events, 3):
            raise RuntimeError("press seen, release not seen")
        return f"events {events}"
    finally:
        _on_main(listener.stop)


def check_paste():
    """Pastes into a text field of our own window — and only if that window
    really is frontmost, so a keystroke can never land in another app."""
    if not _trusted():
        raise Skip("needs Accessibility")
    from AppKit import (
        NSApp,
        NSBackingStoreBuffered,
        NSMakeRect,
        NSTextField,
        NSWindow,
        NSWindowStyleMaskTitled,
    )
    ui = {}

    def make_window():
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(240, 240, 440, 60), NSWindowStyleMaskTitled,
            NSBackingStoreBuffered, False)
        win.setTitle_("PressVoice self-test")
        win.setReleasedWhenClosed_(False)
        field = NSTextField.alloc().initWithFrame_(NSMakeRect(12, 18, 416, 24))
        win.contentView().addSubview_(field)
        win.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        win.makeFirstResponder_(field)
        ui.update(win=win, field=field)

    def frontmost() -> bool:
        return bool(NSApp.isActive() and ui["win"].isKeyWindow())

    _on_main(make_window)
    saved = _on_main(mac_support.save_clipboard)
    try:
        if not _wait(lambda: _on_main(frontmost), 3):
            raise Skip("could not bring the test window to the front")

        def paste():
            if not frontmost():
                raise Skip("lost focus before pasting")
            mac_support.copy_to_clipboard(PASTE_SAMPLE)
            mac_support.send_cmd_v()
        _on_main(paste)
        if not _wait(lambda: _on_main(lambda: str(ui["field"].stringValue())) == PASTE_SAMPLE, 3):
            got = _on_main(lambda: str(ui["field"].stringValue()))
            raise RuntimeError(f"field contains {got!r}")
        return "Cmd+V pasted into a text field (Edit-menu shortcuts work)"
    finally:
        _on_main(lambda: mac_support.restore_clipboard(saved))
        _on_main(lambda: ui["win"].close() if "win" in ui else None)


def check_login_item():
    if not mac_support.login_item_supported():
        raise Skip("only for the packaged app")
    return f"enabled={mac_support.is_login_item_enabled()} ({mac_support.LAUNCH_AGENT_PATH})"


CHECKS = [
    ("imports", check_imports, True),
    ("app_module", check_app_module, True),
    ("translations", check_translations, True),
    ("bundle", check_bundle, True),
    ("codesign", check_codesign, True),
    ("config_dir", check_config_dir, True),
    ("portaudio", check_portaudio, True),
    ("microphone", check_microphone, False),
    ("https", check_https, True),
    ("clipboard", check_clipboard, True),
    ("accessibility", check_accessibility, False),
    ("hotkey", check_hotkey, True),
    ("paste", check_paste, True),
    ("login_item", check_login_item, False),
]


def _run_checks(report_path: str) -> int:
    checks = {}
    failed = False
    for name, fn, critical in CHECKS:
        t0 = time.time()
        try:
            status, detail = "ok", fn()
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

    from version import __version__
    report = {
        "app": f"PressVoice {__version__} (macOS)",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "frozen": mac_support.is_frozen(),
        "executable": sys.executable,
        "ok": not failed,
        "checks": checks,
    }
    path = report_path or os.path.join(tempfile.gettempdir(), "pressvoice-selftest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return 0 if not failed else 1


def run_selftest(report_path: str) -> int:
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    mac_support.install_edit_menu()

    def worker():
        code = 1
        try:
            code = _run_checks(report_path)
        except Exception:
            traceback.print_exc()
        finally:
            os._exit(code)  # the AppKit loop never returns on its own

    threading.Thread(target=worker, name="selftest", daemon=True).start()
    app.run()
    return 1
