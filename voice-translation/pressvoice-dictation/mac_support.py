"""macOS helpers shared by the menu bar app and its self-test: where files
go, the login item, clipboard, paste keystroke, single instance."""
import fcntl
import os
import plistlib
import sys
import time
from typing import Optional

from AppKit import (
    NSURL,
    NSApplication,
    NSEventModifierFlagCommand,
    NSEventModifierFlagShift,
    NSMenu,
    NSMenuItem,
    NSPasteboard,
    NSPasteboardItem,
    NSPasteboardTypeString,
    NSWorkspace,
)
from ApplicationServices import (
    AXIsProcessTrusted,
    AXIsProcessTrustedWithOptions,
    kAXTrustedCheckOptionPrompt,
)
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    CGEventSourceCreate,
    kCGEventFlagMaskCommand,
    kCGEventSourceStateHIDSystemState,
    kCGHIDEventTap,
)

import config

# Deliberately not "com.pressvoice.app": that id belongs to the old
# source-checkout launcher, and two apps sharing one id makes System Settings
# bind the Accessibility switch to the wrong one.
BUNDLE_ID = "com.pressvoice.mac"
LAUNCH_AGENT_PATH = os.path.expanduser(f"~/Library/LaunchAgents/{BUNDLE_ID}.plist")
_LEGACY_LAUNCH_AGENTS = [os.path.expanduser("~/Library/LaunchAgents/com.pressvoice.app.plist")]
ACCESSIBILITY_PANE = ("x-apple.systempreferences:com.apple.preference.security"
                      "?Privacy_Accessibility")
_KEYCODE_V = 9
_lock_file = None


def is_frozen() -> bool:
    """Running as the packaged PressVoice.app (not from source)."""
    return bool(getattr(sys, "frozen", False))


def app_bundle_path() -> Optional[str]:
    """/Applications/PressVoice.app when packaged, else None."""
    if not is_frozen():
        return None
    path = os.path.abspath(sys.executable)  # …/PressVoice.app/Contents/MacOS/PressVoice
    for _ in range(3):
        path = os.path.dirname(path)
    return path if path.endswith(".app") else None


def log_path(source_dir: str) -> str:
    # A signed app bundle must never be written to, so the packaged app logs
    # where Console.app looks for user logs (tests point it at their own dir).
    if os.environ.get("PRESSVOICE_CONFIG_DIR"):
        return os.path.join(config.CONFIG_DIR, "pressvoice.log")
    if is_frozen():
        return os.path.expanduser("~/Library/Logs/PressVoice/pressvoice.log")
    return os.path.join(source_dir, "pressvoice.log")


# ── Single instance ──────────────────────────────────────────────────────────

def acquire_single_instance() -> bool:
    """False if another PressVoice already runs. The lock dies with the
    process (and is not inherited across the self-restart exec)."""
    global _lock_file
    try:
        os.makedirs(config.CONFIG_DIR, exist_ok=True)
        f = open(os.path.join(config.CONFIG_DIR, "pressvoice.lock"), "w")
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    f.write(str(os.getpid()))
    f.flush()
    _lock_file = f
    return True


# ── Login item ───────────────────────────────────────────────────────────────
# A per-user LaunchAgent: starts PressVoice at login and, unlike a plain
# login item, relaunches it if it ever crashes (a clean Quit is left alone).

def login_item_supported() -> bool:
    return is_frozen()


def is_login_item_enabled() -> bool:
    return os.path.exists(LAUNCH_AGENT_PATH)


def set_login_item(enabled: bool) -> None:
    if not enabled:
        if os.path.exists(LAUNCH_AGENT_PATH):
            os.remove(LAUNCH_AGENT_PATH)
        return
    plist = {
        "Label": BUNDLE_ID,
        "ProgramArguments": [os.path.abspath(sys.executable)],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Interactive",
        "LimitLoadToSessionType": "Aqua",
    }
    os.makedirs(os.path.dirname(LAUNCH_AGENT_PATH), exist_ok=True)
    tmp = LAUNCH_AGENT_PATH + ".tmp"
    with open(tmp, "wb") as f:
        plistlib.dump(plist, f)
    os.replace(tmp, LAUNCH_AGENT_PATH)


def remove_legacy_login_items() -> bool:
    """Remove login items written by 1.0.0/1.0.1 under the old bundle id.
    True if one was there (the caller carries the choice over)."""
    removed = False
    for path in _LEGACY_LAUNCH_AGENTS:
        try:
            with open(path, "rb") as f:
                program = (plistlib.load(f).get("ProgramArguments") or [""])[0]
            if program.endswith("PressVoice.app/Contents/MacOS/PressVoice"):
                os.remove(path)
                removed = True
        except (OSError, ValueError, plistlib.InvalidFileException):
            pass
    return removed


def refresh_login_item() -> None:
    """Point an existing login item at wherever the app lives now."""
    if not (login_item_supported() and is_login_item_enabled()):
        return
    try:
        with open(LAUNCH_AGENT_PATH, "rb") as f:
            current = plistlib.load(f).get("ProgramArguments", [None])[0]
        if current != os.path.abspath(sys.executable):
            set_login_item(True)
    except Exception:
        set_login_item(True)


# ── Edit menu ────────────────────────────────────────────────────────────────

def install_edit_menu() -> None:
    """A menu bar app has no main menu, and without one Cmd+C / Cmd+V do
    nothing in text fields — not even pasting the API key into Settings.
    The menu stays invisible (LSUIElement); only its shortcuts matter."""
    main = NSMenu.alloc().init()
    app_item = NSMenuItem.alloc().init()
    app_item.setSubmenu_(NSMenu.alloc().init())
    main.addItem_(app_item)

    edit = NSMenu.alloc().initWithTitle_("Edit")
    for entry in (("Undo", "undo:", "z", 0),
                  ("Redo", "redo:", "z", NSEventModifierFlagShift),
                  None,
                  ("Cut", "cut:", "x", 0),
                  ("Copy", "copy:", "c", 0),
                  ("Paste", "paste:", "v", 0),
                  ("Select All", "selectAll:", "a", 0)):
        if entry is None:
            edit.addItem_(NSMenuItem.separatorItem())
            continue
        title, action, key, extra = entry
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
        item.setKeyEquivalentModifierMask_(NSEventModifierFlagCommand | extra)
        edit.addItem_(item)
    edit_item = NSMenuItem.alloc().init()
    edit_item.setSubmenu_(edit)
    main.addItem_(edit_item)
    NSApplication.sharedApplication().setMainMenu_(main)


# ── Accessibility ────────────────────────────────────────────────────────────
# Without it macOS still delivers the modifier hotkey, so recording works,
# but it silently drops the synthetic Cmd+V: the text never gets pasted.

def is_trusted() -> bool:
    return bool(AXIsProcessTrusted())


def prompt_accessibility() -> None:
    """macOS's own "would like to control this computer" dialog; it also
    puts PressVoice into the Accessibility list, switched off."""
    AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})


def open_accessibility_settings() -> None:
    NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(ACCESSIBILITY_PANE))


# ── Clipboard and paste ──────────────────────────────────────────────────────

def copy_to_clipboard(text: str) -> None:
    # NSPasteboard rather than pbcopy: launched from Finder there is no LANG,
    # and pbcopy then garbles anything that isn't ASCII (Cyrillic included).
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    if not pb.setString_forType_(text, NSPasteboardTypeString):
        raise RuntimeError("the clipboard refused the text")


def clipboard_text() -> str:
    return NSPasteboard.generalPasteboard().stringForType_(NSPasteboardTypeString) or ""


def save_clipboard() -> list:
    """Snapshot of every item and type on the clipboard."""
    saved = []
    for item in NSPasteboard.generalPasteboard().pasteboardItems() or []:
        saved.append({t: item.dataForType_(t) for t in item.types()})
    return saved


def restore_clipboard(saved: list) -> None:
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    items = []
    for types in saved:
        item = NSPasteboardItem.alloc().init()
        for t, data in types.items():
            if data is not None:
                item.setData_forType_(data, t)
        items.append(item)
    if items:
        pb.writeObjects_(items)


def send_cmd_v() -> None:
    """Cmd+V to the frontmost app. Needs the Accessibility permission."""
    time.sleep(0.05)
    src = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
    key_down = CGEventCreateKeyboardEvent(src, _KEYCODE_V, True)
    key_up = CGEventCreateKeyboardEvent(src, _KEYCODE_V, False)
    CGEventSetFlags(key_down, kCGEventFlagMaskCommand)
    CGEventSetFlags(key_up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, key_down)
    CGEventPost(kCGHIDEventTap, key_up)
