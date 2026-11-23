"""Global push-to-talk / toggle hotkey for Windows and Linux (X11).

pynput installs a low-level keyboard hook (WH_KEYBOARD_LL on Windows, XRecord
on X11). Hook callbacks must return within a few hundred milliseconds or
Windows silently unhooks them, so they only emit a Qt signal; all decisions
happen on the main thread in HotkeyLogic."""
import sys
import threading
import time
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal

from i18n import t

IS_WINDOWS = sys.platform == "win32"

# id → (key name, Windows virtual-key codes, X11 keysyms, is_modifier).
# Key names are what's printed on the keys, so they stay untranslated;
# "Right …" and "Menu key" are translated in hotkey_label().
HOTKEYS: dict = {
    "ctrl_r":      ("Ctrl",        {0xA3}, {0xFFE4}, True),
    "alt_r":       ("Alt",         {0xA5}, {0xFFEA, 0xFE03, 0xFF7E}, True),
    "shift_r":     ("Shift",       {0xA1}, {0xFFE2}, True),
    "menu":        ("Menu",        {0x5D}, {0xFF67}, False),
    "pause":       ("Pause/Break", {0x13}, {0xFF13}, False),
    "scroll_lock": ("Scroll Lock", {0x91}, {0xFF14}, False),
    "insert":      ("Insert",      {0x2D}, {0xFF63}, False),
    **{f"f{n}": (f"F{n}", {0x7C + n - 13}, {0xFFCA + n - 13}, False)
       for n in range(13, 20)},
}
DEFAULT_HOTKEY = "ctrl_r"
_RIGHT_HAND = {"ctrl_r", "alt_r", "shift_r"}


def hotkey_label(key_id: str) -> str:
    """Name of a hotkey in the interface language ("Right Ctrl")."""
    key_id = hotkey_id(key_id)
    if key_id in _RIGHT_HAND:
        return t("key.right", key=HOTKEYS[key_id][0])
    if key_id == "menu":
        return t("key.menu")
    return HOTKEYS[key_id][0]

_WM_KEYDOWN, _WM_KEYUP, _WM_SYSKEYDOWN, _WM_SYSKEYUP = 0x100, 0x101, 0x104, 0x105
_VK_LCONTROL = 0xA2
_VK_MENU_MASK = 0xE8
_POLL_SECONDS = 0.15


def hotkey_id(value: str) -> str:
    return value if value in HOTKEYS else DEFAULT_HOTKEY


class HotkeyLogic:
    """Turns raw key events into start / stop / cancel. Feed it from one
    thread only (the Qt main thread).

    push_to_talk: key down starts, key up stops. Another key pressed while a
    modifier hotkey is held means the user is typing a shortcut (Right
    Ctrl+C), so the recording is cancelled.
    toggle: a clean tap (down + up with nothing in between) starts or stops.
    """

    def __init__(self, mode: str, is_modifier: bool,
                 on_start: Callable, on_stop: Callable, on_cancel: Callable):
        self.mode = mode
        self.is_modifier = is_modifier
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_cancel = on_cancel
        self.held = False
        self.recording = False
        self._combo = False
        # Until then, other keys are the app's own paste keystrokes (live
        # dictation pastes while the hotkey is held), not a shortcut.
        self.own_input_until = 0.0

    def reset(self) -> None:
        self.held = False
        self.recording = False
        self._combo = False

    def hotkey_down(self) -> None:
        if self.held:
            return  # auto-repeat
        self.held = True
        self._combo = False
        if self.mode == "push_to_talk" and not self.recording:
            self.recording = True
            self._on_start()

    def other_key_down(self) -> None:
        if not (self.held and self.is_modifier) or self._combo:
            return
        if time.monotonic() < self.own_input_until:
            return
        self._combo = True
        if self.mode == "push_to_talk" and self.recording:
            self.recording = False
            self._on_cancel()

    def hotkey_up(self) -> None:
        if not self.held:
            return
        self.held = False
        if self.mode == "push_to_talk":
            if self.recording:
                self.recording = False
                self._on_stop()
            return
        if self._combo:
            return
        if self.recording:
            self.recording = False
            self._on_stop()
        else:
            self.recording = True
            self._on_start()


def _vk_of(key) -> Optional[int]:
    """Platform key code of a pynput key (Windows VK or X11 keysym)."""
    value = getattr(key, "value", key)  # Key enum member → its KeyCode
    return getattr(value, "vk", None)


class HotkeyListener(QObject):
    """Owns the pynput listener; emits raw events onto the main thread."""

    _raw = Signal(str)  # "down" | "up" | "other"

    def __init__(self, log: Callable[[str], None] = lambda m: None):
        super().__init__()
        self._log = log
        self._listener = None
        self._logic: Optional[HotkeyLogic] = None
        self._hotkey = DEFAULT_HOTKEY
        self._codes: set = set()
        self._win_vks: set = set()
        self._is_modifier = True
        self._poll_active = False
        self._raw.connect(self._on_raw)

    @property
    def logic(self) -> Optional[HotkeyLogic]:
        return self._logic

    def start(self, hotkey: str, mode: str, on_start: Callable,
              on_stop: Callable, on_cancel: Callable) -> None:
        from pynput import keyboard

        self.stop()
        self._hotkey = hotkey_id(hotkey)
        _label, win_vks, x_syms, self._is_modifier = HOTKEYS[self._hotkey]
        self._win_vks = win_vks
        self._codes = win_vks if IS_WINDOWS else x_syms
        self._logic = HotkeyLogic(mode, self._is_modifier,
                                  on_start, on_stop, on_cancel)
        kwargs = {}
        if IS_WINDOWS:
            kwargs["win32_event_filter"] = self._win32_filter
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release, **kwargs
        )
        self._listener.daemon = True
        self._listener.start()
        self._log(f"hotkey: listening for {self._hotkey} ({mode})")

    def stop(self) -> None:
        self._poll_active = False
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def is_alive(self) -> bool:
        return self._listener is not None and self._listener.is_alive()

    def reset(self) -> None:
        self._poll_active = False
        if self._logic:
            self._logic.reset()

    def ignore_own_input(self, seconds: float = 0.5) -> None:
        """Main thread, right before the app sends keystrokes of its own:
        they must not look like a shortcut that cancels the recording."""
        if self._logic:
            self._logic.own_input_until = time.monotonic() + seconds

    # ── Hook thread ──────────────────────────────────────────────────────────
    # Anything raised here would stop the pynput listener for good, so every
    # callback swallows its own errors.

    def _win32_filter(self, msg, data):
        """Windows only: sees the raw event before pynput translates it, so
        the hotkey can be matched by virtual-key code and, for keys that have
        their own effect (Insert, Menu, F-keys), hidden from other apps."""
        try:
            if data.vkCode not in self._win_vks:
                return True
            if msg in (_WM_KEYDOWN, _WM_SYSKEYDOWN):
                self._raw.emit("down")
            elif msg in (_WM_KEYUP, _WM_SYSKEYUP):
                self._raw.emit("up")
            if self._is_modifier:
                return False  # handled; let other apps see it, skip on_press
            listener = self._listener
            if listener is None:
                return False
        except Exception:
            return True
        listener.suppress_event()  # raises; the key goes nowhere else

    def _on_press(self, key, injected=False):
        try:
            vk = _vk_of(key)
            if IS_WINDOWS:
                # The hotkey itself was handled in _win32_filter.
                if vk == _VK_MENU_MASK:
                    return
                if self._hotkey == "alt_r" and vk == _VK_LCONTROL:
                    return  # AltGr's phantom Left Ctrl
                self._raw.emit("other")
            elif vk in self._codes:
                self._raw.emit("down")
            else:
                self._raw.emit("other")
        except Exception:
            pass

    def _on_release(self, key, injected=False):
        try:
            if not IS_WINDOWS and _vk_of(key) in self._codes:
                self._raw.emit("up")
        except Exception:
            pass

    # ── Main thread ──────────────────────────────────────────────────────────

    def _on_raw(self, kind: str) -> None:
        logic = self._logic
        if logic is None:
            return
        try:
            if kind == "down":
                was_held = logic.held
                logic.hotkey_down()
                if not was_held:
                    self._after_key_down()
            elif kind == "up":
                self._poll_active = False
                logic.hotkey_up()
            else:
                logic.other_key_down()
        except Exception as e:
            self._log(f"hotkey handler error: {e}")

    def _after_key_down(self) -> None:
        if not IS_WINDOWS:
            return
        from qtapp import winapi
        if self._hotkey == "alt_r":
            # Without this, releasing Alt alone opens the focused window's
            # menu bar.
            winapi.tap_menu_mask()
        if self._logic and self._logic.mode == "push_to_talk":
            self._start_polling()

    def _start_polling(self) -> None:
        """Safety net for a lost key-up (e.g. the hook missed it while a UAC
        prompt had the secure desktop): poll the physical key state."""
        self._poll_active = True
        vks = tuple(self._win_vks)
        threading.Thread(target=self._poll_loop, args=(vks,),
                         name="hotkey-poll", daemon=True).start()

    def _poll_loop(self, vks: tuple) -> None:
        from qtapp import winapi
        while self._poll_active:
            time.sleep(_POLL_SECONDS)
            if not self._poll_active:
                break
            try:
                if not any(winapi.is_key_down(vk) for vk in vks):
                    self._poll_active = False
                    self._raw.emit("up")
                    break
            except Exception:
                break
