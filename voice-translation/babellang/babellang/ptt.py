from __future__ import annotations

import logging
import threading
from typing import Optional

from pynput import keyboard as _kb

logger = logging.getLogger(__name__)

# Human-readable names for the settings UI
HOTKEY_NAMES: dict[str, str] = {
    "space":   "Пробел",
    "ctrl_l":  "⌃ Ctrl (L)",
    "ctrl_r":  "⌃ Ctrl (R)",
    "cmd":     "⌘ Cmd",
    "cmd_l":   "⌘ Cmd (L)",
    "alt_l":   "⌥ Option (L)",
    "alt_r":   "⌥ Option (R)",
    "shift_l": "⇧ Shift (L)",
    "shift_r": "⇧ Shift (R)",
    **{f"f{i}": f"F{i}" for i in range(1, 13)},
}

# tkinter keysym → our hotkey string
TK_TO_HOTKEY: dict[str, str] = {
    "space":     "space",
    "Control_L": "ctrl_l",
    "Control_R": "ctrl_r",
    "Meta_L":    "cmd_l",
    "Meta_R":    "cmd",
    "Alt_L":     "alt_l",
    "Alt_R":     "alt_r",
    "Option_L":  "alt_l",  # macOS alias
    "Shift_L":   "shift_l",
    "Shift_R":   "shift_r",
    **{f"F{i}": f"f{i}" for i in range(1, 13)},
}

_PYNPUT_KEY: dict[str, _kb.Key | _kb.KeyCode] = {
    "space":   _kb.Key.space,
    "ctrl_l":  _kb.Key.ctrl_l,
    "ctrl_r":  _kb.Key.ctrl_r,
    "cmd":     _kb.Key.cmd,
    "cmd_l":   _kb.Key.cmd_l,
    "alt_l":   _kb.Key.alt_l,
    "alt_r":   _kb.Key.alt_r,
    "shift_l": _kb.Key.shift_l,
    "shift_r": _kb.Key.shift_r,
    **{f"f{i}": getattr(_kb.Key, f"f{i}") for i in range(1, 13)},
}


def hotkey_display(key_str: str) -> str:
    return HOTKEY_NAMES.get(key_str, key_str.upper())


def _resolve(key_str: str) -> _kb.Key | _kb.KeyCode:
    if key_str in _PYNPUT_KEY:
        return _PYNPUT_KEY[key_str]
    return _kb.KeyCode.from_char(key_str)


class PttState:
    """Thread-safe push-to-talk flag shared between the pynput listener thread
    and the asyncio pipeline coroutine."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def press(self) -> None:
        with self._lock:
            self._active = True

    def release(self) -> None:
        with self._lock:
            self._active = False


class PttListener:
    """Global keyboard listener that updates a PttState on key press/release."""

    def __init__(self, hotkey_str: str, state: PttState) -> None:
        self._target = _resolve(hotkey_str)
        self._state  = state
        self._listener: Optional[_kb.Listener] = None

    def start(self) -> None:
        self._listener = _kb.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        logger.info("PTT listener started (key=%s)", self._target)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
            logger.info("PTT listener stopped")

    # ── pynput callbacks (called from listener thread) ────────────────────────

    def _on_press(self, key: _kb.Key | _kb.KeyCode) -> None:
        if self._matches(key):
            self._state.press()

    def _on_release(self, key: _kb.Key | _kb.KeyCode) -> None:
        if self._matches(key):
            self._state.release()

    def _matches(self, key: _kb.Key | _kb.KeyCode) -> bool:
        if key == self._target:
            return True
        if (hasattr(key, "char") and hasattr(self._target, "char")
                and key.char is not None):
            return key.char == self._target.char
        return False
