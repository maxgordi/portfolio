import threading
import time
from typing import Callable, Optional

from AppKit import (
    NSCommandKeyMask,
    NSControlKeyMask,
    NSEvent,
    NSFlagsChangedMask,
    NSKeyDownMask,
    NSKeyUpMask,
    NSShiftKeyMask,
)
from Quartz import (
    CGEventSourceFlagsState,
    kCGEventSourceStateCombinedSessionState,
)

KEY_CODES: dict[str, int] = {
    "cmd_r": 54,
    "cmd_l": 55,
    "ctrl_r": 62,
    "shift_r": 60,
    "f18": 79,
    "f19": 80,
}

# id → (side, key symbol); the side word is translated in hotkey_label().
_HOTKEY_PARTS: dict = {
    "cmd_r": ("right", "⌘"),
    "cmd_l": ("left", "⌘"),
    "ctrl_r": ("right", "⌃"),
    "shift_r": ("right", "⇧"),
    "f18": (None, "F18"),
    "f19": (None, "F19"),
}
HOTKEY_IDS = list(_HOTKEY_PARTS)


def hotkey_label(key_id: str) -> str:
    """Name of a hotkey in the interface language ("Right ⌘")."""
    from i18n import t
    side, symbol = _HOTKEY_PARTS.get(key_id, _HOTKEY_PARTS["cmd_r"])
    return t(f"key.{side}", key=symbol) if side else symbol

_MODIFIER_KEYS: set[str] = {"cmd_r", "cmd_l", "ctrl_r", "shift_r"}

# NSEvent modifier masks
_NS_FLAGS: dict[str, int] = {
    "cmd_r": NSCommandKeyMask,
    "cmd_l": NSCommandKeyMask,
    "ctrl_r": NSControlKeyMask,
    "shift_r": NSShiftKeyMask,
}

# Quartz CGEvent modifier masks (same numeric values, but via different API)
_CG_FLAGS: dict[str, int] = {
    "cmd_r": 0x100000,   # kCGEventFlagMaskCommand
    "cmd_l": 0x100000,
    "ctrl_r": 0x040000,  # kCGEventFlagMaskControl
    "shift_r": 0x020000, # kCGEventFlagMaskShift
}

_TYPE_KEY_DOWN = 10
_TYPE_KEY_UP = 11


class HotkeyListener:
    def __init__(self):
        self._monitor = None
        self._hotkey = "cmd_r"
        self._mode = "push_to_talk"
        self._on_start: Optional[Callable] = None
        self._on_stop: Optional[Callable] = None
        self._dispatch_to_main: Optional[Callable] = None
        self._recording = False
        self._key_is_down = False
        self._lock = threading.Lock()
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_active = False
        self._poll_release = True

    def start(
        self,
        hotkey: str,
        mode: str,
        on_start: Callable,
        on_stop: Callable,
        dispatch_to_main: Optional[Callable] = None,
        poll_release: bool = True,
    ) -> None:
        """Must be called from the main thread after NSApp run loop has started.
        poll_release=False turns off the missed-release poller: live
        dictation types while the key is held, and the typed events reset
        the combined modifier state the poller reads."""
        self.stop()
        self._poll_release = poll_release
        self._hotkey = hotkey
        self._mode = mode
        self._on_start = on_start
        self._on_stop = on_stop
        self._dispatch_to_main = dispatch_to_main
        with self._lock:
            self._recording = False
            self._key_is_down = False

        is_modifier = hotkey in _MODIFIER_KEYS
        mask = NSFlagsChangedMask if is_modifier else (NSKeyDownMask | NSKeyUpMask)
        self._monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
            mask, self._handle_event
        )

    def stop(self) -> None:
        self._poll_active = False
        if self._monitor:
            try:
                NSEvent.removeMonitor_(self._monitor)
            except Exception:
                pass
            self._monitor = None

    # ── NSEvent handler (main thread) ─────────────────────────────────────────

    def _handle_event(self, event) -> None:
        # Called by AppKit on the main thread; an escaping exception would
        # take down the run loop, so the whole handler is guarded.
        try:
            self._handle_event_inner(event)
        except Exception:
            pass

    def _handle_event_inner(self, event) -> None:
        key_code = KEY_CODES.get(self._hotkey, 54)
        if event.keyCode() != key_code:
            return

        if self._hotkey in _MODIFIER_KEYS:
            flag = _NS_FLAGS.get(self._hotkey, NSCommandKeyMask)
            is_pressed = bool(event.modifierFlags() & flag)
            with self._lock:
                was_down = self._key_is_down
                self._key_is_down = is_pressed
            if is_pressed and not was_down:
                self._fire_key_down()
            elif not is_pressed and was_down:
                self._fire_key_up()
        else:
            etype = event.type()
            with self._lock:
                was_down = self._key_is_down
                self._key_is_down = (etype == _TYPE_KEY_DOWN)
            if etype == _TYPE_KEY_DOWN and not was_down:
                self._fire_key_down()
            elif etype == _TYPE_KEY_UP and was_down:
                self._fire_key_up()

    # ── Key state transitions ─────────────────────────────────────────────────

    def _fire_key_down(self) -> None:
        if self._mode == "push_to_talk":
            with self._lock:
                if self._recording:
                    return
                self._recording = True
            self._on_start()
            self._start_polling()
        else:  # toggle
            with self._lock:
                toggle_on = not self._recording
                self._recording = toggle_on
            if toggle_on:
                self._on_start()
            else:
                self._on_stop()

    def _fire_key_up(self) -> None:
        if self._mode == "push_to_talk":
            with self._lock:
                if not self._recording:
                    return
                self._recording = False
            self._poll_active = False
            self._on_stop()

    # ── Polling fallback (background thread) ──────────────────────────────────

    def _start_polling(self) -> None:
        """Start a background thread that watches for missed key-release events."""
        cg_flag = _CG_FLAGS.get(self._hotkey)
        if cg_flag is None or not self._poll_release:
            return  # No polling for non-modifier keys (F18/F19 don't need it)
        self._poll_active = True
        t = threading.Thread(target=self._poll_loop, args=(cg_flag,), daemon=True)
        t.start()

    def _poll_loop(self, cg_flag: int) -> None:
        """Run in background thread; checks every 150 ms if the key is still held."""
        while self._poll_active:
            time.sleep(0.15)
            if not self._poll_active:
                break
            with self._lock:
                if not self._recording:
                    break
            current_flags = CGEventSourceFlagsState(kCGEventSourceStateCombinedSessionState)
            is_held = bool(current_flags & cg_flag)
            if not is_held:
                with self._lock:
                    if not self._recording:
                        break
                    self._recording = False
                    self._key_is_down = False
                self._poll_active = False
                if self._mode == "push_to_talk":
                    if self._dispatch_to_main:
                        self._dispatch_to_main(self._on_stop)
                    else:
                        self._on_stop()
                break
