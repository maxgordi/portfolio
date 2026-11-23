"""Types live dictation into whatever app has the keyboard focus, piece by
piece and in order, from its own thread (so the audio and network threads
never wait).

Synthetic key events carry the text itself (Unicode), so the clipboard is
untouched and any language works. The events come from a private event
source with no modifier flags: the hotkey still being held (Right ⌘) can't
turn the text into shortcuts."""
import queue
import threading
import time

_PIECE = 16  # UTF-16 units per key event (the API limit is 20)


def _utf16_pieces(text: str):
    piece, units = "", 0
    for ch in text:
        n = 2 if ord(ch) > 0xFFFF else 1
        if units + n > _PIECE:
            yield piece, units
            piece, units = "", 0
        piece += ch
        units += n
    if piece:
        yield piece, units


def type_unicode(text: str) -> None:
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventKeyboardSetUnicodeString,
        CGEventPost,
        CGEventSetFlags,
        CGEventSourceCreate,
        kCGEventSourceStatePrivate,
        kCGHIDEventTap,
    )
    src = CGEventSourceCreate(kCGEventSourceStatePrivate)
    for piece, units in _utf16_pieces(text):
        for down in (True, False):
            ev = CGEventCreateKeyboardEvent(src, 0, down)
            CGEventKeyboardSetUnicodeString(ev, units, piece)
            CGEventSetFlags(ev, 0)
            CGEventPost(kCGHIDEventTap, ev)
        time.sleep(0.004)


class MacTyper:
    def __init__(self, log=lambda m: None, type_text=type_unicode):
        self._log = log
        self._type = type_text
        self._enabled = True
        self._q: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._run, name="typer", daemon=True).start()

    def begin(self, enabled: bool) -> None:
        """enabled=False (no Accessibility permission): macOS would drop
        the keystrokes, so nothing is typed and the app hands the text over
        another way when the dictation ends."""
        self._enabled = enabled

    def write(self, text: str) -> None:
        if text and self._enabled:
            self._q.put(text)

    def finish(self, timeout: float = 10.0) -> None:
        """Wait until everything queued has been typed."""
        done = threading.Event()
        self._q.put(done)
        done.wait(timeout)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if isinstance(item, threading.Event):
                item.set()
                continue
            try:
                self._type(item)
            except Exception as e:
                self._log(f"typing failed: {e}")
