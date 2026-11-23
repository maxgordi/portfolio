"""Put text on the clipboard and paste it into the focused window."""
import sys
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QGuiApplication

IS_WINDOWS = sys.platform == "win32"

# Give the clipboard owner change a moment to land before the target app
# asks for the data.
PASTE_DELAY_MS = 60


def copy_to_clipboard(text: str) -> None:
    """Main thread only."""
    QGuiApplication.clipboard().setText(text)


def send_paste_keystroke() -> bool:
    if IS_WINDOWS:
        from qtapp import winapi
        # 0 means Windows refused the input — usually the focused window is
        # elevated (run as administrator) and we are not.
        return winapi.send_ctrl_v() == 4
    from pynput.keyboard import Controller, Key, KeyCode
    kb = Controller()
    with kb.pressed(Key.ctrl):
        v = KeyCode.from_char("v")
        kb.press(v)
        kb.release(v)
    return True


def paste_text(text: str, on_done: Callable[[bool], None] = lambda ok: None) -> None:
    """Main thread only. The keystroke goes out after the event loop has had
    a chance to run: on Windows the clipboard data is served from this
    thread, so blocking here would hang the paste."""
    copy_to_clipboard(text)

    def fire():
        try:
            ok = send_paste_keystroke()
        except Exception:
            ok = False
        on_done(ok)

    QTimer.singleShot(PASTE_DELAY_MS, fire)


class LivePaster(QObject):
    """Pastes live dictation into the focused window piece by piece while
    you speak. Main thread only. Pieces that arrive while a paste is under
    way are joined into the next one; each paste waits for the previous one
    to have been read from the clipboard."""

    GAP_MS = 150

    def __init__(self, paste: Callable = paste_text,
                 before_paste: Callable[[], None] = lambda: None):
        super().__init__()
        self._paste = paste
        self._before_paste = before_paste
        self.begin()

    def begin(self) -> None:
        self._pending = ""
        self._busy = False
        self._refused = False
        self._on_drained: Optional[Callable[[bool], None]] = None

    def write(self, text: str) -> None:
        self._pending += text
        self._pump()

    def finish(self, on_drained: Callable[[bool], None]) -> None:
        """on_drained(ok) once everything is pasted; ok=False if Windows
        refused a keystroke (the focused window runs as administrator)."""
        self._on_drained = on_drained
        self._pump()

    def _pump(self) -> None:
        if self._busy:
            return
        if not self._pending:
            done, self._on_drained = self._on_drained, None
            if done:
                done(not self._refused)
            return
        text, self._pending = self._pending, ""
        self._busy = True
        self._before_paste()
        self._paste(text, self._pasted)

    def _pasted(self, ok: bool) -> None:
        if not ok:
            self._refused = True
        QTimer.singleShot(self.GAP_MS, self._next)

    def _next(self) -> None:
        self._busy = False
        self._pump()
