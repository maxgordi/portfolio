import datetime
import faulthandler
import os
import sys
import threading

from config import DATA_DIR

LOG_PATH = os.path.join(DATA_DIR, "pressvoice.log")
LOG_MAX_BYTES = 2_000_000

_lock = threading.Lock()
_crash_file = None


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    with _lock:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > LOG_MAX_BYTES:
                os.replace(LOG_PATH, LOG_PATH + ".1")
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


def install_crash_logging() -> None:
    """Record native crashes and stray exceptions in the log, so a silent
    disappearance always leaves a trace. A windowed .exe has no console:
    stdout/stderr are None there, and a stray print would raise."""
    global _crash_file
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        _crash_file = open(LOG_PATH, "a", encoding="utf-8")
        faulthandler.enable(file=_crash_file, all_threads=True)
        if sys.stdout is None:
            sys.stdout = _crash_file
        if sys.stderr is None:
            sys.stderr = _crash_file
    except Exception:
        pass

    def _thread_hook(args):
        log(f"UNCAUGHT in {args.thread.name if args.thread else '?'}: "
            f"{args.exc_type.__name__}: {args.exc_value}")

    def _hook(exc_type, exc_value, tb):
        log(f"UNCAUGHT: {exc_type.__name__}: {exc_value}")

    threading.excepthook = _thread_hook
    sys.excepthook = _hook
