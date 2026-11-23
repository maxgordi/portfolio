from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path


class DialogLogger:
    """
    Writes both sides of a translated dialogue to a timestamped text file.
    Thread-safe: log_* methods may be called from asyncio callbacks in a background thread.
    """

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = log_dir
        now = datetime.now()
        filename = f"dialog_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        self._path = log_dir / filename
        self._lock = threading.Lock()
        self._file = self._path.open("w", encoding="utf-8")
        self._file.write(f"=== BabelLang Session: {now.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        self._file.flush()
        self._cleanup_old_logs()

    def _cleanup_old_logs(self, keep: int = 30) -> None:
        files = sorted(self._log_dir.glob("dialog_*.txt"))
        for old in files[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass

    def log_outgoing(self, source: str, translated: str, my_lang: str = "ME", inter_lang: str = "TRANS") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._file.write(f"[{ts}] [{my_lang}]  {source}\n")
            self._file.write(f"[{ts}] [{my_lang}→{inter_lang}]  {translated}\n\n")
            self._file.flush()

    def log_incoming(self, source: str, translated: str, my_lang: str = "ME", inter_lang: str = "INTER") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._file.write(f"[{ts}] [{inter_lang}]  {source}\n")
            self._file.write(f"[{ts}] [{inter_lang}→{my_lang}]  {translated}\n\n")
            self._file.flush()

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._file.close()
