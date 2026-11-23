import json
import os
from datetime import datetime

from config import CONFIG_DIR

HISTORY_PATH = os.path.join(CONFIG_DIR, "history.json")


class HistoryManager:
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._entries: list[dict] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(HISTORY_PATH):
            try:
                with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
            except Exception:
                self._entries = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)

    def add(self, text: str) -> None:
        entry = {"text": text, "timestamp": datetime.now().isoformat(timespec="seconds")}
        self._entries.insert(0, entry)
        self._entries = self._entries[: self.max_size]
        self._save()

    def get_all(self) -> list[dict]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries = []
        self._save()
