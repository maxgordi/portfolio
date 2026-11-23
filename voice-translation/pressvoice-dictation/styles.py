import json
import os
from typing import Optional

from config import CONFIG_DIR

STYLES_PATH = os.path.join(CONFIG_DIR, "styles.json")

CLEAN_NAME = "Clean"


class StylesManager:
    """Styles are post-processing presets: a name plus an LLM prompt applied
    to the transcript. "Clean" is built in (empty prompt = no processing),
    always first, and cannot be renamed or deleted."""

    def __init__(self):
        self._active: str = CLEAN_NAME
        self._styles: list[dict] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        data = {}
        if os.path.exists(STYLES_PATH):
            try:
                with open(STYLES_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        styles = [s for s in data.get("styles", [])
                  if isinstance(s, dict) and s.get("name")]
        # Clean is always present and always first.
        styles = [s for s in styles if s["name"] != CLEAN_NAME]
        self._styles = [{"name": CLEAN_NAME, "prompt": ""}] + [
            {"name": s["name"], "prompt": s.get("prompt", "")} for s in styles
        ]
        active = data.get("active", CLEAN_NAME)
        self._active = active if self._find(active) else CLEAN_NAME

    def _save(self) -> None:
        os.makedirs(os.path.dirname(STYLES_PATH), exist_ok=True)
        with open(STYLES_PATH, "w", encoding="utf-8") as f:
            json.dump({"active": self._active, "styles": self._styles},
                      f, ensure_ascii=False, indent=2)

    # ── Queries ───────────────────────────────────────────────────────────────

    def _find(self, name: str) -> Optional[dict]:
        for s in self._styles:
            if s["name"] == name:
                return s
        return None

    def get_all(self) -> list[dict]:
        return [dict(s) for s in self._styles]

    def get_active(self) -> dict:
        return dict(self._find(self._active) or self._styles[0])

    @property
    def active_name(self) -> str:
        return self._active

    def _unique_name(self, base: str, skip: Optional[str] = None) -> str:
        name = base
        n = 2
        while True:
            existing = self._find(name)
            if existing is None or (skip is not None and name == skip):
                return name
            name = f"{base} {n}"
            n += 1

    # ── Mutations (each one persists immediately) ─────────────────────────────

    def set_active(self, name: str) -> None:
        if self._find(name):
            self._active = name
            self._save()

    def add(self, base_name: str = "New Style") -> str:
        name = self._unique_name(base_name)
        self._styles.append({"name": name, "prompt": ""})
        self._save()
        return name

    def update(self, old_name: str, new_name: str, prompt: str) -> str:
        """Rename/edit a style; returns the final (possibly deduplicated) name."""
        style = self._find(old_name)
        if style is None or old_name == CLEAN_NAME:
            return old_name
        new_name = new_name.strip() or old_name
        if new_name == CLEAN_NAME:
            new_name = old_name
        new_name = self._unique_name(new_name, skip=old_name)
        style["name"] = new_name
        style["prompt"] = prompt
        if self._active == old_name:
            self._active = new_name
        self._save()
        return new_name

    def delete(self, name: str) -> None:
        if name == CLEAN_NAME or self._find(name) is None:
            return
        self._styles = [s for s in self._styles if s["name"] != name]
        if self._active == name:
            self._active = CLEAN_NAME
        self._save()
