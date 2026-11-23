import json
import os
import sys


def _config_dir() -> str:
    override = os.environ.get("PRESSVOICE_CONFIG_DIR")
    if override:  # tests and portable setups
        return override
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
        return os.path.join(base, "PressVoice")
    if sys.platform == "darwin":
        # Kept where the macOS app has always stored it.
        return os.path.expanduser("~/.config/pressvoice")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "pressvoice")


def _data_dir() -> str:
    """Logs and other machine-local state."""
    if os.environ.get("PRESSVOICE_CONFIG_DIR"):
        return os.environ["PRESSVOICE_CONFIG_DIR"]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, "PressVoice")
    return _config_dir()


CONFIG_DIR = _config_dir()
DATA_DIR = _data_dir()
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_MODEL = "gpt-4o-mini-transcribe"

DEFAULTS = {
    "api_key": "",
    "microphone_index": -1,
    "microphone_name": "",
    "hotkey": "cmd_r" if sys.platform == "darwin" else "ctrl_r",
    "recording_mode": "push_to_talk",
    # When the text appears: "whole" after the key is released, "phrases"
    # typed while you speak (live/session.py).
    "text_mode": "whole",
    "history_size": 50,
    "transcription_model": DEFAULT_MODEL,
    # Model ids fetched by the last "Refresh" in Settings.
    "transcription_models": [],
    # Interface language: "auto" follows the system, else a code from i18n.
    "language": "auto",
}


def load() -> dict:
    """Never raises: a missing or damaged config must not stop the app."""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        if not os.path.exists(CONFIG_PATH):
            save(DEFAULTS.copy())
            return DEFAULTS.copy()
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return DEFAULTS.copy()
        return {**DEFAULTS, **data}
    except Exception:
        return DEFAULTS.copy()


def save(cfg: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    # Write to a temporary file first, so an interrupted save can't leave a
    # half-written config behind.
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)
