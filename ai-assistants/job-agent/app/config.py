"""Loads config.yaml + .env. Runtime overrides (from Telegram) live in the DB settings table."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(os.environ.get("AGENT_ROOT", Path(__file__).resolve().parent.parent))
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
CV_DIR = DATA / "cv"
SESSIONS = DATA / "sessions"
OUTPUT = DATA / "output"
for p in (DATA, SESSIONS, OUTPUT):
    p.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    CFG: dict = yaml.safe_load(f)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = int(os.environ.get("TELEGRAM_CHAT_ID", "0") or 0)

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()   # gemini | anthropic | openai | local
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_EFFORT = os.environ.get("LLM_EFFORT", "")          # anthropic: low | medium | high
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_API_KEY = (
    os.environ.get("LLM_API_KEY")
    or os.environ.get({"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                       "openai": "OPENAI_API_KEY"}.get(LLM_PROVIDER, ""), "")
)

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
