"""Interface translations, shared by the macOS app and the Windows/Linux app.

Strings live in i18n/locales/<code>.json, one flat {key: text} file per
language; English (en.json) is the reference and the fallback for anything
missing. Texts may contain {placeholders} filled by t(key, name=value).

Adding a language = adding a JSON file and a line in LANGUAGES; the tests
check that every file has exactly the keys and placeholders of en.json."""
import json
import locale
import os
import sys
from typing import Optional

# code → the language's own name, in the order shown in Settings.
LANGUAGES = {
    "en": "English",
    "uk": "Українська",
    "ru": "Русский",
    "es": "Español",
    "pt-BR": "Português (Brasil)",
    "de": "Deutsch",
    "fr": "Français",
    "it": "Italiano",
    "pl": "Polski",
    "tr": "Türkçe",
    "nl": "Nederlands",
    "zh-CN": "简体中文",
    "zh-TW": "繁體中文",
    "ja": "日本語",
    "ko": "한국어",
    "id": "Bahasa Indonesia",
}
AUTO = "auto"       # config value: follow the operating system
DEFAULT = "en"

_LOCALES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")
_cache: dict = {}
_current = DEFAULT


def _strings(code: str) -> dict:
    """The {key: text} table of a language, loaded once ({} if unreadable)."""
    if code not in _cache:
        try:
            with open(os.path.join(_LOCALES_DIR, f"{code}.json"), encoding="utf-8") as f:
                data = json.load(f)
            _cache[code] = data if isinstance(data, dict) else {}
        except Exception:
            _cache[code] = {}
    return _cache[code]


# ── Picking the language ─────────────────────────────────────────────────────

def match_language(tag: str) -> Optional[str]:
    """Map an OS language tag ("uk_UA.UTF-8", "zh-Hant-TW", "pt-PT") to a
    supported code, or None."""
    if not tag:
        return None
    tag = tag.replace("_", "-").split(".")[0].split("@")[0].strip()
    if not tag or tag.upper() in ("C", "POSIX"):
        return None
    parts = [p.lower() for p in tag.split("-")]
    lang, rest = parts[0], parts[1:]
    if lang == "zh":
        if "hant" in rest or any(r in ("tw", "hk", "mo") for r in rest):
            return "zh-TW"
        return "zh-CN"
    if lang == "pt":
        return "pt-BR"  # the only Portuguese we have
    if lang == "in":
        lang = "id"     # legacy ISO code for Indonesian
    return lang if lang in LANGUAGES else None


def _windows_languages() -> list:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    count, size = ctypes.c_ulong(0), ctypes.c_ulong(0)
    MUI_LANGUAGE_NAME = 0x8
    if kernel32.GetUserPreferredUILanguages(MUI_LANGUAGE_NAME, ctypes.byref(count),
                                            None, ctypes.byref(size)):
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.GetUserPreferredUILanguages(MUI_LANGUAGE_NAME, ctypes.byref(count),
                                                buf, ctypes.byref(size)):
            return [s for s in buf[:size.value].split("\0") if s]
    return [locale.windows_locale.get(kernel32.GetUserDefaultUILanguage(), "")]


def _mac_languages() -> list:
    from Foundation import NSLocale
    return [str(x) for x in NSLocale.preferredLanguages()]


def system_languages() -> list:
    """The user's preferred interface languages, most preferred first."""
    found = []
    try:
        if sys.platform == "win32":
            found = _windows_languages()
        elif sys.platform == "darwin":
            found = _mac_languages()
    except Exception:
        pass
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        found += [x for x in os.environ.get(var, "").split(":") if x]
    try:
        found.append(locale.getlocale()[0] or "")
    except Exception:
        pass
    return found


def detect_system_language() -> str:
    for tag in system_languages():
        code = match_language(tag)
        if code:
            return code
    return DEFAULT


def resolve(setting: Optional[str]) -> str:
    """Config value ("auto" or a code) → the language to use."""
    if setting in LANGUAGES:
        return setting
    return detect_system_language()


def set_language(setting: Optional[str]) -> str:
    global _current
    _current = resolve(setting)
    return _current


def current_language() -> str:
    return _current


# ── Using it ─────────────────────────────────────────────────────────────────

def t(msg_id: str, /, **values) -> str:
    """The text for msg_id in the current language (English if missing
    there, the id itself if missing everywhere), with {placeholders} filled
    in. msg_id is positional-only so a placeholder may be called "key"."""
    text = _strings(_current).get(msg_id) or _strings(DEFAULT).get(msg_id) or msg_id
    if not values:
        return text
    try:
        return text.format(**values)
    except (KeyError, IndexError, ValueError):
        # A broken translation must never take a notification down with it.
        fallback = _strings(DEFAULT).get(msg_id) or msg_id
        try:
            return fallback.format(**values)
        except (KeyError, IndexError, ValueError):
            return fallback


def language_choices() -> list:
    """[(config value, label)] for the Settings dropdown; the first entry
    follows the system and says which language that currently means."""
    system = LANGUAGES[detect_system_language()]
    return [(AUTO, t("settings.language_system", name=system))] + list(LANGUAGES.items())


def locales_dir() -> str:
    return _LOCALES_DIR
