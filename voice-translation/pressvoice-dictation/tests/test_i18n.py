import json
import os
import string

import pytest

import i18n

LOCALES = i18n.locales_dir()


def _load(code):
    with open(os.path.join(LOCALES, f"{code}.json"), encoding="utf-8") as f:
        return json.load(f)


def _fields(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


EN = _load("en")


def test_every_language_has_a_file_and_no_stray_files():
    files = {f[:-5] for f in os.listdir(LOCALES) if f.endswith(".json")}
    assert files == set(i18n.LANGUAGES)


@pytest.mark.parametrize("code", [c for c in i18n.LANGUAGES if c != "en"])
def test_translation_matches_english(code):
    tr = _load(code)
    assert set(tr) - set(EN) == set(), "keys that English doesn't have"
    assert set(EN) - set(tr) == set(), "missing keys"
    for key, text in tr.items():
        assert isinstance(text, str) and text.strip(), key
        assert _fields(text) == _fields(EN[key]), f"{code}:{key} placeholders"
        if key not in ("settings.refresh",):  # "↻" may stay alone in short forms
            assert text != "" and len(text) < 4 * len(EN[key]) + 40, f"{code}:{key} too long"


@pytest.mark.parametrize("tag,expected", [
    ("uk_UA.UTF-8", "uk"), ("uk-UA", "uk"), ("ru", "ru"), ("ru_RU@euro", "ru"),
    ("en-GB", "en"), ("de-AT", "de"), ("pt-PT", "pt-BR"), ("pt_BR", "pt-BR"),
    ("zh-Hans-CN", "zh-CN"), ("zh_CN", "zh-CN"), ("zh-Hant", "zh-TW"),
    ("zh-HK", "zh-TW"), ("zh_TW", "zh-TW"), ("ja_JP", "ja"), ("ko-KR", "ko"),
    ("in_ID", "id"), ("id-ID", "id"), ("be_BY", None), ("C", None), ("", None),
])
def test_match_language(tag, expected):
    assert i18n.match_language(tag) == expected


def test_detection_takes_first_supported(monkeypatch):
    monkeypatch.setattr(i18n, "system_languages", lambda: ["be-BY", "uk-UA", "en-US"])
    assert i18n.detect_system_language() == "uk"
    monkeypatch.setattr(i18n, "system_languages", lambda: ["xx", "C"])
    assert i18n.detect_system_language() == "en"


def test_resolve_and_fallbacks(monkeypatch):
    monkeypatch.setattr(i18n, "system_languages", lambda: ["ru_RU"])
    assert i18n.resolve("auto") == "ru"
    assert i18n.resolve(None) == "ru"
    assert i18n.resolve("de") == "de"
    assert i18n.resolve("klingon") == "ru"
    try:
        i18n.set_language("uk")
        assert i18n.t("settings.save") == "Зберегти"
        assert i18n.t("hint.hold", key="X") == "Утримуйте X і говоріть"
        assert i18n.t("no.such.key") == "no.such.key"
    finally:
        i18n.set_language("en")


def test_missing_or_broken_translation_falls_back_to_english(monkeypatch):
    monkeypatch.setitem(i18n._cache, "de", {"hint.hold": "Halten {wrong}"})
    try:
        i18n.set_language("de")
        assert i18n.t("hint.hold", key="F19") == "Hold F19 and speak"
        assert i18n.t("settings.save") == "Save"
    finally:
        i18n.set_language("en")


def test_language_choices():
    choices = i18n.language_choices()
    assert choices[0][0] == i18n.AUTO
    assert [c for c, _ in choices[1:]] == list(i18n.LANGUAGES)
