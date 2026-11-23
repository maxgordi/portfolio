from datetime import datetime

from app.models import Job
from app.scorer import is_blacklisted, prefilter


def J(title, company=""):
    return Job(source="dou", ext_id="1", url="u", title=title, company=company, posted_at=datetime.now())


def test_prefilter():
    assert prefilter(J("Python Developer - AI Automations and Internal Tools")) is None
    assert prefilter(J("AI Engineer")) is None
    assert prefilter(J("Розробник AI-рішень")) is None
    assert prefilter(J("Фахівець з автоматизації бізнес-процесів")) is None
    assert prefilter(J("Middle QA Engineer (ML / AI & Data)")) == "title_exclude"
    assert prefilter(J("Java Engineer with AI/ML Experience")) == "title_exclude"
    assert prefilter(J("JavaScript Developer, AI agents")) is None
    assert prefilter(J("AI Intern")) == "title_exclude"
    assert prefilter(J("Frontend Developer")) == "title_not_ai"
    assert prefilter(J("Email marketer")) == "title_not_ai"     # 'ai' must start a word


def test_blacklist():
    for c in ["Казино Приклад", "Lucky Casino", "Acme Gambling", "Example Crypto", "CryptoCorp", "Криптобіржа"]:
        assert is_blacklisted(c), c
    for c in ["Monolith", "Sensei Labs", "SoftServe", "EPAM", "Kyivstar", "Nova Poshta", "OKKO", "INFOBUS",
              "Example Bank", "Приклад Банк", "Acme Insurance"]:
        assert not is_blacklisted(c), c


def test_scan_due():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from app.bot import scan_due   # scan_times 09:15, 13:15, 18:15
    tz = ZoneInfo("Europe/Kyiv")
    now = datetime(2026, 11, 2, 14, 0, tzinfo=tz)
    assert scan_due(None, now)                                         # first launch
    assert scan_due("2026-11-02T09:20:00+02:00", now)                  # missed 13:15
    assert not scan_due("2026-11-02T13:16:00+02:00", now)              # already done
    assert scan_due("2026-11-01T18:20:00+02:00", datetime(2026, 11, 2, 9, 16, tzinfo=tz))
    assert not scan_due("2026-11-01T18:20:00+02:00", datetime(2026, 11, 2, 8, 0, tzinfo=tz))
