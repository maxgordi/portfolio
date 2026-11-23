"""Free prefilter + one LLM call per vacancy: score, verdict and a cover letter."""
import re
from datetime import datetime, timedelta

import yaml

from .config import CFG, ROOT
from .llm import ask_json
from .models import Job


def cv_text() -> str:
    cv = yaml.safe_load(open(ROOT / CFG["cv"]["text"], encoding="utf-8"))
    parts = [f"Ім'я: {cv['name']}", cv["headline"], cv["summary"]]
    parts += [f"{s['title']}: {s['text']}" for s in cv["skills"]]
    for e in cv["experience"]:
        parts.append(f"{e['role']} — {e['company']} ({e['period']}, {e['meta']})")
        parts += [f"- {b['text']}" for b in e["bullets"]]
    parts += cv["languages"]
    return "\n".join(parts)


def is_blacklisted(company: str) -> bool:
    name = (company or "").lower()
    for kw in CFG.get("blacklist_companies", []):
        kw = kw.lower()
        if kw in ("crypto", "крипто"):          # substring: Cryptocorp, Криптобіржа
            if kw in name:
                return True
        elif re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", name):
            return True
    return False


def prefilter(job: Job) -> str | None:
    """Returns a reason to drop the job, or None if it should be scored."""
    pf = CFG["prefilter"]
    t = job.title.lower()
    if any(re.search(rf"(?<!\w){re.escape(x.strip().lower())}(?!\w)", t) for x in pf["title_exclude"]):
        return "title_exclude"
    if not any(re.search(rf"(?<!\w){re.escape(x.strip().lower())}", t) for x in pf["title_include"]):
        return "title_not_ai"
    if job.posted_at and job.posted_at < datetime.now() - timedelta(days=pf["max_age_days"]):
        return "too_old"
    if is_blacklisted(job.company):
        return "blacklist"
    return None


def size_bonus(size: int) -> int:
    for threshold, bonus in sorted(CFG["scoring"]["big_company_bonus"].items(), reverse=True):
        if size >= int(threshold):
            return int(bonus)
    return 0


SYSTEM = """Ти — рекрутинговий асистент кандидата. Оціни вакансію щодо резюме і побажань кандидата
і напиши супровідний лист. Відповідай ЛИШЕ JSON за схемою:
{
 "score": 0-100,                       // наскільки варто відгукуватися
 "is_excluded": true/false,            // казино, ставки, гемблінг або криптовалюти (крипто-біржі, токени, web3)
 "summary": "2-3 речення українською: що за компанія, що за продукт і що конкретно робитиме кандидат",
 "stack": ["до 6 ключових технологій з вакансії"],
 "salary_note": "зарплата/вилка якщо є у вакансії, інакше порожньо",
 "second_job_ok": "yes|maybe|no",      // чи реально поєднати з іншою full-time роботою (віддалено, гнучкий графік)
 "english_required": "none|A2|B1|B2|C1|C2|unknown",
 "client_facing": true/false,          // постійне усне спілкування з іноземними замовниками
 "remote": "remote|hybrid|office|unknown",
 "seniority": "junior|middle|senior|lead|unknown",
 "lang": "uk|en",                      // мова тексту вакансії
 "why": ["до 3 коротких причин українською, чому підходить"],
 "risks": ["до 3 ризиків/невідповідностей українською"],
 "letter": "супровідний лист"
}
Правила для листа: мовою вакансії (lang); 300-500 символів; нічого не вигадувати, цифри лише з
резюме; закінчити простою пропозицією поспілкуватись; підпис — лише ім'я кандидата з резюме
(поле name). Стиль листа: див. LETTER_STYLE нижче."""


def system_prompt() -> str:
    return SYSTEM + "\n\nLETTER_STYLE: " + CFG["scoring"]["letter_style"]


async def score(job: Job) -> dict:
    prefs = CFG["scoring"]["candidate_prefs"]
    user = (f"ПОБАЖАННЯ КАНДИДАТА:\n{prefs}\n\nРЕЗЮМЕ:\n{cv_text()}\n\nВАКАНСІЯ ({job.source}):\n"
            f"Назва: {job.title}\nКомпанія: {job.company or 'невідомо'}"
            f"{f' ({job.company_size}+ спеціалістів)' if job.company_size else ''}\n"
            f"Зарплата: {job.salary or '—'}\nЛокація: {job.location or '—'}\n"
            f"Досвід (років): {job.extra.get('experience_years', '—')}\n\n{job.description[:6000]}")
    v = await ask_json(system_prompt(), user)
    v["score"] = max(0, min(100, int(v.get("score", 0)) + size_bonus(job.company_size)))
    return v
