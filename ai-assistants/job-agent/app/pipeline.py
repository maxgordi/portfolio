"""One cycle: collect → prefilter → enrich → LLM score+letter → auto-apply → report."""
import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from playwright.async_api import async_playwright

from .appliers.browser import apply as browser_apply
from .config import CFG, LLM_PROVIDER, ROOT, USER_AGENT
from .db import DB, opt
from .llm import LLMError
from .models import Job
from .scorer import is_blacklisted, prefilter, score
from .sources import djinni, dou, robota, workua

log = logging.getLogger(__name__)
_lock = asyncio.Lock()


@dataclass
class Report:
    found: int = 0
    new: int = 0
    filtered: int = 0
    scored: int = 0
    applied: list = field(default_factory=list)      # dicts from DB
    dry: list = field(default_factory=list)
    manual: list = field(default_factory=list)       # (job, reason)
    maybe: list = field(default_factory=list)        # good but below auto threshold
    errors: list = field(default_factory=list)       # strings
    alerts: list = field(default_factory=list)       # (text, screenshot) session expired etc.
    queued: int = 0                                  # over daily limit → next cycle
    recommend: list = field(default_factory=list)    # recommend mode: cards to send


def cv_for(lang: str):
    mode = CFG["cv"]["language"]
    key = mode if mode in ("uk", "en") else ("en" if lang == "en" else "uk")
    p = ROOT / CFG["cv"]["files"][key]
    return key, (p if p.exists() else None)


async def collect(client, browser, rep: Report) -> list[Job]:
    src = CFG["sources"]
    jobs: list[Job] = []
    tasks = {
        "djinni": lambda: djinni.fetch(client, src["djinni"]["feeds"]),
        "dou": lambda: dou.fetch(client, src["dou"]["feeds"]),
        "robota": lambda: robota.fetch(client, src["robota"]["queries"]),
    }
    for name, fn in tasks.items():
        if not src.get(name, {}).get("enabled"):
            continue
        try:
            got = await fn()
            jobs += got
            log.info("%s: %d", name, len(got))
        except Exception as e:  # noqa: BLE001
            rep.errors.append(f"{name}: {type(e).__name__} {e}"[:200])
    if src.get("workua", {}).get("enabled"):
        page = await (await browser.new_context(user_agent=USER_AGENT, locale="uk-UA")).new_page()
        try:
            jobs += await workua.fetch(page, src["workua"]["queries"])
        except Exception as e:  # noqa: BLE001
            rep.errors.append(f"work.ua: {e}"[:200])
        finally:
            await page.context.close()
    return jobs


async def enrich(job: Job, client, browser, db: DB):
    try:
        if job.source == "djinni":
            await djinni.enrich(client, job)
        elif job.source == "dou" and job.company_slug:
            size = db.company_size(job.company_slug)
            if size is None:
                size = await dou.company_size(client, job.company_slug)
                db.set_company_size(job.company_slug, size)
            job.company_size = size
        elif job.source == "robota":
            await robota.enrich(client, job)
        elif job.source == "workua":
            page = await (await browser.new_context(user_agent=USER_AGENT)).new_page()
            try:
                await workua.enrich(page, job)
            finally:
                await page.context.close()
    except Exception as e:  # noqa: BLE001
        log.warning("enrich %s failed: %s", job.id, e)


async def run_cycle(db: DB, apply_enabled: bool = True) -> Report:
    async with _lock:
        return await _run(db, apply_enabled)


async def _run(db: DB, apply_enabled: bool) -> Report:
    rep = Report()
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8"}
    async with httpx.AsyncClient(headers=headers, timeout=30, follow_redirects=True) as client, \
            async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            jobs = await collect(client, browser, rep)
            rep.found = len(jobs)

            # 1) new jobs → prefilter → enrich (sequential, fast)
            to_score: list[Job] = []
            for job in jobs:
                if not db.insert_if_new(job):
                    continue
                rep.new += 1
                reason = prefilter(job)
                if reason:
                    db.update(job.id, status="filtered", error=reason)
                    rep.filtered += 1
                    continue
                await enrich(job, client, browser, db)
                db.update(job.id, company=job.company, company_size=job.company_size,
                          description=job.description, salary=job.salary, location=job.location,
                          status="pending")
                if is_blacklisted(job.company):
                    db.update(job.id, status="filtered", error="blacklist")
                    rep.filtered += 1
                    continue
                to_score.append(job)
            # retry jobs whose scoring failed in earlier cycles
            fresh = {j.id for j in to_score}
            for row in db.by_status("pending"):
                if row["id"] not in fresh:
                    to_score.append(Job(source=row["source"], ext_id=row["id"].split(":", 1)[1],
                                        url=row["url"], title=row["title"], company=row["company"] or "",
                                        description=row["description"] or "", salary=row["salary"] or "",
                                        location=row["location"] or "",
                                        company_size=row["company_size"] or 0))
            to_score = to_score[:CFG["scoring"].get("max_per_cycle", 60)]

            # 2) LLM score + letter (parallel, limited)
            sem = asyncio.Semaphore(CFG["scoring"].get("parallel", 4))

            async def score_one(job: Job):
                async with sem:
                    try:
                        v = await score(job)
                    except (LLMError, httpx.HTTPError, ValueError, KeyError) as e:
                        rep.errors.append(f"LLM {job.id}: {e}"[:200])
                        db.update(job.id, error=f"llm: {e}"[:300])   # stays 'pending' → retry later
                        return
                rep.scored += 1
                if v.get("is_excluded"):
                    db.update(job.id, status="filtered", error="excluded (llm)", score=v["score"], verdict=v)
                    rep.filtered += 1
                    return
                if v.get("remote") == "office":
                    db.update(job.id, status="filtered", error="office only", score=min(v["score"], 20), verdict=v)
                    rep.filtered += 1
                    return
                low = v["score"] < opt(db, "scoring.min_score_notify")
                db.update(job.id, status="low" if low else "scored", score=v["score"], verdict=v,
                          letter=v.get("letter", ""), cv_lang=v.get("lang", "uk"), error=None)

            # LLM_PROVIDER=none → no paid API: jobs stay 'pending' and are analysed by a Claude
            # session (subscription) via `app.main review` / `app.main publish`.
            if LLM_PROVIDER != "none":
                await asyncio.gather(*(score_one(j) for j in to_score))

            # 3a) recommend mode: no applying, just hand over the best vacancies
            if CFG.get("mode", "apply") == "recommend":
                rep.recommend = sorted(db.by_status("scored"), key=lambda j: -(j["score"] or 0))
                for j in rep.recommend:
                    db.update(j["id"], status="notified")
                return rep

            # 3) apply
            auto_min = opt(db, "apply.auto_min_score")
            dead_sites: set[str] = set()
            candidates = sorted(db.by_status("scored"), key=lambda j: -(j["score"] or 0))
            for j in candidates:
                if j["score"] < auto_min:
                    rep.maybe.append(j)
                    db.update(j["id"], status="notified")
                    continue
                if not (apply_enabled and opt(db, "apply.auto_apply")) or db.setting("paused", False):
                    rep.maybe.append(j)
                    db.update(j["id"], status="notified")
                    continue
                if db.applied_today() >= opt(db, "apply.daily_limit"):
                    rep.queued += 1
                    continue   # stays 'scored' → next cycle
                if db.company_recently_applied(j["company"], CFG["apply"]["same_company_cooldown_days"]):
                    db.update(j["id"], status="skipped", error="cooldown company")
                    continue
                if j["source"] in dead_sites:
                    rep.manual.append((j, f"{j['source']}: немає входу в акаунт"))
                    continue
                if CFG["sources"][j["source"]].get("apply") != "browser":
                    db.update(j["id"], status="manual", error="manual source")
                    rep.manual.append((j, "відгук вручну"))
                    continue

                lang, cv = cv_for(j["cv_lang"])
                dry = opt(db, "apply.dry_run")
                res = await browser_apply(browser, j["source"], j["id"], j["url"], j["letter"], cv, dry)
                log.info("apply %s → %s %s", j["id"], res.status, res.detail)
                now = datetime.now().isoformat(timespec="seconds")
                j["shot"] = res.screenshot
                if res.status in ("applied", "unconfirmed"):
                    db.update(j["id"], status="applied", applied_at=now, cv_path=str(cv or ""),
                              error=None if res.status == "applied" else res.detail)
                    j["note"] = "" if res.status == "applied" else res.detail
                    rep.applied.append(j)
                elif res.status == "dry_run":
                    db.update(j["id"], status="dry", error=res.detail)
                    rep.dry.append(j)
                elif res.status == "already":
                    db.update(j["id"], status="skipped", error=res.detail)
                elif res.status in ("session_expired", "no_session"):
                    # job stays 'scored' → retried automatically after you re-login
                    dead_sites.add(j["source"])
                    rep.alerts.append((res.detail, res.screenshot))
                    rep.manual.append((j, res.detail))
                    continue
                else:   # external / failed
                    db.update(j["id"], status="manual", error=res.detail, applied_at=now)
                    rep.manual.append((j, res.detail))
                await asyncio.sleep(random.uniform(*CFG["apply"]["delay_seconds"]))
        finally:
            await browser.close()
    return rep
