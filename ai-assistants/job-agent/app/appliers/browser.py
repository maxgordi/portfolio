"""One generic Playwright applier for all sites. Site differences live in config.yaml → selectors.

Login sessions: data/sessions/<site>.json (Playwright storage_state), created once with
tools/login.py on your own computer and uploaded to the bot (send the .json file in Telegram).
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from playwright.async_api import Browser, TimeoutError as PWTimeout

from ..config import CFG, OUTPUT, SESSIONS, USER_AGENT

log = logging.getLogger(__name__)


@dataclass
class ApplyResult:
    status: str  # applied | unconfirmed | dry_run | external | already | failed | session_expired | no_session
    detail: str = ""
    screenshot: Path | None = None


async def _first_visible(page, selector: str, timeout=4000):
    loc = page.locator(selector).first
    try:
        await loc.wait_for(state="visible", timeout=timeout)
        return loc
    except PWTimeout:
        return None


async def _shot(page, job_id: str, tag: str) -> Path:
    p = OUTPUT / f"{job_id.replace(':', '_')}_{tag}_{datetime.now():%H%M%S}.png"
    try:
        await page.screenshot(path=str(p), full_page=False)
    except Exception:  # noqa: BLE001
        return None
    return p


async def apply(browser: Browser, site: str, job_id: str, url: str, letter: str,
                cv_path: Path | None, dry_run: bool) -> ApplyResult:
    sel = CFG["selectors"][site]
    state = SESSIONS / f"{site}.json"
    if not state.exists():
        return ApplyResult("no_session", f"немає сесії {site}.json — надішли її боту")

    ctx = await browser.new_context(storage_state=str(state), user_agent=USER_AGENT,
                                    locale="uk-UA", viewport={"width": 1366, "height": 900})
    page = await ctx.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(1500)

        if not await _first_visible(page, sel["logged_in"], 5000):
            return ApplyResult("session_expired", f"{site}: сесія злетіла, потрібен новий логін",
                               await _shot(page, job_id, "login"))

        if sel.get("external") and await _first_visible(page, sel["external"], 1500):
            return ApplyResult("external", "відгук на сайті компанії")

        opener = await _first_visible(page, sel["open_form"], 4000)
        if opener:
            await opener.click()
            await page.wait_for_timeout(1200)

        box = await _first_visible(page, sel["message"], 6000)
        if not box:
            body = (await page.inner_text("body"))[:4000].lower()
            if any(w in body for w in ("ви вже відгукнулися", "you have applied", "ви відгукнулися")):
                return ApplyResult("already", "вже відгукувались раніше")
            return ApplyResult("failed", "не знайшов форму відгуку", await _shot(page, job_id, "noform"))

        await box.fill(letter)

        if cv_path and cv_path.exists():
            file_input = page.locator(sel["file"]).first
            if await file_input.count():
                try:
                    await file_input.set_input_files(str(cv_path))
                    await page.wait_for_timeout(1500)
                except Exception as e:  # noqa: BLE001
                    log.warning("file upload failed on %s: %s", site, e)

        if dry_run:
            return ApplyResult("dry_run", "форма заповнена, не надіслано (dry_run)",
                               await _shot(page, job_id, "dry"))

        submit = await _first_visible(page, sel["submit"], 4000)
        if not submit:
            return ApplyResult("failed", "не знайшов кнопку «Надіслати»", await _shot(page, job_id, "nosubmit"))
        await submit.click()

        if await _first_visible(page, sel["success"], 12000):
            await ctx.storage_state(path=str(state))   # refresh cookies
            return ApplyResult("applied", "надіслано")
        # Probably sent — never retry automatically, to avoid double applications.
        await ctx.storage_state(path=str(state))
        return ApplyResult("unconfirmed", "натиснув «Надіслати», але не побачив підтвердження — перевір",
                           await _shot(page, job_id, "unconfirmed"))
    except Exception as e:  # noqa: BLE001
        log.exception("apply error %s", url)
        return ApplyResult("failed", f"{type(e).__name__}: {e}"[:300], await _shot(page, job_id, "error"))
    finally:
        await ctx.close()
