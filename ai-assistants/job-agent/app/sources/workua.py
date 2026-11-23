"""work.ua: HTML search pages. work.ua blocks some datacenter IPs (403) — if so, this source
logs a warning and returns nothing; the rest of the agent keeps working."""
import re
from datetime import datetime

from ..models import Job
from .common import html_to_text

BASE = "https://www.work.ua"


def parse_search(page_html: str) -> list[Job]:
    jobs = []
    for m in re.finditer(r'<h2[^>]*>\s*<a[^>]+href="/jobs/(\d+)/"[^>]*>(.*?)</a>', page_html, re.S):
        ext_id, title = m.group(1), html_to_text(m.group(2))
        chunk = page_html[m.end(): m.end() + 2500]
        company = re.search(r'<span class="strong-600">(.*?)</span>', chunk, re.S)
        jobs.append(Job(source="workua", ext_id=ext_id, url=f"{BASE}/jobs/{ext_id}/", title=title,
                        company=html_to_text(company.group(1)) if company else "",
                        posted_at=datetime.now()))
    return jobs


def parse_job(page_html: str) -> str:
    m = re.search(r'<div id="job-description"[^>]*>(.*?)</div>\s*</div>', page_html, re.S)
    return html_to_text(m.group(1)) if m else ""


async def fetch(page, queries: list[str]) -> list[Job]:
    """Uses a Playwright page (work.ua is stricter with plain HTTP clients)."""
    seen, out = set(), []
    for q in queries:
        slug = re.sub(r"\s+", "+", q.strip().lower())
        r = await page.goto(f"{BASE}/jobs-{slug}/", wait_until="domcontentloaded")
        if not r or r.status != 200:
            raise RuntimeError(f"work.ua HTTP {r.status if r else '?'}")
        for j in parse_search(await page.content()):
            if j.id not in seen:
                seen.add(j.id)
                out.append(j)
    return out


async def enrich(page, job: Job) -> None:
    r = await page.goto(job.url, wait_until="domcontentloaded")
    if r and r.status == 200:
        job.description = parse_job(await page.content())
