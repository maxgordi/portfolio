"""DOU: RSS feeds. Title format: '<title> в <Company>, $X–Y, <location>'. Company size from company page."""
import html
import re

import httpx

from ..models import Job
from .common import html_to_text, parse_rss


def split_title(raw: str) -> tuple[str, str, str, str]:
    raw = html.unescape(raw)
    if " в " not in raw:
        return raw.strip(), "", "", ""
    title, rest = raw.rsplit(" в ", 1)
    parts = [p.strip() for p in rest.split(", ")]
    company, salary, loc = parts[0], "", ""
    tail = parts[1:]
    if tail and re.match(r"^(\$|до \$|від \$|€|\d)", tail[0]):
        salary = tail.pop(0)
    loc = ", ".join(tail)
    return title.strip(), company, salary, loc


def parse_feed(xml_text: str) -> list[Job]:
    jobs = []
    for d in parse_rss(xml_text):
        link = d.get("link", "")
        m = re.search(r"/companies/([^/]+)/vacancies/(\d+)", link)
        if not m:
            continue
        title, company, salary, loc = split_title(d.get("title", ""))
        jobs.append(Job(source="dou", ext_id=m.group(2), url=link.split("?")[0], title=title,
                        company=company, salary=salary, location=loc, company_slug=m.group(1),
                        description=html_to_text(d.get("description", "")), posted_at=d["pub"]))
    return jobs


def parse_company_size(page_html: str) -> int:
    m = re.search(r"([\d\s.…]+)\s*спеціаліст", page_html)
    if not m:
        return 0
    nums = [int(n) for n in re.findall(r"\d+", m.group(1))]
    return max(nums) if nums else 0


async def fetch(client: httpx.AsyncClient, feeds: list[str]) -> list[Job]:
    seen, out = set(), []
    for url in feeds:
        r = await client.get(url)
        r.raise_for_status()
        for j in parse_feed(r.text):
            if j.id not in seen:
                seen.add(j.id)
                out.append(j)
    return out


async def company_size(client: httpx.AsyncClient, slug: str) -> int:
    r = await client.get(f"https://jobs.dou.ua/companies/{slug}/")
    return parse_company_size(r.text) if r.status_code == 200 else 0
