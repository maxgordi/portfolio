"""Djinni: RSS feeds for discovery, JSON-LD on the job page for company / requirements."""
import html as html_lib
import json
import re

import httpx

from ..models import Job
from .common import html_to_text, parse_rss


def parse_feed(xml_text: str) -> list[Job]:
    jobs = []
    for d in parse_rss(xml_text):
        m = re.search(r"/jobs/(\d+)", d.get("link", ""))
        if not m:
            continue
        jobs.append(Job(source="djinni", ext_id=m.group(1), url=d["link"].split("?")[0],
                        title=html_lib.unescape(d.get("title", "")).strip(),
                        description=html_to_text(d.get("description", "")), posted_at=d["pub"]))
    return jobs


def apply_jsonld(job: Job, page_html: str) -> None:
    for raw in re.findall(r'<script type="application/ld\+json">(.*?)</script>', page_html, re.S):
        try:
            j = json.loads(raw)
        except ValueError:
            continue
        if j.get("@type") != "JobPosting":
            continue
        job.company = (j.get("hiringOrganization") or {}).get("name", "") or job.company
        months = (j.get("experienceRequirements") or {}).get("monthsOfExperience")
        if months:
            job.extra["experience_years"] = round(months / 12, 1)
        if j.get("jobLocationType") == "TELECOMMUTE":
            job.location = job.location or "remote"
        sal = j.get("baseSalary") or {}
        val = sal.get("value") or {}
        if isinstance(val, dict) and (val.get("minValue") or val.get("maxValue")):
            job.salary = f"${val.get('minValue') or ''}–{val.get('maxValue') or ''}".strip("–")
        return


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


async def enrich(client: httpx.AsyncClient, job: Job) -> None:
    r = await client.get(job.url)
    if r.status_code == 200:
        apply_jsonld(job, r.text)
