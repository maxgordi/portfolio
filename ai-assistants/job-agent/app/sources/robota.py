"""robota.ua: public JSON API used by their own site."""
from datetime import datetime

import httpx

from ..models import Job
from .common import html_to_text

API = "https://api.robota.ua"


def parse_search(data: dict) -> list[Job]:
    jobs = []
    for d in data.get("documents", []):
        try:
            posted = datetime.fromisoformat(d.get("date", "")[:19])
        except ValueError:
            posted = None
        sal = ""
        if d.get("salaryFrom") or d.get("salaryTo"):
            sal = f"{d.get('salaryFrom') or ''}–{d.get('salaryTo') or ''} грн".strip("–")
        elif d.get("salary"):
            sal = f"{d['salary']} грн"
        jobs.append(Job(source="robota", ext_id=str(d["id"]),
                        url=f"https://robota.ua/company{d.get('notebookId')}/vacancy{d['id']}",
                        title=d.get("name", ""), company=d.get("companyName", ""),
                        description=html_to_text(d.get("shortDescription", "")), salary=sal,
                        location=d.get("cityName", ""), posted_at=posted,
                        extra={"notebook_id": d.get("notebookId")}))
    return jobs


async def fetch(client: httpx.AsyncClient, queries: list[str]) -> list[Job]:
    seen, out = set(), []
    for q in queries:
        r = await client.get(f"{API}/vacancy/search", params={"keyWords": q, "page": 0, "count": 40})
        r.raise_for_status()
        for j in parse_search(r.json()):
            if j.id not in seen:
                seen.add(j.id)
                out.append(j)
    return out


async def enrich(client: httpx.AsyncClient, job: Job) -> None:
    r = await client.get(f"{API}/vacancy", params={"id": job.ext_id})
    if r.status_code == 200:
        d = r.json()
        job.description = html_to_text(d.get("description") or "") or job.description
        if d.get("isAgency"):
            job.extra["agency"] = True
