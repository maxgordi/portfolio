"""No-API mode: a Claude session (subscription) does the analysis instead of a paid API.

  python -m app.main collect            # scan sites, prefilter, enrich → jobs stay 'pending'
  python -m app.main review             # write data/review.json (pending jobs for Claude to read)
  # Claude writes data/verdicts.json: {"<job id>": {score, is_excluded, summary, ... letter}}
  python -m app.main publish            # save verdicts, send cards to Telegram
"""
import json

from .config import CFG, DATA
from .db import DB, opt
from .scorer import cv_text, size_bonus, system_prompt

REVIEW = DATA / "review.json"
VERDICTS = DATA / "verdicts.json"


def export_review(db: DB, limit: int = 80) -> int:
    jobs = db.by_status("pending")[:limit]
    out = {
        "instructions": system_prompt(),
        "candidate_prefs": CFG["scoring"]["candidate_prefs"],
        "cv": cv_text(),
        "jobs": [{k: j.get(k) for k in ("id", "source", "title", "company", "company_size", "salary",
                                        "location", "url")} | {"description": (j.get("description") or "")[:2500]}
                 for j in jobs],
    }
    REVIEW.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(jobs)


def import_verdicts(db: DB) -> dict:
    verdicts = json.loads(VERDICTS.read_text(encoding="utf-8"))
    stats = {"scored": 0, "low": 0, "excluded": 0, "missing": 0}
    min_notify = opt(db, "scoring.min_score_notify")
    for job_id, v in verdicts.items():
        j = db.get(job_id)
        if not j:
            stats["missing"] += 1
            continue
        v["score"] = max(0, min(100, int(v.get("score", 0)) + size_bonus(j.get("company_size") or 0)))
        if v.get("is_excluded"):
            db.update(job_id, status="filtered", error="excluded (review)", score=v["score"], verdict=v)
            stats["excluded"] += 1
            continue
        if v.get("remote") == "office":
            db.update(job_id, status="filtered", error="office only", score=min(v["score"], 20), verdict=v)
            stats["office"] = stats.get("office", 0) + 1
            continue
        low = v["score"] < min_notify
        db.update(job_id, status="low" if low else "scored", score=v["score"], verdict=v,
                  letter=v.get("letter", ""), cv_lang=v.get("lang", "uk"), error=None)
        stats["low" if low else "scored"] += 1
    return stats
