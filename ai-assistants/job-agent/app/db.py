import json
import sqlite3
from datetime import datetime, timedelta

from .config import DATA, CFG
from .models import Job

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, source TEXT, url TEXT, title TEXT, company TEXT,
  company_size INTEGER DEFAULT 0, salary TEXT, location TEXT, description TEXT,
  posted_at TEXT, found_at TEXT,
  status TEXT DEFAULT 'new',      -- new, filtered, scored, notified, skipped, prepared, applied, manual, failed
  score INTEGER, verdict TEXT,    -- verdict = JSON from LLM
  cv_lang TEXT, cv_path TEXT, letter TEXT,
  tg_message_id INTEGER, applied_at TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS companies (slug TEXT PRIMARY KEY, size INTEGER, checked_at TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""


class DB:
    def __init__(self, path=DATA / "jobs.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ── jobs ──
    def insert_if_new(self, j: Job) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO jobs (id, source, url, title, company, company_size, salary,"
            " location, description, posted_at, found_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (j.id, j.source, j.url, j.title, j.company, j.company_size, j.salary, j.location,
             j.description, j.posted_at.isoformat() if j.posted_at else None,
             datetime.now().isoformat(timespec="seconds")))
        self.conn.commit()
        return cur.rowcount == 1

    def update(self, job_id: str, **fields):
        if "verdict" in fields and not isinstance(fields["verdict"], str):
            fields["verdict"] = json.dumps(fields["verdict"], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))
        self.conn.commit()

    def get(self, job_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["verdict"] = json.loads(d["verdict"]) if d.get("verdict") else {}
        return d

    def by_status(self, *statuses) -> list[dict]:
        q = f"SELECT id FROM jobs WHERE status IN ({','.join('?' * len(statuses))}) ORDER BY found_at"
        return [self.get(r["id"]) for r in self.conn.execute(q, statuses)]

    def applied_today(self) -> int:
        day = datetime.now().date().isoformat()
        return self.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='applied' AND applied_at >= ?", (day,)).fetchone()[0]

    def company_recently_applied(self, company: str, days: int) -> bool:
        if not company:
            return False
        since = (datetime.now() - timedelta(days=days)).isoformat()
        return self.conn.execute(
            "SELECT 1 FROM jobs WHERE status IN ('applied','manual') AND lower(company)=lower(?)"
            " AND applied_at >= ? LIMIT 1", (company, since)).fetchone() is not None

    def stats(self) -> dict:
        rows = self.conn.execute("SELECT status, COUNT(*) c FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["c"] for r in rows}

    # ── companies (DOU size cache) ──
    def company_size(self, slug: str) -> int | None:
        r = self.conn.execute("SELECT size, checked_at FROM companies WHERE slug=?", (slug,)).fetchone()
        if r and datetime.fromisoformat(r["checked_at"]) > datetime.now() - timedelta(days=30):
            return r["size"]
        return None

    def set_company_size(self, slug: str, size: int):
        self.conn.execute("INSERT OR REPLACE INTO companies VALUES (?,?,?)",
                          (slug, size, datetime.now().isoformat(timespec="seconds")))
        self.conn.commit()

    # ── runtime settings (override config.yaml from Telegram) ──
    def setting(self, key: str, default=None):
        r = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def set_setting(self, key: str, value):
        self.conn.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value)))
        self.conn.commit()


def opt(db: DB, dotted: str):
    """Setting value: Telegram override if set, else config.yaml. dotted = 'apply.daily_limit'."""
    cur = CFG
    for part in dotted.split("."):
        cur = cur[part]
    return db.setting(dotted, cur)
