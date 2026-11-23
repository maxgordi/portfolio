"""Entry point.
  python -m app.main          — run the bot + scheduler (production)
  python -m app.main once     — one scan cycle now, print the report (no Telegram)
  python -m app.main report   — one scan cycle now + send report/cards to Telegram, then exit
                                (used by the GitHub Actions schedule — no always-on process needed)
"""
import asyncio
import json
import logging
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        from .db import DB
        from .pipeline import run_cycle
        from .bot import format_report
        import re
        rep = asyncio.run(run_cycle(DB()))
        for m in format_report(rep):
            print(re.sub(r"<[^>]+>", "", m))
        return
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("collect", "review", "publish"):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        from .db import DB
        from . import session_review as sr
        db = DB()
        if cmd == "collect":
            from .pipeline import run_cycle
            rep = asyncio.run(run_cycle(db, apply_enabled=False))
            db.set_setting("last_collect", {"found": rep.found, "new": rep.new, "filtered": rep.filtered,
                                            "errors": rep.errors})
            print(f"found {rep.found}, new {rep.new}, filtered {rep.filtered}, pending {len(db.by_status('pending'))}")
        elif cmd == "review":
            print(f"{sr.export_review(db)} jobs → {sr.REVIEW}")
        else:
            print(sr.import_verdicts(db))
            from telegram.ext import Application
            from .config import TELEGRAM_TOKEN
            from .pipeline import Report
            from . import bot
            lc = db.setting("last_collect", {}) or {}
            rep = Report(found=lc.get("found", 0), new=lc.get("new", 0), filtered=lc.get("filtered", 0),
                         errors=lc.get("errors", []))
            rep.recommend = sorted(db.by_status("scored"), key=lambda j: -(j["score"] or 0))
            rep.scored = len(json.loads(sr.VERDICTS.read_text(encoding="utf-8")))

            async def go():
                app = Application.builder().token(TELEGRAM_TOKEN).build()
                async with app:
                    intro_file = sr.DATA / "intro.html"
                    intro = intro_file.read_text(encoding="utf-8") if intro_file.exists() else None
                    await bot.send_cards(app, rep, intro=intro)
            asyncio.run(go())
            for j in rep.recommend:
                db.update(j["id"], status="notified")
        return
    if cmd == "report":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        from telegram.ext import Application
        from .config import TELEGRAM_TOKEN
        from . import bot

        async def go():
            app = Application.builder().token(TELEGRAM_TOKEN).build()
            async with app:
                await bot.cycle_and_report(app)
        asyncio.run(go())
        return
    from .bot import run
    run()


if __name__ == "__main__":
    main()
