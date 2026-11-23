"""Telegram bot: reports 3×/day + control commands. Only TELEGRAM_CHAT_ID can use it."""
import asyncio
import html
import json
import logging
from datetime import date, datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .config import CFG, ROOT, SESSIONS, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN
from .db import DB, opt
from .pipeline import Report, run_cycle

log = logging.getLogger(__name__)
db = DB()
TZ = ZoneInfo(CFG["timezone"])
SITES = {"djinni", "dou", "robota", "workua"}
SRC_ICON = {"djinni": "Djinni", "dou": "DOU", "robota": "robota.ua", "workua": "work.ua"}

HELP = """<b>Команди</b>
/status — стан і налаштування
/cv — надіслати обидва резюме (UA і EN)
/scan — запустити пошук зараз
/today — що надіслано сьогодні
/job &lt;id&gt; — деталі вакансії і лист
/pause, /resume — зупинити / відновити відгуки
/dryrun on|off — тестовий режим (не натискає «Надіслати»)
/limit N — макс. відгуків на день
/minscore N — мін. бал для автовідгуку
/help — ця довідка

<b>Логін на сайтах:</b> надішли сюди файл djinni.json / dou.json / robota.json / workua.json
(його створює tools/login.py на твоєму комп'ютері)."""


def esc(s) -> str:
    return html.escape(str(s or ""))


def line(j: dict, extra: str = "") -> str:
    comp = esc(j.get("company") or "?")
    if j.get("company_size"):
        comp += f" ({j['company_size']}+)"
    sal = f" · {esc(j['salary'])}" if j.get("salary") else ""
    tail = f"\n   <i>{esc(extra)}</i>" if extra else ""
    return (f"• <b>{j.get('score', '–')}</b> <a href=\"{j['url']}\">{esc(j['title'])}</a> — {comp}{sal}"
            f" · {SRC_ICON.get(j['source'], j['source'])} <code>{j['id']}</code>{tail}")


def score_badge(score: int) -> str:
    return "🟢" if score >= 85 else "🟡" if score >= 75 else "🟠"


SECOND_JOB = {"yes": "✅ можна поєднати", "maybe": "🤔 можливо поєднати", "no": "⛔ важко поєднати"}
REMOTE = {"remote": "🏠 віддалено", "hybrid": "🏢 гібрид", "office": "🏢 офіс", "unknown": "📍 —"}


def format_card(j: dict, n: int, total: int) -> str:
    v = j.get("verdict") or {}
    comp = esc(j.get("company") or "?")
    if j.get("company_size"):
        comp += f" · {j['company_size']}+ людей"
    salary = v.get("salary_note") or j.get("salary") or ""
    meta = [REMOTE.get(v.get("remote", "unknown"), "📍 —"),
            f"🗣 англ. {esc(v.get('english_required') or '—')}",
            f"👤 {esc(v.get('seniority') or '—')}"]
    if salary:
        meta.insert(0, f"💰 {esc(salary)}")
    lines = [f"{score_badge(j['score'])} <b>{j['score']}/100</b>  ·  {n}/{total}  ·  "
             f"{SRC_ICON.get(j['source'], j['source'])}",
             f"<b>{esc(j['title'])}</b>",
             f"🏢 {comp}",
             "  ·  ".join(meta),
             ""]
    if v.get("summary"):
        lines += [f"📝 {esc(v['summary'])}", ""]
    if v.get("stack"):
        lines += ["🧰 " + "  ".join(f"<code>{esc(t)}</code>" for t in v["stack"][:6]), ""]
    if v.get("why"):
        lines += ["<b>Чому підходить</b>"] + [f"  ✓ {esc(x)}" for x in v["why"][:3]] + [""]
    if v.get("risks"):
        lines += ["<b>Ризики</b>"] + [f"  ⚠ {esc(x)}" for x in v["risks"][:3]] + [""]
    if v.get("second_job_ok"):
        lines += [SECOND_JOB.get(v["second_job_ok"], ""), ""]
    if j.get("letter"):
        lang = "🇬🇧 EN" if (j.get("cv_lang") or v.get("lang")) == "en" else "🇺🇦 UA"
        lines += [f"✉️ <b>Лист</b> ({lang}, CV: {lang}) — натисни, щоб розгорнути",
                  f"<blockquote expandable>{esc(j['letter'])}</blockquote>"]
    return "\n".join(lines)[:4000]


async def send_cards(app: Application, r: Report, intro: str | None = None):
    jobs = r.recommend[: CFG.get("recommend", {}).get("max_cards", 15)]
    head = (f"🔎 <b>Нові вакансії</b> — {datetime.now(TZ):%d.%m %H:%M}\n"
            f"Переглянуто {r.found} · нових {r.new} · відсіяно {r.filtered} · оцінено {r.scored}\n")
    if jobs:
        head += f"\nПідходять: <b>{len(r.recommend)}</b>" + (
            f" (показую топ-{len(jobs)})" if len(jobs) < len(r.recommend) else "") + "\n"
        head += "\n".join(f"{score_badge(j['score'])} {j['score']} · {esc(j['title'][:60])} — "
                           f"{esc((j.get('company') or '?')[:30])}" for j in jobs)
    else:
        head += "\nНових підходящих вакансій немає."
    if intro:
        head = intro
    elif r.errors:
        head += f"\n\n⚠️ Помилок: {len(r.errors)}. Перша:\n<code>{esc(r.errors[0][:300])}</code>"
        if any("credit balance" in e for e in r.errors):
            head += "\n\n💳 <b>Закінчились кредити Anthropic API</b> — поповни баланс у console.anthropic.com → Billing. Ці вакансії оціню в наступному скані."
    await send(app, head)
    for i, j in enumerate(jobs, 1):
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔗 Відкрити вакансію", url=j["url"])]])
        await app.bot.send_message(TELEGRAM_CHAT_ID, format_card(j, i, len(jobs)), parse_mode=ParseMode.HTML,
                                   disable_web_page_preview=True, reply_markup=kb)
        await asyncio.sleep(0.4)


def format_report(r: Report) -> list[str]:
    head = "🧪 <b>ТЕСТОВИЙ РЕЖИМ</b> (dry_run)\n" if opt(db, "apply.dry_run") else ""
    if db.setting("paused", False):
        head += "⏸ <b>Відгуки на паузі</b> (/resume)\n"
    parts = [f"{head}📊 <b>Скан</b>: знайдено {r.found}, нових {r.new}, відсіяно {r.filtered}, "
             f"оцінено {r.scored}. Сьогодні надіслано: {db.applied_today()}/{opt(db, 'apply.daily_limit')}"]
    if r.applied:
        parts.append("✅ <b>Надіслано резюме</b>\n" + "\n".join(line(j, j.get("note", "")) for j in r.applied))
    if r.dry:
        parts.append("🧪 <b>Було б надіслано</b> (dry_run)\n" + "\n".join(line(j) for j in r.dry))
    if r.manual:
        parts.append("👋 <b>Потрібно вручну</b>\n" + "\n".join(line(j, why) for j, why in r.manual))
    if r.maybe:
        parts.append("🤔 <b>Можливо цікаві</b> (бал нижче порогу)\n" + "\n".join(line(j) for j in r.maybe[:10]))
    if r.queued:
        parts.append(f"⏳ Ще {r.queued} у черзі — денний ліміт, надішлю наступного разу.")
    if r.errors:
        parts.append("⚠️ <b>Помилки</b>\n" + "\n".join(f"• {esc(e)}" for e in r.errors[:8]))
    if not (r.applied or r.dry or r.manual or r.maybe):
        parts.append("Нових підходящих вакансій немає.")
    # Telegram limit 4096 chars per message → split by sections
    msgs, cur = [], ""
    for p in parts:
        if len(cur) + len(p) > 3800:
            msgs.append(cur)
            cur = ""
        cur += p + "\n\n"
    return msgs + [cur]


def owner_only(fn):
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if update.effective_chat.id != TELEGRAM_CHAT_ID:
            await update.effective_message.reply_text(
                f"Твій chat id: {update.effective_chat.id}\nВпиши його в .env як TELEGRAM_CHAT_ID.")
            return
        return await fn(update, ctx)
    return wrapper


async def send(app: Application, text: str):
    await app.bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode=ParseMode.HTML,
                               disable_web_page_preview=True)


def scan_due(last_scan: str | None, now: datetime) -> bool:
    """True if the latest scheduled slot (today or yesterday) is after the last scan."""
    if not last_scan:
        return True                       # first launch → scan immediately
    last = datetime.fromisoformat(last_scan)
    slots = []
    for day in (now.date() - timedelta(days=1), now.date()):
        for t in CFG["scan_times"]:
            hh, mm = map(int, t.split(":"))
            slots.append(datetime.combine(day, dtime(hh, mm), tzinfo=now.tzinfo))
    passed = [s for s in slots if s <= now]
    return bool(passed) and max(passed) > last


async def cycle_and_report(app: Application):
    db.set_setting("last_scan", datetime.now(TZ).isoformat(timespec="seconds"))
    try:
        rep = await run_cycle(db)
    except Exception as e:  # noqa: BLE001
        log.exception("cycle failed")
        await send(app, f"❌ Скан впав: <code>{esc(e)}</code>")
        return
    for text, shot in rep.alerts:
        await send(app, f"🔑 <b>{esc(text)}</b>\nЗапусти tools/login.py і надішли сюди новий .json")
        if shot:
            await app.bot.send_photo(TELEGRAM_CHAT_ID, open(shot, "rb"))
    if CFG.get("mode") == "recommend":
        await send_cards(app, rep)
        return
    for m in format_report(rep):
        await send(app, m)
    for j in rep.dry[:3]:      # show how the filled form looks in test mode
        if j.get("shot"):
            await app.bot.send_photo(TELEGRAM_CHAT_ID, open(j["shot"], "rb"),
                                     caption=f"dry_run: {j['title'][:80]}")


# ── commands ──
@owner_only
async def cmd_help(update: Update, ctx):
    await update.message.reply_text(HELP, parse_mode=ParseMode.HTML)


@owner_only
async def cmd_cv(update: Update, ctx):
    for key in ("uk", "en"):
        p = ROOT / CFG["cv"]["files"][key]
        if p.exists():
            await update.message.reply_document(open(p, "rb"), filename=p.name)


@owner_only
async def cmd_status(update: Update, ctx):
    sessions = ", ".join(f"{s} {'✅' if (SESSIONS / f'{s}.json').exists() else '❌'}" for s in sorted(SITES))
    st = db.stats()
    txt = (f"<b>Стан</b>: {'⏸ пауза' if db.setting('paused', False) else '▶️ працює'}"
           f"{' · 🧪 dry_run' if opt(db, 'apply.dry_run') else ''}\n"
           f"Скани: {', '.join(CFG['scan_times'])} ({CFG['timezone']})\n"
           f"Автовідгук від балу {opt(db, 'apply.auto_min_score')}, ліміт {opt(db, 'apply.daily_limit')}/день, "
           f"сьогодні {db.applied_today()}\nСесії: {sessions}\n"
           f"База: {esc(json.dumps(st, ensure_ascii=False))}")
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML)


@owner_only
async def cmd_scan(update: Update, ctx):
    await update.message.reply_text("🔎 Сканую… звіт прийде, коли закінчу (кілька хвилин).")
    ctx.application.create_task(cycle_and_report(ctx.application))


@owner_only
async def cmd_today(update: Update, ctx):
    day = date.today().isoformat()
    rows = [db.get(r["id"]) for r in db.conn.execute(
        "SELECT id FROM jobs WHERE applied_at >= ? ORDER BY applied_at", (day,))]
    if not rows:
        await update.message.reply_text("Сьогодні ще нічого не надсилалось.")
        return
    txt = "\n".join(line(j, j["status"] + (f": {j['error']}" if j.get("error") else "")) for j in rows)
    await update.message.reply_text(txt[:4000], parse_mode=ParseMode.HTML, disable_web_page_preview=True)


@owner_only
async def cmd_job(update: Update, ctx):
    if not ctx.args:
        await update.message.reply_text("Формат: /job djinni:123456")
        return
    j = db.get(ctx.args[0])
    if not j:
        await update.message.reply_text("Не знайшов.")
        return
    v = j["verdict"] or {}
    txt = (line(j) + f"\nСтатус: {esc(j['status'])} {esc(j.get('error') or '')}\n"
           f"Англ.: {esc(v.get('english_required'))} · {esc(v.get('remote'))} · {esc(v.get('seniority'))}\n"
           f"➕ {esc('; '.join(v.get('why', [])))}\n➖ {esc('; '.join(v.get('risks', [])))}\n\n"
           f"<b>Лист:</b>\n{esc(j.get('letter'))}")
    await update.message.reply_text(txt[:4000], parse_mode=ParseMode.HTML, disable_web_page_preview=True)


def _toggle(key, value, label):
    @owner_only
    async def h(update: Update, ctx):
        db.set_setting(key, value)
        await update.message.reply_text(label)
    return h


@owner_only
async def cmd_dryrun(update: Update, ctx):
    if not ctx.args or ctx.args[0] not in ("on", "off"):
        await update.message.reply_text("Формат: /dryrun on або /dryrun off")
        return
    db.set_setting("apply.dry_run", ctx.args[0] == "on")
    await update.message.reply_text("🧪 Тестовий режим увімкнено" if ctx.args[0] == "on"
                                    else "🚀 Бойовий режим: агент буде реально надсилати резюме")


def _int_setting(key, lo, hi, label):
    @owner_only
    async def h(update: Update, ctx):
        try:
            n = int(ctx.args[0])
            assert lo <= n <= hi
        except (IndexError, ValueError, AssertionError):
            await update.message.reply_text(f"Формат: число від {lo} до {hi}")
            return
        db.set_setting(key, n)
        await update.message.reply_text(f"{label}: {n}")
    return h


@owner_only
async def on_document(update: Update, ctx):
    doc = update.message.document
    name = (doc.file_name or "").lower()
    site = name.removesuffix(".json")
    if site not in SITES:
        await update.message.reply_text("Очікую файл djinni.json / dou.json / robota.json / workua.json")
        return
    f = await doc.get_file()
    data = await f.download_as_bytearray()
    try:
        assert "cookies" in json.loads(data)
    except Exception:  # noqa: BLE001
        await update.message.reply_text("Це не схоже на файл сесії Playwright.")
        return
    (SESSIONS / f"{site}.json").write_bytes(data)
    await update.message.delete()   # cookies — не тримаємо в історії чату
    await ctx.bot.send_message(update.effective_chat.id, f"🔑 Сесію {site} збережено (файл видалив із чату).")


def build_app() -> Application:
    if not TELEGRAM_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set in .env")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("cv", cmd_cv))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("job", cmd_job))
    app.add_handler(CommandHandler("pause", _toggle("paused", True, "⏸ Відгуки на паузі. Скан і звіти продовжуються.")))
    app.add_handler(CommandHandler("resume", _toggle("paused", False, "▶️ Відгуки відновлено.")))
    app.add_handler(CommandHandler("dryrun", cmd_dryrun))
    app.add_handler(CommandHandler("limit", _int_setting("apply.daily_limit", 0, 50, "Ліміт на день")))
    app.add_handler(CommandHandler("minscore", _int_setting("apply.auto_min_score", 0, 100, "Мін. бал")))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))

    async def tick(ctx: ContextTypes.DEFAULT_TYPE):
        # Catch-up scheduling: every few minutes check whether a scan slot was missed
        # (computer asleep / program not running) — if so, scan right away.
        if scan_due(db.setting("last_scan"), datetime.now(TZ)):
            await cycle_and_report(ctx.application)

    app.job_queue.run_repeating(tick, interval=300, first=10, name="scheduler")
    return app


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app = build_app()

    async def hello(app_: Application):
        if TELEGRAM_CHAT_ID:
            await send(app_, "🤖 Агент запущений. Скани о " + ", ".join(CFG["scan_times"]) + ".\n/help — команди")
    app.post_init = hello
    app.run_polling(drop_pending_updates=True)
