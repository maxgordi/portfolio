"""Run on YOUR computer (not the server) to log in once and save the session.

  pip install playwright && python -m playwright install chromium
  python tools/login.py djinni      # also: dou, robota, workua

A browser window opens → log in as usual → come back to the terminal and press Enter.
The file data/sessions/<site>.json is created (agent on this computer uses it directly;
for a server, send it to your Telegram bot — it stores it and deletes the message). Sessions usually live for weeks; when one expires the
bot tells you, and you repeat this step.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URLS = {
    "djinni": "https://djinni.co/login",
    "dou": "https://jobs.dou.ua/",
    "robota": "https://robota.ua/auth/login",
    "workua": "https://www.work.ua/jobseeker/login/",
}
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

site = sys.argv[1] if len(sys.argv) > 1 else ""
if site not in URLS:
    sys.exit(f"Usage: python tools/login.py [{'|'.join(URLS)}]")

out = Path(__file__).resolve().parent.parent / "data" / "sessions"
out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    ctx = browser.new_context(user_agent=UA, locale="uk-UA")
    page = ctx.new_page()
    page.goto(URLS[site])
    input(f"\nЗалогінься на {site} у вікні браузера, потім натисни Enter тут… ")
    ctx.storage_state(path=str(out / f"{site}.json"))
    browser.close()
print(f"✅ Збережено: {out / f'{site}.json'}")
print("   Якщо агент працює на цьому ж комп'ютері — більше нічого робити не треба.")
print("   Якщо на сервері — надішли цей файл своєму боту в Telegram.")
