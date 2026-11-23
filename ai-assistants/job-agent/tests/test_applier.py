"""Applier flow against a fake local job page (no real site touched)."""
import asyncio
import json

from playwright.async_api import async_playwright

from app.appliers import browser as ap
from app.config import SESSIONS

PAGE = """<html><body><a href="/logout">Вийти</a><h1>AI Engineer</h1>
<button id="open" onclick="document.getElementById('f').style.display='block'">Відгукнутися</button>
<form id="f" style="display:none" onsubmit="event.preventDefault();document.body.innerHTML+='<p>Ви відгукнулися на вакансію</p>'">
<textarea name="message"></textarea><input type="file" name="cv">
<button type="submit">Відгукнутися</button></form></body></html>"""


def run(tmp_path, dry):
    page = tmp_path / "job.html"
    page.write_text(PAGE, encoding="utf-8")
    cv = tmp_path / "cv.pdf"
    cv.write_bytes(b"%PDF-1.4 test")
    (SESSIONS / "djinni.json").write_text(json.dumps({"cookies": [], "origins": []}))

    async def go():
        async with async_playwright() as pw:
            b = await pw.chromium.launch()
            try:
                return await ap.apply(b, "djinni", "djinni:1", page.as_uri(), "Hello letter", cv, dry)
            finally:
                await b.close()
    try:
        return asyncio.run(go())
    finally:
        (SESSIONS / "djinni.json").unlink()


def test_dry_run(tmp_path):
    r = run(tmp_path, True)
    assert r.status == "dry_run", r


def test_submit(tmp_path):
    r = run(tmp_path, False)
    assert r.status == "applied", r


def test_no_session(tmp_path):
    async def go():
        async with async_playwright() as pw:
            b = await pw.chromium.launch()
            r = await ap.apply(b, "dou", "dou:1", "about:blank", "x", None, True)
            await b.close()
            return r
    assert asyncio.run(go()).status == "no_session"
