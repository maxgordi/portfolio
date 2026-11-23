"""report.html: сводка, карта диапазона, доли подач, таблицы фраз с плеером."""
from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from jinja2 import Template  # noqa: E402

from ..config import Config  # noqa: E402
from ..notes import midi_to_name  # noqa: E402
from ..quality import CODES  # noqa: E402
from ..workspace import Paths, load_table  # noqa: E402
from .template import TEMPLATE  # noqa: E402

log = logging.getLogger("voiceset")


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _t(r) -> str:
    return f"{r['file']} · {r['start']:.1f}–{r['end']:.1f} с"


def build_report(paths: Paths, cfg: Config) -> dict:
    df = load_table(paths)
    bj = paths.work / "build.json"
    info = json.loads(bj.read_text(encoding="utf-8")) if bj.exists() else None
    meta_p = paths.work / "meta.yaml"
    meta = yaml.safe_load(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    rvc_p = paths.rvc / "summary.json"
    rvc = json.loads(rvc_p.read_text(encoding="utf-8")) if rvc_p.exists() else {}

    clean = df[df["reject"] == ""]
    sel_ids = info["ids"] if info else []
    sel = df.loc[sel_ids] if sel_ids else df.iloc[0:0]

    # --- карта диапазона ---
    rec, inset = {}, {}
    for _, r in clean.iterrows():
        for m, ms in zip(r["sem_midi"], r["sem_ms"]):
            rec[int(m)] = rec.get(int(m), 0) + ms / 1000
    for _, r in sel.iterrows():
        for m, ms in zip(r["sem_midi"], r["sem_ms"]):
            inset[int(m)] = inset.get(int(m), 0) + ms / 1000
    fig, ax = plt.subplots(figsize=(11, 3.2))
    if rec:
        ms_ = list(range(min(rec), max(rec) + 1))
        ax.bar(ms_, [rec.get(m, 0) for m in ms_], color="#cbd5e1", label="в записи (чистые)")
        ax.bar(ms_, [inset.get(m, 0) for m in ms_], color="#2563eb", width=0.55, label="в наборе")
        if info:
            for m in info["targets"]:
                if m not in info["covered"]:
                    ax.axvspan(m - 0.5, m + 0.5, color="#dc2626", alpha=0.25)
            ax.axvline(info["range_lo"] - 0.5, color="#6b7280", ls=":", lw=1)
            ax.axvline(info["range_hi"] + 0.5, color="#6b7280", ls=":", lw=1)
        ax.set_xticks(ms_[::2])
        ax.set_xticklabels([midi_to_name(m) for m in ms_[::2]], rotation=60, fontsize=8)
    ax.set_ylabel("секунд")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    img_range = _png(fig)

    # --- доли подач ---
    quotas = info["quotas"] if info else cfg.suno.quotas
    fact = info["shares"] if info else {}
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ks = ["normal", "power", "breathy"]
    y = np.arange(len(ks))
    ax.barh(y + 0.18, [quotas.get(k, 0) * 100 for k in ks], height=0.34, color="#cbd5e1", label="цель")
    ax.barh(y - 0.18, [fact.get(k, 0) * 100 for k in ks], height=0.34, color="#2563eb", label="факт")
    ax.set_yticks(y)
    ax.set_yticklabels(ks)
    ax.set_xlabel("% времени набора")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    img_deliv = _png(fig)

    # --- таблицы ---
    selected = []
    for k, pid in enumerate(sel_ids, 1):
        r = df.loc[pid]
        selected.append(dict(order=k, id=pid, song=r["song"], t=_t(r), range=r["range"], delivery=r["delivery"],
                             conf=f"{r['delivery_conf']:.2f}", unsure=bool(r["unsure"]), quality=f"{r['quality']:.2f}",
                             sim="" if not np.isfinite(r["similarity"]) else f"{r['similarity']:.2f}", path=r["phrase_path"]))
    rejected = [dict(id=int(r["id"]), song=r["song"], t=_t(r), path=r["phrase_path"],
                     why="; ".join(f"{c} — {CODES.get(c, c)}" for c in r["reject"].split(";")))
                for _, r in df[df["reject"] != ""].iterrows()]
    sel_set = set(sel_ids)
    reasons = info["reasons"] if info else {}
    spare = [dict(id=int(r["id"]), song=r["song"], t=_t(r), range=r["range"], delivery=r["delivery"],
                  why=reasons.get(str(int(r["id"])), "").replace("НЕ_ВЫБРАНА: ", ""), path=r["phrase_path"])
             for _, r in clean.iterrows() if int(r["id"]) not in sel_set]

    # --- предупреждения простыми словами ---
    warns = list(info["warnings"]) if info else []
    if info and sel_ids:
        top = max(info["covered"]) if info["covered"] else None
        if info["unavailable"]:
            hi_missing = [m for m in info["unavailable"] if m > info["range_hi"] - 3]
            if hi_missing:
                warns.append(f"Нет нот выше {midi_to_name(min(hi_missing) - 1)} — допойте верхние ноты.")
        pool_br = clean[clean["delivery"] == "breathy"]["dur"].sum() / max(clean["dur"].sum(), 1e-9)
        if pool_br < quotas.get("breathy", 0) * 0.7:
            warns.append("Мало придыхания в записи — добавьте несколько фраз «с придыханием».")
        del top
    if rvc:
        warns += rvc.get("warnings", [])
    if int(df["unsure"].sum()):
        warns.append(f"У {int(df['unsure'].sum())} фраз подача определена неуверенно — разметьте по 3–5 примеров в labels.yaml и запустите calibrate.")
    seen = set()
    warns = [w for w in warns if not (w in seen or seen.add(w))]

    diff = None
    if info and info["prev_ids"]:
        diff = dict(added=info["added"], removed=info["removed"])
    html = Template(TEMPLATE).render(
        stamp=datetime.now().strftime("%Y-%m-%d %H:%M"), seed=cfg.seed, speaker=meta.get("speaker_engine", "—"),
        input_min=meta.get("input_minutes", "—"), n_all=len(df), n_clean=len(clean), n_rej=len(df) - len(clean),
        n_sel=len(sel_ids), length=f"{info['length_s']:.1f}" if info else "—", rvc_min=rvc.get("total_minutes", "—"),
        warnings=warns, diff=diff, img_range=img_range, img_deliv=img_deliv, selected=selected,
        rejected=rejected, spare=spare, root=str(paths.root))
    paths.report.mkdir(parents=True, exist_ok=True)
    out = paths.report / "report.html"
    out.write_text(html, encoding="utf-8")
    log.info("Отчёт: %s", out)
    return {"path": str(out), "warnings": warns}
