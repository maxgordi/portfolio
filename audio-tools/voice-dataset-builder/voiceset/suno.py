"""Сборка набора для Suno: отбор → порядок → склейка → manifest.csv + build.json."""
from __future__ import annotations

import json
import logging
import shutil

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from . import assemble as asm
from .audio_io import resample, write_wav24
from .config import Config
from .order import order_phrases
from .select import build_selection
from .workspace import Paths, load_table

log = logging.getLogger("voiceset")


def load_review(paths: Paths) -> dict:
    if paths.review.exists():
        d = yaml.safe_load(paths.review.read_text(encoding="utf-8")) or {}
    else:
        d = {}
    return {"keep": sorted({int(i) for i in d.get("keep", [])}), "ban": sorted({int(i) for i in d.get("ban", [])})}


def save_review(paths: Paths, review: dict) -> None:
    paths.review.write_text(
        "# Ручная правка набора: keep — обязательно взять, ban — не брать. Номера фраз смотрите в report.html\n"
        + yaml.safe_dump(review, allow_unicode=True), encoding="utf-8")


def parse_ids(s: str | None) -> list[int]:
    if not s:
        return []
    return [int(t) for t in s.replace(" ", "").split(",") if t]


def build_suno(paths: Paths, cfg: Config, length: float | None = None, seed: int | None = None,
               exclude: list[int] | None = None, include: list[int] | None = None) -> dict:
    df = load_table(paths)
    s = cfg.suno.model_copy(update={"length_s": length or cfg.suno.length_s})
    seed = cfg.seed if seed is None else seed
    review = load_review(paths)
    for i in exclude or []:
        if i not in review["ban"]:
            review["ban"].append(i)
        if i in review["keep"]:
            review["keep"].remove(i)
    for i in include or []:
        if i not in review["keep"]:
            review["keep"].append(i)
        if i in review["ban"]:
            review["ban"].remove(i)
    if exclude or include or paths.review.exists():
        save_review(paths, {k: sorted(v) for k, v in review.items()})
    unknown = [i for i in review["keep"] + review["ban"] if i not in df.index]
    if unknown:
        log.warning("В правке есть несуществующие номера фраз: %s", unknown)
    paths.suno.mkdir(parents=True, exist_ok=True)
    prev_manifest = paths.suno / "manifest.csv"
    prev_ids: list[int] = []
    if prev_manifest.exists():
        prev_ids = pd.read_csv(prev_manifest)["id"].astype(int).tolist()

    sel = build_selection(df, s, seed, set(review["ban"]), set(review["keep"]))
    for pid in sorted(sel.reasons):
        log.debug("фраза %d: %s", pid, sel.reasons[pid])
    ordered, order_warn = order_phrases(df, sel.ids, seed, s.order_tries)
    sr = cfg.io.sr
    parts, rows = [], []
    for pid in ordered:
        r = df.loc[pid]
        x, sr_in = sf.read(str(paths.root / r["phrase_path"]), dtype="float32")
        parts.append(asm.prepare(resample(x, sr_in, sr) if sr_in != sr else x, sr, s.assemble))
        rows.append(r)
    y, gaps, join_warn = asm.join(parts, sr, s.length_s, s.assemble)
    y = asm.limit_peak(y, s.assemble.peak_dbfs)
    out_wav = paths.suno / "voice_set_2min.wav" if abs(s.length_s - 120) < 1e-6 else paths.suno / f"voice_set_{int(s.length_s)}s.wav"
    write_wav24(out_wav, y, sr)

    manifest = pd.DataFrame([{
        "order": k + 1, "id": int(r["id"]), "file": r["file"], "song": r["song"], "start": round(r["start"], 3),
        "end": round(r["end"], 3), "dur": round(r["dur"], 3), "range": r["range"], "delivery": r["delivery"],
        "delivery_conf": round(r["delivery_conf"], 2), "quality": round(r["quality"], 2),
        "similarity": None if not np.isfinite(r["similarity"]) else round(r["similarity"], 3),
        "gap_after_ms": round(gaps[k] / sr * 1000, 1) if k < len(gaps) else 0.0,
    } for k, r in enumerate(rows)])
    manifest.to_csv(paths.suno / "manifest.csv", index=False)

    warnings = sel.warnings + order_warn + join_warn
    if s.length_s > 130:
        warnings += asm.sliding_window_check(
            [dict(dur=r["dur"], delivery=r["delivery"]) for r in rows],
            [g / sr for g in gaps], s.quotas, s.quota_tolerance)
    n_out = len(y) / sr
    info = {
        "file": str(out_wav.relative_to(paths.root)), "length_s": round(n_out, 3), "sr": sr, "phrases": len(ordered),
        "ids": ordered, "prev_ids": prev_ids,
        "added": sorted(set(ordered) - set(prev_ids)) if prev_ids else [],
        "removed": sorted(set(prev_ids) - set(ordered)) if prev_ids else [],
        "range_lo": sel.range_lo, "range_hi": sel.range_hi, "targets": sel.targets,
        "unavailable": sel.unavailable, "weights": {str(k): v for k, v in sel.weights.items()},
        "covered": sel.covered, "shares": sel.shares, "song_shares": sel.song_shares,
        "quotas": s.quotas, "warnings": warnings, "seed": seed,
        "reasons": {str(k): v for k, v in sel.reasons.items()}, "review": review,
    }
    (paths.work / "build.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Suno: %s — %.2f с, фраз %d, покрыто нот %d/%d", out_wav.name, n_out, len(ordered),
             len(sel.covered), len(sel.targets))
    log.info("Подачи: %s", ", ".join(f"{k} {v*100:.0f}%" for k, v in sel.shares.items()))
    for w in warnings:
        log.warning("⚠ %s", w)
    if prev_ids:
        log.info("Отличия от прошлой сборки: добавлено %s, убрано %s", info["added"] or "—", info["removed"] or "—")
    return info
