"""Режим RVC: только чистка. Все годные фразы → клипы 2–12 с, без перемешивания и бюджета."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import soundfile as sf

from . import loudness as ld
from .assemble import fade
from .audio_io import highpass, resample, write_wav24
from .config import RvcCfg
from .segment import frame_db
from .workspace import Paths

log = logging.getLogger("voiceset")


def trim_silence(x: np.ndarray, sr: int, floor_db: float, margin_db: float, keep_ms: float = 30) -> np.ndarray:
    db = frame_db(x, sr)
    act = np.where(db > floor_db + margin_db)[0]
    if len(act) == 0:
        return x
    a = max(0, int(act[0] * 0.010 * sr) - int(keep_ms / 1000 * sr))
    b = min(len(x), int((act[-1] * 0.010 + 0.025) * sr) + int(keep_ms / 1000 * sr))
    return x[a:b]


def build_rvc(df: pd.DataFrame, paths: Paths, cfg: RvcCfg, sr_out: int | None = None, combined: bool = False) -> dict:
    sr_out = sr_out or cfg.sr
    out = paths.rvc / "dataset"
    if out.exists():
        for f in out.glob("*.wav"):
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)
    clean = df[df["reject"] == ""].sort_values("id")
    clips: list[tuple[list[int], np.ndarray]] = []
    dropped: list[int] = []
    cur_ids: list[int] = []
    cur: list[np.ndarray] = []
    cur_file, cur_end, cur_dur = None, 0.0, 0.0

    def finish():
        nonlocal cur_ids, cur, cur_dur
        if cur:
            if cur_dur >= cfg.clip_min_s:
                clips.append((cur_ids, np.concatenate(cur)))
            else:
                dropped.extend(cur_ids)
        cur_ids, cur, cur_dur = [], [], 0.0

    for _, r in clean.iterrows():
        x, sr = sf.read(str(paths.root / r["phrase_path"]), dtype="float32")
        x = highpass(x, sr, cfg.highpass_hz)
        x = trim_silence(x, sr, r["noise_floor_db"], cfg.trim_margin_db)
        x = ld.normalize_to(x, sr, cfg.target_lufs)
        x = fade(x, sr, 10)
        d = len(x) / sr
        can_merge = (cur and cur_dur < cfg.clip_min_s and r["file"] == cur_file
                     and r["start"] - cur_end <= cfg.merge_gap_s and cur_dur + 0.15 + d <= cfg.clip_max_s)
        if can_merge:
            cur.append(np.zeros(int(0.15 * sr), np.float32))
            cur_dur += 0.15
        else:
            finish()
        cur.append(x)
        cur_ids.append(int(r["id"]))
        cur_file, cur_end, cur_dur = r["file"], r["end"], cur_dur + d
    finish()
    written, total = [], 0.0
    parts = []
    for k, (ids, x) in enumerate(clips, 1):
        y = resample(x, 48000, sr_out) if sr_out != 48000 else x
        name = f"{k:04d}_{'-'.join(str(i) for i in ids[:3])}.wav"
        write_wav24(out / name, y, sr_out)
        total += len(y) / sr_out
        written.append(name)
        parts.append(y)
    if combined and parts:
        sil = np.zeros(int(0.5 * sr_out), np.float32)
        write_wav24(paths.rvc / "all_clips.wav", np.concatenate([p for y in parts for p in (y, sil)]), sr_out)
    kept_df = clean[~clean["id"].isin(dropped)]
    shares = (kept_df.groupby("delivery")["dur"].sum() / max(kept_df["dur"].sum(), 1e-9)).round(3).to_dict()
    notes = {}
    for sm, ms in zip(kept_df["sem_midi"], kept_df["sem_ms"]):
        for m, t in zip(sm, ms):
            notes[int(m)] = notes.get(int(m), 0) + t
    from .notes import midi_to_name
    summary = {
        "clips": len(written), "sample_rate": sr_out, "total_seconds": round(total, 1),
        "total_minutes": round(total / 60, 2), "phrases_used": int(len(kept_df)),
        "phrases_rejected": int((df["reject"] != "").sum()), "phrases_too_short_to_use": dropped,
        "delivery_shares": shares,
        "range_map_ms": {midi_to_name(m): int(t) for m, t in sorted(notes.items())},
        "warnings": [],
    }
    if total < cfg.min_total_s:
        summary["warnings"].append(
            f"Всего {total/60:.1f} мин чистых данных — меньше {cfg.min_total_s/60:.0f} мин. Для RVC лучше 10–30 минут: допишите запись.")
    paths.rvc.mkdir(parents=True, exist_ok=True)
    (paths.rvc / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("RVC: %d клипов, %.1f мин, %d Гц", len(written), total / 60, sr_out)
    for w in summary["warnings"]:
        log.warning("⚠ %s", w)
    return summary
