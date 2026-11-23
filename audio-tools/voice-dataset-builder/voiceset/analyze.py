"""Пайплайн анализа: каждая фраза получает строку признаков и флагов в analysis.parquet."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from . import asr as asr_mod
from . import delivery as dlv
from . import loudness as ld
from . import quality as qt
from . import speaker as spk
from .audio_io import SongInfo, condition, discover, load_raw, write_wav24
from .config import Config
from .notes import hz_to_name, midi_to_name
from .pitch import phrase_pitch_features, track_f0
from .segment import segment
from .workspace import Paths, load_table, save_table

log = logging.getLogger("voiceset")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")[:24] or "song"


def note_range(p5: float, p95: float) -> str:
    if not np.isfinite(p5):
        return ""
    return f"{hz_to_name(p5)}–{hz_to_name(p95)}"


def apply_rejects(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Коды брака по текущим порогам (можно пересчитать без повторного анализа)."""
    q = cfg.quality
    sim = df["similarity"]
    valid = sim.dropna()
    pct = np.percentile(valid, q.similarity_pctl_min) if len(valid) >= 20 else -np.inf
    zs = (sim - valid.mean()) / (valid.std(ddof=0) + 1e-9) if len(valid) >= 3 else sim * 0
    rej = []
    for i, r in df.iterrows():
        c = []
        if r["dur"] < cfg.segment.min_s:
            c.append("SHORT")
        if r["clip"]:
            c.append("CLIP")
        if r["clicks"] > 0:
            c.append("CLICK")
        if np.isfinite(r["snr_db"]) and r["snr_db"] < q.snr_min_db:
            c.append("NOISE")
        if r["tail_reverb"] and np.isfinite(r["tail_db"]) and r["tail_db"] > q.reverb_tail_db:
            c.append("REVERB")
        if r["break_rate"] > q.break_rate_max:
            c.append("PITCH")
        if r["voiced_frac"] < q.voiced_min:
            c.append("UNVOICED")
        if np.isfinite(sim[i]) and (sim[i] < pct or zs[i] < q.similarity_z_min):
            c.append("LOWSIM")
        rej.append(";".join(c))
    df = df.copy()
    df["reject"] = rej
    df["unsure"] = df["delivery_conf"] < cfg.delivery.unsure_below
    return df


def log_rejects(df: pd.DataFrame) -> None:
    for _, r in df.iterrows():
        pos = f"{r['file']} {r['start']:.2f}–{r['end']:.2f}с"
        if r["reject"]:
            why = ", ".join(f"{c} ({qt.CODES[c]})" for c in r["reject"].split(";"))
            log.debug("фраза %d [%s]: ОТБРОШЕНА: %s", r["id"], pos, why)
        else:
            log.debug("фраза %d [%s]: чистая, подача %s (%.2f)", r["id"], pos, r["delivery"], r["delivery_conf"])


def analyze(input_dir: str | Path, paths: Paths, cfg: Config, labels: str | Path | None = None) -> pd.DataFrame:
    t0 = time.time()
    songs = discover(input_dir)
    if not songs:
        raise SystemExit(f"В папке {input_dir} нет WAV/FLAC файлов.")
    sr = cfg.io.sr
    paths.phrases.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pid = 0
    total_in = 0.0
    for si, song in enumerate(songs):
        raw = load_raw(song.path, sr)
        x = condition(raw, sr, cfg.io.highpass_hz)
        total_in += len(x) / sr
        segs = segment(x, sr, cfg.segment)
        floor = qt.noise_floor_db(x, sr, segs)
        log.info("[%d/%d] %s: %.0f с, фраз: %d, фон %.0f дБ", si + 1, len(songs), song.file,
                 len(x) / sr, len(segs), floor)
        for k, (s, e) in enumerate(segs):
            pid += 1
            a, b = int(s * sr), int(e * sr)
            ph, rw = x[a:b], raw[a:b]
            f0 = track_f0(ph, sr, cfg.pitch)
            feat = phrase_pitch_features(f0, cfg.pitch)
            nxt = segs[k + 1][0] if k + 1 < len(segs) else None
            tail_db, tail_ok = qt.reverb_tail_db(x, sr, e, nxt, floor, cfg.quality.reverb_margin_db)
            snr = qt.snr_db(ph, sr, f0, floor)
            clicks = qt.count_clicks(ph, cfg.quality.click_k, cfg.quality.click_min_abs)
            spec = dlv.spectral_features(ph, sr)
            vq = dlv.voice_quality(ph, sr)
            fname = f"{pid:04d}_{_slug(song.song)}.wav"
            write_wav24(paths.phrases / fname, ph, sr)
            row = dict(
                id=pid, file=song.file, song=song.song, song_idx=si, key=song.key, style=song.style,
                start=s, end=e, dur=(b - a) / sr, phrase_path=f"phrases/{fname}",
                lufs=ld.lufs(ph, sr), peak_dbfs=ld.peak_dbfs(ph), crest_db=ld.crest_db(ph),
                clip=qt.clip_run(rw, cfg.quality.clip_level, cfg.quality.clip_run),
                clicks=clicks, snr_db=snr, tail_db=tail_db, tail_reverb=tail_ok,
                noise_floor_db=floor, **spec, **vq, **feat,
            )
            row["range"] = note_range(row["f0_p5"], row["f0_p95"])
            row["quality"] = qt.quality_score(snr, row["break_rate"], clicks, cfg.quality)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(int)
    df = df.set_index("id", drop=False)
    df.index.name = None

    # подача
    knn = paths.work / "delivery_knn.json"
    if knn.exists():
        knn.unlink()
    df = _delivery(df, cfg, paths, labels)

    # сходство с голосом
    emb_fn, engine = spk.get_embedder(cfg.speaker)
    embs: list[np.ndarray | None] = []
    for _, r in df.iterrows():
        if r["dur"] >= cfg.speaker.min_s:
            x_ph, sr_ph = sf.read(str(paths.root / r["phrase_path"]), dtype="float32")
            embs.append(emb_fn(x_ph, sr_ph))
        else:
            embs.append(None)
    df["similarity"] = spk.similarities(embs, engine)
    df.attrs["speaker_engine"] = engine

    # слова (опционально)
    df["text"] = ""
    if cfg.asr.enabled:
        m = asr_mod.load(cfg.asr.model)
        if m is not None:
            for i, r in df.iterrows():
                x_ph, sr_ph = sf.read(str(paths.root / r["phrase_path"]), dtype="float32")
                df.at[i, "text"] = asr_mod.transcribe(m, x_ph, sr_ph)

    df = apply_rejects(df, cfg)
    df["sem_midi"] = df["sem_midi"].apply(lambda v: [int(i) for i in v])
    df["sem_ms"] = df["sem_ms"].apply(lambda v: [int(i) for i in v])
    save_table(paths, df)
    log_rejects(df)
    keep = int((df["reject"] == "").sum())
    log.info("Готово за %.0f с: вход %.1f мин, фраз %d, чистых %d, отброшено %d",
             time.time() - t0, total_in / 60, len(df), keep, len(df) - keep)
    (paths.work / "meta.yaml").write_text(
        yaml.safe_dump({"input_minutes": round(total_in / 60, 2), "speaker_engine": engine,
                        "elapsed_s": round(time.time() - t0, 1)}, allow_unicode=True), encoding="utf-8")
    return df


def _delivery(df: pd.DataFrame, cfg: Config, paths: Paths, labels) -> pd.DataFrame:
    if labels:
        lab = yaml.safe_load(Path(labels).read_text(encoding="utf-8")) or []
        if isinstance(lab, dict):
            lab = lab.get("labels", [])
        labeled = []
        for item in lab:
            m = df[(df["file"] == item["file"]) & (df["start"] < item["end"]) & (df["end"] > item["start"])]
            for i, r in m.iterrows():
                ov = min(r["end"], item["end"]) - max(r["start"], item["start"])
                if ov >= 0.5 * (r["end"] - r["start"]):
                    labeled.append((int(i), item["label"]))
        if len(labeled) >= 3:
            paths.work.mkdir(parents=True, exist_ok=True)
            dlv.train_knn(df, labeled, paths.work / "delivery_knn.json")
            log.info("Подача: kNN обучен по %d размеченным фразам", len(labeled))
            return dlv.classify_knn(df, paths.work / "delivery_knn.json", cfg.delivery.knn_k)
        log.warning("В labels.yaml меньше 3 подходящих фраз — использую правила")
    elif (paths.work / "delivery_knn.json").exists():
        return dlv.classify_knn(df, paths.work / "delivery_knn.json", cfg.delivery.knn_k)
    return dlv.classify_rules(df, cfg.delivery)


def calibrate(paths: Paths, cfg: Config, labels: str | Path) -> pd.DataFrame:
    df = load_table(paths)
    df = _delivery(df, cfg, paths, labels)
    df = apply_rejects(df, cfg)
    save_table(paths, df)
    c = df["delivery"].value_counts().to_dict()
    log.info("Калибровка выполнена. Подачи: %s", c)
    return df
