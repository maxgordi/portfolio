"""Брак: перегруз, щелчки, шум фона (SNR), реверб, нестабильная высота. Каждый — с кодом причины."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import QualityCfg
from .segment import frame_db

CODES = {
    "CLIP": "перегруз (клиппинг)",
    "CLICK": "щелчок",
    "NOISE": "шум фона (низкий SNR)",
    "REVERB": "реверб / длинный хвост",
    "PITCH": "нестабильная высота (срывы)",
    "UNVOICED": "почти нет озвученных кадров",
    "SHORT": "слишком короткая фраза",
    "LOWSIM": "не похоже на голос (сходство)",
}


def clip_run(x: np.ndarray, level: float, run: int) -> bool:
    hit = (np.abs(x) >= level).astype(np.int8)
    if hit.sum() < run:
        return False
    # длина максимальной серии подряд
    d = np.diff(np.concatenate([[0], hit, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return bool(np.any(ends - starts >= run))


def count_clicks(x: np.ndarray, k: float, min_abs: float) -> int:
    dx = np.abs(np.diff(x.astype(np.float64)))
    if len(dx) == 0:
        return 0
    mad = np.median(np.abs(dx - np.median(dx))) * 1.4826
    thr = max(k * mad, min_abs)
    over = dx > thr
    # соседние выбросы — один щелчок
    return int(ndimage.label(ndimage.binary_dilation(over, iterations=8))[1])


def noise_floor_db(x: np.ndarray, sr: int, segs: list[tuple[float, float]]) -> float:
    """Уровень фона: кадры вне фраз; если их нет — 5-й перцентиль энергии."""
    db = frame_db(x, sr)
    mask = np.ones(len(db), bool)
    for s, e in segs:
        mask[int(s / 0.01):int(np.ceil(e / 0.01)) + 1] = False
    if mask.sum() >= 50:
        return float(10 * np.log10(np.mean(10 ** (db[mask] / 10))))
    return float(np.percentile(db, 5))


def snr_db(x: np.ndarray, sr: int, f0: np.ndarray, floor_db: float) -> float:
    db = frame_db(x, sr)
    n = min(len(db), len(f0))
    v = np.isfinite(f0[:n])
    if v.sum() < 3:
        return float("nan")
    sig = 10 * np.log10(np.mean(10 ** (db[:n][v] / 10)))
    return float(sig - floor_db)


def reverb_tail_db(full: np.ndarray, sr: int, end_s: float, next_start_s: float | None,
                   floor_db: float, margin: float) -> tuple[float, bool]:
    """Уровень хвоста 100–250 мс после конца относительно последних 150 мс фразы."""
    a = int((end_s - 0.15) * sr)
    b = int(end_s * sr)
    t0, t1 = int((end_s + 0.10) * sr), int((end_s + 0.25) * sr)
    if t1 > len(full) or (next_start_s is not None and next_start_s < end_s + 0.25) or a < 0:
        return float("nan"), False
    lvl = lambda y: 10 * np.log10(np.mean(y.astype(np.float64) ** 2) + 1e-12)  # noqa: E731
    end_lvl, tail = lvl(full[a:b]), lvl(full[t0:t1])
    return float(tail - end_lvl), bool(tail > floor_db + margin)


def quality_score(snr: float, break_rate: float, clicks: int, cfg: QualityCfg) -> float:
    s = 0.0 if not np.isfinite(snr) else np.clip((snr - cfg.snr_min_db) / 20 + 0.5, 0, 1)
    p = 1 - np.clip(break_rate / max(cfg.break_rate_max, 1e-6), 0, 1)
    c = 1.0 if clicks == 0 else 0.0
    return float(0.5 * s + 0.35 * p + 0.15 * c)
