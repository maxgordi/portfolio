"""Нарезка на фразы: Silero VAD (фолбэк: порог энергии), склейка пауз, деление длинных, запас по краям."""
from __future__ import annotations

import logging

import numpy as np

from .audio_io import resample
from .config import SegmentCfg

log = logging.getLogger("voiceset")
FRAME = 0.010


def frame_db(x: np.ndarray, sr: int, win_s: float = 0.025) -> np.ndarray:
    """RMS в дБ на сетке 10 мс (кадр i покрывает [i*10мс, i*10мс+win])."""
    hop = int(round(sr * FRAME))
    win = int(round(sr * win_s))
    n = max(1, 1 + (len(x) - win) // hop) if len(x) >= win else 1
    pad = np.pad(x.astype(np.float64) ** 2, (0, max(0, win + (n - 1) * hop - len(x))))
    cs = np.concatenate([[0.0], np.cumsum(pad)])
    idx = np.arange(n) * hop
    ms = (cs[idx + win] - cs[idx]) / win
    return 10 * np.log10(ms + 1e-12)


def vad_energy(x: np.ndarray, sr: int, margin_db: float) -> list[tuple[float, float]]:
    db = frame_db(x, sr)
    floor = max(float(np.percentile(db, 2)), -80.0)
    mask = db > floor + margin_db
    return _mask_to_segments(mask)


def _mask_to_segments(mask: np.ndarray) -> list[tuple[float, float]]:
    """Кадр i покрывает [i·10 мс, i·10 мс + 25 мс]: начало ≈ конец первого активного окна, конец ≈ начало последнего."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start * FRAME + 0.020, (i - 1) * FRAME + 0.005))
            start = None
    if start is not None:
        out.append((start * FRAME + 0.020, (len(mask) - 1) * FRAME + 0.005))
    return [(s, e) for s, e in out if e > s]


_silero = None


def vad_silero(x: np.ndarray, sr: int) -> list[tuple[float, float]] | None:
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except Exception as e:  # noqa: BLE001
        log.info("Silero VAD недоступен (%s), использую порог энергии", e)
        return None
    try:
        global _silero
        if _silero is None:
            _silero = load_silero_vad()
        model = _silero
        wav = torch.from_numpy(resample(x, sr, 16000))
        ts = get_speech_timestamps(wav, model, sampling_rate=16000, threshold=0.35,
                                   min_silence_duration_ms=60, speech_pad_ms=0,
                                   min_speech_duration_ms=150, return_seconds=True)
    except Exception as e:  # noqa: BLE001
        log.info("Silero VAD не сработал (%s), использую порог энергии", e)
        return None
    return [(t["start"], t["end"]) for t in ts]


def merge_short_gaps(segs, gap_s):
    out = []
    for s, e in segs:
        if out and s - out[-1][1] < gap_s:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _split_long(seg, db, cfg: SegmentCfg):
    """Делит в минимуме энергии; обе части не короче min_s, провал не короче min_split_ms."""
    a, b = seg
    if b - a <= cfg.max_s:
        return [seg]
    w = max(1, int(round(cfg.min_split_ms / 1000 / FRAME)))
    ia_lo = int(round((a + cfg.min_s) / FRAME))
    ia_hi = int(round(min(a + cfg.max_s, b - cfg.min_s) / FRAME))
    if ia_hi <= ia_lo:
        return [seg]
    kern = np.ones(w) / w
    sm = np.convolve(db, kern, mode="same")
    cand = np.arange(ia_lo, ia_hi + 1)
    cand = cand[cand < len(sm)]
    if len(cand) == 0:
        return [seg]
    cut = cand[int(np.argmin(sm[cand]))] * FRAME
    return _split_long((a, cut), db, cfg) + _split_long((cut, b), db, cfg)


def segment(x: np.ndarray, sr: int, cfg: SegmentCfg) -> list[tuple[float, float]]:
    """x — кондиционированный моно сигнал. Возвращает [(start_s, end_s)] с запасом по краям."""
    segs = None
    if cfg.vad in ("auto", "silero"):
        segs = vad_silero(x, sr)
        if segs is not None:
            # Silero учили на речи: если он видит заметно меньше, чем порог энергии, пению не доверяем
            en = vad_energy(x, sr, cfg.energy_margin_db)
            cov = lambda v: sum(e - s for s, e in v)  # noqa: E731
            if not segs or cov(segs) < 0.6 * cov(en):
                log.info("Silero нашёл мало пения (%.0f с против %.0f с по энергии) — беру порог энергии",
                         cov(segs), cov(en))
                segs = None
    if segs is None:
        segs = vad_energy(x, sr, cfg.energy_margin_db)
    segs = merge_short_gaps(segs, cfg.merge_gap_ms / 1000)
    segs = [s for s in segs if s[1] - s[0] >= 0.3]
    db = frame_db(x, sr)
    out = []
    for s in segs:
        out.extend(_split_long(s, db, cfg))
    # запас по краям, без наложения на соседей
    dur = len(x) / sr
    pad = cfg.pad_ms / 1000
    res = []
    for i, (s, e) in enumerate(out):
        lo = max(0.0, s - pad)
        hi = min(dur, e + pad)
        if i > 0:
            lo = max(lo, (out[i - 1][1] + s) / 2) if s - out[i - 1][1] < 2 * pad else lo
        if i + 1 < len(out):
            hi = min(hi, (e + out[i + 1][0]) / 2) if out[i + 1][0] - e < 2 * pad else hi
        res.append((round(lo, 4), round(hi, 4)))
    return res
