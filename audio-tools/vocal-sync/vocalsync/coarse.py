"""
Грубое выравнивание: начальный сдвиг и глобальный темп.

1. Огибающие онсетов (и громкости) обеих дорожек на шаге 20 мс.
2. Перебор темпа 0.94…1.06: огибающая эталона растягивается, кросс-корреляция
   с огибающей dub (FFT) даёт сдвиг; побеждает пара (темп, сдвиг) с лучшей
   нормированной корреляцией. Затем уточнение мелким шагом.
3. Линия dub_t = tempo · guide_t + offset потом уточняется устойчивой регрессией
   по грубому пути DTW (см. pipeline) — она точнее корреляции.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.signal as ss

from .features import Analysis, smooth

log = logging.getLogger("vocalsync.coarse")

DECIM = 4                        # 20 мс


@dataclass
class Line:
    tempo: float                 # секунд dub на секунду эталона (>1 — dub медленнее)
    offset: float                # где в dub (с) звучит момент 0 эталона
    score: float = 0.0

    def frames(self, n: int, fm: float) -> np.ndarray:
        """Позиция в dub (кадры) для каждого кадра эталона."""
        return self.tempo * np.arange(n) + self.offset * 1000.0 / fm


def envelope(an: Analysis) -> np.ndarray:
    """Огибающая для корреляции: онсеты + рост громкости, только в звучащих местах."""
    on = np.maximum(an.onset, 0.0) * an.loud
    rise = np.maximum(np.diff(an.energy, prepend=an.energy[0]), 0.0) * an.loud
    e = on / (on.std() + 1e-9) + 0.5 * rise / (rise.std() + 1e-9) + 0.3 * an.loud
    e = smooth(e, 6)
    n = len(e) // DECIM
    e = e[: n * DECIM].reshape(n, DECIM).max(axis=1)
    return e - e.mean()


def _xcorr_best(d: np.ndarray, g: np.ndarray):
    c = ss.correlate(d, g, mode="full", method="fft")
    lags = ss.correlation_lags(len(d), len(g), mode="full")
    k = int(np.argmax(c))
    frac = 0.0
    if 0 < k < len(c) - 1:                          # параболическое уточнение пика
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            frac = 0.5 * (y0 - y2) / den
    norm = np.linalg.norm(d) * np.linalg.norm(g) + 1e-9
    return float(lags[k] + frac), float(c[k] / norm)


def _stretch(g: np.ndarray, tempo: float) -> np.ndarray:
    n = int(round(len(g) * tempo))
    return np.interp(np.arange(n) / tempo, np.arange(len(g)), g)


def estimate_line(guide: Analysis, dub: Analysis, tempo_range: float = 0.06) -> Line:
    fm_c = guide.fm * DECIM
    g, d = envelope(guide), envelope(dub)
    best = (0.0, 0.0, -1e9)

    def scan(tempos):
        nonlocal best
        for t in tempos:
            lag, sc = _xcorr_best(d, _stretch(g, t))
            if sc > best[2]:
                best = (t, lag, sc)

    scan(np.arange(1.0 - tempo_range, 1.0 + tempo_range + 1e-9, 0.0025))
    t0 = best[0]
    scan(np.arange(t0 - 0.003, t0 + 0.003 + 1e-9, 0.0005))
    tempo, lag, sc = best
    # сдвиг в огибающей dub: момент 0 эталона = lag кадров dub (20 мс)
    line = Line(tempo=float(tempo), offset=float(lag * fm_c / 1000.0), score=float(sc))
    log.info("грубо: темп %.4f, сдвиг %.3f с, корреляция %.3f", line.tempo, line.offset, sc)
    return line


def robust_line(map_frames: np.ndarray, mask: np.ndarray, fm: float, iters: int = 8) -> Line:
    """Устойчивая (Huber, IRLS) прямая j = a·i + b по точкам карты в звучащих кадрах."""
    i = np.flatnonzero(mask)
    if len(i) < 20:
        i = np.arange(len(map_frames))
    y = map_frames[i]
    x = i.astype(np.float64)
    w = np.ones(len(i))
    a, b = 1.0, float(np.median(y - x))
    delta = 60.0 / fm                                   # 60 мс
    for _ in range(iters):
        W = np.sqrt(w)
        M = np.vstack([x * W, W]).T
        sol, *_ = np.linalg.lstsq(M, y * W, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = np.abs(y - (a * x + b))
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9))
    return Line(tempo=a, offset=b * fm / 1000.0)
