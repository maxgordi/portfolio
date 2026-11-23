"""
Высота тона (необязательно, по умолчанию выключено — для RVC нужен только тайминг):
подтяжка нот к эталону и определение типа дорожки (унисон / гармония / октава).
Перенесено из VoxMatch Простой.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from .features import FRAME_MS, HOP, ANALYSIS_SR, segments, smooth


def f0_track(x: np.ndarray, quality: str = "fast") -> np.ndarray:
    """f0 на кадрах анализа (24 кГц, шаг 5 мс). x — предобработанный сигнал анализа."""
    from .world import analyze_world
    f0, _sp, _ap = analyze_world(x.astype(np.float64), ANALYSIS_SR, HOP, quality, False)
    return f0


def pitch_curve(f0_ref, f0_dub_aligned, amount: float, fm: float = FRAME_MS):
    """Отношение частот на каждый кадр результата. Медленная часть разницы
    эталон−дубль (ноту) правим, быструю (вибрато дубля) оставляем."""
    n = len(f0_ref)
    both = (f0_ref > 0) & (f0_dub_aligned > 0)
    c = np.zeros(n)
    if both.sum() < 10 or amount <= 0:
        return np.ones(n), 0.0, 0.0, 0
    c[both] = 1200.0 * np.log2(f0_ref[both] / f0_dub_aligned[both])
    octave = int(np.round(np.median(c[both]) / 1200.0))
    c[both] -= 1200.0 * octave                               # другая октава — сохраняем
    valid = both & (np.abs(c) < 350.0)                       # больше 3,5 полутона — не та нота
    before = float(np.median(np.abs(c[valid]))) if valid.any() else 0.0
    cm = c.copy()
    k = max(3, int(round(35.0 / fm)) | 1)
    for a, b in segments(valid):
        if b - a >= k:
            cm[a:b] = ss.medfilt(c[a:b], k)
    w = int(round(120.0 / fm))
    corr = np.zeros(n)
    for a, b in segments(valid):
        seg = cm[a:b]
        if b - a > 2:
            ww = min(w, b - a)
            corr[a:b] = smooth(seg, ww) if ww > 2 else seg
        else:
            corr[a:b] = seg
    idx = np.arange(n)
    if valid.any():
        filled = np.interp(idx, idx[valid], corr[valid])
        vi = np.flatnonzero(valid)
        j = np.searchsorted(vi, idx)
        left = np.where(j > 0, idx - vi[np.clip(j - 1, 0, len(vi) - 1)], 1e9)
        right = np.where(j < len(vi), vi[np.clip(j, 0, len(vi) - 1)] - idx, 1e9)
        dist = np.minimum(left, right) * fm
        fade = np.clip(1.0 - (dist - 60.0) / 60.0, 0.0, 1.0)  # 60 мс держим, ещё 60 — гасим
        corr = filled * fade
    corr = smooth(corr, int(round(20.0 / fm)))
    corr *= amount
    after = float(np.median(np.abs(c[valid] - corr[valid]))) if valid.any() else 0.0
    return 2.0 ** (corr / 1200.0), before, after, octave


def classify(f0_ref, f0_al):
    """Унисон, гармония или октава: по интервалу между нотами эталона и дубля."""
    both = (f0_ref > 0) & (f0_al > 0)
    if both.sum() < 40:
        return "унисон", 0.0, 0
    c = 1200.0 * np.log2(f0_al[both] / f0_ref[both])
    octave = int(np.round(np.median(c) / 1200.0))
    r = c - 1200.0 * octave
    r = (r + 600.0) % 1200.0 - 600.0
    near = float(np.mean(np.abs(r) < 100.0))
    med = float(np.median(r))
    if near >= 0.5 and abs(med) < 100.0:
        return ("октава" if octave else "унисон"), float(np.median(c)), octave
    return "гармония", float(np.median(c)), octave
