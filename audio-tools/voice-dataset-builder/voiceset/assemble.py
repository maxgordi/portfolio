"""Склейка набора для Suno: громкость, мягкая компрессия, затухания, зазоры, срез гула, точная длина."""
from __future__ import annotations

import logging

import numpy as np
import soundfile as sf
from scipy import signal

from . import loudness as ld
from .config import AssembleCfg

log = logging.getLogger("voiceset")


def lowshelf_cut(x: np.ndarray, sr: int, hz: float, db: float) -> np.ndarray:
    """Полка ниже hz: ослабление на db (<0) без сдвига фазы."""
    if db == 0:
        return x
    sos = signal.butter(2, hz, btype="lowpass", fs=sr, output="sos")
    low = signal.sosfiltfilt(sos, x)
    return (x + (10 ** (db / 20) - 1) * low).astype(np.float32)


def compress(x: np.ndarray, sr: int, thr_db: float, ratio: float) -> np.ndarray:
    """Мягкая статическая компрессия по огибающей (детерминированная, без look-ahead)."""
    a_env = np.exp(-1 / (0.010 * sr))
    env = signal.lfilter([1 - a_env], [1, -a_env], np.abs(x).astype(np.float64))
    env_db = 20 * np.log10(env + 1e-9)
    over = np.maximum(0.0, env_db - thr_db)
    g_db = -over * (1 - 1 / ratio)
    a_g = np.exp(-1 / (0.020 * sr))
    g_db = signal.lfilter([1 - a_g], [1, -a_g], g_db)
    return (x * 10 ** (g_db / 20)).astype(np.float32)


def fade(x: np.ndarray, sr: int, ms: float) -> np.ndarray:
    n = min(int(sr * ms / 1000), len(x) // 2)
    if n <= 0:
        return x
    w = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    y = x.copy()
    y[:n] *= w
    y[-n:] *= w[::-1]
    return y


def prepare(x: np.ndarray, sr: int, a: AssembleCfg) -> np.ndarray:
    y = lowshelf_cut(x, sr, a.lowshelf_hz, a.lowshelf_db)
    y = ld.normalize_to(y, sr, a.target_lufs)
    if a.compress:
        y = compress(y, sr, a.comp_threshold_db, a.comp_ratio)
        y = ld.normalize_to(y, sr, a.target_lufs)
    return fade(y, sr, a.fade_ms)


def join(parts: list[np.ndarray], sr: int, length_s: float, a: AssembleCfg) -> tuple[np.ndarray, list[int], list[str]]:
    """Склейка до точной длины. Возвращает (аудио, зазоры в сэмплах, предупреждения)."""
    warn: list[str] = []
    total = int(round(length_s * sr))
    body = sum(len(p) for p in parts)
    n = len(parts)
    if n == 1:
        gaps = []
    else:
        left = total - body
        g_lo, g_hi = (int(round(v / 1000 * sr)) for v in a.gap_range_ms)
        if left < g_lo * (n - 1) or left > g_hi * (n - 1):
            warn.append(f"Не удалось уложиться в {length_s:.1f} с зазорами {a.gap_range_ms[0]:.0f}–{a.gap_range_ms[1]:.0f} мс "
                        f"(получилось бы {(left / (n - 1)) / sr * 1000:.0f} мс) — зазор ограничен, длина будет неточной.")
            left = int(np.clip(left, g_lo * (n - 1), g_hi * (n - 1)))
        base, rem = divmod(left, n - 1)
        gaps = [base + (1 if i < rem else 0) for i in range(n - 1)]
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        if i < n - 1:
            out.append(np.zeros(gaps[i], np.float32))
    y = np.concatenate(out)
    if n == 1 and len(y) < total:
        y = np.concatenate([y, np.zeros(total - len(y), np.float32)])
    return y, gaps, warn


def limit_peak(y: np.ndarray, peak_dbfs: float) -> np.ndarray:
    lim = 10 ** (peak_dbfs / 20)
    p = float(np.max(np.abs(y)))
    return (y * (lim / p)).astype(np.float32) if p > lim else y


def sliding_window_check(order_rows: list[dict], gaps_s: list[float], quotas: dict[str, float],
                         tol: float, win_s: float = 120.0) -> list[str]:
    """Для наборов длиннее 2 минут: любое окно win_s должно держать квоты (для --length 240)."""
    t, spans = 0.0, []
    for i, r in enumerate(order_rows):
        spans.append((t, t + r["dur"], r["delivery"]))
        t += r["dur"] + (gaps_s[i] if i < len(gaps_s) else 0)
    total = t
    if total <= win_s + 1:
        return []
    bad = []
    for s in np.arange(0, total - win_s + 1e-9, 10.0):
        e = s + win_s
        acc = {d: 0.0 for d in quotas}
        for a, b, d in spans:
            ov = max(0.0, min(b, e) - max(a, s))
            acc[d] = acc.get(d, 0) + ov
        tot = sum(acc.values()) or 1.0
        if any(abs(acc[d] / tot - q) > tol * 2 for d, q in quotas.items()):
            bad.append(f"{s:.0f}–{e:.0f} с")
    return [f"Окно 2:00 не держит квоты: {', '.join(bad[:6])}{'…' if len(bad) > 6 else ''}"] if bad else []
