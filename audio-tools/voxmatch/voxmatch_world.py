"""
Анализ вокодером WORLD с разбиением длинных файлов на куски,
которые считаются параллельно на всех ядрах процессора.

Модуль намеренно лёгкий (только numpy и pyworld), чтобы рабочие процессы
запускались быстро.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
import numpy as np
import pyworld as pw

F0_FLOOR, F0_CEIL = 60.0, 1100.0
CHUNK_SEC = 12.0      # длина куска
PAD_SEC = 1.0         # запас по краям куска, чтобы на стыках не было ошибок
INLINE_SEC = 20.0     # короче этого — считаем без параллельности

_pool: ProcessPoolExecutor | None = None


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
        _pool = ProcessPoolExecutor(max_workers=workers)
    return _pool


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def _analyze_piece(x: np.ndarray, sr: int, fp: float, quality: str, need_env: bool):
    x = np.ascontiguousarray(x, dtype=np.float64)
    if quality == "fast":
        f0, t = pw.dio(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=fp)
        f0 = pw.stonemask(x, f0, t, sr)
    else:
        f0, t = pw.harvest(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=fp)
    if not need_env:
        return f0, None, None
    sp = pw.cheaptrick(x, f0, t, sr).astype(np.float32)
    ap = pw.d4c(x, f0, t, sr).astype(np.float32)
    return f0, sp, ap


def _job(args):
    return _analyze_piece(*args)


def analyze_world(x: np.ndarray, sr: int, hop: int, quality: str, need_env: bool,
                  progress=None):
    """Возвращает f0 (float64) и, если нужно, огибающую sp и апериодичность ap (float32).
    Кадры идут с шагом hop отсчётов, кадр k соответствует времени k*hop/sr."""
    fp = 1000.0 * hop / sr
    total = int(len(x) // hop) + 1
    if len(x) < INLINE_SEC * sr:
        f0, sp, ap = _analyze_piece(x, sr, fp, quality, need_env)
        return _fit_frames(f0, sp, ap, total)

    cf = int(CHUNK_SEC * 1000 / fp)        # кадров в куске
    pf = int(PAD_SEC * 1000 / fp)          # кадров запаса
    jobs, spans = [], []
    for start in range(0, total, cf):
        end = min(start + cf, total)
        a = max(0, start - pf)
        b = min(total, end + pf)
        jobs.append((x[a * hop: min(len(x), b * hop + hop)], sr, fp, quality, need_env))
        spans.append((start, end, a))

    pool = _get_pool()
    futures = [pool.submit(_job, j) for j in jobs]
    f0 = np.zeros(total)
    sp = ap = None
    for n, (start, end, a) in enumerate(spans):
        cf0, csp, cap = futures[n].result()
        futures[n] = None                   # освободить память куска
        lo, hi = start - a, end - a
        piece = _fit_frames(cf0, csp, cap, hi)
        f0[start:end] = piece[0][lo:hi]
        if need_env:
            if sp is None:
                sp = np.zeros((total, csp.shape[1]), dtype=np.float32)
                ap = np.zeros((total, cap.shape[1]), dtype=np.float32)
            sp[start:end] = piece[1][lo:hi]
            ap[start:end] = piece[2][lo:hi]
        if progress:
            progress((n + 1) / len(futures))
    return f0, sp, ap


def _fit_frames(f0, sp, ap, n):
    def fit(a):
        if a is None or len(a) == n:
            return a
        if len(a) > n:
            return a[:n]
        return np.concatenate([a, np.repeat(a[-1:], n - len(a), axis=0)])
    return fit(f0), fit(sp), fit(ap)
