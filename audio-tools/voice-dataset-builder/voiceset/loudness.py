"""Громкость: integrated LUFS (pyloudnorm), пик в dBFS, крест-фактор."""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln

_meters: dict[int, pyln.Meter] = {}


def lufs(x: np.ndarray, sr: int) -> float:
    if len(x) < int(0.4 * sr) or np.max(np.abs(x)) < 1e-6:
        return float("-inf")
    m = _meters.setdefault(sr, pyln.Meter(sr))
    return float(m.integrated_loudness(x.astype(np.float64)))


def peak_dbfs(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if len(x) else 0.0
    return 20 * np.log10(p + 1e-12)


def crest_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
    return peak_dbfs(x) - 20 * np.log10(rms)


def normalize_to(x: np.ndarray, sr: int, target: float) -> np.ndarray:
    l = lufs(x, sr)
    if not np.isfinite(l):
        return x
    return (x * 10 ** ((target - l) / 20)).astype(np.float32)
