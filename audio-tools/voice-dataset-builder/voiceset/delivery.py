"""Подача: breathy / normal / power. Правила по z-оценкам самого певца, либо kNN по labels.yaml."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DeliveryCfg

log = logging.getLogger("voiceset")
FEATURES = ["hnr", "cpps", "tilt", "sf_ratio", "lufs", "centroid"]
LABELS = ["breathy", "normal", "power"]


def spectral_features(x: np.ndarray, sr: int) -> dict:
    from scipy import signal
    f, p = signal.welch(x, sr, nperseg=4096)
    p = p + 1e-20
    sel = (f >= 100) & (f <= 5000)
    slope = np.polyfit(np.log2(f[sel]), 10 * np.log10(p[sel]), 1)[0]  # дБ/октаву
    lo = p[(f >= 0) & (f < 2000)].sum()
    hi = p[(f >= 2000) & (f <= 4000)].sum()
    return {"tilt": float(slope), "sf_ratio": float(10 * np.log10(hi / lo)),
            "centroid": float((f * p).sum() / p.sum())}


def voice_quality(x: np.ndarray, sr: int) -> dict:
    """HNR и CPPS через Praat."""
    import parselmouth
    from parselmouth.praat import call
    from .audio_io import resample
    s22 = resample(x, sr, 22050)
    snd = parselmouth.Sound(s22.astype(np.float64), sampling_frequency=22050)
    out = {"hnr": np.nan, "cpps": np.nan}
    try:
        h = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=60)
        vals = h.values[h.values > -100]
        if vals.size:
            out["hnr"] = float(np.mean(vals))
    except Exception as e:  # noqa: BLE001
        log.debug("HNR: %s", e)
    try:
        pc = call(snd, "To PowerCepstrogram", 60, 0.01, 5000, 50)
        out["cpps"] = float(call(pc, "Get CPPS", True, 0.02, 0.0005, 60, 1000, 0.05,
                                 "Parabolic", 0.001, 0, "Straight", "Robust"))
    except Exception as e:  # noqa: BLE001
        log.debug("CPPS: %s", e)
    return out


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-9:
        return s * 0.0
    return (s - s.mean()) / sd


def classify_rules(df: pd.DataFrame, cfg: DeliveryCfg) -> pd.DataFrame:
    """Правила: придыхание = низкий HNR/CPPS и тихо; опора = громко, сильная полоса 2–4 кГц, высокий HNR."""
    ok = df["voiced_frac"] > 0
    d = df[ok]
    z = {k: _z(d[k].astype(float).fillna(d[k].median())) for k in FEATURES}
    breathy = (-z["hnr"] - z["cpps"] - z["lufs"]) / 3
    power = (z["lufs"] + z["sf_ratio"] + z["hnr"]) / 3
    label = pd.Series("normal", index=d.index)
    label[(breathy > cfg.z_breathy) & (breathy >= power)] = "breathy"
    label[(power > cfg.z_power) & (power > breathy)] = "power"
    diff = (power - breathy)
    thr = {"normal": None}
    conf = pd.Series(0.0, index=d.index)
    m = label == "breathy"
    conf[m] = np.clip((breathy[m] - cfg.z_breathy) / 1.0 + 0.3, 0, 1)
    m = label == "power"
    conf[m] = np.clip((power[m] - cfg.z_power) / 1.0 + 0.3, 0, 1)
    m = label == "normal"
    far = np.maximum(breathy[m] - cfg.z_breathy, power[m] - cfg.z_power)  # <0: далеко от границ
    conf[m] = np.clip(0.4 + (-far) / 1.5, 0, 1)
    df = df.copy()
    df["delivery"] = "normal"
    df["delivery_conf"] = 0.0
    df.loc[d.index, "delivery"] = label
    df.loc[d.index, "delivery_conf"] = conf
    del diff, thr
    return df


def _matrix(df: pd.DataFrame, mu, sd) -> np.ndarray:
    X = df[FEATURES].astype(float).fillna(df[FEATURES].astype(float).median()).to_numpy()
    return (X - mu) / sd


def train_knn(df: pd.DataFrame, labeled: list[tuple[int, str]], path: Path) -> None:
    ids = [i for i, _ in labeled]
    sub = df.loc[ids]
    allX = df[FEATURES].astype(float)
    mu = allX.mean().to_numpy()
    sd = allX.std(ddof=0).replace(0, 1).to_numpy()
    X = _matrix(sub, mu, sd)
    Path(path).write_text(json.dumps({"mu": mu.tolist(), "sd": sd.tolist(), "X": X.tolist(),
                                      "y": [l for _, l in labeled]}), encoding="utf-8")


def classify_knn(df: pd.DataFrame, path: Path, k: int) -> pd.DataFrame:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    X = np.array(m["X"])
    y = np.array(m["y"])
    Q = _matrix(df, np.array(m["mu"]), np.array(m["sd"]))
    kk = min(k, len(X))
    labels, conf = [], []
    for q in Q:
        d = np.linalg.norm(X - q, axis=1)
        idx = np.argsort(d, kind="stable")[:kk]
        votes = {l: 0.0 for l in LABELS}
        for i in idx:
            votes[y[i]] += 1.0 / (d[i] + 1e-6)
        tot = sum(votes.values())
        best = max(LABELS, key=lambda l: (votes[l], -LABELS.index(l)))
        labels.append(best)
        conf.append(votes[best] / tot if tot else 0.0)
    df = df.copy()
    df["delivery"] = labels
    df["delivery_conf"] = conf
    return df
