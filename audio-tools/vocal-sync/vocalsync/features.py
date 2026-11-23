"""
Предобработка (только для анализа — на выходной звук не влияет) и признаки.

  моно → ресемплинг 24 кГц → полоса 80 Гц – 8 кГц → нормализация →
  кадры по 5 мс: MFCC (+дельты), chroma (CENS-подобная), огибающая онсетов,
  энергия; тишина — по RMS с порогом относительно пика самого файла.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.signal as ss

from . import io as mio

log = logging.getLogger("vocalsync.features")

ANALYSIS_SR = 24000
HOP = 120                       # 5 мс
FRAME_MS = 1000.0 * HOP / ANALYSIS_SR
CHROMA_DECIM = 4                # chroma считается на 20 мс и интерполируется


@dataclass
class Weights:
    chroma: float = 0.5
    mfcc: float = 1.0
    onset: float = 0.4
    energy: float = 0.4

    @classmethod
    def from_settings(cls, s) -> "Weights":
        return cls(s.w_chroma, s.w_mfcc, s.w_onset, s.w_energy)


@dataclass
class Analysis:
    path: str
    n: int
    t0: float                   # начало фрагмента в исходном файле, с
    mfcc: np.ndarray            # (n, 19) z-норм.
    dmfcc: np.ndarray           # (n, 19)
    chroma: np.ndarray          # (n, 12) L2-норм.
    onset: np.ndarray           # (n,) z-норм.
    energy: np.ndarray          # (n,) (дБ − пик)/10, обрезано в [−4, 1]
    db: np.ndarray              # (n,) дБ относительно 95-го перцентиля
    loud: np.ndarray            # (n,) bool — не тишина
    onsets: np.ndarray          # индексы кадров атак
    chroma_profile: np.ndarray  # (12,) средняя chroma по звучащим кадрам
    f0: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)

    fm: float = FRAME_MS        # мс на кадр
    sr: int = ANALYSIS_SR
    hop: int = HOP

    @property
    def duration(self) -> float:
        return self.n * self.fm / 1000.0


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    if a.shape[-1] == n:
        return a
    if a.shape[-1] > n:
        return a[..., :n]
    pad = [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])]
    return np.pad(a, pad, mode="edge")


def smooth(v: np.ndarray, win: int) -> np.ndarray:
    """Сглаживание окном Ханна (win кадров), края — по крайнему значению."""
    if win <= 2:
        return np.asarray(v, dtype=np.float64).copy()
    w = np.hanning(win + 2)[1:-1]
    w /= w.sum()
    pad = win
    return np.convolve(np.pad(np.asarray(v, np.float64), pad, mode="edge"), w, mode="same")[pad:-pad]


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _clean_mask(mask: np.ndarray, min_gap: int, min_len: int) -> np.ndarray:
    m = mask.copy()
    for a, b in segments(~m):                      # короткие паузы (стоп-согласные) — не тишина
        if 0 < a and b < len(m) and b - a < min_gap:
            m[a:b] = True
    for a, b in segments(m):                       # короткие всплески — тишина
        if b - a < min_len:
            m[a:b] = False
    return m


def preprocess(mono: np.ndarray, sr: int) -> np.ndarray:
    x = mio.resample(np.asarray(mono, np.float32), sr, ANALYSIS_SR)
    sos = ss.butter(4, [80.0, 8000.0], btype="bandpass", fs=ANALYSIS_SR, output="sos")
    x = ss.sosfiltfilt(sos, x).astype(np.float32)
    peak = float(np.percentile(np.abs(x), 99.9)) if len(x) else 0.0
    if peak > 1e-6:
        x /= peak
    return x


def analyze(mono: np.ndarray, sr: int, path: str = "", t0: float = 0.0,
            silence_db: float = 35.0) -> Analysis:
    import librosa

    x = preprocess(mono, sr)
    n = 1 + len(x) // HOP
    # --- спектр 1024 (43 мс): MFCC, онсеты, громкость
    X = np.abs(librosa.stft(x, n_fft=1024, hop_length=HOP, window="hann", center=True))
    P = X ** 2
    mel = librosa.feature.melspectrogram(S=P, sr=ANALYSIS_SR, n_mels=40, fmin=80.0, fmax=8000.0)
    logmel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=90.0)
    mf = librosa.feature.mfcc(S=logmel, n_mfcc=20)[1:]
    onset = librosa.onset.onset_strength(S=logmel, sr=ANALYSIS_SR, hop_length=HOP, lag=1,
                                         max_size=3)
    rms = np.sqrt(P.mean(axis=0) * 2.0 / 1024.0 + 1e-12)
    del X, P, mel
    mf, onset, rms = _fit(mf, n), _fit(onset, n), _fit(rms, n)
    db = 20.0 * np.log10(rms + 1e-9)
    level = float(np.percentile(db, 95))
    db = db - level
    sm = smooth(db, int(round(30.0 / FRAME_MS)))
    # порог тишины — от пика файла, но не ниже шумового пола (остатки инструментов
    # после сепаратора, шум): 5-й перцентиль + 8 дБ, не выше −15 дБ от пика
    floor = float(np.percentile(sm, 5)) + 8.0
    thr = max(-abs(silence_db), min(floor, -15.0))
    loud = sm > thr
    loud = _clean_mask(loud, int(round(60.0 / FRAME_MS)), int(round(30.0 / FRAME_MS)))
    if loud.sum() < 10:
        loud = np.ones(n, dtype=bool)

    def znorm(a):
        mu = a[..., loud].mean(axis=-1, keepdims=True)
        sd = a[..., loud].std(axis=-1, keepdims=True) + 1e-6
        return (a - mu) / sd

    mfn = znorm(mf)
    dmf = librosa.feature.delta(mfn, width=9, mode="nearest")
    dmf = dmf / (dmf[:, loud].std(axis=1, keepdims=True) + 1e-6)
    on_l = np.log1p(np.maximum(onset, 0.0))
    on_z = znorm(on_l[None, :])[0]
    energy = np.clip(db / 10.0, -4.0, 1.0)

    # --- chroma: спектр 4096 (170 мс) с шагом 20 мс, CENS-подобная обработка
    hop_c = HOP * CHROMA_DECIM
    Xc = np.abs(librosa.stft(x, n_fft=4096, hop_length=hop_c, window="hann", center=True)) ** 2
    ch = librosa.feature.chroma_stft(S=Xc, sr=ANALYSIS_SR, n_fft=4096, hop_length=hop_c,
                                     tuning=0.0, norm=None)
    del Xc
    ch = ch / (ch.sum(axis=0, keepdims=True) + 1e-9)
    ch = np.log1p(100.0 * ch)
    ch = np.vstack([smooth(r, 3) for r in ch])
    tc = np.arange(ch.shape[1]) * CHROMA_DECIM
    ti = np.arange(n)
    chroma = np.vstack([np.interp(ti, tc, r) for r in ch])
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma[:, ~loud] = 1.0 / np.sqrt(12.0)
    prof = chroma[:, loud].mean(axis=1)
    prof /= np.linalg.norm(prof) + 1e-9

    # --- атаки слогов
    try:
        ons = librosa.onset.onset_detect(onset_envelope=onset, sr=ANALYSIS_SR, hop_length=HOP,
                                         units="frames", backtrack=False)
    except Exception:  # noqa: BLE001
        ons = np.zeros(0, dtype=np.int64)
    ons = np.asarray(ons, dtype=np.int64)
    near_loud = np.convolve(loud.astype(np.float64), np.ones(9), mode="same") > 0
    ons = ons[(ons < n) & near_loud[np.clip(ons, 0, n - 1)]]
    starts = np.array([a for a, _b in segments(loud)], dtype=np.int64)
    ons = np.unique(np.concatenate([ons, starts]))

    return Analysis(path=path, n=n, t0=t0,
                    mfcc=np.ascontiguousarray(mfn.T, np.float32),
                    dmfcc=np.ascontiguousarray(dmf.T, np.float32),
                    chroma=np.ascontiguousarray(chroma.T, np.float32),
                    onset=on_z.astype(np.float32), energy=energy.astype(np.float32),
                    db=db.astype(np.float32), loud=loud, onsets=ons,
                    chroma_profile=prof.astype(np.float32))


def stack(an: Analysis, w: Weights) -> np.ndarray:
    """Вектор признаков на кадр. Блоки масштабированы так, что вклад блока в квадрат
    расстояния пропорционален его весу, а расстояние ≈ √2 для несвязанных кадров."""
    blocks, total = [], 0.0
    if w.mfcc > 0:
        blocks.append(an.mfcc * np.sqrt(w.mfcc / an.mfcc.shape[1]))
        blocks.append(an.dmfcc * np.sqrt(0.4 * w.mfcc / an.dmfcc.shape[1]))
        total += 1.4 * w.mfcc
    if w.chroma > 0:
        blocks.append(an.chroma * np.sqrt(w.chroma / 0.35))   # ≈ средний квадрат разницы 0.7
        total += w.chroma
    if w.onset > 0:
        blocks.append(an.onset[:, None] * np.sqrt(w.onset))
        total += w.onset
    if w.energy > 0:
        e = (an.energy - an.energy[an.loud].mean()) / (an.energy[an.loud].std() + 1e-6)
        blocks.append(e[:, None] * np.sqrt(w.energy))
        total += w.energy
    if not blocks:
        raise ValueError("Все веса признаков равны нулю.")
    F = np.hstack(blocks) / np.sqrt(max(total, 1e-9))
    return np.ascontiguousarray(F, dtype=np.float32)


def key_shift(g: Analysis, d: Analysis) -> tuple[int, float, float]:
    """Насколько полутонов dub выше эталона (по средней chroma) и уверенность.
    Возвращает (сдвиг −5…+6, корреляция при этом сдвиге, корреляция без сдвига)."""
    a = g.chroma_profile - g.chroma_profile.mean()
    b = d.chroma_profile - d.chroma_profile.mean()
    a /= np.linalg.norm(a) + 1e-9
    b /= np.linalg.norm(b) + 1e-9
    sims = [float(np.dot(a, np.roll(b, -k))) for k in range(12)]       # корреляция профилей
    k = int(np.argmax(sims))
    k = k - 12 if k > 6 else k
    return k, max(sims), sims[0]
