"""
Синтетика для проверки качества: «певец» поёт фразы слогами (согласная-шум + гласная
с формантами и вибрато), dub — тот же текст другим голосом, но с известным искажением
времени: сдвиг начала, глобальный темп и кусочно-случайные локальные отклонения.
В эталон подмешиваются реверб и «остатки инструментов» (как после сепаратора).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as ss

SR = 44100
VOWELS = {  # форманты F1, F2, F3 (Гц)
    "a": (800, 1200, 2600), "e": (500, 1800, 2600), "i": (300, 2300, 3000),
    "o": (500, 900, 2500), "u": (330, 800, 2300),
}
SCALE = [0, 2, 4, 5, 7, 9, 11, 12]


@dataclass
class Event:
    t: float          # начало согласной
    cons: float       # длина согласной
    dur: float        # длина гласной
    midi: float
    vowel: str
    cons_band: float  # центр полосы шума согласной


@dataclass
class Warp:
    offset: float
    tempo: float
    knots_t: np.ndarray   # время эталона
    knots_d: np.ndarray   # время dub

    def __call__(self, t):
        return np.interp(t, self.knots_t, self.knots_d)


def make_score(duration: float, rng: np.random.Generator, start: float = 1.0) -> list[Event]:
    ev = []
    t = start
    root = 57 + rng.integers(0, 5)
    while t < duration - 2.0:
        nsyl = int(rng.integers(4, 9))
        for _ in range(nsyl):
            cons = float(rng.uniform(0.025, 0.07)) if rng.random() < 0.75 else 0.0
            dur = float(rng.uniform(0.12, 0.45))
            midi = root + SCALE[int(rng.integers(0, len(SCALE)))]
            ev.append(Event(t, cons, dur, midi, str(rng.choice(list(VOWELS))),
                            float(rng.uniform(2500, 6000))))
            t += cons + dur + float(rng.uniform(0.0, 0.06))
            if t > duration - 2.0:
                break
        t += float(rng.uniform(0.35, 1.2))
    return ev


def make_warp(duration: float, rng: np.random.Generator, offset: float, tempo: float,
              local: tuple[float, float] = (0.05, 0.10), seg: tuple[float, float] = (1.5, 4.0)) -> Warp:
    ts, ds = [0.0], [offset]
    t = 0.0
    while t < duration + 10:
        L = float(rng.uniform(*seg))
        r = 1.0 + float(rng.uniform(*local)) * (1 if rng.random() < 0.5 else -1) if local[1] > 0 else 1.0
        t += L
        ts.append(t)
        ds.append(ds[-1] + L * tempo * r)
    return Warp(offset, tempo, np.array(ts), np.array(ds))


def _voice(ev: list[Event], total: float, rng: np.random.Generator, formant_k: float = 1.0,
           vib_rate: float = 5.5, bright: float = 1.0, breath: float = 0.02, detune_c: float = 0.0,
           time_fn=None) -> np.ndarray:
    n = int(total * SR)
    y = np.zeros(n)
    tf = time_fn or (lambda x: x)
    for e in ev:
        t0 = float(tf(e.t))
        tc = float(tf(e.t + e.cons))
        t1 = float(tf(e.t + e.cons + e.dur))
        # согласная — полосовой шум
        if e.cons > 0:
            a, b = int(t0 * SR), int(tc * SR)
            if b > a + 16 and b < n:
                noise = rng.standard_normal(b - a)
                lo, hi = e.cons_band * 0.7, min(e.cons_band * 1.3, SR / 2 - 100)
                sos = ss.butter(2, [lo, hi], btype="bandpass", fs=SR, output="sos")
                noise = ss.sosfilt(sos, noise)
                env = np.hanning(b - a)
                y[a:b] += 0.25 * noise * env / (np.std(noise) + 1e-9) * 0.3
        a, b = int(tc * SR), int(t1 * SR)
        if b <= a + 32 or b >= n:
            continue
        m = b - a
        tt = np.arange(m) / SR
        f0 = 440.0 * 2 ** ((e.midi - 69 + detune_c / 100.0) / 12.0)
        vib = np.clip((tt - 0.15) / 0.1, 0, 1) * 0.25 * np.sin(2 * np.pi * vib_rate * tt
                                                                + rng.uniform(0, 6.28))
        glide = -0.3 * np.exp(-tt / 0.03)                       # подъезд к ноте
        f = f0 * 2 ** ((vib + glide) / 12.0)
        ph = 2 * np.pi * np.cumsum(f) / SR
        F = np.array(VOWELS[e.vowel]) * formant_k
        sig = np.zeros(m)
        for h in range(1, 40):
            fh = f0 * h
            if fh > 7000:
                break
            g = sum(np.exp(-0.5 * ((fh - Fi) / (60 + 0.06 * Fi)) ** 2) * (1.0 / (k + 1))
                    for k, Fi in enumerate(F)) + 0.02 * bright / h
            sig += g * np.sin(h * ph)
        att = np.clip(tt / 0.015, 0, 1)
        rel = np.clip((m / SR - tt) / 0.04, 0, 1)
        env = att * rel * (0.8 + 0.2 * np.sin(2 * np.pi * 0.7 * tt))
        sig = sig / (np.max(np.abs(sig)) + 1e-9) * env
        sig += breath * rng.standard_normal(m) * env
        y[a:b] += 0.5 * sig
    return y


def _contaminate(y: np.ndarray, rng: np.random.Generator, reverb: float = 0.25,
                 residue_db: float = -26.0) -> np.ndarray:
    L = int(1.2 * SR)
    ir = rng.standard_normal(L) * np.exp(-np.arange(L) / (0.25 * SR))
    ir[0] = 0
    wet = ss.fftconvolve(y, ir)[: len(y)]
    wet *= reverb * np.std(y) / (np.std(wet) + 1e-9)
    out = y + wet
    # остатки инструментов: аккорды-пэды и «бочка»
    n = len(y)
    pad = np.zeros(n)
    t = np.arange(n) / SR
    for k in range(0, int(n / SR), 2):
        root = 110 * 2 ** (rng.integers(0, 7) / 12)
        a, b = k * SR, min(n, (k + 2) * SR)
        for r in (1, 1.25, 1.5, 2):
            pad[a:b] += np.sin(2 * np.pi * root * r * t[a:b])
    kick = np.zeros(n)
    for k in np.arange(0.0, n / SR, 0.5):
        a = int(k * SR)
        m = min(n - a, int(0.15 * SR))
        tt = np.arange(m) / SR
        kick[a:a + m] += np.sin(2 * np.pi * 60 * tt) * np.exp(-tt / 0.04)
    res = pad / (np.std(pad) + 1e-9) + 0.7 * kick / (np.std(kick) + 1e-9)
    out += res * np.std(y) * 10 ** (residue_db / 20)
    return out


@dataclass
class Pair:
    guide: np.ndarray        # (1, n)
    dub: np.ndarray          # (2, m) — стерео, чтобы проверить все каналы
    warp: Warp
    events: list
    sr: int = SR

    def voiced(self, t: np.ndarray) -> np.ndarray:
        m = np.zeros(len(t), bool)
        for e in self.events:
            m |= (t >= e.t) & (t < e.t + e.cons + e.dur)
        return m


def make_pair(duration: float = 60.0, seed: int = 0, offset: float = 1.7, tempo: float = 1.02,
              local=(0.05, 0.10), contaminate: bool = True) -> Pair:
    rng = np.random.default_rng(seed)
    ev = make_score(duration, rng)
    w = make_warp(duration, rng, offset, tempo, local)
    g = _voice(ev, duration, rng)
    if contaminate:
        g = _contaminate(g, rng)
    d_total = float(w(duration)) + 2.0
    d = _voice(ev, d_total, np.random.default_rng(seed + 1000), formant_k=1.07, vib_rate=5.0,
               bright=1.6, breath=0.04, detune_c=8.0, time_fn=w)
    d2 = np.vstack([d, 0.8 * d + 0.002 * rng.standard_normal(len(d))])
    g = g / np.max(np.abs(g)) * 0.8
    d2 = d2 / np.max(np.abs(d2)) * 0.8
    return Pair(g[None, :].astype(np.float32), d2.astype(np.float32), w, ev)
