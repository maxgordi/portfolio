"""Генератор синтетической «записи пения» для проверки пайплайна (не настоящий голос!).

Поёт гласными формантный синтезатор: пила + вибрато + резонансы. Подачи:
normal / power (громче, сильнее полоса ~3 кГц) / breathy (тише, больше шума).
Использование: python scripts/make_samples.py --out samples --songs 7 --minutes 2.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
KEYS = ["Am", "C", "Dm", "G", "Em", "F", "Bm", "D"]
SCALE = [0, 2, 3, 5, 7, 8, 10]  # минор натуральный
VOWELS = {"a": (700, 1200, 2600), "o": (500, 900, 2500), "e": (500, 1800, 2600), "i": (300, 2200, 3000)}


def resonator(x, f, bw=90):
    r = np.exp(-np.pi * bw / SR)
    th = 2 * np.pi * f / SR
    return signal.lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)


def note(rng, midi, dur, kind):
    f = 440 * 2 ** ((midi - 69) / 12)
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib_rate = rng.uniform(5.0, 6.5)
    vib_ext = rng.uniform(15, 45) * np.minimum(1, t / 0.3)  # центы
    fi = f * 2 ** (vib_ext * np.sin(2 * np.pi * vib_rate * t) / 1200)
    ph = 2 * np.pi * np.cumsum(fi) / SR
    nh = int(min(30, 9000 / f))
    src = sum(np.sin(k * ph) / k ** 1.1 for k in range(1, nh + 1))
    vowel = VOWELS[rng.choice(list(VOWELS))]
    y = sum(resonator(src, fm, bw) * g for fm, bw, g in zip(vowel, (80, 100, 140), (1.0, 0.6, 0.3)))
    if kind == "power":
        y = y + 0.6 * resonator(src, 3000, 250)
        gain, noise = 0.30, 0.01
    elif kind == "breathy":
        gain, noise = 0.12, 0.12
    else:
        gain, noise = 0.20, 0.03
    y = y / (np.max(np.abs(y)) + 1e-9) * gain
    nz = resonator(rng.standard_normal(n), 3500, 2500)
    nz = nz / (np.max(np.abs(nz)) + 1e-9) * gain
    y = (1 - noise * 3) * y + noise * 3 * nz
    env = np.minimum(1, np.minimum(t / 0.03, (dur - t) / 0.08))
    return (y * np.clip(env, 0, 1)).astype(np.float32)


def song(rng, minutes, lo, hi, key_root, floor_db=-62):
    total = int(minutes * 60 * SR)
    out = np.zeros(total + SR * 2, np.float32)
    pos = int(0.8 * SR)
    degs = [m for m in range(lo, hi + 1) if (m - key_root) % 12 in SCALE]
    while pos < total - 6 * SR:
        kind = rng.choice(["normal", "power", "breathy"], p=[0.62, 0.2, 0.18])
        cur = rng.integers(len(degs))
        phrase = []
        for _ in range(rng.integers(4, 9)):
            cur = int(np.clip(cur + rng.integers(-3, 4), 0, len(degs) - 1))
            phrase.append(note(rng, degs[cur], rng.uniform(0.35, 0.9), kind))
        if rng.random() < 0.25:  # длинная нота
            phrase.append(note(rng, degs[cur], rng.uniform(1.5, 2.2), kind))
        y = np.concatenate(phrase)
        if pos + len(y) > len(out):
            break
        out[pos:pos + len(y)] += y
        pos += len(y) + int(rng.uniform(0.7, 1.3) * SR)
    noise = rng.standard_normal(len(out)).astype(np.float32) * 10 ** (floor_db / 20)
    return (out + noise)[:total + SR]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    ap.add_argument("--songs", type=int, default=7)
    ap.add_argument("--minutes", type=float, default=2.1)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    names = ["zvezda", "doroga", "ogon", "reka", "veter", "zima", "more", "ptitsa"]
    # диапазон: C3..G4 в основном, у разных песен свои «хвосты»
    ranges = [(48, 67), (50, 69), (45, 64), (52, 71), (47, 66), (43, 62), (50, 72), (46, 65)]
    for i in range(a.songs):
        lo, hi = ranges[i % len(ranges)]
        key = KEYS[i % len(KEYS)]
        root = {"A": 9, "C": 0, "D": 2, "G": 7, "E": 4, "F": 5, "B": 11}[key[0]]
        x = song(rng, a.minutes, lo, hi, root)
        x = x / np.max(np.abs(x)) * 0.6
        p = out / f"{i + 1:02d}_{names[i % len(names)]}_{key}.flac"
        sf.write(str(p), x, SR, subtype="PCM_16")
        print("записано", p, f"{len(x) / SR:.0f} с")


if __name__ == "__main__":
    main()
