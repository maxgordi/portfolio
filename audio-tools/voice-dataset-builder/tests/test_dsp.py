"""M1/M2: высота, нарезка, брак, подача — на синтетическом звуке с известным ответом."""
import numpy as np
import pandas as pd
from scipy import signal

from voiceset import delivery as dlv
from voiceset import loudness as ld
from voiceset import quality as qt
from voiceset.config import load_config
from voiceset.pitch import track_f0
from voiceset.segment import segment

from conftest import SR, tone


def _gliss(kind, f0=100.0, f1=800.0, dur=4.0):
    t = np.arange(int(dur * SR)) / SR
    f = f0 * (f1 / f0) ** (t / dur)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) if kind == "sine" else signal.sawtooth(ph) * 0.8
    return (0.3 * y).astype(np.float32), f


def test_pitch_gliss_error_under_30_cents(cfg):
    for kind in ("sine", "saw"):
        x, f = _gliss(kind)
        est = track_f0(x, SR, cfg.pitch)
        hop = int(SR * cfg.pitch.hop_ms / 1000)
        n = min(len(est), len(f) // hop)
        true = f[::hop][:n]
        e = est[:n]
        ok = np.isfinite(e)
        assert ok.mean() > 0.9, kind
        cents = np.abs(1200 * np.log2(e[ok] / true[ok]))
        assert np.median(cents) < 30, (kind, np.median(cents))


def test_segmentation_exact_count_and_boundaries():
    cfg = load_config(overrides={"segment": {"vad": "energy", "pad_ms": 0}}).segment
    lens, gap = [2.0, 3.5, 1.8], 0.6
    parts, truth, t = [np.zeros(int(0.5 * SR), np.float32)], [], 0.5
    for L in lens:
        parts += [tone(220, L, 0.3), np.zeros(int(gap * SR), np.float32)]
        truth.append((t, t + L))
        t += L + gap
    x = np.concatenate(parts) + np.random.default_rng(0).standard_normal(sum(map(len, parts))).astype(np.float32) * 1e-4
    segs = segment(x, SR, cfg)
    assert len(segs) == 3
    for (s, e), (ts, te) in zip(segs, truth):
        assert abs(s - ts) <= 0.02 and abs(e - te) <= 0.02, (s, e, ts, te)


def test_segmentation_merges_short_gaps_and_splits_long():
    cfg = load_config(overrides={"segment": {"vad": "energy", "pad_ms": 0}}).segment
    x = np.concatenate([tone(220, 2, 0.3), np.zeros(int(0.15 * SR), np.float32), tone(220, 2, 0.3)])
    assert len(segment(x, SR, cfg)) == 1          # пауза 150 мс < 250 мс склеена
    long = np.concatenate([tone(220, 6, 0.3), np.zeros(int(0.3 * SR), np.float32), tone(220, 6, 0.3)])
    long_nogap = long.copy()
    long_nogap[int(6 * SR):int(6.3 * SR)] = tone(220, 0.3, 0.003)  # провал энергии, но не тишина
    segs = segment(long_nogap, SR, cfg)
    assert len(segs) >= 2 and all(e - s <= cfg.max_s + 1e-6 for s, e in segs)
    assert abs(segs[0][1] - 6.15) < 0.4


def test_clip_and_click_detection(cfg):
    q = cfg.quality
    clean = tone(220, 2.0, 0.3)
    assert not qt.clip_run(clean, q.clip_level, q.clip_run)
    assert qt.count_clicks(clean, q.click_k, q.click_min_abs) == 0
    clipped = clean.copy()
    clipped[10000:10004] = 1.0
    assert qt.clip_run(clipped, q.clip_level, q.clip_run)
    two = clean.copy()
    two[10000:10002] = 1.0                      # всего 2 подряд — ещё не перегруз
    assert not qt.clip_run(two, q.clip_level, q.clip_run)
    clicked = clean.copy()
    clicked[30000] += 0.7
    clicked[60000] -= 0.7
    assert qt.count_clicks(clicked, q.click_k, q.click_min_abs) == 2


def test_delivery_more_noise_means_breathy(cfg):
    rng = np.random.default_rng(3)
    rows = []
    for k in range(24):
        breathy = k % 2 == 1
        f = float(rng.uniform(150, 330))
        x = tone(f, 2.0, 0.3 if not breathy else 0.12)
        noise = rng.standard_normal(len(x)).astype(np.float32)
        noise = signal.sosfilt(signal.butter(2, 2000, "hp", fs=SR, output="sos"), noise)
        x = x + noise * (0.004 if not breathy else 0.04)
        r = dict(breathy=breathy, lufs=ld.lufs(x, SR), voiced_frac=1.0)
        r.update(dlv.spectral_features(x, SR))
        r.update(dlv.voice_quality(x, SR))
        rows.append(r)
    df = pd.DataFrame(rows)
    out = dlv.classify_rules(df, cfg.delivery)
    assert (out[out.breathy]["delivery"] == "breathy").mean() >= 0.8
    assert (out[~out.breathy]["delivery"] == "breathy").mean() <= 0.1
    assert out[out.breathy]["hnr"].mean() < out[~out.breathy]["hnr"].mean()


def test_snr_and_reverb(cfg):
    n = int(2 * SR)
    sig = tone(220, 2.0, 0.3)
    noise = np.random.default_rng(1).standard_normal(n).astype(np.float32) * 0.1
    f0 = np.full(200, 220.0)
    floor = 10 * np.log10(np.mean(noise ** 2))
    assert qt.snr_db(sig + noise, SR, f0, floor) < 15            # шумно → ниже порога 25
    quiet = qt.snr_db(sig, SR, f0, -70.0)
    assert quiet > cfg.quality.snr_min_db
    # хвост: после конца фразы звук затухает медленно
    tail = np.concatenate([sig, sig[-4800:].mean() * 0 + tone(220, 0.5, 0.2) * np.exp(-np.arange(int(0.5 * SR)) / (0.4 * SR))])
    db, ok = qt.reverb_tail_db(np.concatenate([tail, np.zeros(SR, np.float32)]), SR, 2.0, None, -70.0, 6)
    assert ok and db > cfg.quality.reverb_tail_db
    dry = np.concatenate([sig, np.zeros(SR, np.float32)])
    db, ok = qt.reverb_tail_db(dry, SR, 2.0, None, -70.0, 6)
    assert not ok
