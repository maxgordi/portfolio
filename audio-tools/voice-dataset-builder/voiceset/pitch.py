"""Высота f0: RMVPE (нужны веса) → torchcrepe → librosa.pyin. Шаг 10 мс, медианный фильтр."""
from __future__ import annotations

import logging

import numpy as np
from scipy import signal

from .audio_io import resample
from .config import PitchCfg
from .notes import hz_to_midi

log = logging.getLogger("voiceset")
SR_P = 16000


def _hop(cfg: PitchCfg) -> int:
    return int(round(SR_P * cfg.hop_ms / 1000))


def _pyin(x16, cfg):
    import librosa
    f0, _, _ = librosa.pyin(x16, fmin=cfg.fmin, fmax=cfg.fmax, sr=SR_P, frame_length=1024,
                            hop_length=_hop(cfg), resolution=0.25, fill_na=np.nan)
    return f0


def _viterbi_nodither(logits):
    """viterbi из torchcrepe, но без случайного дитеринга частоты (иначе результат разный от запуска к запуску)."""
    import torchcrepe
    bins, _ = torchcrepe.decode.viterbi(logits)
    cents = torchcrepe.CENTS_PER_BIN * bins + 1997.3794084376191
    return bins, torchcrepe.convert.cents_to_frequency(cents)


def _torchcrepe(x16, cfg):
    import torch
    import torchcrepe
    torch.set_num_threads(max(1, min(8, __import__("os").cpu_count() or 1)))
    a = torch.from_numpy(x16.astype(np.float32))[None]
    f0, per = torchcrepe.predict(a, SR_P, _hop(cfg), cfg.fmin, cfg.fmax, model=cfg.crepe_model,
                                 batch_size=128, device="cpu", return_periodicity=True,
                                 decoder=_viterbi_nodither)
    per = torchcrepe.filter.median(per, 3)
    f0 = torchcrepe.threshold.At(0.21)(f0, per)
    f0 = f0[0].numpy().astype(float)
    f0[~np.isfinite(f0) | (f0 <= 0)] = np.nan
    return f0


def _rmvpe(x16, cfg):
    raise NotImplementedError("RMVPE пока не реализован в voiceset — используется следующий движок")


ENGINES = {"rmvpe": _rmvpe, "torchcrepe": _torchcrepe, "pyin": _pyin}
_used: set[str] = set()


def median_midi(f0: np.ndarray, k: int) -> np.ndarray:
    """Медианный фильтр по озвученным отрезкам в области полутонов (против октавных выбросов)."""
    if k <= 1:
        return f0
    out = f0.copy()
    v = np.isfinite(f0)
    m = np.full(len(f0), np.nan)
    m[v] = hz_to_midi(f0[v])
    i = 0
    n = len(f0)
    while i < n:
        if not v[i]:
            i += 1
            continue
        j = i
        while j < n and v[j]:
            j += 1
        seg = m[i:j]
        if len(seg) >= k:
            seg = signal.medfilt(seg, k)
            out[i:j] = 440.0 * 2 ** ((seg - 69) / 12)
        i = j
    return out


def track_f0(x: np.ndarray, sr: int, cfg: PitchCfg) -> np.ndarray:
    """f0 в Гц с шагом hop_ms, NaN = неозвучено."""
    x16 = resample(x, sr, SR_P)
    last_err = None
    for name in cfg.engine:
        fn = ENGINES.get(name)
        if fn is None:
            continue
        try:
            f0 = fn(x16, cfg)
            if name not in _used:
                log.info("Движок высоты: %s", name)
                _used.add(name)
            return median_midi(f0, cfg.median_frames)
        except Exception as e:  # noqa: BLE001
            if name not in _used:
                log.info("Движок высоты %s недоступен (%s)", name, e)
                _used.add(name)
            last_err = e
    raise RuntimeError(f"Ни один движок высоты не сработал: {last_err}")


def phrase_pitch_features(f0: np.ndarray, cfg: PitchCfg) -> dict:
    hop_s = cfg.hop_ms / 1000
    v = np.isfinite(f0)
    res: dict = {"voiced_frac": float(v.mean()) if len(f0) else 0.0}
    if v.sum() < 5:
        res.update(f0_p5=np.nan, f0_p50=np.nan, f0_p95=np.nan, sem_midi=[], sem_ms=[],
                   vib_rate_hz=np.nan, vib_extent_cents=np.nan, breaks=0, break_rate=0.0)
        return res
    hz = f0[v]
    res["f0_p5"], res["f0_p50"], res["f0_p95"] = (float(np.percentile(hz, p)) for p in (5, 50, 95))
    midi = np.round(hz_to_midi(hz)).astype(int)
    ms = {}
    for m in midi:
        ms[int(m)] = ms.get(int(m), 0) + cfg.hop_ms
    keep = sorted(m for m, t in ms.items() if t >= cfg.note_min_ms)
    res["sem_midi"] = keep
    res["sem_ms"] = [int(ms[m]) for m in keep]
    res.update(_vibrato(f0, cfg))
    res.update(_breaks(f0, cfg, hop_s))
    return res


def _runs(mask: np.ndarray):
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def _vibrato(f0, cfg: PitchCfg) -> dict:
    fs = 1000 / cfg.hop_ms
    best = None
    for a, b in _runs(np.isfinite(f0)):
        if best is None or b - a > best[1] - best[0]:
            best = (a, b)
    if best is None or (best[1] - best[0]) / fs < 0.5:
        return {"vib_rate_hz": np.nan, "vib_extent_cents": np.nan}
    seg = f0[best[0]:best[1]]
    cents = 1200 * np.log2(seg / np.median(seg))
    lo, hi = cfg.vibrato_band_hz
    sos = signal.butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos")
    padlen = min(len(cents) - 1, 3 * 12)
    band = signal.sosfiltfilt(sos, cents, padlen=padlen)
    n = 1 << int(np.ceil(np.log2(len(band) * 8)))
    spec = np.abs(np.fft.rfft(band * np.hanning(len(band)), n))
    fr = np.fft.rfftfreq(n, 1 / fs)
    sel = (fr >= lo) & (fr <= hi)
    rate = float(fr[sel][np.argmax(spec[sel])])
    extent = float(2 * np.sqrt(2) * np.sqrt(np.mean(band ** 2)))
    return {"vib_rate_hz": rate, "vib_extent_cents": extent}


def _breaks(f0, cfg: PitchCfg, hop_s: float) -> dict:
    """Срывы: резкие скачки f0 и провалы озвучки внутри одной ноты."""
    v = np.isfinite(f0)
    cents = np.full(len(f0), np.nan)
    cents[v] = 1200 * np.log2(f0[v] / 440.0)
    jumps = 0
    for a, b in _runs(v):
        if b - a > 1:
            jumps += int(np.sum(np.abs(np.diff(cents[a:b])) > cfg.jump_cents))
    drops = 0
    lo_ms, hi_ms = cfg.dropout_ms
    runs = list(_runs(v))
    for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
        gap = (a2 - b1) * cfg.hop_ms
        if lo_ms <= gap <= hi_ms and (b1 - a1) * cfg.hop_ms >= 100 and (a2 + 0) < len(f0):
            if abs(cents[a2] - cents[b1 - 1]) < 100 and (b2 - a2) * cfg.hop_ms >= 100:
                drops += 1
    dur = max(len(f0) * hop_s, 1e-3)
    n = jumps + drops
    return {"breaks": int(n), "break_rate": float(n / dur)}
