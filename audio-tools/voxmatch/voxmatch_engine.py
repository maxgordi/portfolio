"""
VoxMatch — ядро обработки.

Подгоняет дубль вокала под эталон по таймингу и питчу:
  1. анализ обеих дорожек вокодером WORLD (высота, огибающая, шумовая часть);
  2. выравнивание по времени алгоритмом DTW по признакам звука (MFCC);
  3. перенос высоты эталона на дубль с сохранением тембра дубля;
  4. обратный синтез и запись WAV.

Можно запускать из командной строки:
  python voxmatch_engine.py эталон.wav дубль1.wav [дубль2.wav ...] --timing 100 --pitch 100
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
import numpy as np
import soundfile as sf
import librosa
import pyworld as pw
import scipy.signal as ss
from numba import njit

from voxmatch_world import analyze_world, shutdown as shutdown_workers  # noqa: F401
import voxmatch_tools as vt

FRAME_MS = 5.0                      # шаг анализа, мс
CACHE_SIZE = 8                      # сколько проанализированных файлов держать в памяти
AUDIO_EXT = {".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}

Progress = Optional[Callable[[str, float], None]]


# ----------------------------------------------------------------------------
# Настройки и результаты
# ----------------------------------------------------------------------------
@dataclass
class Settings:
    timing: float = 1.0             # 0..1 — насколько подтягивать тайминг к эталону
    pitch: float = 1.0              # 0..1 — насколько подтягивать питч к эталону
    keep_vibrato: bool = True       # брать у эталона ноту, а вибрато оставлять своё
    pitch_smooth_ms: float = 30.0   # плавность коррекции питча
    max_correction_st: float = 3.0  # больше этого — считаем другой нотой и не трогаем
    # --- питч подробно
    pitch_tol_c: float = 0.0        # допуск: отклонения меньше — не исправлять, центы
    pitch_attack_ms: float = 0.0    # начало ноты: коррекция нарастает за это время (живой «подъезд»)
    pitch_release_ms: float = 0.0   # конец ноты: коррекция спадает за это время
    vibrato_amount: float = 1.0     # сила своего вибрато: 0 — ровно, 1 — как есть, 2 — вдвое ярче
    detune_c: float = 0.0           # общий сдвиг высоты, центы
    formant_st: float = 0.0         # сдвиг формант (тембр) без смены высоты, полутоны
    max_shift_ms: float = 600.0     # максимальный сдвиг по времени
    same_start: bool = True         # файлы вырезаны с одной позиции в Cubase
    # --- тайминг подробно
    timing_mode: str = "syllables"  # "syllables" — по слогам; "phrases" — фразы целиком (только сдвиг)
    timing_tol_ms: float = 0.0      # допуск: ошибки меньше — не исправлять
    timing_smooth_ms: float = 0.0   # сглаживание карты времени
    timing_flex: float = 50.0       # 0..100 — гибкость выравнивания (смелость DTW)
    max_stretch: float = 0.0        # предел растяжения/сжатия, раз (1.1..4); 0 — без ограничения
    timing_offset_ms: float = 0.0   # сдвиг результата: + позже, − раньше
    keep_endings: bool = False      # не растягивать окончания фраз
    ending_ms: float = 150.0        # длина окончания фразы
    quality: str = "accurate"       # "accurate" (Harvest) или "fast" (DIO)
    protect_breath: bool = True     # вдохи не обрабатывать: вставлять исходный звук
    breath_sens: float = 50.0       # 0..100 — чувствительность поиска вдохов
    breath_gain: float = 1.0        # 0..1.5 — громкость вдохов (0 — убрать)
    breath_mode: str = "original"   # "original" — на своём месте как в дубле; "phrase" — к началу фразы
    breath_min_ms: float = 90.0     # короче — не вдох
    breath_pre_ms: float = 10.0     # запас перед вдохом
    breath_post_ms: float = 10.0    # запас после вдоха
    tail_ms: float = 120.0          # защита окончаний фраз: столько после ноты вдохом не считается
    breath_xfade_ms: float = 10.0   # плавность стыков вдоха
    breath_no_clean: bool = True    # очистка (деэссер, гейт, громкость фраз, щелчки) вдохи не трогает
    dyn_match: bool = False         # переносить динамику (компрессию) эталона
    dyn_amount: float = 1.0         # 0..1 — сила переноса динамики
    dyn_speed_ms: float = 50.0      # скорость (как атака/релиз): меньше — быстрее
    dyn_max_db: float = 12.0        # максимальное изменение громкости, дБ
    clean_clicks: bool = True       # убирать щелчки губ
    clean_deess: bool = True        # деэссер
    deess_db: float = 5.0
    clean_level: bool = True        # ровная громкость фраз
    level_strength: float = 0.7
    clean_gate: bool = False        # тише в паузах (по умолчанию выкл — воздух естественный)
    gate_db: float = 50.0
    out_dir: str = ""               # пусто — папка VoxMatch рядом с дублем
    subtype: str = "PCM_24"         # PCM_24 или FLOAT
    suffix: str = "_voxmatch"


@dataclass
class Analysis:
    key: tuple
    path: str
    sr: int
    hop: int
    frame_ms: float
    x: np.ndarray
    f0: np.ndarray
    sp: np.ndarray
    ap: np.ndarray
    feats: np.ndarray
    loud: np.ndarray
    channels: int

    @property
    def n(self) -> int:
        return len(self.f0)

    @property
    def duration(self) -> float:
        return len(self.x) / self.sr


@dataclass
class Result:
    dub_path: str
    out_path: str
    sr: int
    y: np.ndarray
    frame_ms: float
    f0_ref: np.ndarray
    f0_dub: np.ndarray          # исходный питч дубля на его собственной шкале времени
    f0_out: np.ndarray          # питч результата на шкале эталона
    timing_before_ms: float
    timing_after_ms: float
    pitch_before_c: float
    pitch_after_c: float
    lag_ms: float
    octave_shift: int
    seconds: float
    dyn_before_db: float = float("nan")   # среднее отличие громкости от эталона, дБ
    dyn_after_db: float = float("nan")
    breaths: list = field(default_factory=list)   # вдохи в результате: (начало, конец), с
    y_pre: object = None            # результат до очистки (для компа)
    pos: object = None              # карта времени: кадр результата → кадр дубля
    score: object = None            # vt.TakeScore — оценка дубля по фразам
    clean_notes: list = field(default_factory=list)
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------
# Загрузка и анализ
# ----------------------------------------------------------------------------
def load_audio(path: str, target_sr: Optional[int] = None):
    """Читает файл, сводит в моно, при необходимости меняет частоту."""
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float64")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Не удалось открыть файл «{os.path.basename(path)}»: {exc}") from exc
    channels = data.shape[1]
    x = data.mean(axis=1)
    if target_sr and sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
        sr = target_sr
    if len(x) < sr * 0.3:
        raise RuntimeError(f"Файл «{os.path.basename(path)}» слишком короткий (меньше 0,3 с).")
    return np.ascontiguousarray(x, dtype=np.float64), int(sr), channels


def _hop(sr: int):
    hop = max(1, int(round(sr * FRAME_MS / 1000.0)))
    return hop, 1000.0 * hop / sr


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    """Подгоняет длину по последней оси (обрезка или повтор крайнего кадра)."""
    if a.shape[-1] == n:
        return a
    if a.shape[-1] > n:
        return a[..., :n]
    pad = [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])]
    return np.pad(a, pad, mode="edge")


def _features(x: np.ndarray, sr: int, hop: int, n: int):
    """Признаки для выравнивания: MFCC + их изменение + громкость."""
    y = x.astype(np.float32)
    n_fft = 2048 if sr > 32000 else 1024
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop,
                              n_mels=40, fmax=min(8000.0, sr / 2))[1:]
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    mf, rms = _fit(mf, n), _fit(rms, n)
    db = 20.0 * np.log10(rms + 1e-9)
    level = np.percentile(db, 95)
    loud = db > level - 35.0
    if loud.sum() < 10:
        loud = np.ones(n, dtype=bool)
    mu = mf[:, loud].mean(axis=1, keepdims=True)
    sd = mf[:, loud].std(axis=1, keepdims=True) + 1e-6
    mfn = (mf - mu) / sd
    delta = librosa.feature.delta(mfn, width=9, mode="nearest")
    energy = np.clip((db - level) / 10.0, -4.0, 1.0)
    feats = np.vstack([mfn, delta, 1.5 * energy[None, :]]).T
    return np.ascontiguousarray(feats, dtype=np.float32), loud


def analyze(path: str, quality: str = "accurate", target_sr: Optional[int] = None,
            need_env: bool = True, progress: Progress = None, key: tuple = ()) -> Analysis:
    """need_env=False — только высота и признаки (достаточно для эталона)."""
    name = os.path.basename(path)
    if progress:
        progress(f"Читаю {name}", 0.0)
    x, sr, ch = load_audio(path, target_sr)
    hop, fp = _hop(sr)

    if progress:
        progress(f"Анализирую голос: {name}", 0.05)
    f0, sp, ap = analyze_world(
        x, sr, hop, quality, need_env,
        progress=(lambda f: progress(f"Анализирую голос: {name}", 0.05 + 0.85 * f)) if progress else None)
    feats, loud = _features(x, sr, hop, len(f0))
    return Analysis(key=key, path=path, sr=sr, hop=hop, frame_ms=fp, x=x, f0=f0, sp=sp,
                    ap=ap, feats=feats, loud=loud, channels=ch)


# ----------------------------------------------------------------------------
# Выравнивание по времени
# ----------------------------------------------------------------------------
@njit(cache=True)
def _dtw_band(A, B, loud_a, loud_b, center, band, penalty):
    """DTW в полосе вокруг center[i]. Свободные начало и конец по дублю.
    Возвращает пары (i, j): кадр эталона i ↔ кадр дубля j."""
    n, m, dim = A.shape[0], B.shape[0], A.shape[1]
    w = 2 * band + 1
    inf = np.float32(1e30)
    D = np.full((n, w), inf, dtype=np.float32)
    S = np.zeros((n, w), dtype=np.int8)   # 0 диагональ, 1 сверху, 2 слева, 3 старт
    for i in range(n):
        j0 = center[i] - band
        for k in range(w):
            j = j0 + k
            if j < 0 or j >= m:
                continue
            if (not loud_a[i]) and (not loud_b[j]):
                c = 0.0
            else:
                s = 0.0
                for d in range(dim):
                    diff = A[i, d] - B[j, d]
                    s += diff * diff
                c = np.sqrt(s / dim)
            if i == 0:
                D[0, k] = c
                S[0, k] = 3
                continue
            best = inf
            step = 0
            pk = j - 1 - (center[i - 1] - band)          # (i-1, j-1)
            if 0 <= pk < w and D[i - 1, pk] < best:
                best = D[i - 1, pk]
                step = 0
            pk = j - (center[i - 1] - band)              # (i-1, j)
            if 0 <= pk < w and D[i - 1, pk] + penalty < best:
                best = D[i - 1, pk] + penalty
                step = 1
            if k > 0 and D[i, k - 1] + penalty < best:   # (i, j-1)
                best = D[i, k - 1] + penalty
                step = 2
            if best < inf:
                D[i, k] = best + c
                S[i, k] = step

    # свободный конец: лучшая клетка последней строки
    kbest = -1
    best = inf
    for k in range(w):
        if D[n - 1, k] < best:
            best = D[n - 1, k]
            kbest = k
    pi = np.empty(n + m + 2, dtype=np.int64)
    pj = np.empty(n + m + 2, dtype=np.int64)
    if kbest < 0:
        return pi[:0], pj[:0]
    i, k, cnt = n - 1, kbest, 0
    while True:
        j = center[i] - band + k
        pi[cnt] = i
        pj[cnt] = j
        cnt += 1
        st = S[i, k]
        if st == 3:
            break
        if st == 0:
            i -= 1
            k = j - 1 - (center[i] - band)
        elif st == 1:
            i -= 1
            k = j - (center[i] - band)
        else:
            k -= 1
    return pi[:cnt][::-1].copy(), pj[:cnt][::-1].copy()


def _global_lag(ref: Analysis, dub: Analysis, max_frames: int) -> int:
    """Общий сдвиг дубля относительно эталона (j ≈ i + lag), по огибающей громкости."""
    ea = ref.feats[:, -1].astype(np.float64)
    eb = dub.feats[:, -1].astype(np.float64)
    ea, eb = ea - ea.mean(), eb - eb.mean()
    c = ss.correlate(eb, ea, mode="full", method="fft")
    lags = ss.correlation_lags(len(eb), len(ea), mode="full")
    ok = np.abs(lags) <= max_frames
    return int(lags[ok][np.argmax(c[ok])])


def _limit_rate(d, loud, r, strength):
    """Внутри звучащих кусков скорость (1 + d') держится в [1/r, r]; что не успели
    догнать — равномерно распределяется по куску. В тишине — точно по карте."""
    if strength <= 0:
        return d
    lo, hi = (1.0 / r - 1.0) / strength, (r - 1.0) / strength
    n = len(d)
    out = d.copy()
    for a, b in _segments(loud):
        a0, b0 = max(0, a - 1), min(n, b + 1)
        tgt = d[a0:b0]
        if len(tgt) < 3:
            continue
        steps = np.clip(np.diff(tgt), lo, hi)
        path = tgt[0] + np.concatenate([[0.0], np.cumsum(steps)])
        resid = tgt[-1] - path[-1]
        path = path + np.linspace(0.0, 1.0, len(path)) * resid
        out[a0:b0] = path
    return out


def shape_timing(full_map, identity, s, fm: float, phrases, loud=None):
    """Из точной карты времени (DTW) делает ту, что нужна по настройкам тайминга.
    Возвращает позицию в дубле для каждого кадра результата."""
    n = len(full_map)
    d = full_map - identity                                   # насколько дубль «уехал», кадры
    if s.timing_mode == "phrases" and phrases:                # фразы целиком: один сдвиг на фразу
        dd = np.full(n, np.nan)
        for a, b in phrases:
            dd[a:b] = np.median(d[a:b])
        idx = np.arange(n)
        ok = np.isfinite(dd)
        d = np.interp(idx, idx[ok], dd[ok]) if ok.any() else d
    tol = s.timing_tol_ms / fm
    if tol > 0:                                               # мелкие ошибки оставить как спето
        d = d - np.clip(d, -tol, tol)
    k = int(round(s.timing_smooth_ms / fm))
    if k > 1:
        w = np.hanning(k + 2)[1:-1]
        d = np.convolve(np.pad(d, k, mode="edge"), w / w.sum(), mode="same")[k:-k]
    if s.keep_endings and phrases:                            # окончания фраз — не растягивать
        e = int(round(s.ending_ms / fm))
        for a, b in phrases:
            t0 = max(a, b - e)
            d[t0:b] = d[t0]
    r = float(s.max_stretch)                                  # предел растяжения/сжатия
    if r >= 1.05:
        d = _limit_rate(d, loud if loud is not None else np.ones(n, dtype=bool), r, s.timing)
    pos = identity + s.timing * d
    pos = pos - s.timing_offset_ms / fm                       # + позже, − раньше
    return np.maximum.accumulate(pos)


def _path_to_map(pi: np.ndarray, pj: np.ndarray, n: int) -> np.ndarray:
    """Путь DTW → позиция в дубле (в кадрах) для каждого кадра эталона."""
    sums = np.bincount(pi, weights=pj.astype(np.float64), minlength=n)
    cnt = np.bincount(pi, minlength=n)
    m = sums / np.maximum(cnt, 1)
    k = 5
    m = np.convolve(np.pad(m, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    return np.maximum.accumulate(m)


# ----------------------------------------------------------------------------
# Питч
# ----------------------------------------------------------------------------
def _segments(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _smooth_in_segments(v: np.ndarray, mask: np.ndarray, win: int) -> np.ndarray:
    out = v.copy()
    if win <= 1:
        return out
    half = win // 2
    for s, e in _segments(mask):
        seg = v[s:e]
        cs = np.concatenate([[0.0], np.cumsum(seg)])
        idx = np.arange(e - s)
        lo = np.maximum(idx - half, 0)
        hi = np.minimum(idx + half + 1, e - s)
        out[s:e] = (cs[hi] - cs[lo]) / (hi - lo)
    return out


def _cents(f0: np.ndarray) -> np.ndarray:
    out = np.zeros_like(f0)
    v = f0 > 0
    out[v] = 1200.0 * np.log2(f0[v] / 440.0) + 6900.0
    return out


def _pitch_correction(f0_ref, f0_dub, s: Settings, frame_ms: float):
    vr, vd = f0_ref > 0, f0_dub > 0
    cr, cd = _cents(f0_ref), _cents(f0_dub)
    win_v = int(round(140.0 / frame_ms))     # ~ один период вибрато
    cd_line = _smooth_in_segments(cd, vd, win_v)
    if s.keep_vibrato:
        cr = _smooth_in_segments(cr, vr, win_v)
        cd = cd_line
    both = vr & vd
    corr = np.zeros_like(cr)
    corr[both] = cr[both] - cd[both]

    octave = 0
    if both.sum() > 20:
        med = float(np.median(corr[both]))
        if abs(med) > 600:
            octave = int(np.round(med / 1200.0))
            corr[both] -= 1200.0 * octave
    valid = both & (np.abs(corr) <= s.max_correction_st * 100.0)

    raw = corr.copy()                        # для статистики
    if s.pitch_tol_c > 0:                    # допуск: мелкие отклонения оставить как спето
        corr = corr - np.clip(corr, -s.pitch_tol_c, s.pitch_tol_c)

    fill = np.zeros_like(corr)
    for a, b in _segments(vd):
        idx = np.arange(a, b)
        v = valid[a:b]
        if v.any():
            fill[a:b] = np.interp(idx, idx[v], corr[a:b][v])
    win = int(round(s.pitch_smooth_ms / frame_ms))
    fill = _smooth_in_segments(fill, vd, win)

    att = int(round(s.pitch_attack_ms / frame_ms))
    rel = int(round(s.pitch_release_ms / frame_ms))
    if att > 0 or rel > 0:                   # начало и конец ноты — коррекция плавно входит/уходит
        w = np.ones_like(fill)
        for a, b in _segments(vd):
            L = b - a
            if att > 0:
                k = min(att, L)
                w[a:a + k] = np.minimum(w[a:a + k], np.linspace(0.0, 1.0, k, endpoint=False))
            if rel > 0:
                k = min(rel, L)
                w[b - k:b] = np.minimum(w[b - k:b], np.linspace(1.0, 0.0, k, endpoint=False))
        fill = fill * w
    applied = fill * s.pitch

    extra = np.zeros_like(applied)
    if abs(s.vibrato_amount - 1.0) > 1e-3:   # своё вибрато: усилить или убрать
        vib = np.where(vd, cd_raw_minus_line(f0_dub, cd_line), 0.0)
        extra += (s.vibrato_amount - 1.0) * vib
    extra += s.detune_c
    f0_new = np.where(vd, f0_dub * np.power(2.0, (applied + extra) / 1200.0), 0.0)
    if valid.any():
        before = float(np.median(np.abs(raw[valid])))
        after = float(np.median(np.abs(raw[valid] - applied[valid])))
    else:
        before = after = float("nan")
    return f0_new, before, after, octave


def cd_raw_minus_line(f0, line):
    """Собственное вибрато в центах: отклонение высоты от сглаженной линии."""
    return np.where(f0 > 0, _cents(f0) - line, 0.0)


def shift_formants(sp: np.ndarray, st: float) -> np.ndarray:
    """Сдвиг формант (тембра) без смены высоты: спектральная огибающая растягивается
    по частоте. + светлее/«меньше», − темнее/«больше»."""
    if abs(st) < 1e-3:
        return sp
    ratio = 2.0 ** (st / 12.0)
    nb = sp.shape[1]
    src = np.clip(np.arange(nb) / ratio, 0, nb - 1)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, nb - 1)
    w = (src - i0)[None, :]
    lsp = np.log(np.maximum(sp, 1e-16))
    return np.ascontiguousarray(np.exp((1 - w) * lsp[:, i0] + w * lsp[:, i1]))


# ----------------------------------------------------------------------------
# Перенос параметров WORLD на новую шкалу времени
# ----------------------------------------------------------------------------
def _warp_params(dub: Analysis, pos: np.ndarray, outside: np.ndarray):
    """Пересчитывает параметры дубля на шкалу времени эталона (блоками, чтобы беречь память)."""
    m, n = dub.n, len(pos)
    p = np.clip(pos, 0, m - 1)
    i0 = np.floor(p).astype(np.int64)
    i1 = np.minimum(i0 + 1, m - 1)
    wf = p - i0
    sp = np.empty((n, dub.sp.shape[1]), dtype=np.float64)
    ap = np.empty((n, dub.ap.shape[1]), dtype=np.float64)
    tiny = np.float32(1e-16)
    for a in range(0, n, 2048):
        b = min(n, a + 2048)
        w = wf[a:b, None].astype(np.float32)
        la = np.log(np.maximum(dub.sp[i0[a:b]], tiny))
        lb = np.log(np.maximum(dub.sp[i1[a:b]], tiny))
        sp[a:b] = np.exp((1 - w) * la + w * lb)
        ap[a:b] = (1 - w) * dub.ap[i0[a:b]] + w * dub.ap[i1[a:b]]

    f0a, f0b = dub.f0[i0], dub.f0[i1]
    both = (f0a > 0) & (f0b > 0)
    f0 = np.where(wf < 0.5, f0a, f0b)
    f0[both] = np.exp((1 - wf[both]) * np.log(f0a[both]) + wf[both] * np.log(f0b[both]))

    if outside.any():           # за пределами дубля — тишина
        f0[outside] = 0.0
        sp[outside] = 1e-16
        ap[outside] = 1.0
    return np.ascontiguousarray(f0), sp, ap


# ----------------------------------------------------------------------------
# Дыхание: найти вдохи и не пропускать их через вокодер
# ----------------------------------------------------------------------------
def _segments_merge(mask: np.ndarray, gap: int):
    """Отрезки True; дырки короче gap кадров заполняются."""
    segs = _segments(mask)
    out = []
    for a, b in segs:
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def detect_breaths(an: Analysis, sens: float = 50.0, min_ms: float | None = None,
                   pre_ms: float = 10.0, post_ms: float = 10.0):
    """Вдохи в дорожке: шумовые (без высоты тона) участки от 90 мс, заметно тише пения,
    без резкого «свиста» шипящих. Возвращает список (начало, конец) в кадрах анализа."""
    n, hop, sr, fm = an.n, an.hop, an.sr, an.frame_ms
    x = an.x.astype(np.float32)
    n_fft = 2048 if sr > 32000 else 1024
    rms = _fit(librosa.feature.rms(y=x, frame_length=n_fft, hop_length=hop)[0], n)
    db = 20.0 * np.log10(rms + 1e-9)
    spec = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))
    cent = _fit(librosa.feature.spectral_centroid(S=spec, sr=sr)[0], n)
    if an.ap is not None:
        freqs = np.linspace(0, sr / 2, an.ap.shape[1])
        band = (freqs > 1000) & (freqs < 5000)
        apm = an.ap[:, band].mean(axis=1)
    else:
        apm = np.ones(n)
    peak = float(np.percentile(db[db > -150], 95)) if np.any(db > -150) else -20.0
    sens = float(np.clip(sens, 0, 100))
    level_max = peak - 34.0 + 0.24 * sens          # 0 → −34 дБ, 50 → −22 дБ, 100 → −10 дБ
    if min_ms is None:
        min_ms = 130.0 - 0.8 * sens                      # 0 → 130 мс, 50 → 90 мс, 100 → 50 мс
    min_frames = max(1, int(round(min_ms / fm)))

    noisy = (an.f0 <= 0) | (apm > 0.75)
    active = (db > peak - 75.0) & (db > -140.0)
    cand = noisy & active
    out = []
    for a, b in _segments_merge(cand, int(round(25.0 / fm))):
        if b - a < min_frames:
            continue
        if np.median(db[a:b]) > level_max:
            continue
        if np.median(cent[a:b]) > 5500.0:           # шипящие «с», «ш», «ц»
            continue
        a = max(0, a - int(round(pre_ms / fm)))
        b = min(n, b + int(round(post_ms / fm)))
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def voiced_mask(an: Analysis) -> np.ndarray:
    """Кадры, где уверенно звучит спетая нота. Только их пропускаем через вокодер.
    Всё остальное (вдохи, паузы, шумы, согласные) берётся из оригинала — так нет
    «жужжания» вокодера на границах, где он путает шум с нотой."""
    n, hop, sr, fm = an.n, an.hop, an.sr, an.frame_ms
    n_fft = 2048 if sr > 32000 else 1024
    rms = _fit(librosa.feature.rms(y=an.x.astype(np.float32), frame_length=n_fft,
                                   hop_length=hop)[0], n)
    db = 20.0 * np.log10(rms + 1e-9)
    peak = float(np.percentile(db[db > -150], 95)) if np.any(db > -150) else -20.0
    if an.ap is not None:
        freqs = np.linspace(0, sr / 2, an.ap.shape[1])
        band = (freqs > 1000) & (freqs < 5000)
        apm = an.ap[:, band].mean(axis=1)
    else:
        apm = np.zeros(n)
    v = (an.f0 > 0) & (apm < 0.6) & (db > peak - 40.0)
    # высота, которой у этого голоса не бывает (ниже/выше обычной больше чем вдвое) — не нота
    if v.sum() > 20:
        med = float(np.median(an.f0[v]))
        v &= (an.f0 > med / 2.1) & (an.f0 < med * 2.1)
    # короткие провалы внутри ноты — оставить нотой; короткие «ноты» в шуме — шумом
    gap, short = int(round(25.0 / fm)), int(round(60.0 / fm))
    for a, b in _segments(~v):
        if 0 < a and b < n and b - a < gap:
            v[a:b] = True
    for a, b in _segments(v):
        # слишком короткая или тихая «нота» — это не пение (край вдоха, шорох, призвук)
        if b - a < short or np.median(db[a:b]) < peak - 28.0:
            v[a:b] = False
    return v


def grow_breaths(breaths, voiced: np.ndarray, an: Analysis):
    """Вдох вместе с соседними обрывками «нот» (короче 150 мс) считаем одним вдохом —
    чтобы ни один его кусочек не попал в питч-коррекцию."""
    fm = an.frame_ms
    near, short = int(round(40.0 / fm)), int(round(150.0 / fm))
    segs = _segments(voiced)
    out = []
    for a, b in breaths:
        for va, vb in segs:
            if vb - va < short and (abs(va - b) <= near or abs(vb - a) <= near):
                a, b = min(a, va), max(b, vb)
        out.append((a, b))
    out.sort()
    merged = []
    for a, b in out:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def protect_tails(breaths, voiced: np.ndarray, an: Analysis, tail_ms: float, min_ms: float):
    """Защита окончаний фраз: первые tail_ms после конца ноты — это затухание голоса,
    а не вдох. Такой кусок остаётся на месте вместе с фразой."""
    if tail_ms <= 0:
        return breaths
    fm = an.frame_ms
    tail = int(round(tail_ms / fm))
    min_f = max(1, int(round(min_ms / fm)))
    ends = np.array([vb for _va, vb in _segments(voiced)])
    out = []
    for a, b in breaths:
        before = ends[(ends > a - tail) & (ends < b)] if len(ends) else ends
        if len(before):
            a = max(a, int(before.max()) + tail)
        if b - a >= min_f:
            out.append((a, b))
    return out


@njit(cache=True)
def _wsola_core(x, centers, N, Hs, tol, n_out):
    """WSOLA: каждый кусочек берётся рядом с нужным местом, но со сдвигом (до ±tol),
    при котором он лучше всего продолжает предыдущий по форме волны. Нет «жужжания»
    и фазовых артефактов простой склейки. При скорости 1 — точная копия."""
    half = N // 2
    nx = x.shape[0]
    win = np.empty(N)
    for i in range(N):
        win[i] = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / N)          # периодическое окно Ханна
    size = centers.shape[0] * Hs + N + 1                          # с запасом: без выхода за границы
    out = np.zeros(size)
    norm = np.zeros(size)
    prev = -10 ** 9
    for k in range(centers.shape[0]):
        want = centers[k]
        best = want
        if prev > -10 ** 8:
            nat = prev + Hs                                       # естественное продолжение
            if abs(want - nat) <= 1:
                best = nat
            else:
                bestv = -1e30
                step = 4
                d = -tol
                while d <= tol:
                    c = want + d
                    s1 = 0.0
                    e1 = 0.0
                    e2 = 0.0
                    for i in range(0, N, 2):
                        ia = c - half + i
                        ib = nat - half + i
                        va = x[ia] if 0 <= ia < nx else 0.0
                        vb = x[ib] if 0 <= ib < nx else 0.0
                        s1 += va * vb
                        e1 += va * va
                        e2 += vb * vb
                    v = s1 / np.sqrt(e1 * e2 + 1e-20)
                    if v > bestv:
                        bestv = v
                        best = c
                    d += step
                c0 = best                                         # уточнение ±4
                for d in range(-3, 4):
                    c = c0 + d
                    s1 = 0.0
                    e1 = 0.0
                    e2 = 0.0
                    for i in range(0, N, 2):
                        ia = c - half + i
                        ib = nat - half + i
                        va = x[ia] if 0 <= ia < nx else 0.0
                        vb = x[ib] if 0 <= ib < nx else 0.0
                        s1 += va * vb
                        e1 += va * va
                        e2 += vb * vb
                    v = s1 / np.sqrt(e1 * e2 + 1e-20)
                    if v > bestv:
                        bestv = v
                        best = c
        prev = best
        o = k * Hs
        for i in range(N):
            ix = best - half + i
            if 0 <= ix < nx:
                out[o + i] += x[ix] * win[i]
            norm[o + i] += win[i]
    res = np.zeros(n_out)
    for i in range(n_out):
        j = i + half
        if j < size and norm[j] > 1e-6:
            res[i] = out[j] / norm[j]
    return res


def _wsola_warp(x: np.ndarray, pos: np.ndarray, hop: int, n_samples: int) -> np.ndarray:
    """Оригинальный звук по карте времени pos (кадр результата → кадр дубля), WSOLA."""
    sr_frame = hop
    N = 1024 if hop >= 200 else 512                               # ~21 мс
    Hs = N // 4                                                   # 75% перекрытие
    n_frames = int(np.ceil(n_samples / Hs)) + 1
    t_out = np.arange(n_frames) * Hs / sr_frame                   # в кадрах анализа
    p = np.interp(t_out, np.arange(len(pos)), pos)
    centers = np.round(p * sr_frame).astype(np.int64)
    return _wsola_core(np.asarray(x, dtype=np.float64), centers, N, Hs, N // 4, n_samples)


def _ola_warp(x: np.ndarray, pos: np.ndarray, hop: int, n_samples: int) -> np.ndarray:
    """Оригинальный звук, разложенный по новой шкале времени: короткие кусочки (40 мс)
    берутся из того места дубля, куда указывает карта, и складываются с перекрытием.
    Для вдохов, шумов и пауз звучит как оригинал; при сдвиге без растяжения — один в один."""
    over = 8
    W = over * hop
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(W) / W)).astype(np.float32)   # периодическое окно Ханна
    norm = W / (2.0 * hop)
    out = np.zeros(n_samples + 2 * W, dtype=np.float32)
    xs = np.concatenate([np.zeros(W, np.float32), x.astype(np.float32), np.zeros(W, np.float32)])
    n = len(pos)
    offs = np.arange(W) - W // 2
    chunk = over * 512
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        centers = np.round(pos[c0:c1] * hop).astype(np.int64)
        valid = (centers >= -W // 2) & (centers < len(x) + W // 2)
        idx = np.clip(centers[:, None] + offs[None, :] + W, 0, len(xs) - 1)
        g = xs[idx] * win[None, :]
        g[~valid] = 0.0
        for k in range(over):                     # кадры с шагом over не перекрываются
            sel = g[k::over]
            if len(sel) == 0:
                continue
            start = (c0 + k) * hop - W // 2 + W
            flat = sel.reshape(-1)
            out[start:start + len(flat)] += flat[: max(0, len(out) - start)]
    return out[W: W + n_samples] / norm


def _soft(mask_frames: np.ndarray, hop: int, n_samples: int, ramp: int) -> np.ndarray:
    """Маска по кадрам → плавная маска по сэмплам (стыки ramp сэмплов)."""
    m = np.repeat(mask_frames.astype(np.float32), hop)
    m = _fit(m, n_samples) if len(m) else np.zeros(n_samples, np.float32)
    if ramp > 1:
        k = np.hanning(ramp).astype(np.float32)
        k /= k.sum()
        m = np.convolve(m, k, mode="same")
    return np.clip(m, 0.0, 1.0)


def place_breaths(pos, breaths, voiced: np.ndarray, mode: str, lagoff: int):
    """Вдохи без растяжения: на месте каждого вдоха карта времени делается «один к одному»
    (вдох звучит из оригинала как есть), а всё выравнивание уходит в соседние паузы.
    Ничего не вырезается и не вставляется — дыр и «съеденного» воздуха быть не может.
      mode="original" — вдох там же, где был в дубле (с учётом общего сдвига файлов);
      mode="phrase"   — вдох заканчивается ровно перед следующей (выровненной) фразой.
    Возвращает новую карту и места вдохов в результате (в кадрах)."""
    pos = np.asarray(pos, dtype=np.float64).copy()
    n = len(pos)
    nd = len(voiced)
    merged = []                                                  # вдохи из одной паузы — одним куском
    for a, b in sorted(breaths):
        if merged and not voiced[merged[-1][1]:a].any():
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    breaths = merged
    k = np.clip(np.round(pos).astype(np.int64), 0, nd - 1)
    note_out = voiced[k] & (pos > -0.5) & (pos < nd - 0.5)
    placed = []
    for a, b in breaths:
        L = b - a
        o0 = int(np.searchsorted(pos, a - 0.5))
        if o0 >= n or L < 2:
            continue
        before = np.flatnonzero(note_out[:o0])
        lb = int(before[-1]) + 1 if len(before) else 0           # пауза в результате: [lb, rb)
        after = np.flatnonzero(note_out[o0:])
        rb = o0 + int(after[0]) if len(after) else n
        if placed and lb < placed[-1][1]:
            lb = placed[-1][1]
        if rb - lb < 4:
            continue
        t0 = (a - lagoff) if mode == "original" else (rb - L)
        t0 = min(max(t0, lb), max(lb, rb - L))
        skip = max(0, L - (rb - t0))                            # места мало — берём конец вдоха
        a2 = a + skip
        t1 = t0 + (b - a2)
        if t1 - t0 < 4:
            continue
        left_val = pos[lb - 1] if lb > 0 else min(pos[0], a2)
        right_val = pos[rb] if rb < n else max(pos[-1], b)
        pos[t0:t1] = np.arange(a2, b, dtype=np.float64)[: t1 - t0]  # вдох — один к одному
        if t0 > lb:                                              # пауза до вдоха
            pos[lb:t0] = np.linspace(left_val, a2, t0 - lb + 1, endpoint=False)[1:] \
                if t0 - lb > 0 else pos[lb:t0]
        if rb > t1:                                              # пауза после вдоха
            pos[t1:rb] = np.linspace(b, right_val, rb - t1 + 1, endpoint=False)[: rb - t1]
        placed.append((t0, t1))
    return np.maximum.accumulate(pos), placed


def _enforce_slope(pos, anchors, lo=0.5, hi=2.5):
    """Карта времени без «заморозок»: скорость (шаг по дублю на кадр результата) держится
    в [lo, hi]. Застывшая карта заставляет брать один и тот же кусочек звука снова и снова —
    это и есть металлическое «жужжание». Вдохи (anchors) остаются один к одному.
    Где материала совсем мало, он растягивается равномерно, без остановок."""
    pos = np.asarray(pos, dtype=np.float64).copy()
    n = len(pos)
    fixed = np.zeros(n, dtype=bool)
    for t0, t1 in anchors:
        fixed[max(0, t0):min(n, t1)] = True
    free = _segments(~fixed)
    for a, b in free:
        L = b - a
        if L < 2:
            continue
        has_l, has_r = a > 0, b < n
        left = pos[a - 1] if has_l else None
        right = pos[b] if has_r else None
        tgt = pos[a:b].copy()
        # вперёд
        q = np.empty(L)
        prev = left
        for i in range(L):
            v = tgt[i]
            if prev is not None:
                v = min(max(v, prev + lo), prev + hi)
            q[i] = v
            prev = v
        # назад
        nxt = right
        for i in range(L - 1, -1, -1):
            v = q[i]
            if nxt is not None:
                v = min(max(v, nxt - hi), nxt - lo)
            q[i] = v
            nxt = v
        if has_l and has_r:
            span = right - left
            if span < lo * (L + 1) or span > hi * (L + 1):     # не помещается — равномерно
                q = left + span * np.arange(1, L + 1) / (L + 1)
            else:
                dq = np.diff(np.concatenate([[left], q, [right]]))
                if np.any(dq < lo - 1e-6) or np.any(dq > hi + 1e-6):
                    q = left + span * np.arange(1, L + 1) / (L + 1)
        pos[a:b] = q
    return pos


def _mix_original(y, dub: Analysis, pos, outside, keep_dub, b_out, gain: float,
                  fade_ms: float = 10.0, orig_out=None):
    """Нота — из вокодера, всё остальное (вдохи, паузы, шумы, согласные) — из оригинала
    по той же карте времени. На вдохах карта «один к одному» (см. place_breaths),
    поэтому вдох звучит ровно как в записи."""
    hop, sr = dub.hop, dub.sr
    k = np.clip(np.round(pos).astype(np.int64), 0, dub.n - 1)
    keep_out = keep_dub[k] | outside
    if orig_out is not None:                        # тихие хвосты/атаки и ноты без поправки
        keep_out = keep_out | orig_out[: len(keep_out)]
    for t0, t1 in b_out:                            # вдох целиком из оригинала
        keep_out[t0:t1] = True
    orig = _wsola_warp(dub.x, pos, hop, len(y))
    ramp = int(round(0.015 * sr))                  # стык 15 мс
    m = _soft(keep_out, hop, len(y), ramp)
    if b_out and abs(gain - 1.0) > 1e-3:
        bm = np.zeros(len(keep_out), dtype=bool)
        for t0, t1 in b_out:
            bm[t0:t1] = True
        r2 = max(3, int(round(sr * max(2.0, fade_ms) / 1000.0)))
        orig = orig * (1.0 + (gain - 1.0) * _soft(bm, hop, len(y), r2))
    placed = [(t0 * hop / sr, t1 * hop / sr) for t0, t1 in b_out]
    return y * (1.0 - m) + orig * m, placed


# ----------------------------------------------------------------------------
# Динамика: перенос «компрессии» эталона
# ----------------------------------------------------------------------------
def _env_db(x: np.ndarray, sr: int, win: int, hop: int) -> np.ndarray:
    rms = librosa.feature.rms(y=x.astype(np.float32), frame_length=win, hop_length=hop,
                              center=True)[0]
    return 20.0 * np.log10(rms + 1e-9)


def _level_diff(er: np.ndarray, eo: np.ndarray):
    """Маски громких кадров и среднее отличие громкости (дБ) там, где звучат оба."""
    n = min(len(er), len(eo))
    er, eo = er[:n], eo[:n]
    loud_r = er > np.percentile(er, 95) - 40.0
    loud_o = eo > np.percentile(eo, 95) - 40.0
    both = loud_r & loud_o
    diff = float(np.median(np.abs(er[both] - eo[both]))) if both.sum() > 10 else float("nan")
    return er, eo, loud_r, both, diff


def _match_dynamics(y: np.ndarray, ref: np.ndarray, sr: int, s: Settings):
    """Ведёт громкость результата так же, как меняется громкость эталона.
    Возвращает (новый сигнал, отличие до, отличие после) в дБ."""
    hop = max(1, int(round(sr * 0.005)))
    win = max(2 * hop, int(round(sr * max(10.0, s.dyn_speed_ms) / 1000.0)) // 2 * 2)
    er, eo, loud_r, both, before = _level_diff(_env_db(ref, sr, win, hop),
                                               _env_db(y, sr, win, hop))
    if not s.dyn_match or s.dyn_amount <= 0 or both.sum() <= 10:
        return y, before, before

    n = len(er)
    idx = np.arange(n)
    gain = np.interp(idx, idx[both], (er - eo)[both])       # между словами — плавный переход
    gain[~loud_r] = np.minimum(gain[~loud_r], 0.0)          # где эталон молчит — не поднимать
    gain = np.clip(gain, -s.dyn_max_db, s.dyn_max_db)
    k = max(1, int(round(s.dyn_speed_ms / 5.0)))            # сглаживание = скорость реакции
    if k > 1:
        gain = np.convolve(np.pad(gain, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:n]
    gain *= s.dyn_amount

    centers = idx * hop
    g = np.power(10.0, np.interp(np.arange(len(y)), centers, gain) / 20.0)
    out = y * g
    after = _level_diff(er, _env_db(out, sr, win, hop))[4]
    return out, before, after


# ----------------------------------------------------------------------------
# Сессия: кэш анализа, обработка, запись
# ----------------------------------------------------------------------------
def _file_key(path: str, quality: str, target_sr):
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size, quality, target_sr)


def output_path_for(dub_path: str, s: Settings) -> str:
    folder = s.out_dir or os.path.join(os.path.dirname(os.path.abspath(dub_path)), "VoxMatch")
    stem = os.path.splitext(os.path.basename(dub_path))[0]
    return os.path.join(folder, f"{stem}{s.suffix}.wav")


def _write(path: str, y: np.ndarray, sr: int, subtype: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, ext = os.path.splitext(path)
    for n in range(0, 50):
        target = path if n == 0 else f"{base}_{n + 1}{ext}"
        try:
            sf.write(target, y, sr, subtype=subtype)
            return target
        except (PermissionError, OSError, RuntimeError):
            continue   # файл занят (например, открыт в Cubase) — пробуем другое имя
    raise RuntimeError(f"Не удалось записать файл в папку {os.path.dirname(path)}")


class Session:
    """Хранит результаты анализа, чтобы повторная обработка с новыми ползунками шла быстро."""

    def __init__(self):
        self._analysis: dict = {}
        self._maps: dict = {}
        self._breaths: dict = {}
        self._phrases: dict = {}

    def phrases(self, ref: Analysis):
        """Фразы эталона и места стыков (в кадрах эталона)."""
        if ref.key not in self._phrases:
            ph = vt.find_phrases(ref.loud, ref.frame_ms)
            if not ph:
                ph = [(0, ref.n)]
            self._phrases[ref.key] = (ph, vt.cut_points(ph, ref.n, ref.frame_ms))
        return self._phrases[ref.key]

    def analysis(self, path: str, quality: str, target_sr=None, need_env: bool = True,
                 progress: Progress = None) -> Analysis:
        key = _file_key(path, quality, target_sr)
        an = self._analysis.get(key)
        if an is not None and (an.sp is not None or not need_env):
            self._analysis[key] = self._analysis.pop(key)      # отметить как свежий
            return an
        an = analyze(path, quality, target_sr, need_env, progress, key)
        self._analysis[key] = an
        while len(self._analysis) > CACHE_SIZE:                  # освободить память
            old = next(iter(self._analysis))
            self._analysis.pop(old)
            self._maps = {k: v for k, v in self._maps.items() if old not in k[:2]}
            self._breaths = {k: v for k, v in self._breaths.items() if k[0] != old}
        return an

    def process(self, ref_path: str, dub_path: str, s: Settings,
                progress: Progress = None, write: bool = True) -> Result:
        t0 = time.time()

        def sub(lo, hi):
            if not progress:
                return None
            return lambda msg, f: progress(msg, lo + (hi - lo) * f)

        ref = self.analysis(ref_path, s.quality, None, False, sub(0.0, 0.3))
        dub = self.analysis(dub_path, s.quality, ref.sr, True, sub(0.3, 0.7))
        fm = ref.frame_ms
        n, m = ref.n, dub.n
        band = max(4, int(round(s.max_shift_ms / fm)))

        penalty = float(np.interp(s.timing_flex, [0, 50, 100], [0.6, 0.15, 0.02]))
        mkey = (ref.key, dub.key, band, round(penalty, 3))
        if mkey in self._maps:
            full_map, lag = self._maps[mkey]
        else:
            if progress:
                progress("Выравниваю по времени", 0.72)
            max_lag = max(band, int(round(3000.0 / fm)))
            lag = _global_lag(ref, dub, max_lag)
            center = np.clip(np.arange(n) + lag, 0, m - 1).astype(np.int64)
            pi, pj = _dtw_band(ref.feats, dub.feats, ref.loud, dub.loud, center, band,
                               np.float32(penalty))
            if len(pi) == 0:
                raise RuntimeError("Не удалось сопоставить дубль с эталоном. "
                                   "Проверьте, что это одна и та же часть песни.")
            full_map = _path_to_map(pi, pj, n)
            self._maps[mkey] = (full_map, lag)

        identity = np.arange(n, dtype=np.float64) + (0 if s.same_start else lag)
        score = self._score(ref, dub, full_map, np.arange(n, dtype=np.float64) + lag)
        phrases, _cuts = self.phrases(ref)
        pos = shape_timing(full_map, identity, s, fm, phrases, ref.loud)

        breaths, keep_dub, b_out = [], None, []
        if s.protect_breath:
            bkey = (dub.key, round(s.breath_sens, 1), s.breath_min_ms, s.breath_pre_ms,
                    s.breath_post_ms)
            breaths = self._breaths.get(bkey)
            if breaths is None:
                breaths = detect_breaths(dub, s.breath_sens, s.breath_min_ms,
                                         s.breath_pre_ms, s.breath_post_ms)
                self._breaths[bkey] = breaths
            vkey = (dub.key, "voiced")
            vm = self._breaths.get(vkey)
            if vm is None:
                vm = voiced_mask(dub)
                self._breaths[vkey] = vm
            breaths = grow_breaths(breaths, vm, dub)
            breaths = protect_tails(breaths, vm, dub, s.tail_ms, s.breath_min_ms)
            keep_dub = ~vm
            for a, b in breaths:                         # вдохи — всегда из оригинала
                keep_dub[a:b] = True
            lagoff = 0 if s.same_start else int(round(identity[0]))
            pos, b_out = place_breaths(pos, breaths, vm, s.breath_mode, lagoff)
        pos = _enforce_slope(pos, b_out)
        if not b_out:
            pos = _enforce_slope(pos, [])
        outside = (pos < -0.5) | (pos > m - 0.5)

        if progress:
            progress("Подтягиваю питч", 0.8)
        f0w, spw, apw = _warp_params(dub, pos, outside)
        if keep_dub is not None:
            k = np.clip(np.round(pos).astype(np.int64), 0, dub.n - 1)
            kk = keep_dub[k]
            kk = kk & np.roll(kk, 2) & np.roll(kk, -2)     # у самой границы ноту не трогаем
            f0w = np.where(kk, 0.0, f0w)                 # вне нот вокодер не «поёт»
        f0_new, pb, pa, octave = _pitch_correction(ref.f0, f0w, s, fm)

        if progress:
            progress("Синтезирую результат", 0.85)
        if abs(s.formant_st) > 1e-3:
            spw = shift_formants(spw, s.formant_st)
        y = pw.synthesize(f0_new, spw, apw, ref.sr, fm)
        y = _fit(y, len(ref.x))

        placed = []
        if keep_dub is not None:
            if progress:
                progress("Беру вдохи и паузы из оригинала", 0.92)
            orig_out = self._orig_frames(dub, pos, f0w, f0_new, s)
            y, placed = _mix_original(y, dub, pos, outside, keep_dub, b_out, s.breath_gain,
                                      s.breath_xfade_ms, orig_out)
        y_mix = np.asarray(y, dtype=np.float64).copy()

        if s.dyn_match and progress:
            progress("Переношу динамику эталона", 0.94)
        y, db_before, db_after = _match_dynamics(y, ref.x, ref.sr, s)
        if s.dyn_match and placed and s.breath_no_clean:
            y = keep_breaths_clean(y, y_mix, placed, ref.sr, s.breath_xfade_ms)
        y_pre = np.asarray(y, dtype=np.float32).copy()
        if progress:
            progress("Очистка", 0.96)
        y, clean_notes = vt.finish(y, ref.sr, clean_settings(s), placed)
        if s.breath_no_clean and placed:
            y = keep_breaths_clean(y, y_pre, placed, ref.sr, s.breath_xfade_ms)

        notes = []
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 0.999:
            y = y * (0.999 / peak)
            notes.append("Результат был громче 0 dBFS — громкость немного снижена.")
        if octave:
            notes.append(f"Дубль спет на {abs(octave)} окт. {'ниже' if octave > 0 else 'выше'} "
                         "эталона — октава сохранена.")

        loud = ref.loud
        tb = float(np.median(np.abs(full_map - identity)[loud])) * fm
        ta = float(np.median(np.abs(full_map - (pos + s.timing_offset_ms / fm))[loud])) * fm

        out_path = output_path_for(dub_path, s)
        if write:
            if progress:
                progress("Сохраняю файл", 0.97)
            out_path = _write(out_path, y, ref.sr, s.subtype)

        return Result(dub_path=dub_path, out_path=out_path, sr=ref.sr, y=y.astype(np.float32),
                      frame_ms=fm, f0_ref=ref.f0, f0_dub=dub.f0, f0_out=f0_new,
                      timing_before_ms=tb, timing_after_ms=ta, pitch_before_c=pb,
                      pitch_after_c=pa, lag_ms=lag * fm, octave_shift=octave,
                      seconds=time.time() - t0, dyn_before_db=db_before,
                      dyn_after_db=db_after, breaths=placed, notes=notes, y_pre=y_pre,
                      pos=np.asarray(pos, dtype=np.float32),
                      score=score, clean_notes=clean_notes)


def keep_breaths_clean(y, y_pre, breaths_s, sr, fade_ms):
    """Вернуть вдохи как были до очистки — очистка их не касается."""
    y = np.asarray(y, dtype=np.float64)
    m = np.zeros(len(y))
    for a, b in breaths_s:
        m[int(a * sr): int(b * sr)] = 1.0
    k = max(3, int(round(sr * fade_ms / 1000.0)))
    w = np.hanning(k)
    m = np.clip(np.convolve(m, w / w.sum(), mode="same"), 0, 1)
    return y * (1 - m) + np.asarray(y_pre[: len(y)], dtype=np.float64) * m


def clean_settings(s: Settings) -> vt.CleanSettings:
    return vt.CleanSettings(clicks=s.clean_clicks, deess=s.clean_deess, deess_db=s.deess_db,
                            level=s.clean_level, level_strength=s.level_strength,
                            gate=s.clean_gate, gate_db=s.gate_db)


def _score_method(self, ref: Analysis, dub: Analysis, full_map, identity):
    """Оценка дубля по фразам эталона — по тому, как он спет ДО обработки."""
    fm = ref.frame_ms
    k = np.clip(np.round(full_map).astype(np.int64), 0, dub.n - 1)
    err_t = np.abs(full_map - identity) * fm
    fr, fd = ref.f0, dub.f0[k]
    both = (fr > 0) & (fd > 0)
    err_p = np.full(ref.n, np.nan)
    if both.any():
        c = 1200.0 * np.log2(fd[both] / fr[both])
        c -= 1200.0 * np.round(np.median(c) / 1200.0)          # другая октава — не ошибка
        err_p[both] = np.abs(c)
    ddb, _ = vt._frame_db(dub.x, dub.sr, 0.02, fm / 1000.0)
    ddb = _fit(ddb, dub.n)
    hop = dub.hop
    nn = (len(dub.x) // hop) * hop
    clip_f = np.zeros(dub.n, dtype=bool)
    if nn:
        cf = np.abs(dub.x[:nn]).reshape(-1, hop).max(axis=1) > 0.985
        clip_f[: len(cf)] = cf[: dub.n]
    phrases, _cuts = self.phrases(ref)
    return vt.score_take(phrases, ref.loud, err_t, err_p, ddb[k], clip_f[k])


Session._score = _score_method


def _orig_frames_method(self, dub: Analysis, pos, f0w, f0_new, s: Settings):
    """Где вокодер НЕ нужен (берём оригинал по карте времени, WSOLA):
       • тихие хвосты нот и мягкие атаки (тише пения на 25 дБ) — там вокодер звучит «роботом»;
       • ноты, где поправка высоты почти нулевая (меньше 3 центов) и форманты не трогаются.
    Короткие «островки» вокодера (< 60 мс) тоже заменяются оригиналом — без мерцания."""
    key = (dub.key, "db")
    db = self._breaths.get(key)
    if db is None:
        db, _h = vt._frame_db(dub.x, dub.sr, 0.03, dub.frame_ms / 1000.0)
        db = _fit(db, dub.n)
        self._breaths[key] = db
    peak = vt._peak_db(db)
    k = np.clip(np.round(pos).astype(np.int64), 0, dub.n - 1)
    quiet = db[k] < peak - 25.0
    v = (f0w > 0) & (f0_new > 0)
    corr = np.zeros(len(f0w))
    corr[v] = np.abs(1200.0 * np.log2(f0_new[v] / f0w[v]))
    small = corr < 3.0
    if abs(s.formant_st) > 1e-3:
        small[:] = False
    use_voc = ~(quiet | small)
    fm = dub.frame_ms
    use_voc = _close_gaps(use_voc, int(round(40.0 / fm)))
    for a, b in _segments(use_voc):
        if b - a < int(round(60.0 / fm)):
            use_voc[a:b] = False
    return ~use_voc


def _close_gaps(mask, gap):
    m = mask.copy()
    for a, b in _segments(~m):
        if 0 < a and b < len(m) and b - a < gap:
            m[a:b] = True
    return m


Session._orig_frames = _orig_frames_method


def quiet_cuts(cuts, phrases, choices, good, ref):
    """Стык компа — в самом тихом месте паузы для ОБОИХ дублей (чтобы не оборвать
    окончание фразы и не разрезать вдох посередине)."""
    out = list(cuts)
    fm = ref.frame_ms / 1000.0
    env = {}
    for d in set(choices):
        db, _ = vt._frame_db(np.asarray(good[d].y_pre, dtype=np.float64), ref.sr, 0.02, fm)
        env[d] = _fit(db, ref.n)
    for i in range(len(choices) - 1):
        a_prev, b_prev = phrases[i]
        a_next, _ = phrases[i + 1]
        if a_next - b_prev < 4:
            continue
        ea, eb = env[choices[i]], env[choices[i + 1]]
        if choices[i] == choices[i + 1]:
            continue
        lo, hi = b_prev, a_next
        both = np.maximum(ea[lo:hi], eb[lo:hi])
        floor = float(both.min())
        ok = np.flatnonzero(both <= floor + 3.0)                  # самое раннее тихое место
        out[i + 1] = int(lo + ok[0])
    return out


def make_comp(session: Session, ref_path: str, results: dict, s: Settings,
              choices: list | None = None, write: bool = True):
    """Автокомп из уже выровненных дублей. results: путь дубля → Result.
    Возвращает словарь с сигналом, выбором по фразам, оценками и путём файла."""
    ref = session.analysis(ref_path, s.quality, None, False)
    phrases, cuts = session.phrases(ref)
    good = {d: r for d, r in results.items() if r is not None and r.y_pre is not None}
    if not good:
        raise RuntimeError("Нет выровненных дублей для компа.")
    scores = {d: r.score for d, r in good.items()}
    auto = vt.best_choices(scores, len(phrases))
    if not choices or len(choices) != len(phrases) or any(c not in good for c in choices):
        choices = auto
    cuts = quiet_cuts(cuts, phrases, choices, good, ref)
    y = vt.build_comp({d: r.y_pre for d, r in good.items()}, choices, cuts, ref.hop, ref.sr)
    breaths = []
    for i, d in enumerate(choices):
        t0, t1 = cuts[i] * ref.hop / ref.sr, cuts[i + 1] * ref.hop / ref.sr
        breaths += [(a, b) for a, b in good[d].breaths if t0 <= (a + b) / 2 < t1]
    y_pre = y.copy()
    y, notes = vt.finish(y, ref.sr, clean_settings(s), breaths)
    if s.breath_no_clean and breaths:
        y = keep_breaths_clean(y, y_pre, breaths, ref.sr, s.breath_xfade_ms)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.999:
        y = y * (0.999 / peak)
    folder = s.out_dir or os.path.join(os.path.dirname(os.path.abspath(ref_path)), "VoxMatch")
    stem = os.path.splitext(os.path.basename(ref_path))[0]
    out_path = os.path.join(folder, f"{stem} — КОМП.wav")
    if write:
        out_path = _write(out_path, y, ref.sr, s.subtype)
    fm = ref.frame_ms / 1000.0
    return {"y": y.astype(np.float32), "sr": ref.sr, "choices": choices, "auto": auto,
            "phrases": [(a * fm, b * fm) for a, b in phrases],
            "scores": scores, "breaths": breaths, "out_path": out_path, "notes": notes}


# ----------------------------------------------------------------------------
# Командная строка
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VoxMatch: подгонка дублей вокала под эталон")
    ap.add_argument("reference", help="эталонный вокал")
    ap.add_argument("dubs", nargs="+", help="дубли для подгонки")
    ap.add_argument("--timing", type=float, default=100, help="тайминг, %% (0–100)")
    ap.add_argument("--pitch", type=float, default=100, help="питч, %% (0–100)")
    ap.add_argument("--no-vibrato", action="store_true", help="копировать и вибрато эталона")
    ap.add_argument("--smooth", type=float, default=30, help="плавность питча, мс")
    ap.add_argument("--max-st", type=float, default=3, help="предел коррекции, полутонов")
    ap.add_argument("--max-shift", type=float, default=600, help="макс. сдвиг по времени, мс")
    ap.add_argument("--fast", action="store_true", help="быстрый анализ")
    ap.add_argument("--dyn", action="store_true", help="перенести динамику (компрессию) эталона")
    ap.add_argument("--dyn-amount", type=float, default=100, help="сила переноса динамики, %%")
    ap.add_argument("--dyn-speed", type=float, default=50, help="скорость динамики, мс")
    ap.add_argument("--touch-breath", action="store_true", help="обрабатывать и вдохи (по умолчанию не трогаются)")
    ap.add_argument("--breath-sens", type=float, default=50, help="чувствительность поиска вдохов, 0–100")
    ap.add_argument("--breath-gain", type=float, default=100, help="громкость вдохов, %%")
    ap.add_argument("--breath-mode", choices=["original", "phrase"], default="original",
                    help="original — вдох на своём месте; phrase — вдох перед фразой")
    ap.add_argument("--tail-ms", type=float, default=120, help="защита окончаний фраз, мс")
    ap.add_argument("--comp", action="store_true", help="собрать комп из лучших фраз всех дублей")
    ap.add_argument("--no-clean", action="store_true", help="без очистки (щелчки, деэссер, громкость фраз, паузы)")
    ap.add_argument("-o", "--out", default="", help="папка для результатов")
    a = ap.parse_args()

    s = Settings(timing=a.timing / 100, pitch=a.pitch / 100, keep_vibrato=not a.no_vibrato,
                 pitch_smooth_ms=a.smooth, max_correction_st=a.max_st, max_shift_ms=a.max_shift,
                 quality="fast" if a.fast else "accurate", out_dir=a.out, dyn_match=a.dyn,
                 dyn_amount=a.dyn_amount / 100, dyn_speed_ms=a.dyn_speed,
                 protect_breath=not a.touch_breath, breath_sens=a.breath_sens,
                 breath_gain=a.breath_gain / 100, breath_mode=a.breath_mode,
                 tail_ms=a.tail_ms)
    if a.no_clean:
        s.clean_clicks = s.clean_deess = s.clean_level = s.clean_gate = False
    results = {}
    session = Session()
    for d in a.dubs:
        r = session.process(a.reference, d, s,
                            progress=lambda msg, f: print(f"  [{f * 100:3.0f}%] {msg}"))
        results[d] = r
        print(f"{os.path.basename(d)} → {r.out_path}   оценка {r.score.total:.0f}/100")
        print(f"  тайминг: {r.timing_before_ms:.0f} → {r.timing_after_ms:.0f} мс, "
              f"питч: {r.pitch_before_c:.0f} → {r.pitch_after_c:.0f} центов, "
              f"громкость: {r.dyn_before_db:.1f} → {r.dyn_after_db:.1f} дБ, {r.seconds:.1f} с")
        if r.breaths:
            print(f"  вдохов не тронуто: {len(r.breaths)}")
        for note in r.notes:
            print("  " + note)
        for note in r.clean_notes:
            print("  " + note)
    if a.comp and len(results) > 1:
        c = make_comp(session, a.reference, results, s)
        print(f"\nКОМП → {c['out_path']}")
        for i, ((t0, t1), d) in enumerate(zip(c["phrases"], c["choices"])):
            print(f"  фраза {i + 1}: {t0:5.1f}–{t1:5.1f} с ← {os.path.basename(d)} "
                  f"({c['scores'][d].phrases[i]:.0f}/100)")


if __name__ == "__main__":
    main()
