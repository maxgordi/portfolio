"""
VoxMatch Простой — ядро.

Как у VocAlign / LINE X: дубль НЕ пересинтезируется вокодером. Живая запись
только растягивается/сжимается во времени и сдвигается по высоте движком
Rubber Band (R3, сохранение формант).

  1. Где какой слог: DTW по MFCC (эталон ↔ дубль).
  2. Карта времени сглаживается, в паузах — плавная, скорость ограничена.
  3. Проход 1: Rubber Band растягивает весь дубль по карте (ключевые точки).
  4. Проход 2: Rubber Band подтягивает высоту к эталону плавной кривой
     (своё вибрато дубля остаётся). Если питч выключен — прохода нет.
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
import numpy as np
import soundfile as sf
import librosa
import scipy.signal as ss
from numba import njit
import pylibrb as rb

from vms_world import analyze_world, shutdown as shutdown_workers  # noqa: F401

FRAME_MS = 5.0
PITCH_LAT_SAMPLES = None    # None — по задержке движка
AUDIO_EXT = {".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}
Progress = Optional[Callable[[str, float], None]]


@dataclass
class Settings:
    timing: int = 100          # сила выравнивания по времени, %
    tightness: int = 70        # 0 — свободно (плавно), 100 — жёстко по каждому слогу
    max_stretch: float = 2.0   # не растягивать/сжимать сильнее, чем во столько раз
    pitch: int = 100           # сила подтяжки высоты, % (0 — высота дубля как есть)
    formants: bool = True      # сохранять тембр при сдвиге высоты
    harmony_auto: bool = True  # бэк в гармонию — ноты не трогать, только тайминг
    flexible: bool = False     # гибкий режим: только анализ, звук правится по фразам вручную
    out_dir: str = ""          # пусто — папка «VoxMatch Простой» рядом с дублем


@dataclass
class Track:
    path: str
    sr: int
    hop: int
    fm: float
    data: np.ndarray           # (каналы, сэмплы) float32 — оригинал
    x: np.ndarray              # моно float64
    f0: np.ndarray
    feats: np.ndarray
    loud: np.ndarray

    @property
    def n(self):
        return len(self.f0)


@dataclass
class Result:
    dub_path: str
    out_path: str
    sr: int
    y: np.ndarray              # (каналы, сэмплы) float32
    timing_before_ms: float
    timing_after_ms: float
    pitch_before_c: float
    pitch_after_c: float
    seconds: float
    notes: list = field(default_factory=list)
    pos: Optional[np.ndarray] = None       # кадр результата → кадр дубля
    rate: Optional[np.ndarray] = None      # скорость (1 — без растяжения)
    dissim: Optional[np.ndarray] = None    # непохожесть на эталон по звуку (0 — то же самое)
    f0_al: Optional[np.ndarray] = None     # высота дубля ДО коррекции, на шкале эталона
    kind: str = "унисон"                   # унисон / гармония / октава
    interval_c: float = 0.0                # средний интервал к эталону, центы
    problem: Optional[np.ndarray] = None   # кадры, где стоит послушать внимательнее
    lag_frames: int = 0
    pos_full: Optional[np.ndarray] = None       # карта времени при 100 % тайминга
    corr_c: Optional[np.ndarray] = None         # поправка высоты при 100 % питча, центы
    timing_dev_ms: Optional[np.ndarray] = None  # насколько раньше/позже эталона спето (до выравнивания)


# ----------------------------------------------------------------------------
# Чтение и анализ
# ----------------------------------------------------------------------------
def _fit(a, n):
    if a.shape[-1] == n:
        return a
    if a.shape[-1] > n:
        return a[..., :n]
    pad = [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])]
    return np.pad(a, pad, mode="edge")


def load(path: str, target_sr: Optional[int] = None):
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Не удалось открыть «{os.path.basename(path)}»: {exc}") from exc
    data = data.T.copy()                                     # (каналы, сэмплы)
    if target_sr and sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq",
                                axis=-1).astype(np.float32)
        sr = target_sr
    if data.shape[1] < sr * 0.3:
        raise RuntimeError(f"«{os.path.basename(path)}» слишком короткий (меньше 0,3 с).")
    return np.ascontiguousarray(data), int(sr)


def _features(x, sr, hop, n):
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


def analyze(path: str, target_sr: Optional[int] = None, progress: Progress = None) -> Track:
    name = os.path.basename(path)
    if progress:
        progress(f"Читаю {name}", 0.0)
    data, sr = load(path, target_sr)
    x = data.mean(axis=0).astype(np.float64)
    hop = max(1, int(round(sr * FRAME_MS / 1000.0)))
    fm = 1000.0 * hop / sr
    if progress:
        progress(f"Слушаю высоту: {name}", 0.1)
    f0, _sp, _ap = analyze_world(x, sr, hop, "accurate", False)
    feats, loud = _features(x, sr, hop, len(f0))
    return Track(path, sr, hop, fm, data, x, f0, feats, loud)


def quick_track(path: str, target_sr: Optional[int] = None) -> Track:
    """Только признаки звука (без высоты) — быстро, для выбора эталона."""
    data, sr = load(path, target_sr)
    x = data.mean(axis=0).astype(np.float64)
    hop = max(1, int(round(sr * FRAME_MS / 1000.0)))
    fm = 1000.0 * hop / sr
    n = len(x) // hop + 1
    feats, loud = _features(x, sr, hop, n)
    return Track(path, sr, hop, fm, data, x, np.zeros(n), feats, loud)


# ----------------------------------------------------------------------------
# Карта времени
# ----------------------------------------------------------------------------
@njit(cache=True)
def _dtw_band(A, B, loud_a, loud_b, center, band, penalty):
    n, m, dim = A.shape[0], B.shape[0], A.shape[1]
    w = 2 * band + 1
    inf = np.float32(1e30)
    D = np.full((n, w), inf, dtype=np.float32)
    S = np.zeros((n, w), dtype=np.int8)
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
            pk = j - 1 - (center[i - 1] - band)
            if 0 <= pk < w and D[i - 1, pk] < best:
                best = D[i - 1, pk]
                step = 0
            pk = j - (center[i - 1] - band)
            if 0 <= pk < w and D[i - 1, pk] + penalty < best:
                best = D[i - 1, pk] + penalty
                step = 1
            if k > 0 and D[i, k - 1] + penalty < best:
                best = D[i, k - 1] + penalty
                step = 2
            if best < inf:
                D[i, k] = best + c
                S[i, k] = step
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


def _global_lag(ref: Track, dub: Track, max_frames: int) -> int:
    """Общий сдвиг дубля (j ≈ i + lag). Кандидаты — пики корреляции громкости в окне
    до ±15 с (кусок из Cubase мог начаться на такт-два раньше или позже), а побеждает
    тот, где звук (MFCC) совпадает лучше всего — так повторяющиеся фразы не путают."""
    max_frames = max(max_frames, int(round(15000.0 / ref.fm)))
    ea = ref.feats[:, -1].astype(np.float64)
    eb = dub.feats[:, -1].astype(np.float64)
    ea, eb = ea - ea.mean(), eb - eb.mean()
    c = ss.correlate(eb, ea, mode="full", method="fft")
    lags = ss.correlation_lags(len(eb), len(ea), mode="full")
    ok = np.abs(lags) <= max_frames
    L, C = lags[ok], c[ok]
    order = np.argsort(C)[::-1]
    sep = int(round(150.0 / ref.fm))
    cands = []
    for o in order:
        x = int(L[o])
        if all(abs(x - y) > sep for y in cands):
            cands.append(x)
        if len(cands) >= 8:
            break
    if len(cands) <= 1:
        return cands[0] if cands else 0
    i_all = np.arange(ref.n)
    best, bd = cands[0], 1e9
    for lag in cands:
        j = i_all + lag
        m = (j >= 0) & (j < dub.n) & ref.loud
        if m.sum() < max(20, 0.3 * ref.loud.sum()):         # почти не перекрываются
            continue
        d = float(np.sqrt(np.mean((ref.feats[i_all[m], :19] - dub.feats[j[m], :19]) ** 2,
                                  axis=1)).mean())
        if d < bd:
            best, bd = lag, d
    return int(best)


def _path_to_map(pi, pj, n):
    sums = np.bincount(pi, weights=pj.astype(np.float64), minlength=n)
    cnt = np.bincount(pi, minlength=n)
    m = sums / np.maximum(cnt, 1)
    return np.maximum.accumulate(m)


def _segments(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _smooth(v, win):
    """Сглаживание окном Ханна (win кадров), края — по крайнему значению."""
    if win <= 2:
        return v.copy()
    w = np.hanning(win + 2)[1:-1]
    w /= w.sum()
    pad = win
    return np.convolve(np.pad(v, pad, mode="edge"), w, mode="same")[pad:-pad]


def _limit_slope(pos, lo, hi):
    """Скорость карты (кадров дубля на кадр результата) в [lo, hi]: без «заморозок»
    и без рывков. Проход вперёд и назад — карта остаётся близкой к нужной."""
    q = pos.astype(np.float64).copy()
    for i in range(1, len(q)):
        q[i] = min(max(q[i], q[i - 1] + lo), q[i - 1] + hi)
    for i in range(len(q) - 2, -1, -1):
        q[i] = min(max(q[i], q[i + 1] - hi), q[i + 1] - lo)
    return q


def time_map(ref: Track, dub: Track, s: Settings, cache: dict | None = None):
    """Позиция в дубле (кадры) для каждого кадра результата (= кадра эталона)."""
    n, m, fm = ref.n, dub.n, ref.fm
    key = (ref.path, dub.path, ref.n, dub.n)
    if cache is not None and key in cache:
        full_map, lag = cache[key]
    else:
        band = max(4, int(round(1000.0 / fm)))                 # слог может уехать до 1 с
        lag = _global_lag(ref, dub, max(band, int(round(3000.0 / fm))))
        center = np.clip(np.arange(n) + lag, 0, m - 1).astype(np.int64)
        pi, pj = _dtw_band(ref.feats, dub.feats, ref.loud, dub.loud, center, band,
                           np.float32(0.15))
        if len(pi) == 0:
            raise RuntimeError("Не удалось сопоставить дубль с эталоном — это точно одна "
                               "и та же часть песни?")
        full_map = _path_to_map(pi, pj, n)
        if cache is not None:
            cache[key] = (full_map, lag)

    ident = np.arange(n, dtype=np.float64) + lag
    d = full_map - ident
    # В паузах эталона DTW ничего не знает — там сдвиг просто плавно переходит
    # от конца одной фразы к началу следующей (вдохи едут целиком, без рывков).
    loud = ref.loud.copy()
    idx = np.arange(n)
    if loud.any():
        d = np.interp(idx, idx[loud], d[loud])
    # Жёсткость: 0 → окно 250 мс (плавно), 100 → 15 мс (каждый слог)
    win_ms = float(np.interp(s.tightness, [0, 100], [250.0, 15.0]))
    d = _smooth(d, int(round(win_ms / fm)))
    pos = ident + (s.timing / 100.0) * d
    r = max(1.05, float(s.max_stretch))
    pos = _limit_slope(pos, 1.0 / r, r)

    before = float(np.median(np.abs(full_map - ident)[ref.loud])) * fm
    after = float(np.median(np.abs(full_map - pos)[ref.loud])) * fm
    return pos, before, after, lag, full_map


# ----------------------------------------------------------------------------
# Rubber Band
# ----------------------------------------------------------------------------
def _opts(realtime: bool, formants: bool):
    O = rb.Option
    o = O.ENGINE_FINER | O.CHANNELS_TOGETHER | O.WINDOW_STANDARD
    o |= O.PROCESS_REALTIME if realtime else O.PROCESS_OFFLINE
    o |= O.FORMANT_PRESERVED if formants else O.FORMANT_SHIFTED
    if realtime:
        o |= O.PitchHighConsistency
    return int(o)


def render(data: np.ndarray, sr: int, pos: np.ndarray, hop: int, n_out: int,
           ratio_frames: Optional[np.ndarray], formants: bool,
           progress: Progress = None) -> np.ndarray:
    """Один проход Rubber Band (R3): дубль растягивается по карте pos
    (кадр результата → кадр дубля) и одновременно сдвигается по высоте
    по кривой ratio_frames (на кадр результата). Без склеек и вокодера.
    Скорость задаётся каждые 256 сэмплов с обратной связью, так что
    накопленной ошибки по времени нет. Где дубля нет — тишина."""
    ch, m = data.shape
    pos_s = pos * hop
    ok = (pos_s >= 0) & (pos_s <= m - 1)
    out = np.zeros((ch, n_out), dtype=np.float32)
    if ok.sum() < 4:
        return out
    i0, i1 = int(np.flatnonzero(ok)[0]), int(np.flatnonzero(ok)[-1])
    in0, in1 = int(np.ceil(pos_s[i0])), int(np.floor(pos_s[i1]))
    o0, o1 = i0 * hop, min(n_out, i1 * hop)
    if in1 - in0 < hop * 4 or o1 - o0 < hop * 4:
        return out
    x = np.ascontiguousarray(data[:, in0:in1])
    L = x.shape[1]
    # T(a): где в результате должен прозвучать сэмпл a дубля (относительно начал)
    src_pts = pos_s[i0:i1 + 1] - in0
    dst_pts = (np.arange(i0, i1 + 1) * hop - o0).astype(np.float64)

    def T(a):
        return float(np.interp(a, src_pts, dst_pts))

    def P(o):
        if ratio_frames is None:
            return 1.0
        return float(np.interp((o + o0) / hop, np.arange(len(ratio_frames)), ratio_frames))

    blk = 256
    st = rb.RubberBandStretcher(sr, ch, _opts(True, formants), 1.0, 1.0)
    st.set_max_process_size(blk * 4)
    r0 = max(0.2, min(5.0, T(blk) / blk))
    st.time_ratio = r0
    st.pitch_scale = P(0)
    pad = st.get_preferred_start_pad()
    delay = st.get_start_delay()
    lat = PITCH_LAT_SAMPLES if PITCH_LAT_SAMPLES is not None else int(1.75 * delay)
    pz = np.zeros((ch, pad), np.float32)
    for a in range(0, pad, blk):
        st.process(np.ascontiguousarray(pz[:, a:a + blk]), final=False)
    need = (o1 - o0) + delay
    chunks, got = [], 0

    def pull():
        nonlocal got
        av = st.available()
        if av > 0:
            c = st.retrieve(av)
            chunks.append(c)
            got += c.shape[1]

    pull()
    sched = 0.0                                   # сколько результата уже «заказано»
    a = 0
    step = 0
    while a < L:
        b = min(L, a + blk)
        want = T(b) - sched
        r = max(0.2, min(5.0, want / (b - a)))
        st.time_ratio = r
        st.pitch_scale = P(T(a) - lat)          # движок отдаёт звук с задержкой
        st.process(np.ascontiguousarray(x[:, a:b]), final=False)
        sched += r * (b - a)
        a = b
        pull()
        step += 1
        if progress and step % 400 == 0:
            progress("Выравниваю и подтягиваю высоту", a / L)
    tail = np.zeros((ch, blk), np.float32)
    guard = 0
    while got < need and guard < 4000:
        guard += 1
        st.process(tail, final=False)
        pull()
    y = np.concatenate(chunks, axis=1) if chunks else np.zeros((ch, 0), np.float32)
    y = y[:, delay:delay + (o1 - o0)]
    k = y.shape[1]
    out[:, o0:o0 + k] = y
    f = min(int(0.01 * sr), k // 4)
    if f > 2 and o0 > 0:
        out[:, o0:o0 + f] *= np.linspace(0, 1, f, dtype=np.float32)
    if f > 2 and o0 + k < n_out:
        out[:, o0 + k - f:o0 + k] *= np.linspace(1, 0, f, dtype=np.float32)
    return out


# ----------------------------------------------------------------------------
# Кривая высоты
# ----------------------------------------------------------------------------
def pitch_curve(f0_ref, f0_dub_aligned, amount: float, fm: float):
    """Отношение частот на каждый кадр результата. Медленная часть разницы
    эталон−дубль (ноту) правим, быструю (вибрато дубля) оставляем."""
    n = len(f0_ref)
    both = (f0_ref > 0) & (f0_dub_aligned > 0)
    c = np.zeros(n)
    if both.sum() < 10 or amount <= 0:
        return np.ones(n), 0.0, 0.0, 0
    c[both] = 1200.0 * np.log2(f0_ref[both] / f0_dub_aligned[both])
    octave = int(np.round(np.median(c[both]) / 1200.0))
    c[both] -= 1200.0 * octave                               # другая октава — сохраняем
    valid = both & (np.abs(c) < 350.0)                       # больше 3,5 полутона — не та нота
    before = float(np.median(np.abs(c[valid]))) if valid.any() else 0.0
    # медиана (убираем «прострелы» детектора), затем сглаживание 120 мс внутри нот
    cm = c.copy()
    k = max(3, int(round(35.0 / fm)) | 1)
    for a, b in _segments(valid):
        if b - a >= k:
            cm[a:b] = ss.medfilt(c[a:b], k)
    w = int(round(120.0 / fm))
    corr = np.zeros(n)
    for a, b in _segments(valid):
        seg = cm[a:b]
        if b - a > 2:
            ww = min(w, b - a)
            corr[a:b] = _smooth(seg, ww) if ww > 2 else seg
        else:
            corr[a:b] = seg
    # через короткие дырки (согласные) — тянем, дальше плавно к нулю
    idx = np.arange(n)
    if valid.any():
        filled = np.interp(idx, idx[valid], corr[valid])
        dist = np.full(n, 1e9)
        vi = np.flatnonzero(valid)
        j = np.searchsorted(vi, idx)
        left = np.where(j > 0, idx - vi[np.clip(j - 1, 0, len(vi) - 1)], 1e9)
        right = np.where(j < len(vi), vi[np.clip(j, 0, len(vi) - 1)] - idx, 1e9)
        dist = np.minimum(left, right) * fm
        fade = np.clip(1.0 - (dist - 60.0) / 60.0, 0.0, 1.0)  # 60 мс держим, ещё 60 — гасим
        corr = filled * fade
    corr = _smooth(corr, int(round(20.0 / fm)))
    corr *= amount
    after = float(np.median(np.abs(c[valid] - corr[valid]))) if valid.any() else 0.0
    return 2.0 ** (corr / 1200.0), before, after, octave


def classify(f0_ref, f0_al):
    """Унисон, гармония или октава: по интервалу между нотами эталона и дубля."""
    both = (f0_ref > 0) & (f0_al > 0)
    if both.sum() < 40:
        return "унисон", 0.0, 0
    c = 1200.0 * np.log2(f0_al[both] / f0_ref[both])
    octave = int(np.round(np.median(c) / 1200.0))
    r = c - 1200.0 * octave
    r = (r + 600.0) % 1200.0 - 600.0
    near = float(np.mean(np.abs(r) < 100.0))
    med = float(np.median(r))
    if near >= 0.5 and abs(med) < 100.0:
        kind = "октава" if octave else "унисон"
        return kind, float(np.median(c)), octave
    return "гармония", float(np.median(c)), octave


def frame_dissim(ref: Track, dub: Track, pos):
    """Насколько звук дубля в каждом кадре не похож на эталон (по MFCC), 0…~3."""
    k = np.clip(np.round(pos).astype(np.int64), 0, dub.n - 1)
    A = ref.feats[:, :19]
    B = dub.feats[k, :19]
    d = np.sqrt(np.mean((A - B) ** 2, axis=1))
    d = _smooth(d, int(round(60.0 / ref.fm)))
    inside = (pos >= 0) & (pos <= dub.n - 1)
    return np.where(inside, d, 3.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Сеанс
# ----------------------------------------------------------------------------
def raw_on_ref(dub: Track, lag_frames: int, n_out: int, start_frame: int = 0) -> np.ndarray:
    """Дубль как есть, только сдвинут на общий сдвиг (без растяжения — бит в бит)."""
    hop = dub.hop
    ch, m = dub.data.shape
    out = np.zeros((ch, n_out), np.float32)
    s0 = (start_frame + lag_frames) * hop
    a, b = max(0, s0), min(m, s0 + n_out)
    if b > a:
        out[:, a - s0:b - s0] = dub.data[:, a:b]
    return out


def render_phrase(dub: Track, n_frames: int, lag: int, pos_full, corr_c, a: int, b: int,
                  t_pct: float, p_pct: float, formants: bool, margin_ms: float = 200.0):
    """Одна фраза [a, b) (кадры эталона) с запасом по краям: тайминг t_pct % и питч
    p_pct % — только для неё. 0/0 — оригинал без обработки.
    Возвращает (начальный сэмпл на шкале эталона, звук (каналы, сэмплы))."""
    hop = dub.hop
    mg = int(round(margin_ms / dub.fm))
    fa, fb = max(0, a - mg), min(n_frames, b + mg)
    if fb - fa < 4:
        return fa * hop, np.zeros((dub.data.shape[0], max(1, (fb - fa) * hop)), np.float32)
    if t_pct <= 0 and p_pct <= 0:
        return fa * hop, raw_on_ref(dub, lag, (fb - fa) * hop, fa)
    ident = np.arange(fa, fb, dtype=np.float64) + lag
    pos = ident + (t_pct / 100.0) * (np.asarray(pos_full[fa:fb], np.float64) - ident)
    ratio = None
    if p_pct > 0 and corr_c is not None:
        ratio = 2.0 ** ((p_pct / 100.0) * np.asarray(corr_c[fa:fb], np.float64) / 1200.0)
    y = render(dub.data, dub.sr, pos, hop, (fb - fa) * hop, ratio, formants)
    return fa * hop, y


def output_path_for(dub_path: str, s: Settings) -> str:
    d = s.out_dir or os.path.join(os.path.dirname(dub_path), "VoxMatch Простой")
    os.makedirs(d, exist_ok=True)
    stem = os.path.splitext(os.path.basename(dub_path))[0]
    return os.path.join(d, f"{stem}_выровнен.wav")


class Session:
    def __init__(self):
        self._tracks: dict = {}
        self._maps: dict = {}

    def track(self, path, sr=None, progress: Progress = None) -> Track:
        try:
            st = os.stat(path)
            key = (os.path.abspath(path), st.st_mtime, st.st_size, sr)
        except OSError:
            key = (path, sr)
        t = self._tracks.get(key)
        if t is None:
            t = analyze(path, sr, progress)
            if len(self._tracks) > 8:
                self._tracks.pop(next(iter(self._tracks)))
            self._tracks[key] = t
        return t

    def process(self, ref_path: str, dub_path: str, s: Settings, progress: Progress = None,
                write: bool = True) -> Result:
        t0 = time.time()

        def sub(lo, hi):
            return (lambda msg, f: progress(msg, lo + (hi - lo) * f)) if progress else None

        ref = self.track(ref_path, None, sub(0.0, 0.25))
        dub = self.track(dub_path, ref.sr, sub(0.25, 0.5))
        if progress:
            progress("Ищу, где какой слог", 0.52)
        import dataclasses
        s_full = dataclasses.replace(s, timing=100)
        pos_full, tb, _ta, lag, full_map = time_map(ref, dub, s_full, self._maps)
        if s.flexible:
            pos, ta = pos_full, tb
        else:
            pos, tb, ta, lag, full_map = time_map(ref, dub, s, self._maps)
        n_out = ref.data.shape[1]
        notes = []
        k = np.clip(np.round(pos_full).astype(np.int64), 0, dub.n - 1)
        inside = (pos_full >= 0) & (pos_full <= dub.n - 1)
        f0d = np.where(inside, dub.f0[k], 0.0)
        kind, interval, _oct = classify(ref.f0, f0d)
        dis_full = frame_dissim(ref, dub, pos_full)
        if kind == "гармония" and ref.loud.any() and float(np.median(dis_full[ref.loud])) > 1.1:
            kind = "унисон"                                  # не уверены — лучше не выкидывать
            notes.append("Дубль плохо совпадает с эталоном — проверьте, та ли это часть песни.")
        ratio_full, pb, pa_full, octave = pitch_curve(ref.f0, f0d, 1.0, ref.fm)
        corr_c = (1200.0 * np.log2(ratio_full)).astype(np.float32)
        if kind == "октава":
            notes.append(f"Дубль спет на {abs(octave)} окт. {'ниже' if octave < 0 else 'выше'} "
                         "эталона — октава сохранена.")
        if kind == "гармония" and s.harmony_auto:
            notes.append("Бэк в гармонию — ноты не трогаются, только тайминг.")
            corr_c = np.zeros_like(corr_c)
        if s.flexible:
            y = raw_on_ref(dub, lag, n_out)                   # без обработки — правка по фразам
            ta, pa = tb, pb
        else:
            if s.pitch > 0 and not (kind == "гармония" and s.harmony_auto):
                if pos is not pos_full:
                    kk = np.clip(np.round(pos).astype(np.int64), 0, dub.n - 1)
                    f0p = np.where((pos >= 0) & (pos <= dub.n - 1), dub.f0[kk], 0.0)
                else:
                    f0p = f0d
                ratio, pb, pa, octave = pitch_curve(ref.f0, f0p, s.pitch / 100.0, ref.fm)
            else:
                ratio, pa = None, pb
            y = render(dub.data, ref.sr, pos, ref.hop, n_out, ratio, s.formants, sub(0.55, 0.95))

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0.999:
            y *= 0.999 / peak
            notes.append("Результат был громче 0 dBFS — громкость чуть снижена.")
        out_path = output_path_for(dub_path, s)
        if write and not s.flexible:
            if progress:
                progress("Сохраняю", 0.97)
            sf.write(out_path, y.T, ref.sr, subtype="PCM_24")
        rate = np.gradient(pos).astype(np.float32)
        dis = frame_dissim(ref, dub, pos_full)
        bad = ref.loud & ((np.abs(np.log2(np.maximum(rate, 1e-3))) > np.log2(1.6)) | (dis > 1.6))
        bad = _smooth(bad.astype(np.float64), int(round(80.0 / ref.fm))) > 0.3
        # та же самая запись, что и эталон (другой вырез/копия) — в комп её не берём
        if kind == "унисон" and ref.loud.any():
            if float(np.median(dis[ref.loud])) < 0.3 and pb < 5.0 and tb < 10.0:
                kind = "копия эталона"
                notes.append("Это та же запись, что и эталон (копия) — в финал не берётся.")
        return Result(dub_path, out_path, ref.sr, y, tb, ta, pb, pa, time.time() - t0, notes,
                      pos=pos.astype(np.float32), rate=rate, dissim=dis,
                      f0_al=f0d.astype(np.float32), kind=kind, interval_c=interval,
                      problem=bad, lag_frames=int(lag),
                      pos_full=pos_full.astype(np.float32), corr_c=corr_c,
                      timing_dev_ms=((full_map - (np.arange(ref.n) + lag)) * ref.fm).astype(np.float32))
