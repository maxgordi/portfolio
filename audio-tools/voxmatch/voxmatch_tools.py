"""
VoxMatch — рутина звукорежиссёра после записи вокала:

  * фразы и оценка дублей (ноты, тайминг, полнота, клиппинг);
  * автокомп — сборка лучшей версии из фраз разных дублей;
  * очистка: щелчки губ, деэссер, ровная громкость фраз, тишина в паузах.

Всё работает на шкале времени эталона: выровненные дубли (Result.y_pre) уже на ней.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal as ss

FRAME_S = 0.005          # шаг кадров для очистки, с


# ----------------------------------------------------------------------------
# Общие мелочи
# ----------------------------------------------------------------------------
def _segments(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _frame_db(y: np.ndarray, sr: int, win_s: float = 0.02, hop_s: float = FRAME_S):
    hop = max(1, int(round(sr * hop_s)))
    win = max(hop, int(round(sr * win_s)))
    y2 = np.asarray(y, dtype=np.float64) ** 2
    cs = np.concatenate([[0.0], np.cumsum(y2)])
    n = int(np.ceil(len(y) / hop))
    c = np.arange(n) * hop + hop // 2
    lo = np.clip(c - win // 2, 0, len(y))
    hi = np.clip(c + win // 2, 0, len(y))
    e = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return 10.0 * np.log10(e + 1e-12), hop


def _peak_db(db: np.ndarray) -> float:
    good = db[db > -110]
    return float(np.percentile(good, 95)) if len(good) else -20.0


def _to_samples(frames: np.ndarray, hop: int, n: int) -> np.ndarray:
    x = np.arange(n) / hop - 0.5
    return np.interp(x, np.arange(len(frames)), frames)


def _smooth(v: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return v
    w = np.hanning(k + 2)[1:-1]
    w /= w.sum()
    return np.convolve(np.pad(v, k // 2, mode="edge"), w, mode="valid")[: len(v)]


# ----------------------------------------------------------------------------
# Фразы эталона
# ----------------------------------------------------------------------------
def find_phrases(loud: np.ndarray, frame_ms: float, min_gap_ms: float = 180.0,
                 min_len_ms: float = 250.0):
    """Фразы — звучащие участки эталона, разделённые паузами. Возвращает
    [(начало, конец)] в кадрах анализа эталона."""
    gap = int(round(min_gap_ms / frame_ms))
    segs = []
    for a, b in _segments(loud):
        if segs and a - segs[-1][1] < gap:
            segs[-1] = (segs[-1][0], b)
        else:
            segs.append((a, b))
    short = int(round(min_len_ms / frame_ms))
    out = []
    for a, b in segs:
        if b - a < short and out:
            out[-1] = (out[-1][0], b)          # слишком короткое — к предыдущей фразе
        elif b - a >= short or not out:
            out.append((a, b))
    return out


def cut_points(phrases, n: int, frame_ms: float):
    """Где резать между фразами: сразу после конца фразы (40 мс), чтобы вдох
    перед следующей фразой целиком был из того же дубля, что и фраза."""
    pad = int(round(40.0 / frame_ms))
    cuts = [0]
    for (a0, b0), (a1, _b1) in zip(phrases[:-1], phrases[1:]):
        cuts.append(min(b0 + pad, (b0 + a1) // 2))
    cuts.append(n)
    return cuts


# ----------------------------------------------------------------------------
# Оценка дубля
# ----------------------------------------------------------------------------
@dataclass
class TakeScore:
    total: float                                   # 0..100, среднее по фразам
    phrases: list = field(default_factory=list)    # оценки по фразам 0..100
    details: list = field(default_factory=list)    # (ноты, тайминг, полнота, клиппинг) по фразам


def score_take(phrases, ref_loud, err_t_ms, err_p_c, take_db, clip):
    """Оценка каждой фразы: насколько близко к эталону спето ДО обработки
    (чем ближе — тем меньше правок и артефактов), спеты ли все слова, нет ли перегруза."""
    peak = _peak_db(take_db)
    scores, details = [], []
    for a, b in phrases:
        lm = ref_loud[a:b]
        if lm.sum() < 3:
            lm = np.ones(b - a, dtype=bool)
        t = np.median(err_t_ms[a:b][lm])
        p_seg = err_p_c[a:b][lm]
        p_seg = p_seg[np.isfinite(p_seg)]
        p = float(np.median(p_seg)) if len(p_seg) > 3 else 60.0
        sung = take_db[a:b][lm] > peak - 30.0
        full = float(sung.mean())
        clipped = bool(clip[a:b].any())
        s_pitch = 100.0 * np.exp(-p / 40.0)
        s_time = 100.0 * np.exp(-t / 90.0)
        s = 0.45 * s_pitch + 0.30 * s_time + 0.25 * 100.0 * full ** 2
        if clipped:
            s *= 0.6
        scores.append(float(np.clip(s, 0, 100)))
        details.append((round(p), round(t), round(100 * full), clipped))
    total = float(np.mean(scores)) if scores else 0.0
    return TakeScore(total=total, phrases=scores, details=details)


# ----------------------------------------------------------------------------
# Автокомп
# ----------------------------------------------------------------------------
def best_choices(scores_by_take: dict, n_phrases: int):
    """Для каждой фразы — дубль с лучшей оценкой. scores_by_take: имя → TakeScore."""
    names = list(scores_by_take)
    out = []
    for i in range(n_phrases):
        best = max(names, key=lambda nm: scores_by_take[nm].phrases[i]
                   if i < len(scores_by_take[nm].phrases) else -1)
        out.append(best)
    return out


def build_comp(signals: dict, choices: list, cuts: list, hop: int, sr: int,
               fade_ms: float = 30.0):
    """Склеивает комп: фраза i берётся из signals[choices[i]].
    Стыки — в паузах, равномощные кроссфейды fade_ms."""
    n = min(len(v) for v in signals.values())
    out = np.zeros(n, dtype=np.float64)
    fade = max(8, int(round(sr * fade_ms / 1000.0)))
    t = np.linspace(0, np.pi / 2, fade)
    fin, fout = np.sin(t) ** 2, np.cos(t) ** 2
    for i, name in enumerate(choices):
        a = min(n, cuts[i] * hop)
        b = min(n, cuts[i + 1] * hop)
        w = np.zeros(n)
        w[a:b] = 1.0
        if i > 0:
            s0, s1 = max(0, a - fade // 2), min(n, a - fade // 2 + fade)
            w[s0:s1] = fin[: s1 - s0]
        if i < len(choices) - 1:
            e0, e1 = max(0, b - fade // 2), min(n, b - fade // 2 + fade)
            w[e0:e1] = fout[: e1 - e0]
        out += np.asarray(signals[name][:n], dtype=np.float64) * w
    return out


# ----------------------------------------------------------------------------
# Очистка
# ----------------------------------------------------------------------------
def remove_clicks(y: np.ndarray, sr: int, ratio: float = 6.0):
    """Щелчки губ и чмоки: очень короткие острые всплески высоких частот в паузах
    и на вдохах (там они слышны сильнее всего). В спетых нотах и на согласных
    («т», «к», «п») ничего не трогаем — чтобы не съесть дикцию.
    Убираем только высокочастотную часть в месте щелчка."""
    from scipy.ndimage import median_filter
    if sr < 16000:
        return y, 0
    sos = ss.butter(4, 2500, "highpass", fs=sr, output="sos")
    hp = ss.sosfiltfilt(sos, y)
    step = max(1, int(sr * 0.0005))                          # 0.5 мс
    n = len(hp) // step
    if n < 100:
        return y, 0
    env = np.sqrt((hp[: n * step] ** 2).reshape(n, step).mean(axis=1))
    local = median_filter(env, size=61, mode="nearest")      # фон за 30 мс (медиана не видит щелчка)
    floor = 10 ** (-70 / 20) * max(float(np.abs(y).max()), 1e-6)
    cand = (env > ratio * np.maximum(local, 1e-9)) & (env > floor)
    db, hop = _frame_db(y, sr, 0.04)                         # громкость вокруг (40 мс)
    peak = _peak_db(db)
    db_s = median_filter(db, size=11, mode="nearest")        # ±25 мс, без самого щелчка
    mask = np.zeros(len(y))
    count = 0
    for s0, s1 in _segments(cand):
        if s1 - s0 > 14:                                     # длиннее 7 мс — не щелчок
            continue
        c = (s0 + s1) // 2 * step
        if db_s[min(len(db_s) - 1, c // hop)] > peak - 25.0:  # идёт пение — не трогаем
            continue
        lo = max(0, (s0 - 4) * step)
        hi = min(len(y), (s1 + 4) * step)
        mask[lo:hi] = np.maximum(mask[lo:hi], np.hanning(hi - lo + 2)[1:-1])
        count += 1
    if count == 0:
        return y, 0
    return y - hp * mask, count


def deess(y: np.ndarray, sr: int, amount_db: float = 5.0):
    """Деэссер: в местах, где громкие шипящие («с», «ш», «ц»), приглушаем только верх."""
    if amount_db <= 0 or sr < 16000:
        return y, 0.0
    sos = ss.butter(4, 4500, "highpass", fs=sr, output="sos")
    hb = ss.sosfiltfilt(sos, y)
    lb = y - hb
    db_all, hop = _frame_db(y, sr, 0.01)
    db_hi, _ = _frame_db(hb, sr, 0.01)
    peak = _peak_db(db_all)
    ratio = db_hi - db_all                                   # доля верха, дБ (≤ 0)
    w = np.clip((ratio + 6.0) / 4.0, 0.0, 1.0)               # верх ≥ −2 дБ от целого → шипящая
    w *= (db_hi > peak - 30.0)                               # только заметные
    w = np.maximum(w, _smooth(w, 3))                         # быстрая атака
    w = _smooth(w, 7)                                        # мягкий релиз ~35 мс
    g = 10.0 ** (-amount_db * w / 20.0)
    g = _to_samples(g, hop, len(y))
    return lb + hb * g, float(w.mean())


def level_phrases(y: np.ndarray, sr: int, strength: float = 0.7, max_db: float = 8.0):
    """Ровная громкость фраз (как клип-гейн): тихие фразы чуть поднимаем, громкие чуть
    опускаем к общему уровню. Внутри фразы динамика не меняется."""
    if strength <= 0:
        return y, []
    db, hop = _frame_db(y, sr, 0.03)
    peak = _peak_db(db)
    loud = db > peak - 25.0
    fm = 1000.0 * hop / sr
    phr = find_phrases(loud, fm, min_gap_ms=250.0, min_len_ms=300.0)
    if len(phr) < 2:
        return y, []
    levels = np.array([np.percentile(db[a:b][loud[a:b]], 75) if loud[a:b].any() else db[a:b].max()
                       for a, b in phr])
    target = float(np.median(levels))
    gains = np.clip((target - levels) * strength, -max_db, max_db)
    centers = np.array([(a + b) / 2 for a, b in phr])
    g = np.zeros(len(db))
    for (a, b), gd in zip(phr, gains):
        g[a:b] = gd
    gaps = ~np.zeros(len(db), dtype=bool)
    for a, b in phr:
        gaps[a:b] = False
    idx = np.arange(len(db))
    if gaps.any():                                            # в паузах — плавный переход
        g[gaps] = np.interp(idx[gaps], centers, gains)
    g = _smooth(g, int(round(60.0 / fm)))
    gs = 10.0 ** (_to_samples(g, hop, len(y)) / 20.0)
    return y * gs, [round(float(x), 1) for x in gains]


def gate_pauses(y: np.ndarray, sr: int, breaths_s, below_db: float = 50.0,
                depth_db: float = 30.0):
    """Тише в паузах: шум и гул между фразами приглушаются на depth_db (не в ноль —
    воздух остаётся естественным). Никогда не трогает: вдохи, окончания фраз
    (300 мс после пения) и подготовку к фразе (100 мс до пения)."""
    db, hop = _frame_db(y, sr, 0.02)
    peak = _peak_db(db)
    quiet = db < peak - below_db
    singing = db > peak - 35.0
    protect = np.zeros(len(db), dtype=bool)
    after, before = int(round(0.30 * sr / hop)), int(round(0.10 * sr / hop))
    for a, b in _segments(singing):
        protect[max(0, a - before): b + after] = True
    for a, b in breaths_s or []:
        protect[max(0, int(a * sr / hop) - 8): int(b * sr / hop) + 8] = True
    quiet &= ~protect
    m = np.zeros(len(db))
    min_len = int(round(0.15 * sr / hop))
    for a, b in _segments(quiet):
        if b - a >= min_len:
            m[a:b] = 1.0
    if not m.any():
        return y, 0.0
    m = _smooth(m, int(round(0.06 * sr / hop)))              # плавно: 60 мс
    floor = 10.0 ** (-depth_db / 20.0)
    ms = _to_samples(m, hop, len(y))
    return y * (1.0 - ms * (1.0 - floor)), float(m.mean())


@dataclass
class CleanSettings:
    clicks: bool = True
    deess: bool = True
    deess_db: float = 5.0
    level: bool = True
    level_strength: float = 0.7
    gate: bool = False
    gate_db: float = 50.0


def finish(y: np.ndarray, sr: int, c: CleanSettings, breaths_s=None):
    """Цепочка очистки. Возвращает (сигнал, заметки для отчёта)."""
    notes = []
    y = np.asarray(y, dtype=np.float64)
    if c.clicks:
        y, n = remove_clicks(y, sr)
        if n:
            notes.append(f"щелчков убрано: {n}")
    if c.deess:
        y, _ = deess(y, sr, c.deess_db)
    if c.level:
        y, gains = level_phrases(y, sr, c.level_strength)
        if gains:
            notes.append("громкость фраз: " + ", ".join(f"{g:+.1f}" for g in gains) + " дБ")
    if c.gate:
        y, _ = gate_pauses(y, sr, breaths_s, c.gate_db)
    return y, notes
