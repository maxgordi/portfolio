"""
Постобработка пути DTW → карта времени.

Карта pos[i] — позиция в dub (кадры анализа, 5 мс) для кадра i эталона/результата.

  1. Монотонность (путь DTW уже неубывающий).
  2. Паузы эталона: отклонение от глобальной линии интерполируется линейно между
     краями фраз — тишину не «дёргает», вдохи едут целиком.
  3. Сглаживание отклонения окном Ханна (шире при меньшей tightness).
  4. tightness: pos = линия + α·(DTW − линия), α = tightness/100.
  5. Опорные точки (anchors): карта принудительно проходит через них, поправка
     плавно гаснет за 1,5 с.
  6. Пределы локального коэффициента растяжения (по умолчанию 0.67–1.5).
  7. Плавность смены скорости: сглаживание окном rate_smooth_ms (свёртка с
     положительным окном не выводит наклон за пределы шага 6).
  8. Прореживание до опорных кадров: сетка keyframe_ms в звучащих местах + онсеты
     эталона + границы пауз; между ними — линейно (так же работает Rubber Band).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import segments, smooth

ANCHOR_RADIUS_S = 1.5


@dataclass
class TimeMap:
    pos: np.ndarray              # итоговая карта (кадры dub на кадр результата)
    pos_target: np.ndarray       # до ограничений растяжения
    pos_dtw: np.ndarray          # «сырая» карта DTW (после пауз), 100 % точности
    lin: np.ndarray              # глобальная линия (сдвиг + темп)
    keys: np.ndarray             # индексы опорных кадров
    clipped: np.ndarray          # bool: где сработали ограничения растяжения
    anchors: list = field(default_factory=list)   # [(кадр эталона, кадр dub)]

    @property
    def rate(self) -> np.ndarray:
        """Скорость чтения dub (кадров dub на кадр результата)."""
        return np.gradient(self.pos)

    @property
    def stretch(self) -> np.ndarray:
        """Локальный коэффициент растяжения: длительность результата / исходника."""
        return 1.0 / np.maximum(self.rate, 1e-6)


def fill_pauses(dev: np.ndarray, loud: np.ndarray) -> np.ndarray:
    """В тишине эталона — линейная интерполяция отклонения между краями фраз."""
    idx = np.arange(len(dev))
    if loud.sum() < 2:
        return dev.copy()
    return np.interp(idx, idx[loud], dev[loud])


def smooth_lin(v: np.ndarray, win: int) -> np.ndarray:
    """Сглаживание Ханном с линейным продолжением краёв (наклон на краях сохраняется)."""
    if win <= 2 or len(v) < 4:
        return v.astype(np.float64).copy()
    k = min(len(v) - 1, max(2, win))
    s0 = (v[k] - v[0]) / k
    s1 = (v[-1] - v[-1 - k]) / k
    pre = v[0] - s0 * np.arange(win, 0, -1)
    post = v[-1] + s1 * np.arange(1, win + 1)
    ext = np.concatenate([pre, v, post])
    w = np.hanning(win + 2)[1:-1]
    w /= w.sum()
    return np.convolve(ext, w, mode="same")[win:-win]


def limit_slope(pos: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Наклон карты в [lo, hi]: проход вперёд и назад — остаётся близко к нужной."""
    q = pos.astype(np.float64).copy()
    for _ in range(2):
        for i in range(1, len(q)):
            q[i] = min(max(q[i], q[i - 1] + lo), q[i - 1] + hi)
        for i in range(len(q) - 2, -1, -1):
            q[i] = min(max(q[i], q[i + 1] - hi), q[i + 1] - lo)
    return q


def apply_anchors(pos: np.ndarray, anchors: list, fm: float) -> np.ndarray:
    """anchors — [(кадр эталона, кадр dub)]. Поправка проходит точно через точки и
    линейно гаснет за ANCHOR_RADIUS_S от крайних/далёких точек."""
    if not anchors:
        return pos
    n = len(pos)
    R = ANCHOR_RADIUS_S * 1000.0 / fm
    pts = sorted((float(g), float(d)) for g, d in anchors if 0 <= g < n)
    if not pts:
        return pos
    xs, ys = [], []
    for k, (g, d) in enumerate(pts):
        r = d - float(np.interp(g, np.arange(n), pos))
        if k == 0:
            xs.append(g - R)
            ys.append(0.0)
        elif g - xs[-1] > 2 * R:
            xs.append(xs[-1] + R)
            ys.append(0.0)
            xs.append(g - R)
            ys.append(0.0)
        xs.append(g)
        ys.append(r)
    xs.append(xs[-1] + R)
    ys.append(0.0)
    corr = np.interp(np.arange(n), xs, ys, left=0.0, right=0.0)
    return pos + corr


def keyframes(n: int, loud: np.ndarray, onsets: np.ndarray, step: int,
              extra: list | None = None, use_onsets: bool = True) -> np.ndarray:
    keys = [0, n - 1]
    for a, b in segments(loud):
        keys += [a, max(a, b - 1)]
        keys += list(range(a, b, max(1, step)))
    if use_onsets and len(onsets):
        keys += [int(o) for o in onsets if 0 <= o < n]
    fixed = {0, n - 1}
    if extra:
        ex = [int(round(e)) for e in extra if 0 <= e < n]
        keys += ex
        fixed |= set(ex)
    keys = np.unique(np.asarray(keys, dtype=np.int64))
    # близкие точки дают скачки наклона из-за погрешности ±1 кадр — прореживаем:
    # не чаще чем через 0,6 шага (якоря и края остаются всегда)
    min_gap = max(2, int(round(0.6 * step)))
    out = [int(keys[0])]
    for k in keys[1:]:
        k = int(k)
        if k - out[-1] >= min_gap or k in fixed:
            if k in fixed and k - out[-1] < min_gap and out[-1] not in fixed and len(out) > 1:
                out[-1] = k
            else:
                out.append(k)
    return np.asarray(out, dtype=np.int64)


def build(dtw_map: np.ndarray | None, lin: np.ndarray, loud: np.ndarray, onsets: np.ndarray,
          s, fm: float, anchors: list | None = None, trust: np.ndarray | None = None) -> TimeMap:
    """trust — кадры, где карте DTW можно верить (поют и эталон, и dub); в остальных —
    линейная интерполяция. По умолчанию — звучащие кадры эталона."""
    n = len(lin)
    trust = loud if trust is None or trust.sum() < 2 else trust
    anchors = anchors or []
    alpha = float(np.clip(s.tightness, 0.0, 100.0)) / 100.0
    if dtw_map is None or s.mode == "global":
        pos_dtw = lin.copy() if dtw_map is None else lin + fill_pauses(dtw_map - lin, trust)
        target = lin.copy()
    else:
        dev = fill_pauses(dtw_map - lin, trust)
        pos_dtw = lin + dev
        win_ms = s.smooth_ms * (1.0 + 3.0 * (1.0 - alpha))
        dev = smooth(dev, int(round(win_ms / fm)))
        target = lin + alpha * dev
    target = apply_anchors(target, anchors, fm)
    lo = 1.0 / max(1.01, float(s.stretch_max))         # растяжение 1.5 → читаем dub медленнее
    hi = 1.0 / min(0.99, float(s.stretch_min))
    pos = limit_slope(target, lo, hi)
    pos = smooth_lin(pos, int(round(s.rate_smooth_ms / fm)))
    pos = limit_slope(pos, lo, hi)
    step = int(round(s.keyframe_ms / fm))
    keys = keyframes(n, loud, onsets, step, [g for g, _d in anchors], s.keyframes_onsets)
    final = np.interp(np.arange(n), keys, pos[keys])
    clipped = np.abs(final - target) * fm > 30.0
    return TimeMap(pos=final, pos_target=target, pos_dtw=pos_dtw, lin=lin, keys=keys,
                   clipped=clipped, anchors=list(anchors))
