"""
DTW в полосе вокруг заданной «центральной» линии + многомасштабная схема.

Почему своя реализация, а не librosa.sequence.dtw и не synctoolbox (MrMsDTW):
  • librosa.sequence.dtw строит полную матрицу N×M: для 4 минут при шаге 5 мс это
    48 000 × 48 000 ячеек ≈ 9 ГБ — не помещается; полосы вокруг произвольного пути
    (а не диагонали) там нет;
  • synctoolbox делает то же, что и здесь (грубо → точно в полосе), но тянет тяжёлые
    зависимости (libfmp, pysoundfile-версии, numba-ядра с жёсткими версиями), рассчитан
    на chroma+DLNCO музыкальных записей и не знает про тишину/паузы вокала;
  • здесь: память O(N·полоса) (4 мин, ±250 мс → ~25 МБ), паузы с обеих сторон
    ничего не стоят (путь в тишине не «цепляется» за шум), а полоса идёт вдоль
    грубого пути, учитывающего сдвиг, темп и локальные отклонения.
"""
from __future__ import annotations

import numpy as np
from numba import njit


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
                c = np.sqrt(s)
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


def banded_dtw(A: np.ndarray, B: np.ndarray, loud_a: np.ndarray, loud_b: np.ndarray,
               center: np.ndarray, band: int, penalty: float = 0.15):
    """Путь (i, j) с j ∈ [center[i] − band, center[i] + band]."""
    m = B.shape[0]
    c = np.clip(np.round(center), 0, m - 1).astype(np.int64)
    c = np.maximum.accumulate(c)
    pi, pj = _dtw_band(np.ascontiguousarray(A, np.float32), np.ascontiguousarray(B, np.float32),
                       loud_a.astype(np.bool_), loud_b.astype(np.bool_), c, int(band),
                       np.float32(penalty))
    if len(pi) == 0:
        raise RuntimeError("DTW не нашёл путь — дорожки слишком разные или полоса слишком узкая.")
    return pi, pj


def path_to_map(pi: np.ndarray, pj: np.ndarray, n: int) -> np.ndarray:
    """Средняя позиция j для каждой строки i (неубывающая)."""
    sums = np.bincount(pi, weights=pj.astype(np.float64), minlength=n)
    cnt = np.bincount(pi, minlength=n)
    m = sums / np.maximum(cnt, 1)
    return np.maximum.accumulate(m)


def path_cost(A, B, pi, pj) -> np.ndarray:
    d = A[pi] - B[pj]
    return np.sqrt(np.sum(d * d, axis=1))


def pool(F: np.ndarray, k: int) -> np.ndarray:
    n = F.shape[0] // k
    if n == 0:
        return F[:1].copy()
    return F[: n * k].reshape(n, k, -1).mean(axis=1)


def pool_mask(m: np.ndarray, k: int) -> np.ndarray:
    n = len(m) // k
    if n == 0:
        return m[:1].copy()
    return m[: n * k].reshape(n, k).any(axis=1)


def _level(A, B, la, lb, center_fine, factor, band_fine, penalty):
    """Один уровень: признаки усредняются по factor кадров, DTW в полосе ±band_fine
    (в точных кадрах) вокруг center_fine; результат — карта на точном шаге."""
    n = A.shape[0]
    if factor == 1:
        pi, pj = banded_dtw(A, B, la, lb, center_fine, band_fine, penalty)
        return path_to_map(pi, pj, n), pi, pj
    Ac, Bc = pool(A, factor), pool(B, factor)
    lac, lbc = pool_mask(la, factor), pool_mask(lb, factor)
    nc = Ac.shape[0]
    half = (factor - 1) / 2.0
    cc = (np.interp(np.arange(nc) * factor + half, np.arange(n), center_fine) - half) / factor
    pci, pcj = banded_dtw(Ac, Bc, lac, lbc, cc, max(2, band_fine // factor), penalty)
    mc = path_to_map(pci, pcj, nc)
    m = np.interp((np.arange(n) - half) / factor, np.arange(nc), mc) * factor + half
    return m, pci, pcj


def multiscale(A: np.ndarray, B: np.ndarray, la: np.ndarray, lb: np.ndarray,
               center_fine: np.ndarray, levels: list[tuple[int, int]], penalty: float = 0.15):
    """Многомасштабный DTW (как MrMsDTW): levels — [(шаг ×k, полоса в точных кадрах), …]
    от грубого к точному; каждый уровень ищет путь в полосе вокруг пути предыдущего.
    Возвращает (карта i→j на точном шаге, карта предпоследнего уровня, путь pi, pj)."""
    center = np.asarray(center_fine, np.float64)
    prev = center
    pi = pj = None
    for k, (factor, band) in enumerate(levels):
        m, pi, pj = _level(A, B, la, lb, center, factor, band, penalty)
        prev, center = center, m
    return center, prev, pi, pj
