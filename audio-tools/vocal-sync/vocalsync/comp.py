"""
Комп из лучших фраз нескольких дублей (перенесено из VoxMatch Простой).

• Эталон режется на фразы по паузам (граница — в самой тихой точке паузы).
• Каждая фраза каждого дубля получает оценку 0–100:
    – чисто ли спеты ноты (сравнение с «общей мелодией» всех дублей, ДО коррекции);
    – те ли слова и все ли на месте (похожесть звука на эталон);
    – сильно ли пришлось тянуть/сжимать (риск артефактов);
    – перегруз.
• Комп по умолчанию — лучшая фраза из каждого дубля; выбор можно менять.
• Склейка — короткий равномощный кроссфейд в тишине между фразами.
"""
from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from .features import segments as _segments, smooth as _smooth


@dataclass
class GuideView:
    """Что компу нужно знать об эталоне: кадры анализа (5 мс) и звук результата."""
    fm: float                 # мс на кадр
    n: int                    # кадров
    loud: np.ndarray          # (n,) bool
    db: np.ndarray            # (n,) дБ относительно 95-го перцентиля
    sr: int                   # частота звука
    data: np.ndarray          # (каналы, сэмплы) эталон в частоте результата

    @property
    def hop(self) -> float:   # сэмплов на кадр (может быть дробным: 44,1 кГц → 220,5)
        return self.sr * self.fm / 1000.0


def _s(frame, hop) -> int:
    return int(round(frame * hop))

REF_KEY = "__ref__"


# ----------------------------------------------------------------------------
# Фразы
# ----------------------------------------------------------------------------
def phrases(ref: GuideView, min_gap_ms=200.0, min_len_ms=350.0, max_len_s=5.0):
    """Фразы эталона. Возвращает (tiles, sung): tiles — куски, покрывающие весь файл
    (границы в самой тихой точке паузы), sung — где внутри куска реально поют.
    Порог «тишины» — от уровня самой записи (подклад/наушники в паузах не мешают).
    Слишком длинные фразы делятся по самому тихому месту между словами."""
    fm = ref.fm
    n = ref.n
    db = ref.db.astype(np.float64)                          # уровень относительно пика, дБ
    sm = _smooth(db, int(round(30.0 / fm)))
    p10, p90 = np.percentile(sm, 10), np.percentile(sm, 90)
    thr = float(np.clip(p10 + 0.45 * (p90 - p10), -30.0, -12.0))
    loud = sm > thr
    segs = [list(x) for x in _segments(loud)]
    merged = []
    gap = int(round(min_gap_ms / fm))
    for a, b in segs:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    mn = int(round(min_len_ms / fm))
    out = []
    for a, b in merged:                                      # короткие — к соседу
        if out and (b - a < mn or out[-1][1] - out[-1][0] < mn) and a - out[-1][1] < 4 * gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    if not out:
        return [(0, n)], [(0, n)]
    # длинные — делим по самому тихому месту (между словами)
    mx = int(round(max_len_s * 1000.0 / fm))
    edge = int(round(600.0 / fm))
    sm2 = _smooth(db, int(round(60.0 / fm)))
    changed = True
    while changed:
        changed = False
        res = []
        for a, b in out:
            if b - a > mx and b - a > 3 * edge:
                c = a + edge + int(np.argmin(sm2[a + edge:b - edge]))
                res.extend([[a, c], [c, b]])
                changed = True
            else:
                res.append([a, b])
        out = res
    cuts = [0]
    for (a0, b0), (a1, b1) in zip(out[:-1], out[1:]):
        if a1 <= b0:                                          # деление внутри фразы
            cuts.append(b0)
            continue
        seg = sm2[b0:a1]
        c = b0 + int(np.argmin(seg)) if len(seg) > 2 else (b0 + a1) // 2
        cuts.append(c)
    cuts.append(n)
    tiles = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]
    return tiles, [tuple(x) for x in out]


# ----------------------------------------------------------------------------
# Оценки
# ----------------------------------------------------------------------------
def _cents(f):
    c = np.full(len(f), np.nan)
    v = f > 0
    c[v] = 1200.0 * np.log2(f[v] / 440.0)
    return c


def consensus(ref, results: dict):
    """Цель по нотам — мелодия эталона (дубли под неё и подтягиваются)."""
    return _cents(ref.f0)


def consensus_median(ref, results: dict):
    """(запасной вариант) медиана высоты всех унисонных дублей + эталона."""
    rows = [_cents(ref.f0)]
    for r in results.values():
        if r.kind != "унисон" or r.f0_al is None:
            continue
        rows.append(_cents(r.f0_al.astype(np.float64)))
    M = np.vstack(rows)
    with np.errstate(all="ignore"):
        cnt = np.sum(np.isfinite(M), axis=0)
        med = np.nanmedian(np.where(np.isfinite(M), M, np.nan), axis=0) if M.shape[0] else None
    med = np.where(cnt >= max(2, (M.shape[0] + 1) // 2), med, np.nan)
    return med


def _score_one(ref_db, take_db, dis, rate, clip, loud_ref, present, a, b, is_ref, dev=None):
    """Оценка куска [a, b) одного дубля: (оценка 0–100, пояснение).
    Главное — тайминг (насколько раньше/позже эталона спето) и ВОЛНА: насколько
    форма громкости дубля после выравнивания совпадает с волной эталона
    (те же атаки, ударения, окончания). Высота нот в оценке не участвует.
    Плюс штрафы: другие слова, пропуски, сильное растяжение, перегруз."""
    sl = slice(a, b)
    lr = loud_ref[sl]
    if lr.sum() < 3:
        return 100.0, ""
    info, warn = [], []
    pen = 0.0
    if is_ref:
        return 100.0, "эталон"
    # тайминг
    if dev is not None:
        dv = np.abs(dev[sl][lr])
        terr = float(np.median(dv)) if len(dv) else 0.0
        pen += float(np.clip((terr - 25.0) * 0.3, 0, 35))
        info.append(f"тайм ±{terr:.0f} мс")
    # волна
    r = ref_db[sl]
    t = take_db[sl]
    m = lr | present[sl]
    if m.sum() >= 8:
        rr, tt = np.maximum(r[m], -60.0), np.maximum(t[m], -60.0)
        if np.std(rr) > 1e-6 and np.std(tt) > 1e-6:
            corr = float(np.corrcoef(rr, tt)[0, 1])
        else:
            corr = 0.0
        off = float(np.median(tt[lr[m]] - rr[lr[m]])) if lr[m].any() else 0.0
        shape = float(np.median(np.abs(tt - rr - off)))
        pen += float(np.clip((0.92 - corr) * 90.0, 0, 45))
        pen += float(np.clip((shape - 3.0) * 2.5, 0, 15))
        info.append(f"волна {max(0.0, corr) * 100:.0f}%")
    dd = dis[sl][lr]
    bad = float(np.mean(dd > 1.5))
    p = float(np.clip((bad - 0.1) * 80.0, 0, 40))
    if p > 6:
        warn.append("другие слова/звук")
    pen += p
    rrate = rate[sl][lr]
    far = float(np.mean(np.abs(np.log2(np.maximum(rrate, 1e-3))) > np.log2(1.4)))
    p = float(np.clip(far * 30.0, 0, 15))
    if p > 6:
        warn.append("сильно растянуто")
    pen += p
    miss = float(np.mean(~present[sl][lr]))
    p = float(np.clip((miss - 0.05) * 90.0, 0, 50))
    if p > 6:
        warn.append("не всё спето")
    pen += p
    if clip[sl].any():
        warn.append("перегруз")
        pen += 20.0
    return float(np.clip(100.0 - pen, 0, 100)), " · ".join(info + warn)


def _frame_db(y, hop, n):
    mono = y.mean(axis=0) if y.ndim > 1 else y
    edges = np.round(np.arange(n + 1) * hop).astype(np.int64)
    edges = np.clip(edges, 0, len(mono))
    sq = np.concatenate([[0.0], np.cumsum(mono.astype(np.float64) ** 2)])
    cnt = np.maximum(1, edges[1:] - edges[:-1])
    rms = np.sqrt((sq[edges[1:]] - sq[edges[:-1]]) / cnt + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    db = _smooth(db, 4)                                       # ~20 мс
    return db - np.percentile(db, 97)


def score_all(ref: GuideView, results: dict, tiles, sung):
    """Оценки фраз: {ключ: [(оценка, пояснение), ...]}. Эталон — под ключом REF_KEY."""
    loud_ref = ref.loud
    hop = ref.hop
    n = ref.n

    def clip_frames(y):
        x = np.max(np.abs(y), axis=0) if y.ndim > 1 else np.abs(y)
        idx = np.flatnonzero(x > 0.985)
        cf = np.zeros(n, bool)
        if len(idx):
            cf[np.clip((idx / hop).astype(np.int64), 0, n - 1)] = True
        return cf

    out = {}
    ref_db = _frame_db(ref.data, hop, n)
    clip_r = clip_frames(ref.data)
    ones = np.ones(n, np.float32)
    out[REF_KEY] = [_score_one(ref_db, ref_db, np.zeros(n), ones, clip_r, loud_ref,
                               np.ones(n, bool), a, b, True) for a, b in tiles]
    for key, r in results.items():
        y = r.y
        db = _frame_db(y, hop, n)
        present = db > -30.0
        present = _smooth(present.astype(np.float64), int(round(80.0 / ref.fm))) > 0.2
        dev = getattr(r, "timing_dev_ms", None)
        out[key] = [_score_one(ref_db, db, r.dissim, r.rate, clip_frames(y), loud_ref, present,
                               a, b, False, dev) for a, b in tiles]
    return out


def default_choices(scores: dict, candidates: list, n_tiles: int, prefer_ref_bonus=0.0):
    """Лучший дубль для каждой фразы (эталон в финал не берётся — он только ориентир)."""
    ch = []
    for i in range(n_tiles):
        best, bv = candidates[0], -1e9
        for k in candidates:
            v = scores[k][i][0] + (prefer_ref_bonus if k == REF_KEY else 0.0)
            if v > bv:
                best, bv = k, v
        ch.append(best)
    return ch


def ranked(scores: dict, candidates: list, i: int):
    """Кандидаты для фразы i от лучшего к худшему."""
    return sorted(candidates, key=lambda k: -scores[k][i][0])


# ----------------------------------------------------------------------------
# Сборка
# ----------------------------------------------------------------------------
def build_comp(ref: GuideView, audio: dict, tiles, choices, fade_ms=60.0):
    """audio: {ключ: (каналы, сэмплы)} — все уже на шкале эталона (одной длины).
    Склейка как кроссфейд в Cubase (X): плавная S-кривая, сумма громкостей всегда 1,
    центр — в самой тихой точке между фразами. Если с обеих сторон один и тот же
    дубль — никакой склейки, звук идёт сплошняком."""
    hop, sr = ref.hop, ref.sr
    n_s = ref.data.shape[1]
    ch = max(a.shape[0] for a in audio.values())
    out = np.zeros((ch, n_s), np.float32)
    bounds = [_s(a, hop) for a, _b in tiles[1:]]
    # «ступенька» 0/1 для каждого куска и сглаживание каждой границы
    owner = np.zeros(n_s, np.int32)
    for i, (a, b) in enumerate(tiles):
        owner[_s(a, hop): min(n_s, _s(b, hop))] = i
    if tiles:
        owner[_s(tiles[-1][1], hop):] = len(tiles) - 1
    key_of = [choices[i] for i in range(len(tiles))]
    keys = sorted(set(key_of), key=str)
    W = {k: np.zeros(n_s, np.float32) for k in keys}
    for i in range(len(tiles)):
        W[key_of[i]][owner == i] = 1.0
    f0 = max(16, int(sr * fade_ms / 1000.0))
    for j, c in enumerate(bounds):
        ka, kb = key_of[j], key_of[j + 1]
        if ka == kb:
            continue
        la = (tiles[j][1] - tiles[j][0]) * hop
        lb = (tiles[j + 1][1] - tiles[j + 1][0]) * hop
        f = int(min(f0, 0.8 * la, 0.8 * lb))
        if f < 16:
            continue
        lo = max(0, c - f // 2)
        hi = min(n_s, lo + f)
        t = (np.arange(hi - lo, dtype=np.float32) + 0.5) / (hi - lo)
        up = 0.5 - 0.5 * np.cos(np.pi * t)                    # S-кривая, up + down = 1
        W[kb][lo:hi] = up
        W[ka][lo:hi] = 1.0 - up
    for k in keys:
        src = audio[k]
        if src.shape[0] < ch:
            src = np.repeat(src[:1], ch, axis=0)
        L = min(n_s, src.shape[1])
        out[:, :L] += src[:, :L] * W[k][None, :L]
    return out


def build_from_pieces(ref: GuideView, tiles, pieces, fade_ms=60.0):
    """Финал из кусков: pieces[i] = (начальный сэмпл, звук (каналы, сэмплы)) для фразы i.
    Каждый кусок длиннее своей фразы (запас по краям) — стыки делаются кроссфейдом
    S-кривой (как X в Cubase), середина — в самой тихой точке между фразами."""
    hop, sr = ref.hop, ref.sr
    n_s = ref.data.shape[1]
    ch = max(p[1].shape[0] for p in pieces)
    out = np.zeros((ch, n_s), np.float32)
    f0 = max(16, int(sr * fade_ms / 1000.0))
    nt = len(tiles)
    for i, (a, b) in enumerate(tiles):
        s0, s1 = _s(a, hop), min(n_s, _s(b, hop))
        if i == nt - 1:
            s1 = n_s
        if i == 0:
            s0 = 0
        st, data = pieces[i]
        if data.shape[0] < ch:
            data = np.repeat(data[:1], ch, axis=0)
        # окно куска: 1 внутри, S-кривые на стыках
        lo_f = hi_f = 0
        if i > 0:
            la = (tiles[i - 1][1] - tiles[i - 1][0]) * hop
            lo_f = int(min(f0, 0.8 * la, 0.8 * (s1 - s0)))
        if i < nt - 1:
            lb = (tiles[i + 1][1] - tiles[i + 1][0]) * hop
            hi_f = int(min(f0, 0.8 * lb, 0.8 * (s1 - s0)))
        lo = max(0, s0 - lo_f // 2)
        hi = min(n_s, s1 + hi_f // 2 + (hi_f % 2))
        w = np.ones(hi - lo, np.float32)
        if lo_f >= 16:
            t = (np.arange(lo_f, dtype=np.float32) + 0.5) / lo_f
            k = min(lo_f, len(w))
            w[:k] = (0.5 - 0.5 * np.cos(np.pi * t))[:k]
        if hi_f >= 16:
            t = (np.arange(hi_f, dtype=np.float32) + 0.5) / hi_f
            r0 = max(0, (s1 - hi_f // 2) - lo)
            k = min(hi_f, len(w) - r0)
            w[r0:r0 + k] = (0.5 + 0.5 * np.cos(np.pi * t))[:k]
        # кусок на шкале эталона
        seg = np.zeros((ch, hi - lo), np.float32)
        a0, a1 = max(lo, st), min(hi, st + data.shape[1])
        if a1 > a0:
            seg[:, a0 - lo:a1 - lo] = data[:, a0 - st:a1 - st]
        out[:, lo:hi] += seg * w[None, :]
    return out
