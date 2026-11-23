"""Отбор фраз для Suno: покрытие диапазона + квоты подач + ограничение по песням (жадно + локальный поиск)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SunoCfg
from .notes import midi_to_name

log = logging.getLogger("voiceset")
DELIVERIES = ["normal", "power", "breathy"]


@dataclass
class Selection:
    ids: list[int]
    range_lo: int
    range_hi: int
    targets: list[int]                 # полутоны, которые можно покрыть
    unavailable: list[int]             # полутоны диапазона, которых нет ни в одной чистой фразе
    weights: dict[int, float]
    covered: list[int]
    shares: dict[str, float]
    song_shares: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    reasons: dict[int, str] = field(default_factory=dict)
    objective: float = 0.0


def weighted_range(df: pd.DataFrame, pctl: list[float]) -> tuple[int, int]:
    """Полутоны от p_lo до p_hi перцентиля f0 всей записи (по времени звучания)."""
    acc: dict[int, float] = {}
    for sm, ms in zip(df["sem_midi"], df["sem_ms"]):
        for m, t in zip(sm, ms):
            acc[int(m)] = acc.get(int(m), 0) + t
    if not acc:
        raise ValueError("Ни в одной фразе не найдена высота — проверьте запись.")
    ks = sorted(acc)
    cum = np.cumsum([acc[k] for k in ks]) / sum(acc.values()) * 100
    lo = ks[int(np.searchsorted(cum, pctl[0]))]
    hi = ks[min(len(ks) - 1, int(np.searchsorted(cum, pctl[1])))]
    return int(lo), int(hi)


class _Problem:
    def __init__(self, df: pd.DataFrame, cfg: SunoCfg, pool: pd.DataFrame, lo: int, hi: int, rng):
        self.cfg = cfg
        self.pool = pool
        self.ids = pool["id"].to_numpy()
        self.n = len(pool)
        self.dur = pool["dur"].to_numpy(float)
        songs = sorted(df["song_idx"].unique())
        self.song_names = {int(r["song_idx"]): r["song"] for _, r in df.drop_duplicates("song_idx").iterrows()}
        self.song = pool["song_idx"].to_numpy(int)
        self.n_songs_total = len(songs)
        self.n_songs_pool = len(set(self.song))
        self.deliv = np.array([DELIVERIES.index(d) if d in DELIVERIES else 0 for d in pool["delivery"]])
        self.q = pool["quality"].to_numpy(float) + 1e-6 * rng.random(self.n)
        s = pool["similarity"]
        rank = s.rank(pct=True).fillna(0.5)
        self.sim = rank.to_numpy(float)
        # карта диапазона
        avail = set()
        for sm in pool["sem_midi"]:
            avail.update(int(m) for m in sm)
        rng_all = list(range(lo, hi + 1))
        self.targets = [m for m in rng_all if m in avail]
        self.unavailable = [m for m in rng_all if m not in avail]
        edge = cfg.edge_semitones
        self.w = np.array([cfg.edge_weight if (m - lo < edge or hi - m < edge) else 1.0 for m in self.targets])
        self.wsum = max(self.w.sum(), 1e-9)
        idx = {m: j for j, m in enumerate(self.targets)}
        self.C = np.zeros((self.n, len(self.targets)), bool)
        for i, sm in enumerate(pool["sem_midi"]):
            for m in sm:
                if int(m) in idx:
                    self.C[i, idx[int(m)]] = True
        self.quota = np.array([cfg.quotas.get(d, 0.0) for d in DELIVERIES])
        a = cfg.assemble
        self.g_lo, self.g_hi = a.gap_range_ms[0] / 1000, a.gap_range_ms[1] / 1000
        self.L = cfg.length_s
        self.gap0 = a.gap_ms / 1000
        self.forced: set[int] = set()

    # --- целевая функция ---
    def feas_pen(self, D: float, n: int) -> float:
        if n == 0:
            return 1.0
        hi = self.L - (n - 1) * self.g_lo
        lo = self.L - (n - 1) * self.g_hi
        if n == 1:
            hi, lo = self.L, self.L - 0.0
        if D > hi:
            return D - hi
        if D < lo:
            return lo - D
        return 0.0

    def gap_pen(self, D: float, n: int) -> float:
        """Лучше, когда нужный зазор близок к стандартному gap_ms (а не 0 или максимум)."""
        if n < 2:
            return 0.0
        gap = (self.L - D) / (n - 1)
        return abs(gap - self.gap0) / max(self.g_hi - self.g_lo, 1e-3)

    def J(self, sel) -> float:
        sel = np.asarray(sel, int)
        c, W = self.cfg, self.cfg.weights
        if len(sel) == 0:
            return -1e9
        D = self.dur[sel].sum()
        cov = (self.w * self.C[sel].any(0)).sum() / self.wsum
        sh = np.bincount(self.deliv[sel], weights=self.dur[sel], minlength=3) / D
        dev = np.abs(sh - self.quota)
        quota = 1 - np.clip(dev.sum() / 0.3, 0, 2)
        qpen = np.maximum(0, dev - c.quota_tolerance * 0.8).sum()
        sd = np.bincount(self.song[sel], weights=self.dur[sel]) / D
        cap = c.max_song_share
        over = np.maximum(0, sd - cap).sum() / cap
        used = int((sd > 0).sum())
        need = min(c.min_songs, self.n_songs_pool)
        short = max(0, need - used)
        return float(W.coverage * cov + W.quota * quota + W.quality * self.q[sel].mean()
                     + W.similarity * self.sim[sel].mean() - W.song_overuse * over
                     - 10 * qpen - 2 * short - 0.5 * self.feas_pen(D, len(sel))
                     - self.gap_pen(D, len(sel)))


def build_selection(df: pd.DataFrame, cfg: SunoCfg, seed: int, ban: set[int] | None = None,
                    keep: set[int] | None = None) -> Selection:
    ban, keep = set(ban or ()), set(keep or ())
    rng = np.random.default_rng(seed)
    ok = df[(df["reject"] == "") & (df["dur"] >= cfg.pool_min_s) & (df["dur"] <= cfg.pool_max_s)]
    ok = ok[~ok["id"].isin(ban)]
    extra = df[df["id"].isin(keep) & ~df["id"].isin(ok["id"]) & ~df["id"].isin(ban)]
    pool = pd.concat([ok, extra]).sort_values("id")
    if len(pool) == 0:
        raise ValueError("Нет подходящих чистых фраз для набора.")
    lo, hi = weighted_range(df, cfg.range_pctl)
    P = _Problem(df, cfg, pool, lo, hi, rng)
    forced = [int(np.where(P.ids == i)[0][0]) for i in sorted(keep) if i in set(P.ids)]
    P.forced = set(forced)
    warn: list[str] = []

    sel: list[int] = list(forced)
    D = float(P.dur[sel].sum()) if sel else 0.0
    gap0 = cfg.assemble.gap_ms / 1000
    W = cfg.weights
    cov_now = P.C[sel].any(0) if sel else np.zeros(len(P.targets), bool)
    sh_dur = np.zeros(3)
    sg_dur: dict[int, float] = {}
    for i in sel:
        sh_dur[P.deliv[i]] += P.dur[i]
        sg_dur[P.song[i]] = sg_dur.get(P.song[i], 0) + P.dur[i]
    # --- жадный отбор ---
    while D + max(len(sel) - 1, 0) * gap0 < P.L - 3.0:
        best, bs = None, -1e18
        for i in range(P.n):
            if i in sel:
                continue
            new = (P.w * (P.C[i] & ~cov_now)).sum() / P.wsum
            d = P.deliv[i]
            tgt = P.quota[d]
            deficit = np.clip((tgt - sh_dur[d] / P.L) / tgt, -1, 1) if tgt > 0 else -1.0
            over = max(0.0, (sg_dur.get(P.song[i], 0) + P.dur[i]) / P.L - cfg.max_song_share) / cfg.max_song_share
            s = (W.coverage * new * 3 + W.quota * deficit + W.quality * P.q[i]
                 + W.similarity * P.sim[i] - W.song_overuse * over * 2)
            if s > bs:
                best, bs = i, s
        if best is None:
            break
        sel.append(best)
        D += P.dur[best]
        cov_now = cov_now | P.C[best]
        sh_dur[P.deliv[best]] += P.dur[best]
        sg_dur[P.song[best]] = sg_dur.get(P.song[best], 0) + P.dur[best]

    # --- локальный поиск: добавить/убрать, замены 1↔1 и 1↔2 ---
    cur = P.J(sel)
    for _ in range(cfg.local_search_passes):
        improved = False
        rest = [i for i in range(P.n) if i not in sel]
        cand: tuple[float, list[int]] | None = None

        def consider(new):
            nonlocal cand
            if new and len(new) >= 1:
                v = P.J(new)
                if v > cur + 1e-9 and (cand is None or v > cand[0] + 1e-12):
                    cand = (v, list(new))
        for c in rest:
            consider(sel + [c])
        for pos, i in enumerate(sel):
            if i in P.forced:
                continue
            base = sel[:pos] + sel[pos + 1:]
            consider(base)
            for c in rest:
                consider(base + [c])
        top = sorted(rest, key=lambda c: -(P.w * (P.C[c] & ~P.C[sel].any(0))).sum())[:cfg.pair_candidates]
        for pos, i in enumerate(sel):
            if i in P.forced:
                continue
            base = sel[:pos] + sel[pos + 1:]
            for a in range(len(top)):
                for b in range(a + 1, len(top)):
                    consider(base + [top[a], top[b]])
        if cand is not None:
            cur, sel = cand[0], cand[1]
            improved = True
        if not improved:
            break
        # повторять, пока есть улучшение в рамках числа проходов
    while True:  # дожимаем простые улучшения (добавить/заменить) до сходимости, ограниченно
        best = None
        rest = [i for i in range(P.n) if i not in sel]
        for pos, i in enumerate(sel):
            if i in P.forced:
                continue
            base = sel[:pos] + sel[pos + 1:]
            for c in rest:
                v = P.J(base + [c])
                if v > cur + 1e-9 and (best is None or v > best[0]):
                    best = (v, base + [c])
        for c in rest:
            v = P.J(sel + [c])
            if v > cur + 1e-9 and (best is None or v > best[0]):
                best = (v, sel + [c])
        if best is None:
            break
        cur, sel = best
        if len(sel) > 80:
            break

    # --- итог и предупреждения ---
    sel_ids = [int(P.ids[i]) for i in sel]
    Dsel = float(P.dur[sel].sum())
    n = len(sel)
    if P.feas_pen(Dsel, n) > 1e-6:
        warn.append(f"Длительность выбранных фраз {Dsel:.1f} с не укладывается в {P.L:.0f} с даже при зазорах "
                    f"{cfg.assemble.gap_range_ms[0]:.0f}–{cfg.assemble.gap_range_ms[1]:.0f} мс — записи не хватает.")
    shares = dict(zip(DELIVERIES, (np.bincount(P.deliv[sel], weights=P.dur[sel], minlength=3) / Dsel).tolist()))
    for d, q in zip(DELIVERIES, P.quota):
        if abs(shares[d] - q) > cfg.quota_tolerance + 1e-9:
            warn.append(f"Доля подачи «{d}» {shares[d]*100:.0f}% при цели {q*100:.0f}% — в записи не хватает чистых фраз этой подачи.")
    sd = np.bincount(P.song[sel], weights=P.dur[sel]) / Dsel
    song_shares = {P.song_names[int(k)]: float(v) for k, v in enumerate(sd) if v > 0}
    for k, v in enumerate(sd):
        if v > cfg.max_song_share + 1e-9:
            warn.append(f"Из песни «{P.song_names[k]}» {v*100:.0f}% набора (лимит {cfg.max_song_share*100:.0f}%).")
    used = sum(1 for v in sd if v > 0)
    if used < min(cfg.min_songs, P.n_songs_pool):
        warn.append(f"В наборе только {used} песен(и) вместо {cfg.min_songs}.")
    covered = [P.targets[j] for j in np.where(P.C[sel].any(0))[0]]
    frac = len(covered) / max(len(P.targets), 1)
    if frac < 0.9:
        warn.append(f"Покрыто {frac*100:.0f}% нот диапазона (нужно ≥ 90%).")
    if P.unavailable:
        names = ", ".join(midi_to_name(m) for m in P.unavailable)
        warn.append(f"Нет чистых фраз с нотами: {names} — допойте эти ноты.")
    # причины для лога
    reasons: dict[int, str] = {}
    sel_set = set(sel)
    covered_set = set(covered)
    for i in range(P.n):
        pid = int(P.ids[i])
        if i in sel_set:
            gain = [midi_to_name(P.targets[j]) for j in np.where(P.C[i])[0]]
            reasons[pid] = f"ВЫБРАНА: {df.loc[pid, 'delivery']}, песня «{df.loc[pid, 'song']}», ноты {', '.join(gain[:6])}{'…' if len(gain) > 6 else ''}"
        else:
            new = [midi_to_name(P.targets[j]) for j in np.where(P.C[i] & ~P.C[sel].any(0))[0]]
            reasons[pid] = "НЕ_ВЫБРАНА: " + (f"дала бы новые ноты {', '.join(new)}, но не вошла по квотам/длине/лимиту песни" if new
                                              else "ноты уже покрыты, подача/песня заполнены (код REDUNDANT)")
    for pid in ban:
        reasons[int(pid)] = "ИСКЛЮЧЕНА вручную (review.yaml / --exclude)"
    return Selection(ids=sel_ids, range_lo=lo, range_hi=hi, targets=P.targets, unavailable=P.unavailable,
                     weights=dict(zip(P.targets, P.w.tolist())), covered=covered, shares=shares,
                     song_shares=song_shares, warnings=warn, reasons=reasons, objective=float(cur))
