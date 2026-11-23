"""Порядок фраз: соседи из разных песен, не больше двух одной подачи подряд, разные тональности соседей."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("voiceset")
W_SONG, W_DELIV, W_KEY = 100, 10, 1


def _viol(rows, seq, i) -> tuple[bool, bool, bool]:
    """Нарушения при добавлении rows[i] в конец seq: (песня, подача×3, тональность)."""
    if not seq:
        return False, False, False
    p, r = rows[seq[-1]], rows[i]
    song = r["song_idx"] == p["song_idx"]
    key = bool(r["key"] and p["key"] and r["key"] == p["key"])
    deliv = len(seq) >= 2 and r["delivery"] == p["delivery"] == rows[seq[-2]]["delivery"]
    return song, deliv, key


def _build(rows, rng):
    rem = list(range(len(rows)))
    seq: list[int] = []
    score = [0, 0, 0]
    while rem:
        info = {i: _viol(rows, seq, i) for i in rem}
        # приоритет: песня > подача > тональность; берём кандидатов с наименьшим «штрафом»
        best = min(W_SONG * v[0] + W_DELIV * v[1] + W_KEY * v[2] for v in info.values())
        cands = [i for i in rem if W_SONG * info[i][0] + W_DELIV * info[i][1] + W_KEY * info[i][2] == best]
        left: dict = {}
        for i in rem:
            left[rows[i]["song_idx"]] = left.get(rows[i]["song_idx"], 0) + 1
        mx = max(left[rows[i]["song_idx"]] for i in cands)
        top = [i for i in cands if left[rows[i]["song_idx"]] == mx]
        pool = top if rng.random() < 0.6 else cands
        pick = pool[int(rng.integers(len(pool)))]
        for k in range(3):
            score[k] += int(info[pick][k])
        seq.append(pick)
        rem.remove(pick)
    return seq, score


def order_phrases(df: pd.DataFrame, ids: list[int], seed: int, tries: int = 400) -> tuple[list[int], list[str]]:
    rows = [df.loc[i].to_dict() for i in ids]
    rng = np.random.default_rng(seed + 1)
    best, best_s, best_c = None, None, None
    for _ in range(tries):
        seq, sc = _build(rows, rng)
        cost = W_SONG * sc[0] + W_DELIV * sc[1] + W_KEY * sc[2]
        if best is None or cost < best_s:
            best, best_s, best_c = seq, cost, sc
        if cost == 0:
            break
    warn = []
    if best_c[0]:
        warn.append(f"В порядке {best_c[0]} раз подряд идут фразы одной песни (не удалось развести).")
    if best_c[1]:
        warn.append(f"В порядке {best_c[1]} раз подряд идут 3 фразы одной подачи (при такой доле «обычной» подачи это неизбежно).")
    if best_c[2]:
        warn.append(f"В порядке {best_c[2]} раз соседние фразы в одной тональности.")
    return [ids[i] for i in best], warn
