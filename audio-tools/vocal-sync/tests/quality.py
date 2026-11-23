"""
Проверка качества на синтетике (и, по желанию, на реальном вокале).

    .venv/bin/python tests/quality.py            # синтетика, таблица метрик
    .venv/bin/python tests/quality.py --real ~/Music/vocal.wav

Ошибка = |карта программы(t) − известная карта(t)| в звучащих местах эталона, мс.
Критерии: медиана < 20 мс, 95-й перцентиль < 50 мс.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocalsync.config import Settings  # noqa: E402
from vocalsync import pipeline  # noqa: E402
import synth  # noqa: E402

CASES = [
    # имя, сид, сдвиг, темп, локальные ±
    ("сдвиг+1.7 темп+2% лок±5-10%", 1, 1.7, 1.02, (0.05, 0.10)),
    ("сдвиг-0.6 темп-3% лок±5-10%", 2, -0.6, 0.97, (0.05, 0.10)),
    ("сдвиг+3.0 темп+1% лок±5-10%", 3, 3.0, 1.01, (0.05, 0.10)),
    ("сдвиг+0.5 темп-1% лок±5-10%", 4, 0.5, 0.99, (0.05, 0.10)),
    ("сдвиг+2.2 темп+3% без локальных", 5, 2.2, 1.03, (0.0, 0.0)),
]


def map_error(res, true_fn, voiced_fn) -> dict:
    tm, fm = res.tm, res.fm
    n = len(tm.pos)
    t = res.t0 + np.arange(n) * fm / 1000.0
    v = voiced_fn(t)
    est = tm.pos * fm / 1000.0
    err = np.abs(est - true_fn(t))[v] * 1000.0
    return {"median": float(np.median(err)), "p95": float(np.percentile(err, 95)),
            "max": float(np.max(err))}


def render_error(res, true_fn, guide_events_t, sr) -> dict:
    """Насколько сам звук результата совпадает с картой: атаки результата против
    атак эталона (синтетические события известны точно)."""
    from vocalsync import features as ft
    an = ft.analyze(res.y.mean(axis=0), res.sr)
    on = an.onsets * an.fm / 1000.0
    g = np.asarray(guide_events_t)
    j = np.clip(np.searchsorted(on, g), 1, len(on) - 1)
    near = np.minimum(np.abs(on[j - 1] - g), np.abs(on[j] - g)) * 1000.0
    ok = near < 100
    return {"median": float(np.median(near[ok])) if ok.any() else None,
            "matched": float(ok.mean())}


def clicks(y: np.ndarray, sr: int) -> float:
    """Самый резкий скачок между соседними сэмплами относительно типичного (щелчки)."""
    d = np.abs(np.diff(y, axis=1)).max(axis=0)
    typical = np.percentile(d[d > 0], 99.9) if (d > 0).any() else 1.0
    return float(d.max() / (typical + 1e-12))


def run_case(name, seed, offset, tempo, local, duration=60.0, engine="auto", out=None):
    p = synth.make_pair(duration, seed, offset, tempo, local)
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = os.path.join(tmp, "guide.wav"), os.path.join(tmp, "dub.wav")
        sf.write(gp, p.guide.T, p.sr)
        sf.write(dp, p.dub.T, p.sr)
        s = Settings(engine=engine)
        t0 = time.time()
        res = pipeline.align(gp, dp, s)
        dt = time.time() - t0
        e = map_error(res, p.warp, p.voiced)
        r = render_error(res, p.warp, [ev.t + ev.cons for ev in p.events], p.sr)
        if out:
            pipeline.export(res, out)
    ok_len = res.y.shape[1] == p.guide.shape[1]
    return {"name": name, "map": e, "render": r, "len_ok": ok_len, "channels": res.y.shape[0],
            "engine": res.engine, "seconds": dt, "clicks": clicks(res.y, res.sr),
            "offset_est": res.line.offset, "tempo_est": res.line.tempo,
            "warnings": len(res.warnings)}


def real_case(path: str, seed=7, offset=1.3, tempo=1.015, local=(0.05, 0.10), seconds=60.0):
    """Реальный вокал: dub = тот же файл, растянутый Rubber Band по известной карте."""
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    a = int(min(30.0, max(0.0, data.shape[0] / sr - seconds) / 2) * sr)
    x = data[a:a + int(seconds * sr)]
    rng = np.random.default_rng(seed)
    w = synth.make_warp(seconds, rng, offset, tempo, local)
    # dub(t_d) = guide(w⁻¹(t_d)): рендер: выход (dub) t_d ↔ вход (guide) t_g
    pre = int(max(0.0, offset) * sr)
    with tempfile.TemporaryDirectory() as tmp:
        gi, go, mp = (os.path.join(tmp, n) for n in ("g.wav", "d.wav", "m.txt"))
        sf.write(gi, x, sr, subtype="FLOAT")
        tg = np.arange(0.05, seconds - 0.05, 0.05)
        td = w(tg) - w(0.0)
        with open(mp, "w") as f:
            for s_, d_ in zip(tg, td):
                f.write(f"{int(s_ * sr)} {int(d_ * sr)}\n")
        dur = (w(seconds) - w(0.0))
        subprocess.run(["rubberband", "-3", "-q", "--timemap", mp, "-D", f"{dur:.6f}", gi, go],
                       check=True)
        dd, _ = sf.read(go, always_2d=True, dtype="float32")
        dub = np.concatenate([np.zeros((pre, dd.shape[1]), np.float32), dd])
        # dub_t = w(g) − w(0) + offset_pre
        true = (lambda t, w=w, pre=pre: w(t) - w(0.0) + pre / sr)
        dp = os.path.join(tmp, "dub.wav")
        gp = os.path.join(tmp, "guide.wav")
        sf.write(dp, dub, sr)
        sf.write(gp, x, sr)
        t0 = time.time()
        res = pipeline.align(gp, dp, Settings())
        dt = time.time() - t0
    from vocalsync import features as ft
    an = ft.analyze(x.mean(axis=1), sr)
    voiced = lambda t, an=an: an.loud[np.clip((t * 1000 / an.fm).astype(int), 0, an.n - 1)]  # noqa: E731
    e = map_error(res, true, voiced)
    return {"name": "реальный вокал " + os.path.basename(path), "map": e, "len_ok":
            res.y.shape[1] == x.shape[0], "seconds": dt, "engine": res.engine,
            "clicks": clicks(res.y, res.sr), "offset_est": res.line.offset,
            "tempo_est": res.line.tempo, "warnings": len(res.warnings), "render": {}, "channels": res.y.shape[0]}


def show(rows):
    print(f"{'случай':40s} {'медиана':>8s} {'p95':>7s} {'макс':>7s} {'звук мед':>9s} "
          f"{'длина':>5s} {'щелч':>5s} {'сек':>5s}  движок")
    for r in rows:
        m = r["map"]
        rm = r["render"].get("median")
        print(f"{r['name']:40s} {m['median']:8.1f} {m['p95']:7.1f} {m['max']:7.0f} "
              f"{(f'{rm:.1f}' if rm is not None else '—'):>9s} {'ok' if r['len_ok'] else 'НЕТ':>5s} "
              f"{r['clicks']:5.1f} {r['seconds']:5.1f}  {r['engine']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--out", default="")
    ap.add_argument("--only", type=int, default=-1)
    a = ap.parse_args()
    rows = []
    for i, c in enumerate(CASES):
        if a.only >= 0 and i != a.only:
            continue
        rows.append(run_case(*c, engine=a.engine, out=a.out or None))
        show(rows[-1:])
    if a.real:
        rows.append(real_case(os.path.expanduser(a.real)))
    print()
    show(rows)
    allm = np.array([r["map"]["median"] for r in rows])
    allp = np.array([r["map"]["p95"] for r in rows])
    print(f"\nХудшая медиана {allm.max():.1f} мс (критерий < 20), худший p95 {allp.max():.1f} мс (< 50)")
