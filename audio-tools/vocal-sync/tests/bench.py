"""
Производительность: 4-минутный трек (стерео dub, 44,1 кГц), время и пиковая память.

    .venv/bin/python tests/bench.py [--seconds 240] [--engine auto]
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocalsync import pipeline  # noqa: E402
from vocalsync.config import Settings  # noqa: E402
import synth  # noqa: E402


def peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def child(gp: str, dp: str, engine: str, out: str):
    """Выравнивание в чистом процессе — чтобы пиковая память не включала генерацию."""
    t0 = time.time()
    res = pipeline.align(gp, dp, Settings(engine=engine))
    t_align = time.time() - t0
    t1 = time.time()
    pipeline.export(res, out)
    np.save(os.path.join(out, "pos.npy"), res.tm.pos)
    print(f"{t_align:.2f} {time.time() - t1:.2f} {peak_mb():.0f} {res.engine} "
          f"{int(res.y.shape[1])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--child", nargs=3, default=None)
    a = ap.parse_args()
    if a.child:
        child(a.child[0], a.child[1], a.engine, a.child[2])
        sys.exit(0)
    import subprocess
    p = synth.make_pair(a.seconds, 11, 2.1, 1.015)
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = os.path.join(tmp, "g.wav"), os.path.join(tmp, "d.wav")
        sf.write(gp, p.guide.T, p.sr)
        sf.write(dp, p.dub.T, p.sr)
        out = os.path.join(tmp, "out")
        os.makedirs(out)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--engine", a.engine,
                            "--child", gp, dp, out], capture_output=True, text=True)
        line = [x for x in r.stdout.strip().splitlines() if x][-1]
        t_align, t_export, mem, engine, n_out = line.split()
        pos = np.load(os.path.join(out, "pos.npy"))
    fm = 5.0
    t = np.arange(len(pos)) * fm / 1000.0
    v = p.voiced(t)
    err = np.abs(pos * fm / 1000.0 - p.warp(t))[v] * 1000.0
    print(f"Длительность: {a.seconds:.0f} с (стерео dub, {p.sr} Гц), движок {engine}, "
          f"CPU: {os.cpu_count()} потоков")
    print(f"Выравнивание: {float(t_align):.1f} с, экспорт (WAV, превью, отчёт, график): "
          f"{float(t_export):.1f} с")
    print(f"Пиковая память процесса выравнивания: {float(mem):.0f} МБ")
    print(f"Ошибка карты: медиана {np.median(err):.1f} мс, p95 {np.percentile(err, 95):.1f} мс")
    print(f"Длина выхода = длине эталона: {int(n_out) == p.guide.shape[1]}")
