"""
Растяжение dub по карте времени. Высота тона не меняется (кроме режима «Питч»).

Движки (auto — первый доступный):
  cli      — Rubber Band CLI, движок R3 (--fine) с --timemap: офлайн, лучшее качество;
  pylibrb  — та же библиотека Rubber Band R3 из Python, потоковый режим: скорость
             задаётся каждые 256 сэмплов с обратной связью (без накопленной ошибки).
             Нужен и для подтяжки нот (питч меняется плавно по кривой);
  wsola    — запасной вариант без Rubber Band: WSOLA с кроссфейдами 75 %. Хуже по
             качеству на сильных растяжениях — выводится предупреждение.

Карта задаётся опорными точками: t_out (время результата, с) → t_in (время dub, с).
Все каналы исходного dub обрабатываются вместе, в исходном качестве.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Callable, Optional

import numpy as np
import soundfile as sf
from numba import njit

log = logging.getLogger("vocalsync.warp")
Progress = Optional[Callable[[str, float], None]]
EDGE_FADE_S = 0.005


def rubberband_cli() -> Optional[str]:
    p = shutil.which("rubberband")
    if p:
        return p
    for c in ("/opt/homebrew/bin/rubberband", "/usr/local/bin/rubberband"):
        if os.path.isfile(c):
            return c
    return None


def has_pylibrb() -> bool:
    try:
        import pylibrb  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def available_engines() -> list[str]:
    out = []
    if rubberband_cli():
        out.append("cli")
    if has_pylibrb():
        out.append("pylibrb")
    out.append("wsola")
    return out


def pick_engine(engine: str, need_pitch: bool) -> str:
    av = available_engines()
    if need_pitch:
        return "pylibrb" if "pylibrb" in av else "none"
    if engine in av:
        return engine
    return av[0]


# ----------------------------------------------------------------------------
# Общая часть: какой кусок dub реально звучит в результате
# ----------------------------------------------------------------------------
def _crop(t_out: np.ndarray, t_in: np.ndarray, dur_in: float, dur_out: float):
    """Ограничить карту участком, где dub существует (0 ≤ t_in ≤ dur_in) и результат
    в пределах [0, dur_out]. Возвращает (t_out, t_in) с точными краями или None."""
    t_out = np.asarray(t_out, np.float64)
    t_in = np.asarray(t_in, np.float64)
    lo_out = max(0.0, float(np.interp(0.0, t_in, t_out)) if t_in[0] < 0 else float(t_out[0]))
    hi_out = min(dur_out, float(np.interp(dur_in, t_in, t_out)) if t_in[-1] > dur_in
                 else float(t_out[-1]))
    if hi_out - lo_out < 0.02:
        return None
    inside = (t_out > lo_out) & (t_out < hi_out)
    to = np.concatenate([[lo_out], t_out[inside], [hi_out]])
    ti = np.interp(to, t_out, t_in)
    ti = np.clip(ti, 0.0, dur_in)
    return to, ti


def _fade_edges(out: np.ndarray, a: int, b: int, sr: int, fade_in: bool, fade_out: bool):
    f = min(int(EDGE_FADE_S * sr), max(0, (b - a) // 4))
    if f < 2:
        return
    if fade_in:
        out[:, a:a + f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
    if fade_out:
        out[:, b - f:b] *= np.linspace(1.0, 0.0, f, dtype=np.float32)


def _strict(src: np.ndarray, dst: np.ndarray):
    """Строго возрастающие целые пары без (0, 0)."""
    keep_s, keep_d = [], []
    ls, ld = 0, 0
    for s, d in zip(src, dst):
        if s > ls and d > ld:
            keep_s.append(int(s))
            keep_d.append(int(d))
            ls, ld = s, d
    return np.asarray(keep_s, np.int64), np.asarray(keep_d, np.int64)


# ----------------------------------------------------------------------------
# Rubber Band CLI (R3, --timemap)
# ----------------------------------------------------------------------------
def _render_cli(x: np.ndarray, sr: int, src: np.ndarray, dst: np.ndarray, n_out: int) -> np.ndarray:
    exe = rubberband_cli()
    if not exe:
        raise RuntimeError("Rubber Band CLI не найден")
    with tempfile.TemporaryDirectory(prefix="vocalsync_") as tmp:
        fin, fout, fmap = (os.path.join(tmp, n) for n in ("in.wav", "out.wav", "map.txt"))
        sf.write(fin, x.T, sr, subtype="FLOAT")
        s, d = _strict(src, dst)
        k = (s < x.shape[1]) & (d < n_out)
        with open(fmap, "w") as f:
            for a, b in zip(s[k], d[k]):
                f.write(f"{a} {b}\n")
        cmd = [exe, "-3", "-q", "--timemap", fmap, "-D", f"{n_out / sr:.6f}", fin, fout]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(fout):
            raise RuntimeError("Rubber Band CLI: " + (r.stderr or r.stdout).strip()[-400:])
        y, _ = sf.read(fout, always_2d=True, dtype="float32")
    y = y.T
    if y.shape[1] >= n_out:
        return np.ascontiguousarray(y[:, :n_out])
    return np.pad(y, ((0, 0), (0, n_out - y.shape[1])))


def _render_cli_parallel(x: np.ndarray, sr: int, src: np.ndarray, dst: np.ndarray, n_out: int,
                         cuts: list[int], progress: Progress = None) -> np.ndarray:
    """R3 однопоточный, поэтому длинный файл режется в паузах (cuts — сэмплы результата
    в тишине) на куски, которые растягиваются параллельно отдельными процессами. У каждого
    куска запас по 0,5 с с обеих сторон, стык — кроссфейд 20 мс в середине паузы."""
    from concurrent.futures import ThreadPoolExecutor
    workers = max(1, min(6, os.cpu_count() or 2))
    piece = max(15.0 * sr, n_out / (2 * workers))
    bounds = [0]
    cand = sorted(c for c in cuts if 0.5 * sr < c < n_out - 0.5 * sr)
    while cand:
        want = bounds[-1] + piece
        if want > n_out - 0.6 * piece:
            break
        c = min(cand, key=lambda v: abs(v - want))
        if c <= bounds[-1] + 5 * sr:
            cand = [v for v in cand if v > bounds[-1] + 5 * sr]
            continue
        bounds.append(int(c))
        cand = [v for v in cand if v > c + 5 * sr]
    bounds.append(n_out)
    if len(bounds) <= 2:
        return _render_cli(x, sr, src, dst, n_out)
    M = int(0.5 * sr)
    F = int(0.02 * sr)
    srcf, dstf = src.astype(np.float64), dst.astype(np.float64)
    jobs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        oa, ob = max(0, a - M), min(n_out, b + M)
        ia = int(np.floor(np.interp(oa, dstf, srcf)))
        ib = int(np.ceil(np.interp(ob, dstf, srcf)))
        ib = min(ib, x.shape[1])
        k = (dst > oa) & (dst < ob)
        s_sub = np.concatenate([np.round(np.interp(dst[k], dstf, srcf)).astype(np.int64) - ia,
                                [ib - ia]])
        d_sub = np.concatenate([dst[k] - oa, [ob - oa]])
        jobs.append((a, b, oa, ob, ia, ib, s_sub, d_sub))
    out = np.zeros((x.shape[0], n_out), np.float32)
    done = [0]

    def run(j):
        a, b, oa, ob, ia, ib, s_sub, d_sub = j
        y = _render_cli(np.ascontiguousarray(x[:, ia:ib]), sr, s_sub, d_sub, ob - oa)
        done[0] += 1
        if progress:
            progress("Растягиваю (Rubber Band R3, параллельно)", done[0] / len(jobs))
        return y

    with ThreadPoolExecutor(max_workers=workers) as ex:
        ys = list(ex.map(run, jobs))
    for idx, ((a, b, oa, ob, *_r), y) in enumerate(zip(jobs, ys)):
        w = np.ones(ob - oa, np.float32)
        if idx > 0:                                   # вход: 0 до a−F/2, кривая, 1
            lo = a - F // 2 - oa
            w[:max(0, lo)] = 0.0
            t = (np.arange(F) + 0.5) / F
            seg = slice(max(0, lo), max(0, lo) + F)
            w[seg] = (0.5 - 0.5 * np.cos(np.pi * t))[: w[seg].shape[0]]
        if idx < len(jobs) - 1:
            hi = b - F // 2 - oa
            t = (np.arange(F) + 0.5) / F
            seg = slice(hi, hi + F)
            w[seg] *= (0.5 + 0.5 * np.cos(np.pi * t))[: w[seg].shape[0]]
            w[hi + F:] = 0.0
        out[:, oa:ob] += y * w[None, :]
    return out


# ----------------------------------------------------------------------------
# pylibrb (R3, потоковый режим с обратной связью) — из VoxMatch
# ----------------------------------------------------------------------------
def _render_pylibrb(x: np.ndarray, sr: int, src: np.ndarray, dst: np.ndarray, n_out: int,
                    pitch: Optional[Callable[[float], float]], formants: bool,
                    progress: Progress = None) -> np.ndarray:
    import pylibrb as rb
    O = rb.Option
    opts = O.ENGINE_FINER | O.CHANNELS_TOGETHER | O.WINDOW_STANDARD | O.PROCESS_REALTIME
    opts |= O.FORMANT_PRESERVED if formants else O.FORMANT_SHIFTED
    opts |= O.PitchHighConsistency
    ch, L = x.shape
    src_f, dst_f = src.astype(np.float64), dst.astype(np.float64)

    def T(a):                     # где в результате должен прозвучать сэмпл a dub
        return float(np.interp(a, src_f, dst_f))

    blk = 256
    st = rb.RubberBandStretcher(sr, ch, int(opts), 1.0, 1.0)
    st.set_max_process_size(blk * 4)
    st.time_ratio = max(0.2, min(5.0, T(blk) / blk))
    st.pitch_scale = pitch(0.0) if pitch else 1.0
    pad = st.get_preferred_start_pad()
    delay = st.get_start_delay()
    lat = int(1.75 * delay)
    pz = np.zeros((ch, pad), np.float32)
    for a in range(0, pad, blk):
        st.process(np.ascontiguousarray(pz[:, a:a + blk]), final=False)
    need = n_out + delay
    chunks, got = [], 0

    def pull():
        nonlocal got
        av = st.available()
        if av > 0:
            c = st.retrieve(av)
            chunks.append(c)
            got += c.shape[1]

    pull()
    sched, a, step = 0.0, 0, 0
    while a < L:
        b = min(L, a + blk)
        r = max(0.2, min(5.0, (T(b) - sched) / (b - a)))
        st.time_ratio = r
        if pitch:
            st.pitch_scale = pitch(T(a) - lat)
        st.process(np.ascontiguousarray(x[:, a:b]), final=False)
        sched += r * (b - a)
        a = b
        pull()
        step += 1
        if progress and step % 400 == 0:
            progress("Растягиваю (Rubber Band R3)", a / L)
    tail = np.zeros((ch, blk), np.float32)
    guard = 0
    while got < need and guard < 4000:
        guard += 1
        st.process(tail, final=False)
        pull()
    y = np.concatenate(chunks, axis=1) if chunks else np.zeros((ch, 0), np.float32)
    y = y[:, delay:delay + n_out]
    if y.shape[1] < n_out:
        y = np.pad(y, ((0, 0), (0, n_out - y.shape[1])))
    return y


# ----------------------------------------------------------------------------
# WSOLA — запасной вариант (из VoxMatch, сдвиги общие для всех каналов)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _wsola_choose(x, centers, N, Hs, tol):
    half = N // 2
    nx = x.shape[0]
    best_c = centers.copy()
    prev = -10 ** 9
    for k in range(centers.shape[0]):
        want = centers[k]
        best = want
        if prev > -10 ** 8:
            nat = prev + Hs
            if abs(want - nat) <= 1:
                best = nat
            else:
                bestv = -1e30
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
                    d += 4
                c0 = best
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
        best_c[k] = best
    return best_c


@njit(cache=True)
def _ola(x, best, N, Hs, n_out):
    half = N // 2
    nx = x.shape[0]
    win = np.empty(N)
    for i in range(N):
        win[i] = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / N)
    size = best.shape[0] * Hs + N + 1
    out = np.zeros(size)
    norm = np.zeros(size)
    for k in range(best.shape[0]):
        o = k * Hs
        for i in range(N):
            ix = best[k] - half + i
            if 0 <= ix < nx:
                out[o + i] += x[ix] * win[i]
            norm[o + i] += win[i]
    res = np.zeros(n_out)
    for i in range(n_out):
        j = i + half
        if j < size and norm[j] > 1e-6:
            res[i] = out[j] / norm[j]
    return res


def _render_wsola(x: np.ndarray, sr: int, src: np.ndarray, dst: np.ndarray, n_out: int) -> np.ndarray:
    N = 1024 if sr > 32000 else 512
    Hs = N // 4
    nf = int(np.ceil(n_out / Hs)) + 1
    centers = np.round(np.interp(np.arange(nf) * Hs, dst.astype(np.float64),
                                 src.astype(np.float64))).astype(np.int64)
    mono = x.mean(axis=0).astype(np.float64)
    best = _wsola_choose(mono, centers, N, Hs, N // 4)
    return np.vstack([_ola(x[c].astype(np.float64), best, N, Hs, n_out) for c in range(x.shape[0])]
                     ).astype(np.float32)


# ----------------------------------------------------------------------------
def render(data: np.ndarray, sr: int, t_out: np.ndarray, t_in: np.ndarray, n_out: int,
           engine: str = "auto", pitch_curve: Optional[tuple] = None, formants: bool = True,
           progress: Progress = None, cuts_s: Optional[list] = None
           ) -> tuple[np.ndarray, str, list[str]]:
    """data — (каналы, сэмплы) dub в частоте sr. Возвращает (результат (каналы, n_out),
    использованный движок, предупреждения). pitch_curve — (кадры результата, с; отношения
    частот) или None. cuts_s — моменты результата в тишине, где можно резать для
    параллельной обработки. Где dub нет — тишина."""
    notes: list[str] = []
    ch, m = data.shape
    out = np.zeros((ch, n_out), np.float32)
    eng = pick_engine(engine, pitch_curve is not None)
    if eng == "none":
        notes.append("Подтяжка нот требует pylibrb — она пропущена, сделан только тайминг.")
        pitch_curve = None
        eng = pick_engine(engine, False)
    cr = _crop(t_out, t_in, m / sr, n_out / sr)
    if cr is None:
        notes.append("Дорожка dub не пересекается с эталоном — результат пустой.")
        return out, eng, notes
    to, ti = cr
    o0, o1 = int(round(to[0] * sr)), min(n_out, int(round(to[-1] * sr)))
    i0, i1 = int(np.ceil(ti[0] * sr)), min(m, int(np.floor(ti[-1] * sr)))
    if o1 - o0 < 64 or i1 - i0 < 64:
        notes.append("Слишком короткий общий участок — результат пустой.")
        return out, eng, notes
    x = np.ascontiguousarray(data[:, i0:i1])
    src = np.round((ti - ti[0]) * sr).astype(np.int64)
    dst = np.round((to - to[0]) * sr).astype(np.int64)
    src[-1], dst[-1] = i1 - i0, o1 - o0
    seg_n = o1 - o0

    pitch_fn = None
    if pitch_curve is not None:
        pt, pr = pitch_curve

        def pitch_fn(o, pt=pt, pr=pr, base=to[0]):
            return float(np.interp(base + o / sr, pt, pr))

    tried = [eng] + [e for e in available_engines() if e != eng]
    y = None
    for e in tried:
        try:
            if e == "cli":
                cuts = [int(round(c * sr)) - o0 for c in (cuts_s or [])]
                if cuts and seg_n > 30 * sr:
                    y = _render_cli_parallel(x, sr, src, dst, seg_n, cuts, progress)
                else:
                    y = _render_cli(x, sr, src, dst, seg_n)
            elif e == "pylibrb":
                y = _render_pylibrb(x, sr, src, dst, seg_n, pitch_fn, formants, progress)
            else:
                y = _render_wsola(x, sr, src, dst, seg_n)
            if e != eng:
                notes.append(f"Движок «{eng}» не сработал — использован «{e}».")
            eng = e
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("движок %s: %s", e, exc)
            notes.append(f"Движок «{e}»: ошибка ({exc}).")
    if y is None:
        raise RuntimeError("Ни один движок растяжения не сработал.")
    if eng == "wsola":
        notes.append("Rubber Band недоступен: растяжение сделано запасным методом WSOLA — "
                     "качество хуже на сильных растяжениях. Установите Rubber Band (см. README).")
    out[:, o0:o0 + y.shape[1]] = y[:, : n_out - o0]
    _fade_edges(out, o0, o0 + min(y.shape[1], n_out - o0), sr, fade_in=i0 > 0 or o0 > 0,
                fade_out=i1 < m or o1 < n_out)
    return out, eng, notes
