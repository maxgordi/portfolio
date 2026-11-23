"""
Конвейер: эталон + dub → выровненный dub, карта времени, метрики, предупреждения.

    io → features → coarse (сдвиг + темп) → dtw (грубо → точно) → path → warp → report
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import coarse, dtw, features as ft, io as mio, path as mpath, warp
from .config import Settings

log = logging.getLogger("vocalsync")
Progress = Optional[Callable[[str, float], None]]
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Track:
    path: str
    data: np.ndarray             # (каналы, сэмплы) оригинал (фрагмент, если задан)
    sr: int
    an: ft.Analysis
    t0: float = 0.0

    @property
    def duration(self) -> float:
        return self.data.shape[1] / self.sr


@dataclass
class Result:
    guide_path: str
    dub_path: str
    settings: Settings
    sr: int                       # частота результата
    y: np.ndarray                 # (каналы, сэмплы) выровненный dub
    guide_y: np.ndarray           # (каналы, сэмплы) эталон (фрагмент) в той же частоте
    tm: mpath.TimeMap
    line: coarse.Line             # глобальная линия (по DTW, устойчивая регрессия)
    line_xcorr: coarse.Line       # первая оценка по корреляции огибающих
    engine: str
    t0: float                     # начало фрагмента эталона, с
    fm: float                     # мс на кадр карты
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)   # [{"t0", "t1", "kind", "text"}]
    notes: list = field(default_factory=list)
    seconds: float = 0.0
    kind: str = "унисон"
    interval_c: float = 0.0
    dissim: Optional[np.ndarray] = None
    corr_c: Optional[np.ndarray] = None
    out_dir: str = ""
    files: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    # совместимость с компом из VoxMatch
    @property
    def rate(self) -> np.ndarray:
        return self.tm.rate.astype(np.float32)

    @property
    def timing_dev_ms(self) -> np.ndarray:
        return ((self.tm.pos_dtw - self.tm.lin) * self.fm).astype(np.float32)

    @property
    def problem(self) -> np.ndarray:
        n = len(self.tm.pos)
        m = np.zeros(n, bool)
        for w in self.warnings:
            if w.get("t0") is None:
                continue
            a = int((w["t0"] - self.t0) * 1000.0 / self.fm)
            b = int((w["t1"] - self.t0) * 1000.0 / self.fm)
            m[max(0, a):min(n, max(a + 1, b))] = True
        return m

    @property
    def out_path(self) -> str:
        return self.files.get("dub_aligned", "")


# ----------------------------------------------------------------------------
# Опорные точки
# ----------------------------------------------------------------------------
def parse_time(s: str) -> float:
    """«12.5», «12,5», «1:02.5», «0:01:02,5» → секунды."""
    s = s.strip().replace(",", ".")
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        v = 0.0
        for p in parts:
            v = v * 60.0 + p
        return v
    return float(s)


def read_anchors(path: str) -> list[tuple[float, float]]:
    """CSV: «время в эталоне; время в dub» (секунды или м:сс.мс). Разделитель ; , или
    табуляция, строки с текстом (заголовок, комментарии #) пропускаются."""
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if ";" in line or "\t" in line:
                parts = re.split(r"[;\t]", line)
            else:
                parts = line.split(",") if line.count(",") == 1 else re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            try:
                out.append((parse_time(parts[0]), parse_time(parts[1])))
            except ValueError:
                continue
    return sorted(out)


# ----------------------------------------------------------------------------
# Сеанс: кэш анализа и путей DTW (окно пересчитывает только то, что изменилось)
# ----------------------------------------------------------------------------
class Session:
    def __init__(self):
        self._tracks: dict = {}
        self._paths: dict = {}
        self._f0: dict = {}

    def track(self, path: str, start=None, end=None, silence_db=35.0,
              progress: Progress = None) -> Track:
        try:
            st = os.stat(path)
            key = (os.path.abspath(path), st.st_mtime, st.st_size, start, end, silence_db)
        except OSError:
            key = (path, start, end, silence_db)
        t = self._tracks.get(key)
        if t is None:
            if progress:
                progress(f"Читаю {os.path.basename(path)}", 0.0)
            data, sr = mio.read_audio(path, start, end)
            if progress:
                progress(f"Анализирую {os.path.basename(path)}", 0.3)
            an = ft.analyze(mio.to_mono(data), sr, path, float(start or 0.0), silence_db)
            t = Track(path, data, sr, an, float(start or 0.0))
            if len(self._tracks) > 10:
                self._tracks.pop(next(iter(self._tracks)))
            self._tracks[key] = t
        return t

    def f0(self, t: Track) -> np.ndarray:
        key = (t.path, t.t0, t.an.n)
        if key not in self._f0:
            from .pitch import f0_track
            x = ft.preprocess(mio.to_mono(t.data), t.sr)
            self._f0[key] = ft._fit(f0_track(x), t.an.n)
        return self._f0[key]

    def align(self, guide_path: str, dub_path: str, s: Settings, progress: Progress = None,
              need_kind: bool = False) -> Result:
        return align(guide_path, dub_path, s, progress, self, need_kind)


# ----------------------------------------------------------------------------
def _sub(progress: Progress, lo: float, hi: float) -> Progress:
    return (lambda msg, f: progress(msg, lo + (hi - lo) * f)) if progress else None


def _frame_dissim(g: ft.Analysis, d: ft.Analysis, pos: np.ndarray) -> np.ndarray:
    """Непохожесть звука dub на эталон в каждом кадре (MFCC), 0…~3 (как в VoxMatch)."""
    k = np.clip(np.round(pos).astype(np.int64), 0, d.n - 1)
    dd = np.sqrt(np.mean((g.mfcc - d.mfcc[k]) ** 2, axis=1))
    dd = ft.smooth(dd, int(round(60.0 / g.fm)))
    inside = (pos >= 0) & (pos <= d.n - 1)
    return np.where(inside, dd, 3.0).astype(np.float32)


def _merge_regions(mask: np.ndarray, fm: float, min_ms: float, gap_ms: float = 200.0):
    segs = ft.segments(mask)
    out = []
    gap = gap_ms / fm
    for a, b in segs:
        if out and a - out[-1][1] < gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if (b - a) * fm >= min_ms]


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def diagnose(g: ft.Analysis, d: ft.Analysis, tm: mpath.TimeMap, dis: np.ndarray, t0: float,
             s: Settings) -> list[dict]:
    """Ненадёжные участки и расхождения структуры (с таймкодами эталона)."""
    fm = g.fm
    n = g.n
    out = []

    def add(a, b, kind, text):
        ta, tb = t0 + a * fm / 1000.0, t0 + b * fm / 1000.0
        out.append({"t0": round(ta, 3), "t1": round(tb, 3), "kind": kind,
                    "text": f"{_fmt(ta)}–{_fmt(tb)}: {text}"})

    k = np.clip(np.round(tm.pos_dtw).astype(np.int64), 0, d.n - 1)
    inside = (tm.pos_dtw >= 0) & (tm.pos_dtw <= d.n - 1)
    dub_loud = d.loud[k] & inside
    # 1. в эталоне поют, а в dub на этом месте тишина — пропущена фраза
    miss = ft.smooth((g.loud & ~dub_loud).astype(np.float64), int(round(100 / fm))) > 0.8
    for a, b in _merge_regions(miss, fm, 600.0):
        add(a, b, "structure", "в эталоне есть пение, в dub на этом месте тишина — "
                               "возможно, фраза пропущена или спета сильно не там")
    # 2. наклон сырого пути: dub «стоит» или «бежит» дольше 0,5 с — лишний/пропущенный кусок
    win = int(round(500.0 / fm))
    if n > 2 * win:
        sl = (tm.pos_dtw[win:] - tm.pos_dtw[:-win]) / win
        slope = np.concatenate([np.full(win // 2, sl[0]), sl, np.full(n - len(sl) - win // 2, sl[-1])])
        weird = (slope < 0.4) | (slope > 2.5)
        loud_any = g.loud | dub_loud
        for a, b in _merge_regions(weird & loud_any, fm, 500.0):
            add(a, b, "structure",
                "структура расходится: в dub здесь лишний кусок или, наоборот, пропуск "
                f"(наклон пути {np.median(slope[a:b]):.2f})")
    # 3. звук dub не похож на эталон
    bad = g.loud & (dis > 1.6)
    bad = ft.smooth(bad.astype(np.float64), int(round(80.0 / fm))) > 0.5
    for a, b in _merge_regions(bad, fm, 400.0):
        add(a, b, "unreliable", "dub плохо похож на эталон — проверьте на слух (другие "
                                "слова, шум, артефакты сепаратора)")
    # 4. сработали ограничения растяжения
    sung = g.loud & dub_loud & (g.db > -25.0)
    for a, b in _merge_regions(tm.clipped & sung, fm, 150.0):
        err = float(np.max(np.abs(tm.pos[a:b] - tm.pos_target[a:b]))) * fm
        add(a, b, "limit", f"упёрлись в предел растяжения ({s.stretch_min}–{s.stretch_max}×), "
                           f"остаток до {err:.0f} мс — нужна опорная точка или шире пределы")
    out.sort(key=lambda w: w["t0"])
    return out


def onset_error(g_on: np.ndarray, d_on: np.ndarray, fm: float, window_ms: float = 100.0):
    """Для каждой атаки эталона — ближайшая атака dub. Возвращает (средняя |Δ| мс,
    медиана |Δ| мс, доля атак, нашедших пару в окне)."""
    if len(g_on) == 0 or len(d_on) == 0:
        return None, None, 0.0
    d_on = np.sort(d_on)
    j = np.clip(np.searchsorted(d_on, g_on), 1, len(d_on) - 1)
    near = np.minimum(np.abs(d_on[j - 1] - g_on), np.abs(d_on[j] - g_on)) * fm
    ok = near <= window_ms
    if not ok.any():
        return None, None, 0.0
    return float(near[ok].mean()), float(np.median(near[ok])), float(ok.mean())


def _onsets_of(y_mono: np.ndarray, sr: int) -> np.ndarray:
    an = ft.analyze(y_mono, sr)
    return an.onsets


def align(guide_path: str, dub_path: str, s: Settings, progress: Progress = None,
          session: Optional[Session] = None, need_kind: bool = False) -> Result:
    t_start = time.time()
    session = session or Session()
    notes: list[str] = []
    G = session.track(guide_path, s.start, s.end, s.silence_db, _sub(progress, 0.0, 0.12))
    D = session.track(dub_path, None, None, s.silence_db, _sub(progress, 0.12, 0.25))
    g, d = G.an, D.an
    fm = g.fm
    n = g.n
    warns: list[dict] = []

    # --- тональность
    shift, sim_k, sim_0 = ft.key_shift(g, d)
    key_info = {"shift_semitones": shift, "similarity": round(sim_k, 3),
                "similarity_same_key": round(sim_0, 3)}
    if shift != 0 and sim_k - sim_0 > 0.15 and sim_k > 0.5:
        hint = ("признаки chroma уже выключены" if s.w_chroma <= 0 else
                "рекомендуется режим без chroma: пресет «Другая тональность» или --no-chroma")
        warns.append({"t0": None, "t1": None, "kind": "key",
                      "text": f"Тональность dub отличается от эталона примерно на {shift:+d} полут. "
                              f"— {hint}."})

    # --- признаки
    w = ft.Weights.from_settings(s)
    A, B = ft.stack(g, w), ft.stack(d, w)

    # --- грубо: сдвиг + темп
    if progress:
        progress("Ищу сдвиг начала и темп", 0.27)
    line_x = coarse.estimate_line(g, d)
    if line_x.score < 0.15:
        notes.append(f"Огибающие похожи слабо (корреляция {line_x.score:.2f}) — сдвиг найден "
                     "неуверенно, проверьте результат.")
    # время dub в абсолютном времени его файла; эталон — от начала фрагмента
    anchors_s = list(s.anchors)
    if s.anchors_path:
        anchors_s += read_anchors(s.anchors_path)
    anchors_f = [((ga - G.t0) * 1000.0 / fm, da * 1000.0 / fm) for ga, da in anchors_s]
    anchors_f = [(a, b) for a, b in anchors_f if 0 <= a < n]
    if s.mode == "anchors" and not anchors_f:
        notes.append("Режим «опорные точки», но точек нет (или все вне фрагмента) — работаю как «авто».")
    center = line_x.frames(n, fm)
    if anchors_f and s.mode == "anchors":
        ga = np.array([a for a, _b in anchors_f])
        da = np.array([b for _a, b in anchors_f])
        if len(ga) == 1:
            center = line_x.tempo * (np.arange(n) - ga[0]) + da[0]
        else:
            center = np.interp(np.arange(n), ga, da)
            t_l = (da[1] - da[0]) / max(1e-9, ga[1] - ga[0])
            t_r = (da[-1] - da[-2]) / max(1e-9, ga[-1] - ga[-2])
            left, right = np.arange(n) < ga[0], np.arange(n) > ga[-1]
            center[left] = da[0] + t_l * (np.arange(n)[left] - ga[0])
            center[right] = da[-1] + t_r * (np.arange(n)[right] - ga[-1])

    # --- DTW
    if progress:
        progress("Сопоставляю слоги (DTW)", 0.32)
    # ширина самого грубого поиска: не меньше coarse_band_s и 4 % длины (накопленный
    # «уход» отдельных фраз от общей линии на длинном треке)
    wide_s = max(s.coarse_band_s, 0.12 * G.duration)
    if s.mode == "anchors" and anchors_f:
        wide_s = min(wide_s, max(1.5, 0.03 * G.duration))
    fr = 1000.0 / fm
    for attempt in range(3):
        levels = [(16, int(wide_s * fr)), (4, int(max(1.0, 0.15 * wide_s) * fr)),
                  (1, int(s.band_ms / fm))]
        dmap, coarse_map, pi, pj = dtw.multiscale(A, B, g.loud, d.loud, center, levels)
        # путь прижался к краю самой широкой полосы — значит, она узка: расширяем
        edge = float(np.max(np.abs(coarse_map - center)[g.loud])) if g.loud.any() else 0.0
        if edge < 0.8 * wide_s * fr or wide_s >= G.duration:
            break
        wide_s *= 2.0
        notes.append(f"Тайминг сильно «гуляет» — поиск расширен до ±{wide_s:.0f} с.")
    # устойчивая глобальная линия по точному пути
    line = coarse.robust_line(dmap, g.loud, fm)
    if not (0.8 < line.tempo < 1.25):
        notes.append(f"Темп по DTW вышел странным ({line.tempo:.3f}) — беру оценку по корреляции.")
        line = line_x
    lin = line.frames(n, fm)

    # --- карта
    if progress:
        progress("Строю карту времени", 0.45)
    kd = np.clip(np.round(dmap).astype(np.int64), 0, d.n - 1)
    # карте верим только там, где поют оба: хвосты реверба эталона и затухания dub
    # (тише −25 дБ от пика) DTW сопоставляет плохо — там линейная интерполяция
    trust = (g.loud & d.loud[kd] & (g.db > -25.0) & (d.db[kd] > -25.0)
             & (dmap >= 0) & (dmap <= d.n - 1))
    tm = mpath.build(dmap, lin, g.loud, g.onsets, s, fm, anchors_f, trust)
    dis = _frame_dissim(g, d, tm.pos_dtw)
    warns += diagnose(g, d, tm, dis, G.t0, s)

    # --- высота (необязательно)
    out_sr = int(s.out_sr or G.sr)
    kind, interval, corr_c, pitch_curve = "унисон", 0.0, None, None
    if s.pitch > 0 or need_kind:
        if progress:
            progress("Слушаю высоту тона", 0.5)
        from . import pitch as mp
        f0g, f0d_all = session.f0(G), session.f0(D)
        k = np.clip(np.round(tm.pos_dtw).astype(np.int64), 0, d.n - 1)
        inside = (tm.pos_dtw >= 0) & (tm.pos_dtw <= d.n - 1)
        f0d = np.where(inside, f0d_all[k], 0.0)
        kind, interval, octave = mp.classify(f0g, f0d)
        if kind == "гармония" and g.loud.any() and float(np.median(dis[g.loud])) > 1.1:
            kind = "унисон"
        if kind == "октава":
            notes.append(f"Dub спет на {abs(octave)} окт. {'ниже' if octave < 0 else 'выше'} — "
                         "октава сохранена.")
        if s.pitch > 0 and not (kind == "гармония" and s.harmony_auto):
            kp = np.clip(np.round(tm.pos).astype(np.int64), 0, d.n - 1)
            f0p = np.where((tm.pos >= 0) & (tm.pos <= d.n - 1), f0d_all[kp], 0.0)
            ratio, pb, pa, _o = mp.pitch_curve(f0g, f0p, s.pitch / 100.0, fm)
            corr_c = (1200.0 * np.log2(ratio)).astype(np.float32)
            pitch_curve = (np.arange(n) * fm / 1000.0, ratio)
            notes.append(f"Ноты подтянуты к эталону на {s.pitch} %: ошибка {pb:.0f} → {pa:.0f} центов.")
        elif s.pitch > 0:
            notes.append("Бэк в гармонию — ноты не трогаются, только тайминг.")

    # --- растяжение
    if progress:
        progress("Растягиваю dub (Rubber Band R3)", 0.55)
    dub_data = mio.resample(D.data, D.sr, out_sr)
    guide_y = mio.resample(G.data, G.sr, out_sr)
    n_out = guide_y.shape[1]
    keys = tm.keys
    t_out = keys * fm / 1000.0
    t_in = tm.pos[keys] * fm / 1000.0
    # места для разрезания при параллельном растяжении: середины пауз ≥ 250 мс, где
    # молчат и эталон, и dub
    kq = np.clip(np.round(tm.pos).astype(np.int64), 0, d.n - 1)
    quiet = ~d.loud[kq]
    cuts = [(a + b) / 2 * fm / 1000.0 for a, b in ft.segments(quiet) if (b - a) * fm >= 250.0]
    y, engine, rn = warp.render(dub_data, out_sr, t_out, t_in, n_out, s.engine, pitch_curve,
                                s.formants, _sub(progress, 0.55, 0.88), cuts)
    notes += rn
    y = mio.fit_length(y, n_out)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if s.out_format.upper() != "FLOAT" and peak > 0.999:
        y *= 0.999 / peak
        notes.append(f"Пик результата {20 * np.log10(peak):+.1f} dBFS — громкость снижена до "
                     "−0,01 dBFS, чтобы не было перегруза (24 бит). Во float32 не снижается.")

    # --- метрики
    if progress:
        progress("Считаю метрики", 0.9)
    loud = g.loud
    ident = np.arange(n) + G.t0 * 1000.0 / fm          # dub «как есть» под фрагментом
    shift_only = np.arange(n) + line.offset * 1000.0 / fm

    def resid(p):
        return float(np.median(np.abs(tm.pos_dtw - p)[loud])) * fm

    st = tm.stretch
    stl = st[loud] if loud.any() else st
    w100 = max(1, int(round(100.0 / fm)))
    dk = np.abs(st[w100:] - st[:-w100]) if n > w100 else np.zeros(1)
    d_on_file = d.onsets
    g_on = g.onsets
    rawpos = d_on_file - G.t0 * 1000.0 / fm
    on_before = onset_error(g_on, rawpos, fm)
    on_shift = onset_error(g_on, d_on_file - line.offset * 1000.0 / fm, fm)
    try:
        y_on = _onsets_of(mio.to_mono(y), out_sr)
        on_after = onset_error(g_on, y_on, fm)
    except Exception:  # noqa: BLE001
        on_after = (None, None, 0.0)

    def r1(v):
        return None if v is None else round(float(v), 1)

    m = {
        # сдвиг dub относительно эталона в одном и том же абсолютном времени
        "offset_s": round(line.offset - G.t0, 4),
        "offset_xcorr_s": round(line_x.offset - G.t0, 4),
        "tempo_ratio": round(line.tempo, 5),
        "tempo_percent": round((1.0 / line.tempo - 1.0) * 100.0, 3),
        "tempo_xcorr": round(line_x.tempo, 5),
        "stretch_mean": round(float(np.mean(stl)), 4),
        "stretch_min": round(float(np.min(stl)), 4),
        "stretch_max": round(float(np.max(stl)), 4),
        "stretch_over_20pct_share": round(float(np.mean(np.abs(np.log(stl)) > np.log(1.2))), 4),
        "stretch_change_max_100ms": round(float(np.max(dk)) if len(dk) else 0.0, 3),
        "dtw_resid_ms": {"as_is": r1(resid(ident)), "shift_only": r1(resid(shift_only)),
                         "global": r1(resid(tm.lin)), "after": r1(resid(tm.pos))},
        "onset_error_ms": {
            "as_is": {"mean": r1(on_before[0]), "median": r1(on_before[1]), "matched": round(on_before[2], 3)},
            "shift_only": {"mean": r1(on_shift[0]), "median": r1(on_shift[1]), "matched": round(on_shift[2], 3)},
            "after": {"mean": r1(on_after[0]), "median": r1(on_after[1]), "matched": round(on_after[2], 3)},
        },
        "key": key_info,
        "guide_duration_s": round(G.duration, 3),
        "dub_duration_s": round(D.duration, 3),
        "output_samples": int(y.shape[1]),
        "guide_samples_out_sr": int(n_out),
        "out_sr": out_sr,
        "channels": int(y.shape[0]),
        "keyframes": int(len(keys)),
        "anchors_used": len(anchors_f),
        "line_score": round(line_x.score, 3),
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-9)), 2),
    }
    res = Result(guide_path=guide_path, dub_path=dub_path, settings=s, sr=out_sr, y=y,
                 guide_y=guide_y, tm=tm, line=line, line_xcorr=line_x, engine=engine, t0=G.t0,
                 fm=fm, metrics=m, warnings=warns, notes=notes, kind=kind, interval_c=interval,
                 dissim=dis, corr_c=corr_c)
    res.seconds = time.time() - t_start
    m["seconds"] = round(res.seconds, 2)
    res.extra = {"coarse_map": coarse_map, "guide_env": g.db, "dub_db": d.db, "dub_n": d.n}
    if progress:
        progress("Готово", 1.0)
    return res


def default_out_root() -> str:
    return os.path.join(APP_DIR, "output")


def export(res: Result, out_root: str = "", plot: bool = True) -> dict:
    """Пишет пакет файлов в out_root/<имя>_<дата-время>/ и возвращает пути."""
    from . import report
    stem = os.path.splitext(os.path.basename(res.dub_path))[0]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = out_root or res.settings.out_dir or default_out_root()
    out_dir = os.path.join(root, f"{stem}_{stamp}")
    k = 2
    while os.path.exists(out_dir):
        out_dir = os.path.join(root, f"{stem}_{stamp}-{k}")
        k += 1
    os.makedirs(out_dir)
    res.out_dir = out_dir
    res.files = report.write_all(res, out_dir, plot=plot)
    return res.files
