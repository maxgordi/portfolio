"""Выходные файлы: dub_aligned.wav, превью, timemap.csv, report.txt/json, alignment.png."""
from __future__ import annotations

import json
import logging
import os

import numpy as np

from . import APP_NAME, __version__, io as mio

log = logging.getLogger("vocalsync.report")

# палитра графика (проверенная по CVD пара: слоты 1–2 эталонной палитры)
C_GUIDE = "#2a78d6"
C_DUB = "#eb6834"
C_WARN = "#d03b3b"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def write_timemap(res, path: str):
    """Опорные точки карты: время в эталоне; время в dub (секунды, абсолютные).
    Формат совпадает с CSV опорных точек — файл можно поправить и подать в --anchors."""
    tm, fm = res.tm, res.fm
    st = tm.stretch
    with open(path, "w", encoding="utf-8") as f:
        f.write("guide_s;dub_s;stretch\n")
        for k in tm.keys:
            f.write(f"{res.t0 + k * fm / 1000.0:.4f};{tm.pos[k] * fm / 1000.0:.4f};{st[k]:.4f}\n")


def _onset_line(d: dict) -> str:
    if d.get("mean") is None:
        return "нет совпадений (сдвиг больше окна 100 мс)"
    return f"средняя {d['mean']:.1f} мс, медиана {d['median']:.1f} мс, совпало {d['matched'] * 100:.0f} % атак"


def report_text(res) -> str:
    m = res.metrics
    s = res.settings
    L = []
    L.append(f"{APP_NAME} {__version__} — отчёт о выравнивании")
    L.append("=" * 64)
    L.append(f"Эталон: {res.guide_path}")
    L.append(f"Dub:    {res.dub_path}")
    if s.start is not None or s.end is not None:
        L.append(f"Фрагмент эталона: {_fmt(res.t0)} – {_fmt(res.t0 + m['guide_duration_s'])} "
                 f"(в DAW ставить файл на позицию {_fmt(res.t0)})")
    L.append(f"Режим: {s.mode}, tightness {s.tightness:.0f} %, растяжение {s.stretch_min}–{s.stretch_max}×, "
             f"движок: {res.engine}")
    L.append("")
    L.append("ГЛОБАЛЬНО")
    off = m["offset_s"]
    side = "позже" if off > 0 else "раньше"
    L.append(f"  Сдвиг начала: {off:+.3f} с — в dub тот же момент звучит на {abs(off):.3f} с {side}"
             f" (по корреляции огибающих: {m['offset_xcorr_s']:+.3f} с)")
    tp = m["tempo_percent"]
    L.append(f"  Темп dub: {'медленнее' if tp < 0 else 'быстрее'} эталона на {abs(tp):.2f} % "
             f"(коэффициент времени {m['tempo_ratio']:.5f})")
    k = m["key"]
    sh = k["shift_semitones"]
    key_txt = "та же" if sh == 0 else f"dub выше на {sh:+d} полут. (по средней chroma)"
    L.append(f"  Тональность: {key_txt}, сходство профилей {k['similarity']:.2f}")
    L.append("")
    L.append("РАСТЯЖЕНИЕ (длительность результата / исходника, по звучащим местам)")
    L.append(f"  среднее {m['stretch_mean']:.3f}, мин {m['stretch_min']:.3f}, макс {m['stretch_max']:.3f}")
    L.append(f"  сильнее ±20 %: {m['stretch_over_20pct_share'] * 100:.1f} % времени; "
             f"наибольшее изменение за 100 мс: {m['stretch_change_max_100ms']:.3f}")
    L.append("")
    L.append("КАЧЕСТВО")
    r = m["dtw_resid_ms"]
    L.append("  Расхождение с найденным путём DTW (медиана, мс):")
    L.append(f"    как есть {r['as_is']} → только сдвиг {r['shift_only']} → сдвиг+темп {r['global']} → "
             f"после {r['after']}")
    o = m["onset_error_ms"]
    L.append("  Ошибка совпадения атак (онсетов) с эталоном, окно ±100 мс:")
    L.append(f"    как есть:      {_onset_line(o['as_is'])}")
    L.append(f"    только сдвиг:  {_onset_line(o['shift_only'])}")
    L.append(f"    после:         {_onset_line(o['after'])}")
    L.append("")
    L.append(f"Длина результата: {m['output_samples']} сэмплов = длине эталона "
             f"({m['guide_samples_out_sr']}), {m['out_sr']} Гц, каналов: {m['channels']}, "
             f"пик {m['peak_dbfs']:+.2f} dBFS")
    L.append(f"Опорных точек карты: {m['keyframes']}, пользовательских якорей: {m['anchors_used']}")
    L.append(f"Время обработки: {m.get('seconds', 0):.1f} с")
    L.append("")
    if res.warnings:
        L.append(f"ПРЕДУПРЕЖДЕНИЯ ({len(res.warnings)})")
        for w in res.warnings:
            L.append(f"  • {w['text']}")
    else:
        L.append("ПРЕДУПРЕЖДЕНИЙ НЕТ")
    if res.notes:
        L.append("")
        L.append("ЗАМЕТКИ")
        for n in res.notes:
            L.append(f"  • {n}")
    return "\n".join(L) + "\n"


def plot_alignment(res, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tm, fm = res.tm, res.fm
    n = len(tm.pos)
    t = res.t0 + np.arange(n) * fm / 1000.0
    step = max(1, n // 6000)
    ts = t[::step]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.titleweight": "bold", "axes.titlesize": 10})
    fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True, facecolor=SURFACE,
                           gridspec_kw={"height_ratios": [1.3, 1, 1, 1]})
    for a in ax:
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.6)
        a.spines[["top", "right"]].set_visible(False)
        for w in res.warnings:
            if w.get("t0") is not None:
                a.axvspan(w["t0"], w["t1"], color=C_WARN, alpha=0.10, linewidth=0)

    dev_dtw = (tm.pos_dtw - tm.lin) * fm
    dev_fin = (tm.pos - tm.lin) * fm
    ax[0].plot(ts, dev_dtw[::step], color=C_GUIDE, linewidth=1.0, label="путь DTW")
    ax[0].plot(ts, dev_fin[::step], color=C_DUB, linewidth=2.0, label="итоговая карта")
    for g, d in tm.anchors:
        ax[0].plot(res.t0 + g * fm / 1000.0, (d - np.interp(g, np.arange(n), tm.lin)) * fm, "o",
                   color=INK, markersize=8, markerfacecolor=SURFACE, markeredgewidth=2)
    ax[0].set_ylabel("мс")
    ax[0].set_title(f"Путь DTW относительно глобальной линии (сдвиг {res.line.offset:+.3f} с, "
                    f"темп {res.line.tempo:.4f})", loc="left")
    ax[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=2)

    st = tm.stretch
    ax[1].plot(ts, st[::step], color=C_GUIDE, linewidth=1.5)
    for v in (res.settings.stretch_min, res.settings.stretch_max):
        ax[1].axhline(v, color=INK2, linewidth=1, linestyle="--")
    ax[1].axhline(1.0, color=GRID, linewidth=1)
    ax[1].set_ylabel("× (результат / исходник)")
    ax[1].set_title("Локальный коэффициент растяжения (пунктир — пределы)", loc="left")

    gdb = np.maximum(res.extra["guide_env"], -50.0)
    ddb = np.maximum(res.extra["dub_db"], -50.0)
    dn = res.extra["dub_n"]
    shift = np.arange(n) * 1.0 + res.line.offset * 1000.0 / fm
    for a, pos, title in ((ax[2], shift, "Огибающие ДО (только сдвиг начала)"),
                          (ax[3], tm.pos, "Огибающие ПОСЛЕ выравнивания")):
        k = np.clip(np.round(pos).astype(np.int64), 0, dn - 1)
        dv = np.where((pos >= 0) & (pos < dn), ddb[k], -50.0)
        a.plot(ts, gdb[::step], color=C_GUIDE, linewidth=1.0, label="эталон")
        a.plot(ts, dv[::step], color=C_DUB, linewidth=1.0, label="dub")
        a.set_ylabel("дБ")
        a.set_ylim(-52, 3)
        a.set_title(title, loc="left")
        a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=2)
    ax[3].set_xlabel("время эталона, с   (красные полосы — предупреждения)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def write_all(res, out_dir: str, plot: bool = True) -> dict:
    make_plot = plot
    s = res.settings
    files = {}
    files["dub_aligned"] = mio.write_audio(os.path.join(out_dir, "dub_aligned.wav"), res.y, res.sr,
                                           s.out_format)
    g = mio.to_mono(res.guide_y)
    dmono = mio.to_mono(res.y)
    n = min(len(g), len(dmono))
    st = np.vstack([g[:n], dmono[:n]])
    files["preview_stereo"] = mio.write_audio(os.path.join(out_dir, "preview_stereo.wav"), st,
                                              res.sr, "PCM_24")
    mix = g[:n] + dmono[:n]
    pk = float(np.max(np.abs(mix))) if n else 0.0
    if pk > 0.891:
        mix = mix * (0.891 / pk)                          # −1 dBFS
    files["preview_mix"] = mio.write_audio(os.path.join(out_dir, "preview_mix.wav"), mix, res.sr,
                                           "PCM_24")
    files["timemap"] = os.path.join(out_dir, "timemap.csv")
    write_timemap(res, files["timemap"])
    files["report_txt"] = os.path.join(out_dir, "report.txt")
    with open(files["report_txt"], "w", encoding="utf-8") as f:
        f.write(report_text(res))
    files["report_json"] = os.path.join(out_dir, "report.json")
    data = {"app": APP_NAME, "version": __version__, "guide": res.guide_path, "dub": res.dub_path,
            "engine": res.engine, "fragment_start_s": res.t0, "settings": s.to_dict(),
            "metrics": res.metrics, "warnings": res.warnings, "notes": res.notes, "kind": res.kind}
    with open(files["report_json"], "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=float)
    if make_plot:
        try:
            files["alignment_png"] = os.path.join(out_dir, "alignment.png")
            plot_alignment(res, files["alignment_png"])
        except Exception as exc:  # noqa: BLE001
            log.warning("график не построен: %s", exc)
            files.pop("alignment_png", None)
    return files
