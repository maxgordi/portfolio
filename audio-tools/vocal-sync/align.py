"""
VocalSync — командная строка.

    python align.py --guide g.wav --dub d.wav [--dub d2.wav ...] [--tightness 90]
                    [--mode auto|global|anchors] [--anchors a.csv] [--start 60 --end 90]

Для каждого dub создаётся папка output/<имя>_<дата-время>/ с dub_aligned.wav,
preview_stereo.wav, preview_mix.wav, timemap.csv, report.txt/json, alignment.png.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocalsync import APP_NAME, __version__  # noqa: E402
from vocalsync.config import MODES, PRESETS, Settings, preset_values  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="align.py", description=f"{APP_NAME} {__version__}: подгонка тайминга вокала под эталон",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = Settings()
    ap.add_argument("--guide", required=True, help="эталон (вокал оригинала)")
    ap.add_argument("--dub", action="append", required=True,
                    help="подгоняемая дорожка; можно несколько раз (пакетный режим)")
    ap.add_argument("--preset", choices=[n for n, _d, _v in PRESETS], default=None,
                    help="пресет (остальные параметры командной строки — поверх)")
    ap.add_argument("--mode", choices=MODES, default=None, help=f"режим (по умолчанию {d.mode})")
    ap.add_argument("--anchors", default="", help="CSV «время в эталоне; время в dub»")
    ap.add_argument("--tightness", type=float, default=None,
                    help=f"0–100 %%: 0 — только сдвиг и темп, 100 — каждый слог (по умолчанию {d.tightness:.0f})")
    ap.add_argument("--start", type=float, default=None, help="начало фрагмента эталона, с")
    ap.add_argument("--end", type=float, default=None, help="конец фрагмента эталона, с")
    ap.add_argument("--stretch-min", type=float, default=None, help=f"мин. коэф. растяжения ({d.stretch_min})")
    ap.add_argument("--stretch-max", type=float, default=None, help=f"макс. коэф. растяжения ({d.stretch_max})")
    ap.add_argument("--smooth-ms", type=float, default=None, help=f"сглаживание карты DTW, мс ({d.smooth_ms:.0f})")
    ap.add_argument("--rate-smooth-ms", type=float, default=None,
                    help=f"за сколько мс может смениться растяжение ({d.rate_smooth_ms:.0f})")
    ap.add_argument("--keyframe-ms", type=float, default=None, help=f"шаг опорных точек, мс ({d.keyframe_ms:.0f})")
    ap.add_argument("--no-onset-keys", action="store_true", help="не добавлять опорные точки на онсетах")
    ap.add_argument("--w-chroma", type=float, default=None, help=f"вес chroma ({d.w_chroma})")
    ap.add_argument("--w-mfcc", type=float, default=None, help=f"вес MFCC ({d.w_mfcc})")
    ap.add_argument("--w-onset", type=float, default=None, help=f"вес онсетов ({d.w_onset})")
    ap.add_argument("--w-energy", type=float, default=None, help=f"вес энергии ({d.w_energy})")
    ap.add_argument("--no-chroma", action="store_true", help="без chroma (если другая тональность)")
    ap.add_argument("--band-ms", type=float, default=None, help=f"полоса точного DTW, ±мс ({d.band_ms:.0f})")
    ap.add_argument("--silence-db", type=float, default=None, help=f"порог тишины ниже пика, дБ ({d.silence_db:.0f})")
    ap.add_argument("--sr", type=int, default=None, help="частота результата (по умолчанию как у эталона)")
    ap.add_argument("--float", action="store_true", help="писать float32 вместо 24 бит")
    ap.add_argument("--engine", choices=["auto", "cli", "pylibrb", "wsola"], default="auto",
                    help="движок растяжения")
    ap.add_argument("--pitch", type=int, default=0, help="подтянуть ноты к эталону, %% (0 — выкл.)")
    ap.add_argument("--out", default="", help="папка для результатов (по умолчанию ./output)")
    ap.add_argument("--no-plot", action="store_true", help="не строить alignment.png")
    ap.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    return ap


def settings_from_args(a) -> Settings:
    s = Settings()
    if a.preset:
        s = Settings.from_dict({**s.to_dict(), **preset_values(a.preset)})
    m = {"mode": a.mode, "tightness": a.tightness, "start": a.start, "end": a.end,
         "stretch_min": a.stretch_min, "stretch_max": a.stretch_max, "smooth_ms": a.smooth_ms,
         "rate_smooth_ms": a.rate_smooth_ms, "keyframe_ms": a.keyframe_ms,
         "w_chroma": a.w_chroma, "w_mfcc": a.w_mfcc, "w_onset": a.w_onset,
         "w_energy": a.w_energy, "band_ms": a.band_ms, "silence_db": a.silence_db,
         "out_sr": a.sr}
    s = s.replace(**{k: v for k, v in m.items() if v is not None})
    if a.no_chroma:
        s = s.replace(w_chroma=0.0)
    if a.no_onset_keys:
        s = s.replace(keyframes_onsets=False)
    if a.anchors:
        s = s.replace(anchors_path=a.anchors, mode=a.mode or "anchors")
    return s.replace(out_format="FLOAT" if a.float else "PCM_24", engine=a.engine, pitch=a.pitch,
                     out_dir=a.out)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    from vocalsync import pipeline, warp
    s = settings_from_args(a)
    eng = warp.pick_engine(s.engine, s.pitch > 0)
    print(f"{APP_NAME} {__version__} · движок растяжения: {eng} (доступны: "
          f"{', '.join(warp.available_engines())})")
    if eng == "wsola":
        print("ВНИМАНИЕ: Rubber Band не найден — запасной метод WSOLA, качество хуже. См. README.")
    session = pipeline.Session()
    failed = 0
    for i, dub in enumerate(a.dub):
        name = os.path.basename(dub)
        print(f"\n[{i + 1}/{len(a.dub)}] {name}")
        last = [-1]

        def prog(msg, f):
            pct = int(f * 100)
            if pct // 10 != last[0]:
                last[0] = pct // 10
                print(f"  {pct:3d}%  {msg}", flush=True)
        try:
            res = pipeline.align(a.guide, dub, s, prog, session)
            files = pipeline.export(res, a.out, plot=not a.no_plot)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logging.getLogger("vocalsync").exception("ошибка")
            print(f"  ОШИБКА: {exc}")
            continue
        m = res.metrics
        oe = m["onset_error_ms"]
        print(f"  сдвиг {m['offset_s']:+.3f} с · темп dub {m['tempo_percent']:+.2f} % · "
              f"растяжение {m['stretch_min']:.2f}–{m['stretch_max']:.2f}× · "
              f"атаки после: {oe['after']['median'] if oe['after']['median'] is not None else '—'} мс · "
              f"{res.seconds:.1f} с")
        for w in res.warnings[:12]:
            print(f"  ! {w['text']}")
        if len(res.warnings) > 12:
            print(f"  ! …и ещё {len(res.warnings) - 12} (см. report.txt)")
        for n in res.notes:
            print(f"  · {n}")
        print(f"  → {res.out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
