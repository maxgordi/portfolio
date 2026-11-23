"""
Настройки и пресеты.

Единицы:
  tightness — %, 0 — только общий сдвиг и темп, 100 — каждый слог по эталону;
  stretch_min/max — пределы локального коэффициента растяжения (длительность
  результата / длительность исходника), по умолчанию 0.67–1.5;
  *_ms — миллисекунды; веса признаков — относительные (0 — признак выключен).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

MODES = ("auto", "global", "anchors")
MODE_NAMES = {"auto": "Авто", "global": "Только сдвиг + темп", "anchors": "По опорным точкам"}


@dataclass
class Settings:
    mode: str = "auto"                 # auto | global | anchors
    tightness: float = 100.0           # смесь глобальной (линейной) карты и DTW, %
    stretch_min: float = 0.67          # не сжимать сильнее, чем в 1/0.67 раза
    stretch_max: float = 1.5           # не растягивать сильнее, чем в 1.5 раза
    smooth_ms: float = 40.0            # сглаживание карты DTW (при tightness 100)
    rate_smooth_ms: float = 150.0      # за сколько мс коэффициент растяжения может смениться
    keyframe_ms: float = 60.0          # шаг опорных точек карты (плюс точки на онсетах)
    keyframes_onsets: bool = True
    w_chroma: float = 0.5              # вес признаков: chroma (ноты)
    w_mfcc: float = 1.0                # MFCC (тембр/фонемы)
    w_onset: float = 0.4               # огибающая онсетов (атаки слогов)
    w_energy: float = 0.4              # громкость
    band_ms: float = 250.0             # ширина полосы точного DTW вокруг грубого пути
    coarse_band_s: float = 3.0         # ширина полосы грубого DTW вокруг линии сдвиг+темп
    silence_db: float = 35.0           # тишина — ниже пика файла на столько дБ
    start: Optional[float] = None      # обработать только фрагмент эталона, с
    end: Optional[float] = None
    anchors: list = field(default_factory=list)   # [(время в эталоне, время в dub), ...], с
    anchors_path: str = ""
    out_sr: Optional[int] = None       # None — как у эталона
    out_format: str = "PCM_24"         # PCM_24 | FLOAT
    engine: str = "auto"               # auto | cli | pylibrb | wsola
    pitch: int = 0                     # подтяжка нот к эталону, % (по умолчанию выкл. — для RVC)
    formants: bool = True              # сохранять тембр при сдвиге высоты
    harmony_auto: bool = True          # бэк в гармонию — ноты не трогать
    out_dir: str = ""                  # пусто — папка output рядом с программой

    def replace(self, **kw) -> "Settings":
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def analysis_key(self) -> tuple:
        """То, от чего зависит путь DTW (кэшируется в окне)."""
        return (self.mode, self.w_chroma, self.w_mfcc, self.w_onset, self.w_energy, self.band_ms,
                self.coarse_band_s, self.silence_db, self.start, self.end,
                tuple(map(tuple, self.anchors)))


# ----------------------------------------------------------------------------
# Пресеты: (название, объяснение, значения)
# ----------------------------------------------------------------------------
PRESETS = [
    ("Для RVC — точно и чисто",
     "Стандарт. Каждый слог по эталону, растяжение не больше 1,5×, плавная смена скорости — "
     "минимум артефактов перед конвертацией голоса.",
     dict(mode="auto", tightness=100, stretch_min=0.67, stretch_max=1.5, smooth_ms=40,
          rate_smooth_ms=150, keyframe_ms=60)),
    ("Точно в ноль — по слогам",
     "Максимально плотно: меньше сглаживания, чаще опорные точки. Для даблов «как один голос».",
     dict(mode="auto", tightness=100, stretch_min=0.6, stretch_max=1.7, smooth_ms=20,
          rate_smooth_ms=60, keyframe_ms=30)),
    ("Плотный дабл",
     "Почти в ноль, но длинные слоги тянутся мягче — меньше «резины».",
     dict(mode="auto", tightness=90, stretch_min=0.7, stretch_max=1.45, smooth_ms=50,
          rate_smooth_ms=180, keyframe_ms=60)),
    ("Естественный дабл",
     "Фразы встают на место, мелкие расхождения слогов остаются — звучит живо.",
     dict(mode="auto", tightness=70, stretch_min=0.75, stretch_max=1.35, smooth_ms=100,
          rate_smooth_ms=250, keyframe_ms=100)),
    ("Только сдвиг и темп",
     "Без локальной подгонки: общий сдвиг начала и постоянный темп. Для дорожек, которые "
     "равномерно «уплывают» к концу.",
     dict(mode="global", tightness=0)),
    ("Другая тональность (без chroma)",
     "Сравнение только по тембру и атакам (MFCC/онсеты), ноты не учитываются. Если кавер "
     "спет в другой тональности.",
     dict(mode="auto", w_chroma=0.0, w_mfcc=1.0, w_onset=0.5, w_energy=0.4)),
    ("Грязный эталон (сепаратор, реверб)",
     "Меньше доверия тембру эталона, больше — нотам и атакам; шире поиск.",
     dict(mode="auto", w_chroma=0.8, w_mfcc=0.7, w_onset=0.6, w_energy=0.2, band_ms=350,
          silence_db=30)),
]


def preset_values(name: str) -> dict:
    for n, _d, v in PRESETS:
        if n == name:
            return dict(v)
    return {}
