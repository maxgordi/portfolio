"""Загрузка: декодирование, моно, 48 кГц, удаление DC, срез ниже highpass_hz."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

AUDIO_EXT = {".wav", ".flac"}


@dataclass
class SongInfo:
    file: str
    song: str
    key: str
    style: str
    path: Path


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def highpass(x: np.ndarray, sr: int, hz: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def load_raw(path: Path, sr: int) -> np.ndarray:
    """Моно float32 нужной частоты, без фильтрации (для проверки перегруза)."""
    x, sr_in = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return resample(np.ascontiguousarray(x), sr_in, sr)


def condition(x: np.ndarray, sr: int, hp_hz: float) -> np.ndarray:
    x = x - float(np.mean(x))
    return highpass(x, sr, hp_hz)


_NAME = re.compile(r"^(?P<nn>\d+)_(?P<song>.+?)(?:_(?P<key>[A-Ga-g][#b]?m?))?$")


def parse_name(stem: str) -> tuple[str, str]:
    m = _NAME.match(stem)
    if not m:
        return stem, ""
    return m.group("song"), (m.group("key") or "")


def discover(input_dir: str | Path) -> list[SongInfo]:
    d = Path(input_dir)
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO_EXT)
    table: dict[str, dict] = {}
    csv_path = d / "songs.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                table[row["file"].strip()] = row
    out = []
    for p in files:
        if p.name in table:
            r = table[p.name]
            out.append(SongInfo(p.name, r.get("song") or p.stem, (r.get("key") or "").strip(),
                                r.get("style") or "", p))
        else:
            song, key = parse_name(p.stem)
            out.append(SongInfo(p.name, song, key, "", p))
    return out


def write_wav24(path: str | Path, x: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_24")
