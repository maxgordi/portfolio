"""Чтение и запись звука: WAV/FLAC/AIFF/OGG/MP3, любая частота, моно или стерео."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

import numpy as np
import soundfile as sf

log = logging.getLogger("vocalsync.io")

AUDIO_EXT = {".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".m4a"}


def _read_ffmpeg(path: str) -> tuple[np.ndarray, int]:
    """Запасной путь для форматов, которые не открывает libsndfile (старые MP3, M4A)."""
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("файл не читается, а ffmpeg не найден")
    probe = subprocess.run([shutil.which("ffprobe") or "ffprobe", "-v", "error", "-select_streams",
                            "a:0", "-show_entries", "stream=sample_rate,channels", "-of",
                            "csv=p=0", path], capture_output=True, text=True)
    try:
        sr_s, ch_s = probe.stdout.strip().split(",")[:2]
        sr, ch = int(sr_s), int(ch_s)
    except ValueError as exc:
        raise RuntimeError(f"ffprobe не понял файл: {probe.stderr.strip()}") from exc
    raw = subprocess.run([ff, "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
                          "-"], capture_output=True, check=True).stdout
    data = np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).T.copy()
    return data, sr


def read_audio(path: str, start: Optional[float] = None,
               end: Optional[float] = None) -> tuple[np.ndarray, int]:
    """Возвращает (каналы, сэмплы) float32 и частоту. start/end — секунды (фрагмент)."""
    name = os.path.basename(path)
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        data = data.T
    except Exception:  # noqa: BLE001
        try:
            data, sr = _read_ffmpeg(path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Не удалось открыть «{name}»: {exc}") from exc
    if start is not None or end is not None:
        a = int(round(max(0.0, start or 0.0) * sr))
        b = data.shape[1] if end is None else int(round(end * sr))
        b = min(data.shape[1], b)
        if b - a < sr * 0.5:
            raise RuntimeError(f"«{name}»: фрагмент {start}–{end} с пустой или короче 0,5 с.")
        data = data[:, a:b]
    if data.shape[1] < sr * 0.3:
        raise RuntimeError(f"«{name}» слишком короткий (меньше 0,3 с).")
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def resample(data: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Качественный ресемплинг (soxr VHQ) по последней оси."""
    if sr == target:
        return data
    import soxr
    x = np.ascontiguousarray(np.atleast_2d(data).T, dtype=np.float32)
    y = soxr.resample(x, sr, target, quality="VHQ")
    out = np.ascontiguousarray(y.T, dtype=np.float32)
    return out if data.ndim > 1 else out[0]


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=0) if data.ndim > 1 else data


def fit_length(data: np.ndarray, n: int) -> np.ndarray:
    """Дополнить тишиной или обрезать до n сэмплов."""
    if data.shape[-1] == n:
        return data
    if data.shape[-1] > n:
        return np.ascontiguousarray(data[..., :n])
    pad = [(0, 0)] * (data.ndim - 1) + [(0, n - data.shape[-1])]
    return np.pad(data, pad)


def write_audio(path: str, data: np.ndarray, sr: int, fmt: str = "PCM_24") -> str:
    """data — (каналы, сэмплы). fmt: PCM_24 или FLOAT."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sub = "FLOAT" if fmt.upper() in ("FLOAT", "FLOAT32", "F32") else "PCM_24"
    y = np.atleast_2d(data).T
    if sub == "PCM_24":
        y = np.clip(y, -1.0, 1.0)
    sf.write(path, y, sr, subtype=sub)
    return path
