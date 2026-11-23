"""Слова (опционально): faster-whisper. Если модель не скачалась — шаг пропускается."""
from __future__ import annotations

import logging

import numpy as np

from .audio_io import resample

log = logging.getLogger("voiceset")


def load(model: str):
    try:
        from faster_whisper import WhisperModel
        return WhisperModel(model, device="cpu", compute_type="int8")
    except Exception as e:  # noqa: BLE001
        log.warning("Whisper недоступен (%s) — шаг «слова» пропущен", str(e)[:80])
        return None


def transcribe(model, x: np.ndarray, sr: int) -> str:
    segs, _ = model.transcribe(resample(x, sr, 16000), word_timestamps=True, beam_size=1)
    return " ".join(s.text.strip() for s in segs)
