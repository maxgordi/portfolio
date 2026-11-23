"""Ноты: A4 = 440 Гц, MIDI 69."""
from __future__ import annotations

import numpy as np

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_midi(hz):
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=float) / 440.0)


def midi_to_name(m: int | float) -> str:
    m = int(round(m))
    return f"{NAMES[m % 12]}{m // 12 - 1}"


def hz_to_name(hz: float) -> str:
    if not np.isfinite(hz) or hz <= 0:
        return ""
    return midi_to_name(hz_to_midi(hz))
