#!/usr/bin/env python3
"""Генерирует тестовые записи: чистую и «с проглоченными р» (синтезатор facebook/mms-tts-ukr).

  python make_tests.py            → test/clean.wav, test/swallowed.wav, test/lyrics.txt
"""
from pathlib import Path

import numpy as np
import soundfile as sf

LYRICS = [
    "Падав град на стару дорогу",
    "Грім гримів над темною рікою",
    "Довга дорога веде додому",
    "Знову град і знову дорога",
]
# то, что «спел» Suno: в трёх словах проглочена «р»
SWALLOWED = [
    "Падав гад на стару дорогу",
    "Грім гримів над темною рікою",
    "Довга доога веде додому",
    "Знову гад і знову дорога",
]
OUT_SR = 44100  # как у стемов Suno
PAUSE_S = 0.6


def synth(lines, tts, tok, seed=1):
    import librosa
    import torch

    parts = []
    for i, line in enumerate(lines):
        torch.manual_seed(seed + i)
        with torch.inference_mode():
            wav = tts(**tok(line, return_tensors="pt")).waveform[0].numpy()
        parts += [np.zeros(int(PAUSE_S * 16000), np.float32), wav.astype(np.float32)]
    parts.append(np.zeros(int(PAUSE_S * 16000), np.float32))
    mono = librosa.resample(np.concatenate(parts), orig_sr=tts.config.sampling_rate, target_sr=OUT_SR)
    mono *= 0.7 / np.abs(mono).max()
    return np.stack([mono, mono], axis=1)  # стерео, как стем


def main():
    from transformers import AutoTokenizer, VitsModel

    tts = VitsModel.from_pretrained("facebook/mms-tts-ukr").eval()
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ukr")
    tts.speaking_rate = 0.45  # медленно, ближе к пению
    tts.noise_scale = 0.4
    out = Path("test")
    out.mkdir(exist_ok=True)
    (out / "lyrics.txt").write_text("[Verse]\n" + "\n".join(LYRICS) + "\n", encoding="utf-8")
    sf.write(out / "clean.wav", synth(LYRICS, tts, tok), OUT_SR, subtype="PCM_24")
    sf.write(out / "swallowed.wav", synth(SWALLOWED, tts, tok), OUT_SR, subtype="PCM_24")
    print("→ test/clean.wav, test/swallowed.wav, test/lyrics.txt")


if __name__ == "__main__":
    main()
