"""Сходство с голосом: эмбеддинг ECAPA (SpeechBrain), фолбэк — MFCC-статистики. Косинус к медиане."""
from __future__ import annotations

import logging

import numpy as np

from .audio_io import resample
from .config import SpeakerCfg

log = logging.getLogger("voiceset")


def _ecapa_embedder():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir=".cache/ecapa", run_opts={"device": "cpu"})

    def emb(x, sr):
        w = torch.from_numpy(resample(x, sr, 16000))[None]
        return clf.encode_batch(w).squeeze().detach().numpy()
    return emb


def _mfcc_embedder():
    import librosa

    def emb(x, sr):
        y = resample(x, sr, 16000)
        m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20, n_fft=512, hop_length=160)[1:]
        d = librosa.feature.delta(m)
        return np.concatenate([m.mean(1), m.std(1), d.std(1)])
    return emb


def get_embedder(cfg: SpeakerCfg):
    if cfg.engine in ("auto", "ecapa"):
        try:
            e = _ecapa_embedder()
            log.info("Сходство: ECAPA (SpeechBrain)")
            return e, "ecapa"
        except Exception as ex:  # noqa: BLE001
            if cfg.engine == "ecapa":
                raise
            log.info("ECAPA недоступна (%s) — использую MFCC-фолбэк", str(ex)[:80])
    return _mfcc_embedder(), "mfcc"


def similarities(embs: list[np.ndarray | None], engine: str) -> np.ndarray:
    """Косинус к медианному эмбеддингу; NaN там, где эмбеддинга нет."""
    idx = [i for i, e in enumerate(embs) if e is not None]
    out = np.full(len(embs), np.nan)
    if len(idx) < 3:
        return out
    E = np.stack([embs[i] for i in idx]).astype(float)
    if engine == "mfcc":  # признаки разного масштаба
        E = (E - E.mean(0)) / (E.std(0) + 1e-9)
    med = np.median(E, axis=0)
    cos = (E @ med) / (np.linalg.norm(E, axis=1) * np.linalg.norm(med) + 1e-12)
    out[idx] = cos
    return out
