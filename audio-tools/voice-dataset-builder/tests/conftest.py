import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from voiceset.config import load_config
from voiceset.workspace import Paths, save_table

ROOT = Path(__file__).resolve().parents[1]
SR = 48000


def _load_gen():
    spec = importlib.util.spec_from_file_location("make_samples", ROOT / "scripts" / "make_samples.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="session")
def gen():
    return _load_gen()


def tone(f, dur, amp=0.3, sr=SR, harmonics=6):
    t = np.arange(int(dur * sr)) / sr
    y = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, harmonics + 1))
    return (amp * y / np.max(np.abs(y))).astype(np.float32)


@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture()
def fake_ws(tmp_path):
    """Рабочая папка без анализа: 70 фраз-тонов с известными признаками (для проверки отбора и склейки)."""
    rng = np.random.default_rng(7)
    paths = Paths(tmp_path / "out")
    paths.phrases.mkdir(parents=True)
    rows = []
    keys = ["Am", "C", "Dm", "G", "Em", "F"]
    for i in range(1, 71):
        song = int(rng.integers(0, 6))
        deliv = rng.choice(["normal", "power", "breathy"], p=[0.6, 0.2, 0.2])
        lo = int(rng.integers(43, 68))
        sem = list(range(lo, lo + int(rng.integers(3, 8))))
        dur = float(rng.uniform(2.0, 7.0))
        x = tone(440 * 2 ** ((lo + 2 - 69) / 12), dur, amp=float(rng.uniform(0.1, 0.5)))
        x = x * np.minimum(1, np.minimum(np.arange(len(x)) / 480, (len(x) - np.arange(len(x))) / 480)).astype(np.float32)
        name = f"{i:04d}_s{song}.wav"
        sf.write(str(paths.phrases / name), x, SR, subtype="PCM_24")
        rows.append(dict(
            id=i, file=f"{song + 1:02d}_s{song}_{keys[song]}.wav", song=f"s{song}", song_idx=song, key=keys[song], style="",
            start=float(i * 10), end=float(i * 10 + dur), dur=dur, phrase_path=f"phrases/{name}", lufs=-20.0,
            peak_dbfs=-6.0, crest_db=10.0, clip=False, clicks=0, snr_db=40.0, tail_db=np.nan, tail_reverb=False,
            noise_floor_db=-60.0, f0_p5=200.0, f0_p50=250.0, f0_p95=300.0, sem_midi=sem, sem_ms=[400] * len(sem),
            voiced_frac=0.9, breaks=0, break_rate=0.0, vib_rate_hz=5.5, vib_extent_cents=40.0,
            range="", quality=float(rng.uniform(0.7, 1.0)), delivery=str(deliv), delivery_conf=0.8, unsure=False,
            similarity=float(rng.uniform(-0.2, 0.8)), text="", reject="",
        ))
    df = pd.DataFrame(rows)
    save_table(paths, df)
    (paths.work / "meta.yaml").write_text("input_minutes: 5.0\nspeaker_engine: mfcc\n", encoding="utf-8")
    return paths
