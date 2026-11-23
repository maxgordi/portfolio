"""Автотесты: .venv/bin/python -m pytest -q tests"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocalsync import pipeline, warp  # noqa: E402
from vocalsync.config import Settings  # noqa: E402
import quality  # noqa: E402
import synth  # noqa: E402


def _files(tmp_path, p):
    gp, dp = str(tmp_path / "guide.wav"), str(tmp_path / "dub.wav")
    sf.write(gp, p.guide.T, p.sr)
    sf.write(dp, p.dub.T, p.sr)
    return gp, dp


@pytest.mark.parametrize("seed,offset,tempo", [(21, 1.2, 1.02), (22, -0.8, 0.975), (23, 2.8, 1.01)])
def test_synthetic_criteria(tmp_path, seed, offset, tempo):
    p = synth.make_pair(45.0, seed, offset, tempo, (0.05, 0.10))
    gp, dp = _files(tmp_path, p)
    res = pipeline.align(gp, dp, Settings())
    e = quality.map_error(res, p.warp, p.voiced)
    assert e["median"] < 20.0, e
    assert e["p95"] < 50.0, e
    assert res.y.shape[1] == p.guide.shape[1]          # длина = длине эталона
    assert res.y.shape[0] == 2                         # все каналы dub
    assert quality.clicks(res.y, res.sr) < 4.0         # нет щелчков
    assert abs(res.line.offset - offset) < 0.3


def test_global_mode_constant_tempo(tmp_path):
    p = synth.make_pair(40.0, 31, 1.5, 1.025, (0.0, 0.0))
    gp, dp = _files(tmp_path, p)
    res = pipeline.align(gp, dp, Settings(mode="global", tightness=0))
    e = quality.map_error(res, p.warp, p.voiced)
    assert abs(res.line.tempo - 1.025) < 0.003
    assert e["median"] < 20.0 and e["p95"] < 50.0, e
    st = res.tm.stretch
    assert np.ptp(st[100:-100]) < 1e-6                   # постоянный темп


def test_anchors_mode_and_timemap_roundtrip(tmp_path):
    p = synth.make_pair(40.0, 41, 2.0, 1.02, (0.05, 0.10))
    gp, dp = _files(tmp_path, p)
    csv = tmp_path / "anchors.csv"
    pts = [5.0, 15.0, 25.0, 35.0]
    csv.write_text("guide;dub\n" + "".join(f"{t:.3f};{float(p.warp(t)):.3f}\n" for t in pts)
                   .replace(".", ","), encoding="utf-8")
    assert len(pipeline.read_anchors(str(csv))) == 4    # десятичная запятая, заголовок
    res = pipeline.align(gp, dp, Settings(mode="anchors", anchors_path=str(csv)))
    for t in pts:
        k = int(t * 1000 / res.fm)
        assert abs(res.tm.pos[k] * res.fm / 1000 - float(p.warp(t))) < 0.02
    files = pipeline.export(res, str(tmp_path / "out"), plot=False)
    back = pipeline.read_anchors(files["timemap"])     # timemap.csv читается как anchors
    assert len(back) == len(res.tm.keys)
    rep = json.load(open(files["report_json"], encoding="utf-8"))
    assert rep["metrics"]["anchors_used"] == 4
    for k in ("dub_aligned", "preview_stereo", "preview_mix", "report_txt"):
        assert os.path.isfile(files[k])
    st, _ = sf.read(files["preview_stereo"])
    assert st.shape[1] == 2 and st.shape[0] == p.guide.shape[1]


def test_fragment(tmp_path):
    p = synth.make_pair(50.0, 51, 1.0, 1.01, (0.05, 0.10))
    gp, dp = _files(tmp_path, p)
    res = pipeline.align(gp, dp, Settings(start=20.0, end=32.0))
    assert res.y.shape[1] == int(round(12.0 * p.sr))
    e = quality.map_error(res, p.warp, p.voiced)
    assert e["median"] < 20.0 and e["p95"] < 50.0, e


def test_key_warning(tmp_path):
    p = synth.make_pair(30.0, 61, 1.0, 1.0, (0.0, 0.0), contaminate=False)
    import librosa
    up = np.vstack([librosa.effects.pitch_shift(c, sr=p.sr, n_steps=3) for c in p.dub])
    gp, dp = _files(tmp_path, synth.Pair(p.guide, up.astype(np.float32), p.warp, p.events))
    res = pipeline.align(gp, dp, Settings())
    assert any(w["kind"] == "key" for w in res.warnings)
    assert res.metrics["key"]["shift_semitones"] == 3


def test_wsola_fallback(tmp_path):
    p = synth.make_pair(25.0, 71, 0.8, 1.02, (0.05, 0.10), contaminate=False)
    gp, dp = _files(tmp_path, p)
    res = pipeline.align(gp, dp, Settings(engine="wsola"))
    assert res.engine == "wsola"
    assert any("WSOLA" in n for n in res.notes)
    assert res.y.shape == (2, p.guide.shape[1])
    assert quality.clicks(res.y, res.sr) < 4.0


def test_engines_listed():
    assert "wsola" in warp.available_engines()
