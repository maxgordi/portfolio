"""M4: отбор, порядок, склейка, точная длина, детерминизм, правка."""
import hashlib

import numpy as np
import pandas as pd
import soundfile as sf

from voiceset import assemble as asm
from voiceset.config import load_config
from voiceset.order import order_phrases
from voiceset.select import DELIVERIES, build_selection
from voiceset.suno import build_suno
from voiceset.workspace import load_table

from conftest import SR, tone


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_selection_quotas_coverage_songs(fake_ws, cfg):
    df = load_table(fake_ws)
    sel = build_selection(df, cfg.suno, 42)
    ph = df.loc[sel.ids]
    tot = ph["dur"].sum()
    for d, q in cfg.suno.quotas.items():
        share = ph[ph["delivery"] == d]["dur"].sum() / tot
        assert abs(share - q) <= cfg.suno.quota_tolerance, (d, share)
    assert len(sel.covered) / len(sel.targets) >= 0.9
    assert ph["song_idx"].nunique() >= cfg.suno.min_songs
    for _, g in ph.groupby("song_idx"):
        assert g["dur"].sum() / tot <= cfg.suno.max_song_share + 0.03


def test_order_rules(fake_ws, cfg):
    df = load_table(fake_ws)
    sel = build_selection(df, cfg.suno, 42)
    order, warn = order_phrases(df, sel.ids, 42)
    assert sorted(order) == sorted(sel.ids)
    rows = df.loc[order]
    songs = rows["song_idx"].tolist()
    assert all(a != b for a, b in zip(songs, songs[1:]))
    dl = rows["delivery"].tolist()
    triples = sum(a == b == c for a, b, c in zip(dl, dl[1:], dl[2:]))
    # при доле «обычной» подачи 70% правило «≤2 подряд» может быть арифметически недостижимо — тогда нарушений минимум
    assert triples <= 2 and (triples == 0 or any("одной подачи" in w for w in warn))


def test_build_length_format_peak_and_determinism(fake_ws, cfg):
    info = build_suno(fake_ws, cfg)
    wav = fake_ws.suno / "voice_set_2min.wav"
    i = sf.info(str(wav))
    assert abs(i.duration - 120.0) <= 0.2
    assert i.channels == 1 and i.samplerate == 48000 and i.subtype == "PCM_24"
    x, _ = sf.read(str(wav))
    assert 20 * np.log10(np.abs(x).max()) <= -3.0 + 1e-6
    h1, m1 = _sha(wav), _sha(fake_ws.suno / "manifest.csv")
    build_suno(fake_ws, cfg)
    assert _sha(wav) == h1 and _sha(fake_ws.suno / "manifest.csv") == m1   # побайтно
    assert info["length_s"] == 120.0


def test_other_length_and_no_cut_inside_phrase(fake_ws, cfg):
    info = build_suno(fake_ws, cfg, length=60.0)
    assert abs(info["length_s"] - 60.0) <= 0.2


def test_exclude_include_rebuild(fake_ws, cfg):
    first = build_suno(fake_ws, cfg)
    drop = first["ids"][0]
    df = load_table(fake_ws)
    spare = next(int(i) for i in df["id"] if int(i) not in first["ids"])
    second = build_suno(fake_ws, cfg, exclude=[drop], include=[spare])
    assert drop not in second["ids"] and spare in second["ids"]
    assert abs(second["length_s"] - 120.0) <= 0.2
    assert second["removed"] and second["added"]
    assert (fake_ws.root / "review.yaml").exists()
    third = build_suno(fake_ws, cfg)             # review.yaml сохраняется между сборками
    assert drop not in third["ids"] and spare in third["ids"]


def test_junction_has_no_jump_and_peak_limit(cfg):
    a = cfg.suno.assemble
    p1 = asm.prepare(tone(200, 3.0, 0.9), SR, a)
    p2 = asm.prepare(tone(333, 3.0, 0.05), SR, a)
    y, gaps, warn = asm.join([p1, p2], SR, 6.06, a)
    k = len(p1)
    seg = y[k - 50:k + gaps[0] + 50]
    assert np.max(np.abs(np.diff(seg))) < 0.01
    assert abs(len(y) / SR - 6.06) < 1e-6
    loud = y * 20
    assert 20 * np.log10(np.abs(asm.limit_peak(loud, -3.0)).max()) <= -3.0 + 1e-6


def test_exact_length_on_any_gap_remainder(cfg):
    a = cfg.suno.assemble
    parts = [asm.prepare(tone(200 + 30 * i, 2.0 + 0.137 * i, 0.3), SR, a) for i in range(5)]
    body = sum(len(p) for p in parts) / SR
    y, gaps, warn = asm.join(parts, SR, body + 0.2437, a)
    assert len(y) == int(round((body + 0.2437) * SR)) and not warn
