"""M5/M6: RVC-датасет, отчёт; M1-M3: сквозной прогон на синтетической записи."""
import hashlib
from pathlib import Path

import pandas as pd
import soundfile as sf

from voiceset.analyze import analyze
from voiceset.config import load_config
from voiceset.report import build_report
from voiceset.rvc import build_rvc
from voiceset.suno import build_suno
from voiceset.workspace import Paths, load_table


def test_rvc_clips_summary_and_warning(fake_ws, cfg):
    df = load_table(fake_ws)
    s = build_rvc(df, fake_ws, cfg.rvc, 40000, combined=True)
    clips = sorted((fake_ws.rvc / "dataset").glob("*.wav"))
    assert clips and s["clips"] == len(clips)
    for c in clips:
        i = sf.info(str(c))
        assert i.samplerate == 40000 and i.channels == 1
        assert cfg.rvc.clip_min_s - 0.2 <= i.duration <= cfg.rvc.clip_max_s + 0.2
    assert s["total_minutes"] < 10 and any("меньше" in w for w in s["warnings"])
    assert (fake_ws.rvc / "all_clips.wav").exists() and (fake_ws.rvc / "summary.json").exists()


def test_report_renders(fake_ws, cfg):
    build_suno(fake_ws, cfg)
    build_rvc(load_table(fake_ws), fake_ws, cfg.rvc)
    out = build_report(fake_ws, cfg)
    html = Path(out["path"]).read_text(encoding="utf-8")
    assert "<audio" in html and "Карта диапазона" in html and "Выбранные фразы" in html


def _tree_hash(p: Path):
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((p.phrases).glob("*.wav"))}


def test_end_to_end_small_recording_and_determinism(tmp_path, gen):
    inp = tmp_path / "in"
    import sys
    sys.argv = ["x", "--out", str(inp), "--songs", "1", "--minutes", "0.55", "--seed", "5"]
    gen.main()
    cfg = load_config()
    p1, p2 = Paths(tmp_path / "o1"), Paths(tmp_path / "o2")
    d1 = analyze(inp, p1, cfg)
    d2 = analyze(inp, p2, cfg)
    assert len(d1) >= 5 and (d1["range"] != "").all()
    assert set(d1["delivery"]) <= {"breathy", "normal", "power"}
    assert {"reject", "similarity", "snr_db", "lufs", "vib_rate_hz"} <= set(d1.columns)
    cols = [c for c in d1.columns if d1[c].dtype != object]
    pd.testing.assert_frame_equal(d1[cols], d2[cols])
    assert _tree_hash(p1) == _tree_hash(p2)
