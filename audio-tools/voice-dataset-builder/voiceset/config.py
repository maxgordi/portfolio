"""Конфигурация: все пороги и веса живут в default_config.yaml / config.yaml пользователя."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_PATH = Path(__file__).with_name("default_config.yaml")


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IOCfg(_M):
    sr: int = 48000
    highpass_hz: float = 70


class SegmentCfg(_M):
    min_s: float = 1.5
    max_s: float = 8.0
    merge_gap_ms: float = 250
    pad_ms: float = 40
    min_split_ms: float = 120
    vad: str = "auto"
    energy_margin_db: float = 15


class PitchCfg(_M):
    engine: list[str] = ["rmvpe", "torchcrepe", "pyin"]
    hop_ms: float = 10
    note_min_ms: float = 150
    fmin: float = 60
    fmax: float = 1200
    median_frames: int = 5
    rmvpe_weights: str | None = None
    crepe_model: str = "tiny"
    vibrato_band_hz: list[float] = [4.0, 8.0]
    jump_cents: float = 350
    dropout_ms: list[float] = [30, 120]


class DeliveryCfg(_M):
    z_breathy: float = 0.5
    z_power: float = 0.5
    unsure_below: float = 0.3
    knn_k: int = 3


class QualityCfg(_M):
    clip_run: int = 3
    clip_level: float = 0.999
    click_k: float = 25
    click_min_abs: float = 0.05
    snr_min_db: float = 25
    reverb_tail_db: float = -20
    reverb_margin_db: float = 6
    break_rate_max: float = 1.5
    voiced_min: float = 0.35
    similarity_pctl_min: float = 5
    similarity_z_min: float = -2.0


class SpeakerCfg(_M):
    engine: str = "auto"
    min_s: float = 2.0


class AsrCfg(_M):
    enabled: bool = False
    model: str = "small"


class AssembleCfg(_M):
    target_lufs: float = -20
    fade_ms: float = 15
    gap_ms: float = 60
    gap_range_ms: list[float] = [0, 200]
    peak_dbfs: float = -3
    compress: bool = True
    comp_threshold_db: float = -26
    comp_ratio: float = 2.0
    lowshelf_hz: float = 120
    lowshelf_db: float = -12


class Weights(_M):
    coverage: float = 1.0
    quota: float = 0.8
    quality: float = 0.5
    similarity: float = 0.5
    song_overuse: float = 1.0


class SunoCfg(_M):
    length_s: float = 120.0
    pool_min_s: float = 2.0
    pool_max_s: float = 8.0
    quotas: dict[str, float] = {"normal": 0.70, "power": 0.15, "breathy": 0.15}
    quota_tolerance: float = 0.05
    max_song_share: float = 0.20
    min_songs: int = 5
    range_pctl: list[float] = [2, 98]
    edge_semitones: int = 3
    edge_weight: float = 2.0
    weights: Weights = Weights()
    local_search_passes: int = 3
    pair_candidates: int = 60
    order_tries: int = 400
    assemble: AssembleCfg = AssembleCfg()


class RvcCfg(_M):
    sr: int = 48000
    target_lufs: float = -20
    highpass_hz: float = 70
    clip_min_s: float = 2.0
    clip_max_s: float = 12.0
    merge_gap_s: float = 1.0
    trim_margin_db: float = 6
    min_total_s: float = 600


class Config(_M):
    seed: int = 42
    io: IOCfg = IOCfg()
    segment: SegmentCfg = SegmentCfg()
    pitch: PitchCfg = PitchCfg()
    delivery: DeliveryCfg = DeliveryCfg()
    quality: QualityCfg = QualityCfg()
    speaker: SpeakerCfg = SpeakerCfg()
    asr: AsrCfg = AsrCfg()
    suno: SunoCfg = SunoCfg()
    rvc: RvcCfg = RvcCfg()


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    data = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    if path:
        data = _merge(data, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if overrides:
        data = _merge(data, overrides)
    return Config.model_validate(data)


def dump_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(cfg.model_dump(), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
