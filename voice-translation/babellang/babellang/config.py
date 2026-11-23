from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import keyring
from rich.prompt import Prompt

logger = logging.getLogger(__name__)

LANGUAGE_NAMES: dict[str, str] = {
    "ru": "Russian / Русский",
    "uk": "Ukrainian / Українська",
    "en": "English",
    "fr": "French / Français",
    "es": "Spanish / Español",
    "de": "German / Deutsch",
    "it": "Italian / Italiano",
    "pt": "Portuguese / Português",
    "zh": "Chinese / 中文",
    "ja": "Japanese / 日本語",
    "ko": "Korean / 한국어",
    "vi": "Vietnamese / Tiếng Việt",
    "th": "Thai / ภาษาไทย",
    "ar": "Arabic / العربية",
    "tr": "Turkish / Türkçe",
    "pl": "Polish / Polski",
    "nl": "Dutch / Nederlands",
    "hi": "Hindi / हिंदी",
}

_KEYCHAIN_SERVICE = {
    "deepgram": "babellang_deepgram",
    "openai": "babellang_openai",
    "elevenlabs": "babellang_elevenlabs",
}
_KEYCHAIN_SERVICE_OLD = {
    "deepgram": "seqvoya_deepgram",
    "openai": "seqvoya_openai",
    "elevenlabs": "seqvoya_elevenlabs",
}
_KEYCHAIN_USER = "api_key"

SETTINGS_PATH = Path.home() / ".config" / "babellang" / "settings.json"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    # 512 samples = 32ms at 16kHz — required by Silero VAD
    chunk_samples: int = 512
    vad_threshold: float = 0.5
    # 3 × 32ms = 96ms to confirm speech start
    vad_speech_start_chunks: int = 3
    # 12 × 32ms = 384ms silence to end segment — faster response while avoiding mid-clause splits
    vad_speech_end_chunks: int = 12
    max_segment_seconds: float = 30.0
    # PTT auto-split: 25 × 32ms = 800ms pause splits a held-key recording into sub-segments,
    # so long speech starts translating before the key is released
    ptt_split_silence_chunks: int = 25
    # Minimum sub-segment before a PTT auto-split is allowed (20 × 32ms = 640ms of audio)
    ptt_split_min_chunks: int = 20


@dataclass(frozen=True)
class PipelineConfig:
    stt_language: str = "ru"                              # language of incoming audio for STT
    # Mode: audio — combined STT+translation in one call
    # (gpt-4o-audio-preview was retired; gpt-audio-mini is its successor)
    openai_audio_model: str = "gpt-audio-mini"
    # Mode: classic — separate transcription then translation
    openai_stt_model: str = "gpt-4o-mini-transcribe"
    openai_model: str = "gpt-4o-mini"
    openai_context_turns: int = 3
    # ElevenLabs TTS
    elevenlabs_model: str = "eleven_flash_v2_5"
    elevenlabs_voice_id: str = "7ajqUfdTnjp4oZ6AK5Tg"
    # pcm_24000: raw int16 PCM at 24kHz — available on Creator tier and above
    elevenlabs_output_format: str = "pcm_24000"
    tts_sample_rate: int = 24000
    # "strict": language param + prompt anchor — locked to stt_language
    # "hint":   language param only — model may still drift
    # "auto":   nothing passed — raw transcription, model detects the language
    stt_language_mode: str = "strict"


@dataclass
class Settings:
    audio: AudioConfig = field(default_factory=AudioConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    input_device_name: str = ""
    output_device_name: str = ""
    incoming_device: Optional[int] = None       # BlackHole 16ch — captures interlocutor's audio
    incoming_device_name: str = ""
    speaker_device: Optional[int] = None        # User's real speakers — plays incoming TTS
    speaker_device_name: str = ""
    outgoing_original_ratio: float = 0.0        # 0.0 = pure TTS; 1.0 = pure original mic voice
    incoming_original_ratio: float = 0.0        # 0.0 = pure TTS; 1.0 = pure interlocutor voice
    outgoing_tts_volume: float = 1.0            # ElevenLabs volume sent to the interlocutor
    incoming_tts_volume: float = 1.0            # ElevenLabs volume played to the user
    smooth_ducking: bool = True                  # True = smooth 1.5s fade; False = instant mute
    stt_mode: str = "audio"                      # "audio" = gpt-4o-audio-preview; "classic" = transcribe+translate
    my_language: str = "ru"                     # language the user speaks (for outgoing STT + incoming TTS)
    interlocutor_language: str = "en"           # language the interlocutor speaks (for incoming STT + outgoing TTS)
    languages_configured: bool = False
    theme: str = "dark"                          # "dark" or "light"
    window_geometry: str = "580x980"             # saved window size, restored on next launch
    vad_mode: str = "auto"                       # "auto" = VAD; "manual" = push-to-talk
    ptt_hotkey: str = "space"                    # hotkey used in manual/PTT mode
    stt_language_mode: str = "strict"            # "strict" | "hint" | "auto" (см. PipelineConfig)

    def to_dict(self) -> dict:
        return {
            "input_device": self.input_device,
            "output_device": self.output_device,
            "input_device_name": self.input_device_name,
            "output_device_name": self.output_device_name,
            "elevenlabs_voice_id": self.pipeline.elevenlabs_voice_id,
            "elevenlabs_model": self.pipeline.elevenlabs_model,
            "incoming_device": self.incoming_device,
            "incoming_device_name": self.incoming_device_name,
            "speaker_device": self.speaker_device,
            "speaker_device_name": self.speaker_device_name,
            "outgoing_original_ratio": self.outgoing_original_ratio,
            "incoming_original_ratio": self.incoming_original_ratio,
            "outgoing_tts_volume": self.outgoing_tts_volume,
            "incoming_tts_volume": self.incoming_tts_volume,
            "smooth_ducking": self.smooth_ducking,
            "stt_mode": self.stt_mode,
            "my_language": self.my_language,
            "interlocutor_language": self.interlocutor_language,
            "languages_configured": self.languages_configured,
            "theme": self.theme,
            "window_geometry": self.window_geometry,
            "vad_mode": self.vad_mode,
            "ptt_hotkey": self.ptt_hotkey,
            "stt_language_mode": self.stt_language_mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        pipeline = PipelineConfig(
            elevenlabs_voice_id=data.get("elevenlabs_voice_id", "7ajqUfdTnjp4oZ6AK5Tg"),
            elevenlabs_model=data.get("elevenlabs_model", "eleven_flash_v2_5"),
        )
        return cls(
            pipeline=pipeline,
            input_device=data.get("input_device"),
            output_device=data.get("output_device"),
            input_device_name=data.get("input_device_name", ""),
            output_device_name=data.get("output_device_name", ""),
            incoming_device=data.get("incoming_device"),
            incoming_device_name=data.get("incoming_device_name", ""),
            speaker_device=data.get("speaker_device"),
            speaker_device_name=data.get("speaker_device_name", ""),
            outgoing_original_ratio=data.get("outgoing_original_ratio", 0.0),
            incoming_original_ratio=data.get("incoming_original_ratio", 0.0),
            outgoing_tts_volume=data.get("outgoing_tts_volume", 1.0),
            incoming_tts_volume=data.get("incoming_tts_volume", 1.0),
            smooth_ducking=data.get("smooth_ducking", True),
            stt_mode=data.get("stt_mode", "audio"),
            my_language=data.get("my_language", "ru"),
            interlocutor_language=data.get("interlocutor_language", "en"),
            languages_configured=data.get("languages_configured", False),
            theme=data.get("theme", "dark"),
            window_geometry=data.get("window_geometry", "580x980"),
            vad_mode=data.get("vad_mode", "auto"),
            ptt_hotkey=data.get("ptt_hotkey", "space"),
            stt_language_mode=data.get("stt_language_mode", "strict"),
        )


def _migrate_keychain_key(service: str) -> Optional[str]:
    old_svc = _KEYCHAIN_SERVICE_OLD.get(service)
    if not old_svc:
        return None
    old_key = keyring.get_password(old_svc, _KEYCHAIN_USER)
    if not old_key:
        return None
    new_svc = _KEYCHAIN_SERVICE[service]
    keyring.set_password(new_svc, _KEYCHAIN_USER, old_key)
    try:
        keyring.delete_password(old_svc, _KEYCHAIN_USER)
    except Exception:
        pass
    logger.info("Migrated Keychain key '%s' → '%s'", old_svc, new_svc)
    return old_key


def get_api_key(service: str) -> Optional[str]:
    svc = _KEYCHAIN_SERVICE.get(service, service)
    key = keyring.get_password(svc, _KEYCHAIN_USER)
    if not key:
        key = _migrate_keychain_key(service)
    return key


def set_api_key(service: str, key: str) -> None:
    svc = _KEYCHAIN_SERVICE.get(service, service)
    keyring.set_password(svc, _KEYCHAIN_USER, key)
    logger.debug("Stored key for %s in Keychain", service)


def require_api_key(service: str, label: str) -> str:
    key = get_api_key(service)
    if key:
        return key
    key = Prompt.ask(f"[bold]{label}[/bold]", password=True)
    if not key.strip():
        raise ValueError(f"API key for {service} cannot be empty")
    set_api_key(service, key.strip())
    return key.strip()


def load_settings() -> Settings:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text())
            return Settings.from_dict(data)
        except Exception as exc:
            # Keep the evidence instead of silently resetting everything
            broken = SETTINGS_PATH.with_suffix(".json.broken")
            try:
                SETTINGS_PATH.replace(broken)
            except OSError:
                pass
            logger.error("settings.json is corrupt (%s) — backed up to %s, using defaults",
                         exc, broken)
    return Settings()


def save_settings(settings: Settings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a crash mid-write must never leave a corrupt settings.json
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings.to_dict(), indent=2))
    os.replace(tmp, SETTINGS_PATH)
