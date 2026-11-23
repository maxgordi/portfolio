import array
import io
import re
import sys
import time
import wave
from typing import Optional

import openai

from i18n import t

CHUNK_SECONDS = 20

# Shown in Settings before the first "Refresh", best first.
KNOWN_MODELS = [
    "gpt-transcribe",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-transcribe-2025-12-15",
    "whisper-1",
]
# Used when the chosen model is gone or this API key has no access to it.
FALLBACK_MODEL = "gpt-4o-mini-transcribe"

# Models on /v1/models that can transcribe a recorded file. Realtime/live
# models only work over streaming sessions, diarize needs different
# parameters, and tts goes the other way.
_MODEL_INCLUDE = re.compile(r"transcribe|whisper")
_MODEL_EXCLUDE = re.compile(r"realtime|live|diarize|tts")


class ModelUnavailable(Exception):
    """The transcription model doesn't exist or this key can't use it."""


def split_wav(wav_bytes: bytes) -> list[bytes]:
    chunks = []
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        chunk_frames = rate * CHUNK_SECONDS
        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                break
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(width)
                out.setframerate(rate)
                out.writeframes(frames)
            chunks.append(buf.getvalue())
    return chunks


def wav_duration(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate() or 1)
    except Exception:
        return 0.0


def is_digital_silence(wav_bytes: bytes) -> bool:
    """True if every sample is exactly zero. A real microphone always picks
    up some noise; pure zeros mean the OS is withholding the audio (on macOS:
    microphone permission denied)."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            if wf.getsampwidth() != 2:
                return False
            samples = array.array("h", wf.readframes(wf.getnframes()))
    except Exception:
        return False
    if sys.byteorder == "big":
        samples.byteswap()
    return not any(samples)


def _is_model_error(e: Exception) -> bool:
    if isinstance(e, (openai.NotFoundError, openai.PermissionDeniedError)):
        return True
    if isinstance(e, openai.BadRequestError):
        code = str(getattr(e, "code", "") or "")
        msg = str(e).lower()
        return "model" in code or ("model" in msg and (
            "does not exist" in msg or "not found" in msg
            or "not supported" in msg or "invalid" in msg))
    return False


def _is_retryable(e: Exception) -> bool:
    # A bad key or a malformed request fails the same way every time.
    if isinstance(e, openai.APIStatusError):
        return e.status_code in (408, 409, 429) or e.status_code >= 500
    return True


def transcribe_chunk(client: openai.OpenAI, wav_bytes: bytes,
                     model: str = FALLBACK_MODEL, prompt: str = "") -> str:
    """prompt: the text said just before this piece, so names and sentence
    boundaries survive when speech is sent phrase by phrase."""
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"
            extra = {"prompt": prompt} if prompt else {}
            result = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                **extra,
            )
            return result.text.strip()
        except Exception as e:
            if _is_model_error(e):
                raise ModelUnavailable(f"{model}: {e}") from e
            last_err = e
            if not _is_retryable(e):
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise last_err


def list_transcription_models(client: openai.OpenAI) -> list[str]:
    """Transcription model ids this API key can use, best known first."""
    ids = {m.id for m in client.models.list()}
    usable = [i for i in ids
              if _MODEL_INCLUDE.search(i) and not _MODEL_EXCLUDE.search(i)]
    known = [m for m in KNOWN_MODELS if m in usable]
    rest = sorted(m for m in usable if m not in KNOWN_MODELS)
    return known + rest


def model_choices(cfg: dict) -> list[str]:
    """What Settings offers: the last fetched list (or the built-in one),
    always including the currently selected model."""
    models = [m for m in cfg.get("transcription_models") or [] if isinstance(m, str)]
    if not models:
        models = list(KNOWN_MODELS)
    current = cfg.get("transcription_model") or ""
    if current and current not in models:
        models.insert(0, current)
    return models


STYLE_MODEL = "gpt-4o-mini"


def apply_style(client: openai.OpenAI, prompt: str, text: str) -> str:
    # Kept in English whatever the interface language: the model follows
    # it equally well, and it says nothing about the language of the text.
    system = (
        f"{prompt.strip()}\n\n"
        "Apply this to the text below. Reply with the resulting text only — "
        "no explanations, preface or quotes."
    )
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            result = client.chat.completions.create(
                model=STYLE_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
            )
            return (result.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if not _is_retryable(e):
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise last_err


def friendly_error(e: Exception) -> str:
    """One short line for a notification or a status label."""
    if isinstance(e, ModelUnavailable):
        return t("error.model_unavailable", detail=str(e))[:160]
    if isinstance(e, openai.AuthenticationError):
        return t("error.invalid_key")
    if isinstance(e, openai.RateLimitError):
        msg = str(e).lower()
        if "quota" in msg or "billing" in msg:
            return t("error.quota")
        return t("error.rate_limit")
    if isinstance(e, openai.APITimeoutError):
        return t("error.timeout")
    if isinstance(e, openai.APIConnectionError):
        return t("error.connection")
    return f"{type(e).__name__}: {e}"[:160]


def make_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, timeout=openai.Timeout(60.0, connect=10.0))
