from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import wave
from typing import Callable, Optional

import numpy as np
from openai import AsyncOpenAI

from babellang.config import PipelineConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2
_CHANNELS = 1
_MIN_SAMPLES = 3200  # 200ms at 16kHz

_LANG_NAMES: dict[str, str] = {
    "ru": "Russian", "uk": "Ukrainian", "en": "English",
    "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "vi": "Vietnamese",
    "th": "Thai", "ar": "Arabic", "tr": "Turkish",
    "pl": "Polish", "nl": "Dutch", "hi": "Hindi",
}


class OpenAIAudioTranslator:
    """
    Combined STT + Translation in a single API call via gpt-4o-audio-preview.

    Replaces the sequential OpenAISTT → OpenAITranslator chain with one call,
    saving one full API round-trip (~300-500ms of latency).

    stt_queue protocol:
      np.ndarray  — 512-sample int16 chunk; accumulate into buffer
      False       — end of utterance (VAD silence); transcribe+translate and emit
      None        — stop signal

    token_queue protocol (same as ElevenLabsTTS expects):
      str         — complete translated text
      None        — sentinel after each utterance
    """

    MAX_RETRIES = 2
    BACKOFF_BASE = 0.5

    def __init__(
        self,
        api_key: str,
        stt_queue: asyncio.Queue,
        token_queue: asyncio.Queue,
        config: PipelineConfig,
        source_lang: str = "ru",
        target_lang: str = "en",
        on_transcript: Optional[Callable[[str], None]] = None,
        on_translation: Optional[Callable[[str], None]] = None,
        transcript_filter: Optional[Callable[[str], bool]] = None,
        input_sample_rate: int = _SAMPLE_RATE,
    ) -> None:
        # 45s: audio payloads are larger than text. max_retries=0 — own retry loop below.
        self._client = AsyncOpenAI(api_key=api_key, timeout=45.0, max_retries=0)
        # Native capture rate — audio reaches the API bit-perfect, no local resampling
        self._input_rate = input_sample_rate
        self._min_samples = int(0.2 * input_sample_rate)
        self._stt_queue = stt_queue
        self._token_queue = token_queue
        self._config = config
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._on_transcript = on_transcript
        self._on_translation = on_translation
        # Optional callable(text) -> True to DROP the utterance (e.g. echo of our own TTS)
        self._transcript_filter = transcript_filter
        self._running = False

        src = _LANG_NAMES.get(source_lang, source_lang)
        tgt = _LANG_NAMES.get(target_lang, target_lang)
        mode = config.stt_language_mode
        if mode == "strict":
            self._system_prompt = (
                f"The speaker speaks {src} ONLY. Transcribe+translate audio. Output ONLY:\n"
                f"ORIGINAL: <verbatim transcript in {src} — never any other language>\n"
                f"TRANSLATION: <{tgt}>\n"
                f"Even if a phrase sounds like a related language, transcribe it as {src}.\n"
                "No speech → both empty. No other text."
            )
        elif mode == "hint":
            self._system_prompt = (
                f"Transcribe+translate audio. The speaker usually speaks {src}. Output ONLY:\n"
                f"ORIGINAL: <{src} verbatim>\n"
                f"TRANSLATION: <{tgt}>\n"
                "No speech → both empty. No other text."
            )
        else:  # auto — no language processing, transcribe whatever is spoken
            self._system_prompt = (
                "Transcribe+translate audio. Output ONLY:\n"
                "ORIGINAL: <verbatim transcript in the language actually spoken>\n"
                f"TRANSLATION: <{tgt}>\n"
                "No speech → both empty. No other text."
            )

    async def run(self) -> None:
        self._running = True
        chunks: list[np.ndarray] = []

        while self._running:
            try:
                item = await self._stt_queue.get()
            except asyncio.CancelledError:
                break

            if item is None:
                break

            if item is False:
                if chunks:
                    await self._translate_and_emit(chunks)
                    chunks.clear()
                continue

            chunks.append(item)

        logger.info("OpenAIAudioTranslator stopped")

    async def _translate_and_emit(self, chunks: list[np.ndarray]) -> None:
        audio = np.concatenate(chunks)
        if len(audio) < self._min_samples:
            logger.debug("Utterance too short (%d samples) — skipping", len(audio))
            return

        wav_bytes = _pcm_to_wav(audio, self._input_rate)
        b64_audio = base64.b64encode(wav_bytes).decode()
        t0 = time.monotonic()

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._config.openai_audio_model,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": b64_audio, "format": "wav"},
                                }
                            ],
                        },
                    ],
                    max_tokens=600,
                    temperature=0.1,
                )
                raw = response.choices[0].message.content or ""
                original, translation = _parse_response(raw)
                elapsed = time.monotonic() - t0
                logger.info(
                    "[AudioTranslator] %.2fs  original=%r  translation=%r",
                    elapsed, original, translation,
                )
                # If original is empty the model hallucinated — discard entirely
                if not original:
                    logger.warning("[AudioTranslator] empty original, discarding response: %r", raw)
                    return
                if self._transcript_filter and self._transcript_filter(original):
                    logger.info("[AudioTranslator] dropped by filter (own echo?): %r", original)
                    return
                if self._on_transcript:
                    self._on_transcript(original)
                if translation:
                    if self._on_translation:
                        self._on_translation(translation)
                    await self._token_queue.put(translation)
                    await self._token_queue.put(None)
                return
            except Exception as exc:
                if attempt < self.MAX_RETRIES:
                    delay = self.BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "[AudioTranslator] attempt %d/%d: %s — retrying in %.1fs",
                        attempt, self.MAX_RETRIES, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "[AudioTranslator] failed after %d attempts: %s — dropping utterance",
                        self.MAX_RETRIES, exc,
                    )
                    await self._token_queue.put(None)


def _parse_response(text: str) -> tuple[str, str]:
    """Parse 'ORIGINAL: ...\nTRANSLATION: ...' format. Falls back to full text as translation."""
    original = ""
    translation = ""
    for line in text.splitlines():
        if line.startswith("ORIGINAL:"):
            original = line[len("ORIGINAL:"):].strip()
        elif line.startswith("TRANSLATION:"):
            translation = line[len("TRANSLATION:"):].strip()
    if not original and not translation:
        translation = text.strip()
    return original, translation


def _pcm_to_wav(audio: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()
