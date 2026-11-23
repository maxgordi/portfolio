from __future__ import annotations

import asyncio
import concurrent.futures
import io
import logging
import time
import wave
from typing import Optional

import numpy as np
from openai import OpenAI

from babellang.config import PipelineConfig

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2       # int16 = 2 bytes
_CHANNELS = 1
_MIN_SAMPLES = 3200     # 200ms at 16kHz — skip shorter utterances

# The `language` API parameter is only a soft hint — for closely related languages
# (uk/ru/be) the model still drifts. A short prompt written IN the target language
# anchors both the language and its spelling much harder.
_LANG_PROMPTS: dict[str, str] = {
    "ru": "Это разговор на русском языке.",
    "uk": "Це розмова українською мовою. Транскрибуй українською.",
    "en": "This is a conversation in English.",
    "fr": "Ceci est une conversation en français.",
    "es": "Esta es una conversación en español.",
    "de": "Dies ist ein Gespräch auf Deutsch.",
    "it": "Questa è una conversazione in italiano.",
    "pt": "Esta é uma conversa em português.",
    "zh": "这是一段中文对话。",
    "ja": "これは日本語での会話です。",
    "ko": "이것은 한국어 대화입니다.",
    "vi": "Đây là cuộc trò chuyện bằng tiếng Việt.",
    "th": "นี่คือบทสนทนาภาษาไทย",
    "ar": "هذه محادثة باللغة العربية.",
    "tr": "Bu Türkçe bir konuşmadır.",
    "pl": "To jest rozmowa w języku polskim.",
    "nl": "Dit is een gesprek in het Nederlands.",
    "hi": "यह हिंदी में बातचीत है।",
}

# Whisper-family models hallucinate these phrases on silence/noise —
# they must never reach translation/TTS. Compared lowercase, punctuation stripped.
_HALLUCINATIONS: frozenset[str] = frozenset({
    # uk
    "дякую за перегляд", "дякуємо за перегляд", "дякую за увагу",
    "субтитри створив dimatorzok", "субтитри зробив dimatorzok",
    # ru
    "спасибо за просмотр", "благодарю за просмотр", "спасибо за внимание",
    "субтитры делал dimatorzok", "субтитры сделал dimatorzok",
    "продолжение следует", "редактор субтитров а семкин корректор а егорова",
    # en
    "thank you for watching", "thanks for watching", "thank you",
    "subtitles by the amara org community", "you",
    # nl
    "bedankt voor het kijken", "ondertitels ingediend door de amara org gemeenschap",
    "ondertiteld door de amara org gemeenschap", "dank u wel",
})


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy text comparison."""
    cleaned = "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace())
    return " ".join(cleaned.split())


def _is_hallucination(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return True
    if cleaned in _HALLUCINATIONS:
        return True
    # Single token repeated many times ("бла бла бла бла...") is noise, not speech
    words = cleaned.split()
    if len(words) >= 4 and len(set(words)) == 1:
        return True
    return False


class OpenAISTT:
    """
    Batch REST STT using OpenAI audio transcriptions API.

    stt_queue protocol:
      np.ndarray  — 512-sample int16 chunk (32ms); accumulate into buffer
      False       — end of utterance (VAD silence or force-finalize); transcribe and clear buffer
      None        — stop signal

    transcript_queue emits: str (complete transcribed utterance)

    The sync OpenAI call runs in a ThreadPoolExecutor (max_workers=1) to avoid
    blocking the asyncio event loop. stdlib `wave` wraps raw PCM into WAV in-memory.
    """

    MAX_RETRIES = 3
    BACKOFF_BASE = 1.0

    def __init__(
        self,
        api_key: str,
        stt_queue: asyncio.Queue,
        transcript_queue: asyncio.Queue,
        config: PipelineConfig,
        on_transcript=None,
        transcript_filter=None,
        input_sample_rate: int = _SAMPLE_RATE,
    ) -> None:
        # timeout=30s: default is ~10 min, and one hung request would stall the
        # single-worker executor (and the whole channel) for that long.
        # max_retries=0: this class has its own retry loop with backoff.
        self._client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
        self._stt_queue = stt_queue
        self._transcript_queue = transcript_queue
        self._config = config
        self._on_transcript = on_transcript
        # Optional callable(text) -> True to DROP the transcript (e.g. echo of our own TTS)
        self._transcript_filter = transcript_filter
        # Native capture rate: audio goes to the API bit-perfect at this rate,
        # no local resampling in the quality path
        self._input_rate = input_sample_rate
        self._min_samples = int(0.2 * input_sample_rate)  # skip utterances < 200ms
        self._running = False
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="openai_stt"
        )

    async def run(self) -> None:
        self._running = True
        chunks: list[np.ndarray] = []

        try:
            while self._running:
                try:
                    item = await self._stt_queue.get()
                except asyncio.CancelledError:
                    break

                if item is None:
                    break

                if item is False:
                    if chunks:
                        await self._transcribe_and_emit(chunks)
                        chunks.clear()
                    else:
                        logger.debug("[OpenAISTT] False received but buffer empty — skipping")
                    continue

                chunks.append(item)
        finally:
            self._executor.shutdown(wait=False)

        logger.info("OpenAISTT stopped")

    async def _transcribe_and_emit(self, chunks: list[np.ndarray]) -> None:
        audio = np.concatenate(chunks)

        if len(audio) < self._min_samples:
            logger.debug(
                "[OpenAISTT] utterance too short (%d samples, %.0fms) — skipping",
                len(audio), len(audio) * 1000 / self._input_rate,
            )
            return

        wav_bytes = _pcm_to_wav(audio, self._input_rate)
        t0 = time.monotonic()

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(self._executor, self._call_api, wav_bytes)
                if text and _is_hallucination(text):
                    logger.info("[OpenAISTT] hallucination filtered: %r", text)
                    return
                # Whisper prompt-echo: on silence/noise the model may return the
                # language-anchor prompt itself as the "transcript" — never real speech.
                if text and self._config.stt_language_mode == "strict":
                    anchor = _LANG_PROMPTS.get(self._config.stt_language, "")
                    if anchor and normalize_text(text) == normalize_text(anchor):
                        logger.info("[OpenAISTT] prompt echo filtered: %r", text)
                        return
                if text and self._transcript_filter and self._transcript_filter(text):
                    logger.info("[OpenAISTT] transcript dropped by filter (own echo?): %r", text)
                    return
                if text:
                    logger.info("[OpenAISTT] %.2fs: %r", time.monotonic() - t0, text)
                    await self._transcript_queue.put(text)
                    if self._on_transcript:
                        self._on_transcript(text)
                else:
                    logger.debug("[OpenAISTT] empty transcript returned")
                return

            except Exception as exc:
                if attempt < self.MAX_RETRIES:
                    delay = self.BACKOFF_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "[OpenAISTT] error attempt %d/%d: %s — retrying in %.1fs",
                        attempt, self.MAX_RETRIES, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "[OpenAISTT] failed after %d attempts: %s — dropping utterance",
                        self.MAX_RETRIES, exc,
                    )

    def _build_api_kwargs(self, wav_bytes: bytes) -> dict:
        """
        stt_language_mode:
          strict — language param + prompt anchor (uk never drifts into ru/be)
          hint   — language param only (model may drift)
          auto   — raw transcription, model detects the language itself
        """
        buf = io.BytesIO(wav_bytes)
        kwargs: dict = {
            "model": self._config.openai_stt_model,
            "file": ("audio.wav", buf, "audio/wav"),
        }
        mode = self._config.stt_language_mode
        if mode in ("strict", "hint"):
            kwargs["language"] = self._config.stt_language
        if mode == "strict":
            lang_prompt = _LANG_PROMPTS.get(self._config.stt_language)
            if lang_prompt:
                kwargs["prompt"] = lang_prompt
        return kwargs

    def _call_api(self, wav_bytes: bytes) -> str:
        response = self._client.audio.transcriptions.create(**self._build_api_kwargs(wav_bytes))
        return response.text.strip()

    async def stop(self) -> None:
        self._running = False
        await self._stt_queue.put(None)


def _pcm_to_wav(audio: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()
