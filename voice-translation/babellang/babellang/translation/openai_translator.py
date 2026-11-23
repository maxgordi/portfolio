from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional

from openai import AsyncOpenAI

from babellang.config import PipelineConfig

logger = logging.getLogger(__name__)

_LANG_NAMES: dict[str, str] = {
    "ru": "Russian",
    "uk": "Ukrainian",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "ar": "Arabic",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "hi": "Hindi",
}


def _build_system_prompt(source_lang: str, target_lang: str) -> str:
    src = _LANG_NAMES.get(source_lang, source_lang)
    tgt = _LANG_NAMES.get(target_lang, target_lang)
    return (
        "You are a simultaneous interpreter. "
        f"Translate the following from {src} to {tgt}. "
        "Preserve tone and meaning. "
        f"The input is speech-to-text output and may contain mis-transcriptions or "
        f"words that drifted into a related language — still translate the intended "
        f"meaning into {tgt}. The output must ALWAYS be in {tgt} and nothing else. "
        "Output ONLY the translation — no explanations, no notes, no quotes."
    )


class OpenAITranslator:
    """
    Streaming GPT-4o-mini translator.
    Reads transcripts from transcript_queue.
    Streams translated tokens to token_queue.
    Sends None sentinel after each utterance completes (signals TTS to flush).
    Direction is controlled by source_lang / target_lang (default: ru → en).
    """

    def __init__(
        self,
        api_key: str,
        transcript_queue: asyncio.Queue,
        token_queue: asyncio.Queue,
        config: PipelineConfig,
        on_translation=None,
        source_lang: str = "ru",
        target_lang: str = "en",
    ) -> None:
        # timeout applies to connect AND gaps between stream chunks — a hung
        # stream otherwise blocks translation forever (default is ~10 min)
        self._client = AsyncOpenAI(api_key=api_key, timeout=30.0, max_retries=1)
        self._transcript_queue = transcript_queue
        self._token_queue = token_queue
        self._config = config
        self._on_translation = on_translation
        self._system_prompt = _build_system_prompt(source_lang, target_lang)
        self._context: deque[dict] = deque(maxlen=config.openai_context_turns * 2)
        self._running = False

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                text: Optional[str] = await self._transcript_queue.get()
                if text is None:
                    break
                await self._translate(text)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Translator error: %s", exc)
        logger.info("OpenAITranslator stopped")

    async def _translate(self, text: str) -> None:
        t0 = time.monotonic()
        logger.info("[TRANSLATE] start: %r", text)
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._context)
        messages.append({"role": "user", "content": text})

        full_tokens: list[str] = []
        try:
            stream = await self._client.chat.completions.create(
                model=self._config.openai_model,
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.1,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_tokens.append(delta)
        except Exception as exc:
            logger.error("OpenAI stream error: %s", exc)
        finally:
            elapsed = time.monotonic() - t0
            translation = "".join(full_tokens)
            if full_tokens:
                logger.info("[TRANSLATE] done in %.2fs: %r → %r", elapsed, text, translation)
                await self._token_queue.put(translation)
            else:
                logger.warning("[TRANSLATE] empty result in %.2fs for: %r", elapsed, text)
            await self._token_queue.put(None)

        if full_tokens:
            translation = "".join(full_tokens)
            self._context.append({"role": "user", "content": text})
            self._context.append({"role": "assistant", "content": translation})
            if self._on_translation:
                self._on_translation(translation)

    async def stop(self) -> None:
        self._running = False
        await self._transcript_queue.put(None)


if __name__ == "__main__":
    import os
    import sys

    async def _test() -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("Set OPENAI_API_KEY env var to test")
            sys.exit(1)

        tx_q: asyncio.Queue = asyncio.Queue()
        tok_q: asyncio.Queue = asyncio.Queue()
        cfg = PipelineConfig()
        translator = OpenAITranslator(api_key=key, transcript_queue=tx_q, token_queue=tok_q, config=cfg)

        task = asyncio.create_task(translator.run())
        await tx_q.put("Привет, как дела?")
        await asyncio.sleep(0.1)

        tokens = []
        while True:
            tok = await tok_q.get()
            if tok is None:
                break
            tokens.append(tok)

        print("Translation:", "".join(tokens))
        await translator.stop()
        await task

    asyncio.run(_test())
