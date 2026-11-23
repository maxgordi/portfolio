from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np
from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.listen.v1.types.listen_v1speech_started import ListenV1SpeechStarted
from deepgram.listen.v1.types.listen_v1utterance_end import ListenV1UtteranceEnd

from babellang.config import PipelineConfig

logger = logging.getLogger(__name__)


class DeepgramSTT:
    """
    Streaming WebSocket STT using Deepgram SDK v7.
    Uses client.listen.v1.connect() async context manager.

    Reads complete speech segments (np.ndarray) from stt_queue.
    Emits final transcripts (str) to transcript_queue on sentence-ending punctuation (.?!)
    or via ListenV1UtteranceEnd (utterance_end_ms safety net for fragments without punctuation).
    """

    MAX_RECONNECT = 5
    BACKOFF_BASE = 1.0

    def __init__(
        self,
        api_key: str,
        stt_queue: asyncio.Queue,
        transcript_queue: asyncio.Queue,
        config: PipelineConfig,
        on_transcript=None,
    ) -> None:
        self._api_key = api_key
        self._stt_queue = stt_queue
        self._transcript_queue = transcript_queue
        self._on_transcript = on_transcript
        self._config = config
        self._running = False
        self._reconnect_count = 0
        self._finalize_pending = False   # True after we call send_finalize(); cleared on response

    async def run(self) -> None:
        self._running = True
        while self._running and self._reconnect_count < self.MAX_RECONNECT:
            try:
                await self._run_session()
                self._reconnect_count = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._reconnect_count += 1
                delay = self.BACKOFF_BASE * (2 ** (self._reconnect_count - 1))
                logger.warning(
                    "Deepgram error (%s), reconnecting in %.1fs (attempt %d/%d)",
                    exc, delay, self._reconnect_count, self.MAX_RECONNECT,
                )
                await asyncio.sleep(delay)
        logger.info("DeepgramSTT stopped")

    async def _run_session(self) -> None:
        # Drain audio chunks that accumulated during the reconnect gap.
        # Sending stale audio to a fresh Deepgram session produces out-of-context transcripts.
        drained = 0
        while True:
            try:
                item = self._stt_queue.get_nowait()
                if item is None:
                    await self._stt_queue.put(None)  # restore stop signal
                    return
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.info("Drained %d stale items from stt_queue before new session", drained)

        client = AsyncDeepgramClient(api_key=self._api_key)
        transcript_buf: list[str] = []

        async with client.listen.v1.connect(
            model=self._config.deepgram_model,
            language=self._config.deepgram_language,
            encoding="linear16",
            sample_rate=16000,
            interim_results="true",   # SDK v7 bug: bool True → "True", API needs "true"
            punctuate="true",         # adds . ? ! to transcripts for sentence detection
            endpointing=self._config.deepgram_endpointing,
            utterance_end_ms=self._config.utterance_end_ms,
            vad_events="true",
        ) as socket:
            logger.info("Deepgram WebSocket connected (model=%s, lang=%s)",
                        self._config.deepgram_model, self._config.deepgram_language)
            self._reconnect_count = 0

            recv_task = asyncio.create_task(
                self._receive_loop(socket, transcript_buf), name="dg_recv"
            )
            send_task = asyncio.create_task(
                self._send_loop(socket), name="dg_send"
            )

            done, pending = await asyncio.wait(
                [recv_task, send_task],
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            for t in done:
                exc = t.exception()
                if exc:
                    raise exc

    async def _send_loop(self, socket) -> None:
        # stt_queue protocol: np.ndarray (512-sample chunk) | False (finalize) | None (stop)
        while self._running:
            try:
                item = await asyncio.wait_for(self._stt_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                await socket.send_keep_alive()
                continue

            if item is None:
                return  # stop signal

            if item is False:
                logger.debug("Deepgram send_finalize (VAD silence)")
                self._finalize_pending = True
                await socket.send_finalize()
                continue

            # np.ndarray — 512-sample (32ms) chunk, stream directly
            await socket.send_media(item.tobytes())

    async def _receive_loop(self, socket, transcript_buf: list[str]) -> None:
        async for message in socket:
            if isinstance(message, ListenV1SpeechStarted):
                # Only flush when we sent send_finalize() — means the previous utterance truly ended.
                # Ignore mid-utterance speech_started (micro-pauses within a sentence).
                if self._finalize_pending and transcript_buf:
                    full = " ".join(transcript_buf).strip()
                    transcript_buf.clear()
                    self._finalize_pending = False
                    if full:
                        logger.info("[DISPATCH speech_started+finalize] %r", full)
                        await self._transcript_queue.put(full)
                        if self._on_transcript:
                            self._on_transcript(full)
                else:
                    logger.debug("[speech_started] finalize_pending=%s buf_len=%d — skipping",
                                 self._finalize_pending, len(transcript_buf))

            elif isinstance(message, ListenV1Results):
                text = ""
                try:
                    text = message.channel.alternatives[0].transcript.strip()
                except (IndexError, AttributeError):
                    pass

                if not message.is_final:
                    if text:
                        logger.debug("[interim] %r", text)
                    continue

                # is_final=True — empty means Deepgram had nothing left to flush
                if not text:
                    if message.speech_final and self._finalize_pending:
                        self._finalize_pending = False
                        # Deepgram buffer was empty, but we may have buffered earlier is_finals
                        if transcript_buf:
                            full = " ".join(transcript_buf).strip()
                            transcript_buf.clear()
                            if full:
                                logger.info("[DISPATCH finalize+empty] %r", full)
                                await self._transcript_queue.put(full)
                                if self._on_transcript:
                                    self._on_transcript(full)
                        else:
                            logger.debug("[is_final] finalize acknowledged (empty buf)")
                    else:
                        logger.debug("[is_final] empty — skipping")
                    continue

                transcript_buf.append(text)
                ends_with_punct = text.rstrip()[-1] in ('.', '?', '!')
                logger.debug("[is_final] %r  punct=%s  finalize_pending=%s  buf_len=%d",
                             text, ends_with_punct, self._finalize_pending, len(transcript_buf))

                if ends_with_punct:
                    # Sentence-ending punctuation — dispatch immediately regardless of finalize state
                    full = " ".join(transcript_buf).strip()
                    transcript_buf.clear()
                    self._finalize_pending = False
                    if full:
                        logger.info("[DISPATCH punct] %r", full)
                        await self._transcript_queue.put(full)
                        if self._on_transcript:
                            self._on_transcript(full)
                elif message.speech_final and self._finalize_pending:
                    # Our send_finalize() produced this speech_final — flush buffer as-is
                    full = " ".join(transcript_buf).strip()
                    transcript_buf.clear()
                    self._finalize_pending = False
                    if full:
                        logger.info("[DISPATCH finalize+speech_final] %r", full)
                        await self._transcript_queue.put(full)
                        if self._on_transcript:
                            self._on_transcript(full)
                else:
                    logger.debug("[is_final] buffering (buf: %r)", transcript_buf)

            elif isinstance(message, ListenV1UtteranceEnd):
                # Safety net: fires utterance_end_ms after last is_final with no new words.
                self._finalize_pending = False
                if transcript_buf:
                    full = " ".join(transcript_buf).strip()
                    transcript_buf.clear()
                    if full:
                        logger.info("[DISPATCH utterance_end] %r", full)
                        await self._transcript_queue.put(full)
                        if self._on_transcript:
                            self._on_transcript(full)
                else:
                    logger.debug("[utterance_end] buf empty")

            else:
                logger.debug("[unhandled] %s", type(message).__name__)

    async def stop(self) -> None:
        self._running = False
        await self._stt_queue.put(None)


if __name__ == "__main__":
    import os
    import sys

    async def _test() -> None:
        key = os.environ.get("DEEPGRAM_API_KEY")
        if not key:
            print("Set DEEPGRAM_API_KEY env var to test")
            sys.exit(1)

        from babellang.config import PipelineConfig
        stt_q: asyncio.Queue = asyncio.Queue()
        tx_q: asyncio.Queue = asyncio.Queue()
        cfg = PipelineConfig()
        stt = DeepgramSTT(api_key=key, stt_queue=stt_q, transcript_queue=tx_q, config=cfg)

        task = asyncio.create_task(stt.run())
        await asyncio.sleep(2)

        silence = np.zeros(32000, dtype=np.int16)
        await stt_q.put(silence)
        await asyncio.sleep(3)

        await stt.stop()
        await task
        print("STT test complete")

    asyncio.run(_test())
