from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Callable, Optional

import websockets
import websockets.exceptions

from babellang.config import PipelineConfig

logger = logging.getLogger(__name__)

_WS_BASE = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"


def _build_url(config: PipelineConfig) -> str:
    return (
        _WS_BASE.format(voice_id=config.elevenlabs_voice_id)
        + f"?model_id={config.elevenlabs_model}"
        + f"&output_format={config.elevenlabs_output_format}"
        + "&optimize_streaming_latency=3"
    )


class ElevenLabsTTS:
    """
    ElevenLabs streaming TTS via raw WebSocket.

    Opens the WebSocket on the FIRST token from GPT — without waiting for the
    full translation to finish — so ElevenLabs starts generating audio while
    GPT is still streaming the remaining tokens. This removes GPT generation
    time from the latency budget entirely.
    """

    def __init__(
        self,
        api_key: str,
        token_queue: asyncio.Queue,
        audio_out_queue: asyncio.Queue,
        config: PipelineConfig,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._api_key = api_key
        self._token_queue = token_queue
        self._audio_out_queue = audio_out_queue
        self._config = config
        self._on_error = on_error
        self._running = False

    async def run(self) -> None:
        self._running = True
        while self._running:
            try:
                text: Optional[str] = await self._token_queue.get()
                if not self._running:
                    break
                if text is None:
                    continue  # empty utterance — skip
                # Drain the None sentinel that translator always puts after complete_text
                sentinel = await self._token_queue.get()
                if sentinel is not None:
                    logger.warning("ElevenLabs: expected None sentinel, got: %r", sentinel)
                try:
                    # Hard deadline per utterance: a hung WebSocket would otherwise
                    # block this single-consumer loop forever and silently kill the
                    # whole channel via queue backpressure.
                    deadline = 15.0 + len(text) / 20.0
                    await asyncio.wait_for(self._stream_live(text), timeout=deadline)
                except asyncio.TimeoutError:
                    logger.error("ElevenLabs: no response in %.0fs — skipping utterance %r",
                                 deadline, text[:60])
                    if self._on_error:
                        self._on_error("Озвучка не ответила — фраза пропущена")
                except (websockets.exceptions.WebSocketException, OSError) as exc:
                    logger.warning("ElevenLabs WS error: %s", exc)
                    if self._on_error:
                        self._on_error(str(exc))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("TTS error: %s", exc)
                if self._on_error:
                    self._on_error(str(exc))
        logger.info("ElevenLabsTTS stopped")

    async def _stream_live(self, complete_text: str) -> None:
        t0 = time.monotonic()
        logger.info("[TTS] start: %r", complete_text)
        url = _build_url(self._config)
        headers = {"xi-api-key": self._api_key}

        async with websockets.connect(url, additional_headers=headers) as ws:
            await ws.send(json.dumps({
                "text": " ",
                "try_trigger_generation": True,
                "generation_config": {"chunk_length_schedule": [50]},
            }))

            send_task = asyncio.create_task(self._send_live(ws, complete_text))
            recv_task = asyncio.create_task(self._receive_audio(ws, t0))

            try:
                await asyncio.gather(send_task, recv_task)
            except Exception:
                send_task.cancel()
                recv_task.cancel()
                raise

        logger.info("[TTS] done: %.2fs total for %r", time.monotonic() - t0, complete_text)

    async def _send_live(self, ws, complete_text: str) -> None:
        await ws.send(json.dumps({"text": complete_text + " ", "try_trigger_generation": True}))
        await ws.send(json.dumps({"text": ""}))

    async def _receive_audio(self, ws, t0: float) -> None:
        first_chunk = True
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "audio" in data and data["audio"]:
                if first_chunk:
                    logger.info("[TTS] first audio chunk in %.2fs (TTFB)", time.monotonic() - t0)
                    first_chunk = False
                pcm_bytes = base64.b64decode(data["audio"])
                await self._audio_out_queue.put(pcm_bytes)
            elif "message" in data:
                raise RuntimeError(f"ElevenLabs error: {data['message']}")

    async def stop(self) -> None:
        self._running = False
        await self._token_queue.put(None)


if __name__ == "__main__":
    import os
    import sys

    import numpy as np
    import sounddevice as sd

    async def _test() -> None:
        key = os.environ.get("ELEVENLABS_API_KEY")
        voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
        if not key or not voice_id:
            print("Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID env vars to test")
            sys.exit(1)

        from babellang.config import PipelineConfig
        cfg = PipelineConfig(elevenlabs_voice_id=voice_id)

        tok_q: asyncio.Queue = asyncio.Queue()
        audio_q: asyncio.Queue = asyncio.Queue()

        tts = ElevenLabsTTS(api_key=key, token_queue=tok_q, audio_out_queue=audio_q, config=cfg)
        task = asyncio.create_task(tts.run())

        test_text = "Hello, this is a test of ElevenLabs streaming TTS."
        for word in test_text.split():
            await tok_q.put(word + " ")
        await tok_q.put(None)

        audio_chunks = []
        while True:
            try:
                chunk = await asyncio.wait_for(audio_q.get(), timeout=5.0)
                audio_chunks.append(chunk)
            except asyncio.TimeoutError:
                break

        await tts.stop()
        await task

        if audio_chunks:
            pcm = b"".join(audio_chunks)
            audio_int16 = np.frombuffer(pcm, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            print(f"Received {len(audio_float) / cfg.tts_sample_rate:.2f}s of audio. Playing...")
            sd.play(audio_float, samplerate=cfg.tts_sample_rate)
            sd.wait()
        else:
            print("No audio received")

    asyncio.run(_test())
