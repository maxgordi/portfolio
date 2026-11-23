from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD wrapper for async use.
    Runs model inference in a dedicated ThreadPoolExecutor (not MPS — thread safety).
    Accepts 512-sample int16 arrays at 16kHz (32ms per chunk).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._model: Optional[object] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="silero_vad"
        )

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        logger.info("Loading Silero VAD model (first run may download ~100MB)...")
        self._model = await loop.run_in_executor(self._executor, self._load_model)
        logger.info("Silero VAD model loaded")

    def _load_model(self) -> object:
        model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        model.eval()
        return model

    async def process_chunk(self, audio_int16: np.ndarray) -> float:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._infer,
            audio_int16.copy(),
        )

    def _infer(self, audio_int16: np.ndarray) -> float:
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_float32)
        with torch.no_grad():
            return self._model(tensor, 16000).item()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


if __name__ == "__main__":
    async def _test() -> None:
        vad = SileroVAD()
        await vad.load()

        silence = np.zeros(512, dtype=np.int16)
        prob = await vad.process_chunk(silence)
        print(f"Silence  → VAD prob: {prob:.4f}  ({'speech' if prob > 0.5 else 'silence'})")

        t = np.linspace(0, 0.032, 512)
        sine = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        prob = await vad.process_chunk(sine)
        print(f"440Hz tone → VAD prob: {prob:.4f}  ({'speech' if prob > 0.5 else 'not speech'})")

        print("VAD test complete.")

    asyncio.run(_test())
