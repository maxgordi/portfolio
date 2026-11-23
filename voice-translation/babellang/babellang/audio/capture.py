from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Optional

import numpy as np
import sounddevice as sd

from babellang.config import AudioConfig

logger = logging.getLogger(__name__)

# int16 peak above this counts as "signal present" (~1% of full scale)
_SIGNAL_PEAK_THRESHOLD = 330


class AudioCapture:
    """
    Wraps sounddevice.InputStream for async use.
    PortAudio callback writes to a deque ring buffer (non-blocking);
    call_soon_threadsafe drains it into an asyncio.Queue.

    Captures at the DEVICE'S NATIVE sample rate (usually 44100/48000) so the
    signal reaches STT bit-perfect — no local downsampling in the quality path.
    Chunks are still 32ms each; `sample_rate` tells consumers the actual rate.
    """

    def __init__(
        self,
        device: Optional[int],
        config: AudioConfig,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._device = device
        self._config = config
        self._queue = queue
        self._loop = loop
        self._stream: Optional[sd.InputStream] = None
        self._ring: deque[np.ndarray] = deque(maxlen=200)
        self._overflow_count = 0
        self._callback_error_count = 0
        # Wall-clock time audible signal last hit this input — lets the GUI warn
        # "no sound from the interlocutor" when call-app routing is wrong.
        self.last_signal_time: float = time.time()
        # Native device rate, known at construction so STT components can be
        # built with the right WAV header rate before the stream even starts.
        try:
            info = sd.query_devices(device) if device is not None \
                else sd.query_devices(kind="input")
            self.sample_rate: int = int(info.get("default_samplerate") or config.sample_rate)
        except Exception:
            self.sample_rate = config.sample_rate
        # 32ms worth of samples at the native rate (e.g. 1536 @ 48kHz)
        self.chunk_samples: int = max(1, round(self.sample_rate * config.chunk_samples
                                               / config.sample_rate))
        logger.info("AudioCapture: device=%s native_rate=%d chunk=%d",
                    device, self.sample_rate, self.chunk_samples)

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        # An exception escaping a PortAudio callback aborts the stream forever.
        try:
            if status.input_overflow:
                self._overflow_count += 1
                if self._overflow_count % 10 == 1:
                    self._loop.call_soon_threadsafe(
                        logger.warning, "Audio input overflow (count=%d)", self._overflow_count
                    )
            mono = indata[:, 0]
            if np.max(np.abs(mono)) > _SIGNAL_PEAK_THRESHOLD:
                self.last_signal_time = time.time()
            self._ring.append(mono.copy())
            self._loop.call_soon_threadsafe(self._drain_ring)
        except Exception:
            self._callback_error_count += 1

    def _drain_ring(self) -> None:
        while self._ring:
            frame = self._ring.popleft()
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    async def start(self) -> None:
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self.sample_rate,
            channels=self._config.channels,
            dtype=self._config.dtype,
            blocksize=self.chunk_samples,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Audio capture started: device=%s, rate=%d, chunk=%d",
            self._device,
            self.sample_rate,
            self.chunk_samples,
        )

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped")


def list_devices() -> list[dict]:
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        devices.append({
            "index": i,
            "name": dev["name"],
            "max_input_channels": dev["max_input_channels"],
            "max_output_channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
        })
    return devices


def find_device_by_name(name_substr: str, require_input: bool = False, require_output: bool = False) -> Optional[int]:
    substr_lower = name_substr.lower()
    for dev in list_devices():
        if substr_lower not in dev["name"].lower():
            continue
        if require_input and dev["max_input_channels"] == 0:
            continue
        if require_output and dev["max_output_channels"] == 0:
            continue
        return dev["index"]
    return None


if __name__ == "__main__":
    import asyncio

    async def _test() -> None:
        q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=100)
        loop = asyncio.get_event_loop()
        cfg = AudioConfig()
        cap = AudioCapture(device=None, config=cfg, queue=q, loop=loop)
        print("Starting capture for 2 seconds...")
        await cap.start()
        await asyncio.sleep(2.0)
        await cap.stop()
        frames = []
        while not q.empty():
            frames.append(q.get_nowait())
        print(f"Captured {len(frames)} frames ({len(frames) * 32}ms of audio)")

    asyncio.run(_test())
