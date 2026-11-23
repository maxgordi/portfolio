from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Optional

import numpy as np
import sounddevice as sd
import torch
import torchaudio

logger = logging.getLogger(__name__)


class _SampleBuffer:
    """
    Single-producer / single-consumer float32 sample buffer.

    Producer (asyncio thread) appends whole numpy arrays with write();
    consumer (PortAudio callback thread) takes exactly `frames` samples with read().
    deque.append/popleft are GIL-atomic, so no lock is needed. The read-side
    remainder (partially consumed chunk) is touched only by the callback thread.
    """

    def __init__(self, max_chunks: int = 0) -> None:
        self._chunks: deque[np.ndarray] = (
            deque(maxlen=max_chunks) if max_chunks > 0 else deque()
        )
        self._rem: Optional[np.ndarray] = None  # callback-thread only

    def write(self, samples: np.ndarray) -> None:
        if len(samples):
            self._chunks.append(samples)

    def extend(self, samples) -> None:
        """List/iterable convenience (used by tests)."""
        self.write(np.asarray(list(samples), dtype=np.float32))

    def __len__(self) -> int:
        n = sum(len(c) for c in self._chunks)
        if self._rem is not None:
            n += len(self._rem)
        return n

    def __iter__(self):
        if self._rem is not None:
            yield from self._rem
        for c in list(self._chunks):
            yield from c

    def clear_pending(self) -> None:
        self._chunks.clear()

    @property
    def has_data(self) -> bool:
        return bool(self._chunks) or self._rem is not None

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        """Return exactly `frames` samples (zero-padded) + flag whether any real data was read."""
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        if self._rem is not None:
            take = min(frames, len(self._rem))
            out[:take] = self._rem[:take]
            self._rem = self._rem[take:] if take < len(self._rem) else None
            filled = take
        while filled < frames and self._chunks:
            try:
                chunk = self._chunks.popleft()
            except IndexError:
                break
            take = min(frames - filled, len(chunk))
            out[filled:filled + take] = chunk[:take]
            if take < len(chunk):
                self._rem = chunk[take:]
            filled += take
        return out, filled > 0


class AudioPlayback:
    """
    Writes PCM audio to a sounddevice output device.
    Accepts raw signed-16-bit PCM bytes from an asyncio.Queue (TTS stream).
    Optionally mixes in a pass-through of original audio at a configurable ratio
    via feed_passthrough() — called from the asyncio event loop.
    """

    def __init__(
        self,
        device: Optional[int],
        sample_rate: int,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        original_ratio: float = 0.0,
        smooth_ducking: bool = True,
        tts_gain: float = 1.0,
    ) -> None:
        self._device = device
        self._tts_sample_rate = sample_rate  # ElevenLabs output rate (e.g. 24000)
        self._sample_rate = sample_rate      # overridden to device native rate in start()
        self._queue = queue
        self._loop = loop
        self._original_ratio = max(0.0, min(1.0, original_ratio))
        self._smooth_ducking = smooth_ducking
        self._tts_gain = max(0.0, min(1.0, tts_gain))
        self._stream: Optional[sd.OutputStream] = None
        self._play_buffer = _SampleBuffer()
        self._passthrough_buffer = _SampleBuffer()
        self._drain_task: Optional[asyncio.Task] = None
        self._underflow_count = 0
        self._channels: int = 1  # resolved in start()
        self._duck_ratio: float = 1.0   # 1.0 = full passthrough, 0.0 = ducked to silence
        self._duck_out_step: float = 0.15  # recalculated in start()
        self._duck_in_step: float = 0.05   # recalculated in start()
        self._last_tts_active: float = 0.0  # wall-clock time TTS samples last hit the device
        self._pt_resampler: Optional[torchaudio.transforms.Resample] = None
        self._tts_resampler: Optional[torchaudio.transforms.Resample] = None
        self._init_resamplers()

    def set_original_ratio(self, ratio: float) -> None:
        self._original_ratio = max(0.0, min(1.0, ratio))
        if self._original_ratio > 0.0 and self._pt_resampler is None:
            self._init_resamplers()

    def set_smooth_ducking(self, enabled: bool) -> None:
        self._smooth_ducking = enabled

    def set_tts_gain(self, gain: float) -> None:
        self._tts_gain = max(0.0, min(1.0, gain))

    @property
    def is_playing(self) -> bool:
        """True while TTS audio is actually being written to the output device."""
        return self._play_buffer.has_data

    def tts_active_within(self, seconds: float) -> bool:
        """True if TTS samples hit the device within the last `seconds` (echo-gate hangover)."""
        return self.is_playing or (time.monotonic() - self._last_tts_active) < seconds

    def _init_resamplers(self) -> None:
        self._pt_src_rate = getattr(self, "_pt_src_rate", 16000)
        self._pt_resampler = (
            torchaudio.transforms.Resample(self._pt_src_rate, self._sample_rate)
            if self._original_ratio > 0.0 else None
        )
        self._tts_resampler = (
            torchaudio.transforms.Resample(self._tts_sample_rate, self._sample_rate)
            if self._tts_sample_rate != self._sample_rate else None
        )

    def feed_passthrough(self, audio_int16: np.ndarray, src_sample_rate: int = 16000) -> None:
        """
        Resample raw mic/interlocutor audio to playback rate and add to passthrough buffer.
        Called from the asyncio event loop. No-op when original_ratio is 0.0.
        Handles any source rate — capture now runs at the device's native rate.
        """
        if self._original_ratio <= 0.0:
            return
        if self._pt_resampler is None or src_sample_rate != self._pt_src_rate:
            self._pt_src_rate = src_sample_rate
            self._pt_resampler = torchaudio.transforms.Resample(src_sample_rate, self._sample_rate)
        if src_sample_rate == self._sample_rate:
            self._passthrough_buffer.write(audio_int16.astype(np.float32) / 32768.0)
            return
        audio_t = torch.from_numpy(audio_int16.astype(np.float32)).unsqueeze(0) / 32768.0
        resampled = self._pt_resampler(audio_t).squeeze(0).numpy()
        self._passthrough_buffer.write(resampled)

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        # An exception escaping a PortAudio callback aborts the stream FOREVER —
        # emit silence and keep the stream alive instead.
        try:
            self._callback_inner(outdata, frames, status)
        except Exception:
            outdata.fill(0)
            self._callback_error_count = getattr(self, "_callback_error_count", 0) + 1

    def _callback_inner(self, outdata: np.ndarray, frames: int, status: sd.CallbackFlags) -> None:
        if status.output_underflow:
            self._underflow_count += 1

        # ── TTS drain ──────────────────────────────────────────────────
        tts, tts_active = self._play_buffer.read(frames)
        if tts_active:
            self._last_tts_active = time.monotonic()
            if self._tts_gain != 1.0:
                tts *= self._tts_gain

        # ── Update duck ratio ───────────────────────────────────────────
        if tts_active:
            if self._smooth_ducking:
                self._duck_ratio = max(0.0, self._duck_ratio - self._duck_out_step)
            else:
                self._duck_ratio = 0.0
        else:
            if self._smooth_ducking:
                self._duck_ratio = min(1.0, self._duck_ratio + self._duck_in_step)
            else:
                self._duck_ratio = 1.0

        # ── Mix passthrough with duck applied ──────────────────────────
        if self._original_ratio > 0.0:
            effective_ratio = self._original_ratio * self._duck_ratio  # 0..original_ratio
            pt, _ = self._passthrough_buffer.read(frames)
            # TTS + original voice ducked while TTS is active
            mixed = np.clip(tts + pt * effective_ratio, -1.0, 1.0)
        else:
            mixed = tts

        outdata[:, 0] = mixed
        # Duplicate mono to all channels (BlackHole 2ch requires stereo)
        if outdata.shape[1] > 1:
            outdata[:, 1:] = outdata[:, :1]

    async def start(self) -> None:
        dev_info = sd.query_devices(self._device)
        self._channels = max(1, min(2, dev_info["max_output_channels"]))
        self._sample_rate = int(dev_info.get("default_samplerate", self._sample_rate))
        cb_interval = 1024 / self._sample_rate
        self._duck_out_step = cb_interval / 1.5    # reach 0 in ~1.5s
        self._duck_in_step  = cb_interval / 0.8    # reach 1 in ~0.8s
        # Cap passthrough to ~1 second of 512-sample chunks — stale audio never plays after long TTS
        self._passthrough_buffer = _SampleBuffer(max_chunks=self._sample_rate // 512 + 1)
        # Recreate resamplers with actual device rate (may differ from init-time sample_rate)
        self._init_resamplers()

        self._stream = sd.OutputStream(
            device=self._device,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()
        self._drain_task = asyncio.create_task(self._drain_queue(), name="playback_drain")
        logger.info(
            "Audio playback started: device=%s (%s), device_rate=%d, tts_rate=%d, channels=%d",
            self._device, dev_info["name"], self._sample_rate, self._tts_sample_rate, self._channels,
        )

    async def _drain_queue(self) -> None:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                continue
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            if self._tts_resampler is not None:
                audio_t = torch.from_numpy(audio_int16.astype(np.float32)).unsqueeze(0) / 32768.0
                audio_float32 = self._tts_resampler(audio_t).squeeze(0).numpy()
            else:
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
            self._play_buffer.write(audio_float32)

    async def stop(self) -> None:
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio playback stopped")
