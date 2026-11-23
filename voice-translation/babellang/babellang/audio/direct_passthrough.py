from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class DirectPassthrough:
    """
    Zero-processing mic → output passthrough at the output device's native sample rate.

    Captures microphone at the output device's default rate (typically 44100 or 48000 Hz),
    scales by volume, and writes directly to the output — bypassing the 16kHz VAD pipeline
    entirely. This gives the interlocutor the highest possible voice quality.

    volume is set from the asyncio event loop via call_soon_threadsafe — it is written as
    a plain float which Python's GIL makes atomic for simple assignments.
    """

    # While outgoing TTS is playing, the live voice ducks to this fraction so the
    # interlocutor hears the translation clearly instead of two voices at once.
    _DUCKED_LEVEL = 0.15

    def __init__(self, input_device: int, output_device: int, volume: float = 0.0) -> None:
        self._input_device = input_device
        self._output_device = output_device
        self._volume: float = max(0.0, min(1.0, volume))
        self._stream: Optional[sd.Stream] = None
        self._out_channels: int = 1
        self._duck_target: float = 1.0   # 1.0 = no duck; _DUCKED_LEVEL while TTS plays
        self._duck_cur: float = 1.0      # smoothed in the callback

    # ── Volume control (thread-safe for simple float write) ───────────────────

    def set_volume(self, v: float) -> None:
        self._volume = max(0.0, min(1.0, v))

    def set_ducked(self, ducked: bool) -> None:
        """Duck the live voice while translated TTS is being sent to the interlocutor."""
        self._duck_target = self._DUCKED_LEVEL if ducked else 1.0

    # ── PortAudio callback ────────────────────────────────────────────────────

    def _callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time,
        status: sd.CallbackFlags,
    ) -> None:
        # An exception escaping a PortAudio callback aborts the stream forever —
        # fall back to silence instead.
        try:
            if status:
                logger.debug("DirectPassthrough status: %s", status)
            # Smooth duck transition (~100ms at 512-frame blocks / 48kHz)
            self._duck_cur += (self._duck_target - self._duck_cur) * 0.1
            # indata: (frames, 1) mono mic
            scaled = indata[:, 0:1] * (self._volume * self._duck_cur)
            # Duplicate mono to all output channels
            if self._out_channels > 1:
                outdata[:] = np.tile(scaled, (1, self._out_channels))
            else:
                outdata[:] = scaled
        except Exception:
            outdata.fill(0)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        out_info = sd.query_devices(self._output_device)
        in_info  = sd.query_devices(self._input_device)

        # Use output device native rate — both BlackHole and MacBook mic support 44100/48000
        native_rate = int(out_info.get("default_samplerate", 48000))
        in_ch  = max(1, min(int(in_info["max_input_channels"]),  1))
        out_ch = max(1, min(int(out_info["max_output_channels"]), 2))
        self._out_channels = out_ch

        self._stream = sd.Stream(
            device=(self._input_device, self._output_device),
            samplerate=native_rate,
            blocksize=512,
            dtype="float32",
            channels=(in_ch, out_ch),
            callback=self._callback,
            latency="low",
        )
        self._stream.start()
        logger.info(
            "DirectPassthrough started: [%s] → [%s]  %dHz  in_ch=%d out_ch=%d  vol=%.0f%%",
            in_info["name"], out_info["name"], native_rate, in_ch, out_ch,
            self._volume * 100,
        )

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("DirectPassthrough stopped")
