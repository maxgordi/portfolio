"""The microphone as a stream of 20 ms chunks of 24 kHz 16-bit mono PCM,
at the device's lowest latency, so a phrase is cut and sent as soon as
the pause after it is over."""
import io
import sys
import threading
import wave
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from audio import (
    AudioRecorder,
    _close_stream,
    _live_streams,
    _pa_lock,
    _reset_portaudio,
    _windows_fallback_inputs,
)

RATE = 24000
BLOCK = RATE // 50  # 20 ms

# Every device takes 48 kHz; it is captured there and halved when the
# audio system can't give 24 kHz directly (some Linux ALSA devices).
_FALLBACK_RATE = RATE * 2


def to_wav(pcm: bytes, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _halve(pcm: bytes) -> bytes:
    x = np.frombuffer(pcm, dtype="<i2")
    x = x[: len(x) // 2 * 2].astype(np.int32)
    return ((x[0::2] + x[1::2]) // 2).astype("<i2").tobytes()


class MicStream:
    def __init__(self):
        self._stream = None
        self._on_chunk: Optional[Callable[[bytes], None]] = None
        self._halve = False
        self.backend = ""  # how the stream was opened, for the log

    def start(self, on_chunk: Callable[[bytes], None],
              device_index: int = -1, device_name: str = "") -> None:
        self.stop()
        self._on_chunk = on_chunk
        device = AudioRecorder._resolve_device(device_index, device_name)
        attempts = []
        for dev in ([device] if device is not None else []) + [None]:
            attempts.append((dev, None, RATE, "default"))
            attempts.append((dev, None, _FALLBACK_RATE, "48 kHz"))
        if sys.platform == "win32":
            attempts += [(dev, extra, RATE, name)
                         for dev, extra, name in _windows_fallback_inputs()]
        last_err: Optional[Exception] = None
        for i, (dev, extra, rate, name) in enumerate(attempts):
            if i == 2:
                _reset_portaudio(force=True)  # the device list may be stale
            try:
                self._open(dev, extra, rate)
                self.backend = name
                return
            except Exception as e:
                last_err = e
        self._on_chunk = None
        raise last_err

    def _open(self, device, extra_settings, rate: int) -> None:
        stream = None
        try:
            with _pa_lock:
                stream = sd.RawInputStream(
                    samplerate=rate, channels=1, dtype="int16",
                    blocksize=rate // 50, latency="low", device=device,
                    extra_settings=extra_settings, callback=self._callback)
                _live_streams.add(stream)  # keep a ref while the driver calls back
                self._halve = rate == _FALLBACK_RATE
                stream.start()
            self._stream = stream
        except Exception:
            _close_stream(stream)
            raise

    def _callback(self, indata, frames, time_info, status) -> None:
        # Realtime audio thread: never raise, never block.
        try:
            cb = self._on_chunk
            if cb is not None:
                pcm = bytes(indata)
                cb(_halve(pcm) if self._halve else pcm)
        except Exception:
            pass

    def stop(self) -> None:
        self._on_chunk = None
        stream, self._stream = self._stream, None
        _close_stream(stream)


class FileMic:
    """Stand-in microphone for end-to-end tests (PRESSVOICE_FAKE_MIC):
    delivers the WAV file once, as soon as it is opened."""

    backend = "file"

    def __init__(self, wav_path: str):
        self._path = wav_path

    def start(self, on_chunk: Callable[[bytes], None],
              device_index: int = -1, device_name: str = "") -> None:
        with wave.open(self._path, "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        threading.Thread(target=on_chunk, args=(pcm,), daemon=True).start()

    def stop(self) -> None:
        pass
