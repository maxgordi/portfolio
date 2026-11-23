import io
import sys
import threading
import time
import wave
from typing import Optional

import numpy as np
import sounddevice as sd

# Every PortAudio call in this process goes through this lock: opening,
# starting, stopping, closing, terminating and querying devices. PortAudio's
# device list and global state are not thread-safe, and this app touches them
# from the hotkey thread, the wake handler and the settings window.
_pa_lock = threading.RLock()

# Strong references to streams that PortAudio may still be calling back into.
# A stream must never be garbage collected while running: CoreAudio would keep
# invoking a freed callback closure and the process would segfault.
_live_streams: set = set()

_IS_WINDOWS = sys.platform == "win32"

MAX_SECONDS = 900  # hard cap on one recording, so a stuck key can't eat all RAM

# Terminating and reinitializing PortAudio churns CoreAudio's HAL state, which
# can wedge if done in a tight loop, so rescans are rate limited.
_RESET_COOLDOWN = 2.0
_last_reset = 0.0


def _default_hostapi() -> Optional[int]:
    """Index of the platform's default audio API. Windows lists every
    microphone once per API (MME, DirectSound, WASAPI, WDM-KS); only the
    default one (MME) is offered, and it also resamples to 16 kHz for us.
    macOS has only CoreAudio, so this changes nothing there."""
    try:
        return sd.default.hostapi
    except Exception:
        return None


def _usable_input(dev, hostapi: Optional[int]) -> bool:
    if dev["max_input_channels"] <= 0:
        return False
    return hostapi is None or dev.get("hostapi", hostapi) == hostapi


def _windows_fallback_inputs() -> list:
    """Default inputs of the other Windows audio APIs, as
    (device index, extra settings, API name). Some drivers refuse MME but
    work through WASAPI or DirectSound."""
    out = []
    try:
        apis = sd.query_hostapis()
    except Exception:
        return out
    for wanted in ("Windows WASAPI", "Windows DirectSound"):
        for api in apis:
            dev = api.get("default_input_device", -1)
            if api.get("name") != wanted or dev is None or dev < 0:
                continue
            extra = None
            if wanted == "Windows WASAPI":
                try:
                    # Shared-mode WASAPI only opens at the device's own rate
                    # unless Windows is allowed to resample to our 16 kHz.
                    extra = sd.WasapiSettings(auto_convert=True)
                except Exception:
                    pass
            out.append((dev, extra, wanted))
    return out


def wav_peak(wav_bytes: bytes) -> float:
    """Loudest sample of a 16-bit WAV, 0.0–1.0."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    except Exception:
        return 0.0
    if samples.size == 0:
        return 0.0
    return float(np.abs(samples.astype(np.int32)).max()) / 32768.0


def _close_stream(stream) -> None:
    """Tear a stream down completely. Safe to call on a dead device."""
    if stream is None:
        return
    with _pa_lock:
        try:
            stream.abort(ignore_errors=True)
        except Exception:
            pass
        try:
            stream.close(ignore_errors=True)
        except Exception:
            pass
        _live_streams.discard(stream)


def _reset_portaudio(force: bool = False) -> None:
    """Reinitialize PortAudio so it re-scans audio devices.

    PortAudio caches the device list at init; after sleep/wake or after a
    microphone is plugged/unplugged the cached handles go stale. Never runs
    while a stream is open — terminating under a live stream crashes the
    CoreAudio IO thread."""
    global _last_reset
    with _pa_lock:
        if _live_streams:
            return
        now = time.monotonic()
        if not force and now - _last_reset < _RESET_COOLDOWN:
            return
        _last_reset = now
        try:
            sd._terminate()
        except Exception:
            pass
        try:
            sd._initialize()
        except Exception:
            pass


class AudioRecorder:
    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = "int16"

    def __init__(self):
        self._frames: list = []
        self._samples = 0
        self._stream = None
        self._frames_lock = threading.Lock()
        self._max_frames = self.SAMPLE_RATE * MAX_SECONDS
        self.backend = ""  # audio API of the open stream, for the log

    @staticmethod
    def reset() -> None:
        _reset_portaudio(force=True)

    @staticmethod
    def _resolve_device(device_index: int, device_name: str) -> Optional[int]:
        """Pick a usable input device: by saved name first (indices shift when
        devices come and go), then by index, else the system default."""
        with _pa_lock:
            try:
                devices = sd.query_devices()
            except Exception:
                return None
            hostapi = _default_hostapi()
            if device_name:
                for i, dev in enumerate(devices):
                    if dev["name"] == device_name and _usable_input(dev, hostapi):
                        return i
            if (0 <= device_index < len(devices)
                    and _usable_input(devices[device_index], hostapi)):
                return device_index
            return None

    def start(self, device_index: int = -1, device_name: str = "") -> None:
        with _pa_lock:
            # Never leave a previous stream running: dropping its last
            # reference would free the callback PortAudio is still calling.
            _close_stream(self._stream)
            self._stream = None

            with self._frames_lock:
                self._frames = []
                self._samples = 0

            last_err: Optional[Exception] = None
            for attempt in range(3):
                if attempt > 0:
                    # A failed open usually means the device list is stale.
                    _reset_portaudio(force=True)
                    time.sleep(0.3)
                device = self._resolve_device(device_index, device_name)
                if attempt == 2:
                    device = None  # last resort: system default input
                try:
                    self._open(device, None, "default")
                    return
                except Exception as e:
                    last_err = e
            if _IS_WINDOWS:
                for device, extra, name in _windows_fallback_inputs():
                    try:
                        self._open(device, extra, name)
                        return
                    except Exception:
                        pass  # keep reporting the primary (MME) error
            raise last_err

    def _open(self, device, extra_settings, backend: str) -> None:
        stream = None
        try:
            stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
                device=device,
                callback=self._callback,
                extra_settings=extra_settings,
            )
            _live_streams.add(stream)  # hold a ref before it can fire
            stream.start()
            self._stream = stream
            self.backend = backend
        except Exception:
            _close_stream(stream)
            self._stream = None
            raise

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        # Runs on the realtime CoreAudio thread: it must never raise and must
        # never block for long.
        try:
            with self._frames_lock:
                if self._samples >= self._max_frames:
                    return
                self._frames.append(indata.copy())
                self._samples += len(indata)
        except Exception:
            pass

    def stop(self) -> bytes:
        with _pa_lock:
            had_stream = self._stream is not None
            _close_stream(self._stream)
            self._stream = None

        # Only touch the frames once the callback can no longer run.
        with self._frames_lock:
            frames = self._frames
            self._frames = []
            self._samples = 0

        if not frames:
            if had_stream:
                # A stream that produced no audio usually means the device
                # died mid-recording — re-scan so the next start works.
                _reset_portaudio()
            return b""

        try:
            audio = np.concatenate(frames, axis=0).flatten()
        except Exception:
            return b""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    @staticmethod
    def list_devices() -> list[dict]:
        with _pa_lock:
            _reset_portaudio()
            devices = []
            try:
                hostapi = _default_hostapi()
                for i, dev in enumerate(sd.query_devices()):
                    if _usable_input(dev, hostapi):
                        devices.append({"index": i, "name": dev["name"]})
            except Exception:
                pass
            return devices


class FileRecorder:
    """Stand-in microphone for end-to-end tests: every recording "captures"
    the same WAV file. Used only when PRESSVOICE_FAKE_MIC is set."""

    def __init__(self, wav_path: str):
        self._path = wav_path

    def start(self, device_index: int = -1, device_name: str = "") -> None:
        pass

    def stop(self) -> bytes:
        with open(self._path, "rb") as f:
            return f.read()

    def reset(self) -> None:
        pass

    @staticmethod
    def list_devices() -> list[dict]:
        return []
