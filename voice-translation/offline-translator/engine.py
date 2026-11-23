"""Free, offline phrase translation on this Mac: recognise → translate → speak.

- Recognition: Whisper large-v3-turbo (whisper-server keeps the model loaded)
  or Vosk (small models, Russian and English only).
- Translation: Argos (through English when there is no direct pair).
- Voice: Piper.
- Audio: the microphone and the headphones are opened ONCE, at the device's
  own rate; all rate conversion happens here. (Opening the output at Piper's
  22 kHz made PortAudio switch the device's rate, and a microphone on the same
  device then went silent — only the first phrase ever worked.)
"""
import io
import json
import subprocess
import threading
import time
import urllib.request
import wave
from collections import deque
from typing import List, Optional
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soxr

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
RATE = 16000  # what recognisers take
WHISPER_MODEL = MODELS / "ggml-large-v3-turbo-q5_0.bin"
WHISPER_PORT = 8178
VOSK_MODELS = {"ru": "vosk-model-small-ru-0.22", "en": "vosk-model-small-en-us-0.15"}

# code: (name, [(voice label, path in huggingface.co/rhasspy/piper-voices)])
LANGUAGES = {
    "ru": ("Русский", [("Денис (муж.)", "ru/ru_RU/denis/medium/ru_RU-denis-medium"),
                       ("Дмитрий (муж.)", "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium"),
                       ("Руслан (муж.)", "ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium"),
                       ("Ирина (жен.)", "ru/ru_RU/irina/medium/ru_RU-irina-medium")]),
    "en": ("English", [("Ryan (муж., США)", "en/en_US/ryan/high/en_US-ryan-high"),
                       ("Alan (муж., Британия)", "en/en_GB/alan/medium/en_GB-alan-medium"),
                       ("Lessac (жен., США)", "en/en_US/lessac/medium/en_US-lessac-medium"),
                       ("Amy (жен., США)", "en/en_US/amy/medium/en_US-amy-medium")]),
    "uk": ("Українська", [("Ukrainian TTS", "uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium"),
                          ("Лада (жін.)", "uk/uk_UA/lada/x_low/uk_UA-lada-x_low")]),
    "de": ("Deutsch", [("Thorsten (муж.)", "de/de_DE/thorsten/medium/de_DE-thorsten-medium"),
                       ("Eva (жен.)", "de/de_DE/eva_k/x_low/de_DE-eva_k-x_low")]),
    "fr": ("Français", [("Siwis (жен.)", "fr/fr_FR/siwis/medium/fr_FR-siwis-medium")]),
    "es": ("Español", [("Davefx (муж.)", "es/es_ES/davefx/medium/es_ES-davefx-medium")]),
    "it": ("Italiano", [("Paola (жен.)", "it/it_IT/paola/medium/it_IT-paola-medium")]),
    "pl": ("Polski", [("Gosia (жен.)", "pl/pl_PL/gosia/medium/pl_PL-gosia-medium")]),
    "nl": ("Nederlands", [("MLS", "nl/nl_NL/mls/medium/nl_NL-mls-medium")]),
    "tr": ("Türkçe", [("DFKI", "tr/tr_TR/dfki/medium/tr_TR-dfki-medium")]),
    "pt": ("Português", [("Faber (муж.)", "pt/pt_BR/faber/medium/pt_BR-faber-medium")]),
    "zh": ("中文", [("Huayan (жен.)", "zh/zh_CN/huayan/medium/zh_CN-huayan-medium")]),
}


def base_lang(code: str) -> str:
    """PressVoice uses pt-BR, zh-Hans…; Argos and Piper plain codes."""
    return code.split("-")[0].lower()


# ── Recognition ──────────────────────────────────────────────────────────────

def _wav(pcm16: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE), w.writeframes(pcm16.tobytes())
    return buf.getvalue()


class Whisper:
    name = "Whisper"

    def __init__(self):
        self.proc = subprocess.Popen(
            ["whisper-server", "-m", str(WHISPER_MODEL), "--port", str(WHISPER_PORT), "-nt"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            if self.proc.poll() is not None:
                raise RuntimeError("whisper-server не запустился")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{WHISPER_PORT}/", timeout=1)
                return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("whisper-server не отвечает")

    def __call__(self, pcm16: np.ndarray, lang: str) -> str:
        r = requests.post(f"http://127.0.0.1:{WHISPER_PORT}/inference",
                          files={"file": ("phrase.wav", _wav(pcm16), "audio/wav")},
                          data={"response_format": "json", "language": lang, "temperature": "0"}, timeout=60)
        return " ".join(r.json().get("text", "").split())

    def close(self):
        self.proc.terminate()


class Vosk:
    name = "Vosk"

    def __init__(self):
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        self.models = {lang: Model(str(MODELS / name)) for lang, name in VOSK_MODELS.items()}
        self.Recognizer = KaldiRecognizer

    def __call__(self, pcm16: np.ndarray, lang: str) -> str:
        if lang not in self.models:
            raise RuntimeError("Vosk знает только русский и английский")
        rec = self.Recognizer(self.models[lang], RATE)
        rec.AcceptWaveform(pcm16.tobytes())
        return json.loads(rec.FinalResult())["text"]

    def close(self):
        pass


# ── Translation ──────────────────────────────────────────────────────────────

def ensure_pair(src: str, dst: str, say=lambda message: None) -> None:
    """Downloads Argos dictionaries for src→dst (via English if needed)."""
    import argostranslate.package as pkg
    have = {(p.from_code, p.to_code) for p in pkg.get_installed_packages()}
    want = {(src, dst)} if "en" in (src, dst) else {(src, "en"), ("en", dst)}
    missing = sorted(want - have)
    if not missing:
        return
    say(f"Скачиваю словари: {', '.join(f'{a}→{b}' for a, b in missing)}…")
    pkg.update_package_index()
    available = {(p.from_code, p.to_code): p for p in pkg.get_available_packages()}
    for pair in missing:
        if pair not in available:
            raise RuntimeError(f"у Argos нет перевода {pair[0]}→{pair[1]}")
        pkg.install_from_path(available[pair].download())


def translate(text: str, src: str, dst: str) -> str:
    import argostranslate.translate as argos
    return argos.translate(text, src, dst) if src != dst else text


# ── Voice ────────────────────────────────────────────────────────────────────

_voices = {}


def voice(path: str, say=lambda message: None):
    """A Piper voice by its path in the voices catalogue (downloaded once)."""
    if path in _voices:
        return _voices[path]
    from piper import PiperVoice
    name = path.rsplit("/", 1)[1]
    for ext in (".onnx", ".onnx.json"):
        local = MODELS / f"{name}{ext}"
        if not local.exists():
            say(f"Скачиваю голос {name}…")
            urllib.request.urlretrieve(f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{path}{ext}", local)
    # espeak-ng refuses long data paths: a short link to Piper's own data.
    link = Path.home() / ".cache/pv-espeak"
    link.parent.mkdir(exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(__import__("piper").__file__).parent / "espeak-ng-data")
    _voices[path] = PiperVoice.load(str(MODELS / f"{name}.onnx"), espeak_data_dir=link)
    return _voices[path]


def speak(piper_voice, text: str):
    """(float32 samples −1…1, rate)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        piper_voice.synthesize_wav(text, w)
    buf.seek(0)
    with wave.open(buf) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return pcm.astype(np.float32) / 32768, w.getframerate()


# ── Audio ────────────────────────────────────────────────────────────────────

def devices(output: bool):
    key = "max_output_channels" if output else "max_input_channels"
    api = sd.default.hostapi
    return [d["name"] for d in sd.query_devices() if d[key] > 0 and d["hostapi"] == api]


def _device(name, output: bool):
    if name:
        for i, d in enumerate(sd.query_devices()):
            if d["name"] == name and d["max_output_channels" if output else "max_input_channels"] > 0:
                return i
    return None  # system default


def _level_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)) if x.size else -120.0


class Audio:
    """The microphone and the headphones, both at their own rate, open for good."""

    def __init__(self, mic, out):
        mic_dev, out_dev = _device(mic, False), _device(out, True)
        self.mic_rate = int(sd.query_devices(mic_dev if mic_dev is not None else sd.default.device[0])["default_samplerate"])
        out_info = sd.query_devices(out_dev if out_dev is not None else sd.default.device[1])
        self.out_rate = int(out_info["default_samplerate"])
        self.out_channels = min(2, out_info["max_output_channels"])
        self.lock = threading.Lock()
        self.recording = False
        self.chunks = []
        self.queue = deque()
        self.mic_level = self.out_level = -120.0
        self.inp = sd.InputStream(samplerate=self.mic_rate, channels=1, dtype="float32", device=mic_dev,
                                  callback=self._on_mic)
        self.outp = sd.OutputStream(samplerate=self.out_rate, channels=self.out_channels, dtype="float32",
                                    device=out_dev, callback=self._on_out)
        self.inp.start()
        self.outp.start()

    def _on_mic(self, indata, frames, t, status):
        block = indata[:, 0]
        self.mic_level = _level_db(block)
        with self.lock:
            if self.recording:
                self.chunks.append(block.copy())

    def _on_out(self, outdata, frames, t, status):
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self.lock:
            while filled < frames and self.queue:
                piece = self.queue[0]
                n = min(frames - filled, len(piece))
                out[filled:filled + n] = piece[:n]
                filled += n
                if n == len(piece):
                    self.queue.popleft()
                else:
                    self.queue[0] = piece[n:]
        self.out_level = _level_db(out[:filled]) if filled else -120.0
        outdata[:] = out[:, None]  # every channel: both ears

    def start(self):
        with self.lock:
            self.chunks, self.recording = [], True

    def drain(self) -> np.ndarray:
        """What was recorded since the last call (float32, the microphone's rate)."""
        with self.lock:
            chunks, self.chunks = self.chunks, []
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def is_recording(self) -> bool:
        with self.lock:
            return self.recording

    def take(self) -> np.ndarray:
        """Stops recording; the phrase as 16 kHz int16."""
        with self.lock:
            self.recording = False
            audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
        audio = soxr.resample(audio, self.mic_rate, RATE) if audio.size else audio
        return (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    def play(self, samples: np.ndarray, rate: int):
        converted = soxr.resample(samples, rate, self.out_rate).astype(np.float32)
        with self.lock:
            self.queue.append(converted)

    def playing_seconds(self) -> float:
        with self.lock:
            return sum(len(p) for p in self.queue) / self.out_rate

    def close(self):
        for stream in (self.inp, self.outp):
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


# ── Phrases ──────────────────────────────────────────────────────────────────

class PhraseCutter:
    """Cuts a live recording into phrases at pauses, so each one is translated
    while the speaker goes on.

    Speech is found by WebRTC's voice detector on 20 ms frames. A phrase
    ends after `pause` of no speech (once it has at least `min_speech` of
    speech), or at `max_phrase` at the latest. 200 ms before the first speech
    are kept (soft first syllables). The detector hears the sound boosted
    towards −26 dBFS — quiet microphones (a Scarlett at −46) are judged like
    loud ones; the phrase itself is passed on untouched.
    """

    FRAME = RATE // 50  # 20 ms
    TARGET_DB = -26.0

    def __init__(self, in_rate: int, pause: float = 0.7, min_speech: float = 0.4, max_phrase: float = 20.0):
        import webrtcvad
        self.vad = webrtcvad.Vad(2)
        self.resampler = soxr.ResampleStream(in_rate, RATE, 1, dtype="float32")
        self.pause_frames = int(pause * 50)
        self.min_speech = int(min_speech * 50)
        self.max_frames = int(max_phrase * 50)
        self.pending = np.zeros(0, np.int16)
        self.preroll: deque = deque(maxlen=10)
        self.frames: List[np.ndarray] = []
        self.speech = self.silence = 0
        self.loudest_db = self.TARGET_DB

    def _is_speech(self, frame: np.ndarray) -> bool:
        x = frame.astype(np.float32)
        db = float(20 * np.log10(np.sqrt(np.mean(x * x)) / 32768 + 1e-9))
        # The loudest recent speech, decaying ~1 dB/s, sets the boost.
        self.loudest_db = max(db, self.loudest_db - 0.02)
        gain = 10 ** (min(30.0, max(0.0, self.TARGET_DB - self.loudest_db)) / 20)
        boosted = np.clip(x * gain, -32768, 32767).astype(np.int16)
        return self.vad.is_speech(boosted.tobytes(), RATE)

    def feed(self, samples: np.ndarray) -> List[np.ndarray]:
        """Recorded audio in (float32 at the input rate); finished phrases out (16 kHz int16)."""
        x = self.resampler.resample_chunk(samples)
        self.pending = np.concatenate([self.pending, (np.clip(x, -1, 1) * 32767).astype(np.int16)])
        done = []
        while len(self.pending) >= self.FRAME:
            frame, self.pending = self.pending[:self.FRAME], self.pending[self.FRAME:]
            phrase = self._frame(frame)
            if phrase is not None:
                done.append(phrase)
        return done

    def _frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        speech = self._is_speech(frame)
        if not self.frames:
            self.preroll.append(frame)
            if speech:
                self.frames, self.speech, self.silence = list(self.preroll), 1, 0
                self.preroll.clear()
            return None
        self.frames.append(frame)
        if speech:
            self.speech, self.silence = self.speech + 1, 0
        else:
            self.silence += 1
        if self.silence >= self.pause_frames or len(self.frames) >= self.max_frames:
            return self._cut()
        return None

    def _cut(self) -> Optional[np.ndarray]:
        frames, speech, silence = self.frames, self.speech, self.silence
        self.frames, self.speech, self.silence = [], 0, 0
        if speech < self.min_speech:
            return None  # a cough, a click: not a phrase
        keep = len(frames) - max(0, silence - 10)  # the pause itself isn't sent (but 200 ms are)
        return np.concatenate(frames[:keep])

    def flush(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """The last of the recording (key released): whatever phrase is open."""
        x = self.resampler.resample_chunk(samples, last=True)
        rest = np.concatenate([self.pending, (np.clip(x, -1, 1) * 32767).astype(np.int16)])
        self.pending = np.zeros(0, np.int16)
        for i in range(0, len(rest) - self.FRAME + 1, self.FRAME):
            phrase = self._frame(rest[i:i + self.FRAME])
            if phrase is not None:
                return phrase  # (a phrase that ended exactly here)
        return self._cut() if self.frames else None
