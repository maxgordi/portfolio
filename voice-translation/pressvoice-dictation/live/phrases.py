"""Phrase-by-phrase transcription: speech is cut at natural pauses and each
finished phrase goes to the file transcription model (the one chosen in
Settings) while you keep talking. The text appears about a second after
each phrase, with the accuracy of the normal mode; the previous text is
passed along as context so names and sentence boundaries survive the cuts."""
import queue
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from live.mic import RATE, to_wav
from transcribe import FALLBACK_MODEL, ModelUnavailable, make_client, transcribe_chunk


class PhraseSplitter:
    """Energy-based voice activity detection on 20 ms frames.

    A phrase ends after `pause` seconds below the speech level, once it has
    at least `min_speech` seconds of speech; a phrase longer than `max_len`
    is cut anyway. The speech level adapts to the room's noise floor."""

    FRAME = RATE // 50  # 20 ms

    def __init__(self, on_phrase: Callable[[bytes], None], pause: float = 0.6,
                 min_speech: float = 0.3, max_len: float = 15.0):
        self._on_phrase = on_phrase
        self._pause_frames = int(pause * 50)
        self._min_speech_frames = int(min_speech * 50)
        self._max_frames = int(max_len * 50)
        self._pending = b""
        self._noise_db = -60.0
        self._frames: List[bytes] = []
        self._preroll: List[bytes] = []
        self._speech = 0
        self._silence = 0

    def _is_speech(self, frame: bytes) -> bool:
        x = np.frombuffer(frame, dtype="<i2").astype(np.float32)
        db = 20 * np.log10(np.sqrt(np.mean(x * x)) / 32768 + 1e-9)
        if db < self._noise_db:
            self._noise_db = db                       # follow quiet rooms quickly
        else:
            self._noise_db += (db - self._noise_db) * 0.002  # and noise slowly
        return db > max(self._noise_db + 10, -50)

    def feed(self, pcm: bytes) -> None:
        data = self._pending + pcm
        step = self.FRAME * 2
        n = len(data) // step * step
        self._pending = data[n:]
        for i in range(0, n, step):
            self._frame(data[i:i + step])

    def _frame(self, frame: bytes) -> None:
        speech = self._is_speech(frame)
        if not self._frames:
            if not speech:
                self._preroll = (self._preroll + [frame])[-10:]  # keep 200 ms
                return
            self._frames = self._preroll
            self._preroll = []
        self._frames.append(frame)
        if speech:
            self._speech += 1
            self._silence = 0
        else:
            self._silence += 1
        ended = self._silence >= self._pause_frames and self._speech >= self._min_speech_frames
        if ended or len(self._frames) >= self._max_frames:
            self._emit()
        elif self._silence >= self._pause_frames:  # a cough or click, not a phrase
            self._reset()

    def _emit(self) -> None:
        # Keep ~200 ms of the trailing pause, drop the rest.
        keep = len(self._frames) - max(0, self._silence - 10)
        self._on_phrase(b"".join(self._frames[:keep]))
        self._reset()

    def _reset(self) -> None:
        self._frames, self._speech, self._silence = [], 0, 0

    def flush(self) -> None:
        if self._frames and self._speech >= self._min_speech_frames:
            self._emit()
        self._reset()
        self._pending = b""


class PhraseTranscriber:
    """One engine for every dictation: begin / feed / end."""

    def __init__(self, api_key: str, model: str, pause: float = 0.6,
                 transcribe: Callable = transcribe_chunk,
                 log: Callable[[str], None] = lambda m: None):
        self._client = make_client(api_key)
        self.model = model or FALLBACK_MODEL
        self._pause = pause
        self._transcribe = transcribe
        self._log = log
        self._q: "queue.Queue" = queue.Queue()
        self._texts: List[str] = []
        self._on_text: Optional[Callable[[str], None]] = None
        self._splitter: Optional[PhraseSplitter] = None
        self._errors: List[Exception] = []
        # Models found unusable during this dictation (Settings named one
        # this key can't use); the fallback model took over.
        self.unavailable: List[str] = []
        # The first phrase that failed, when others made it: that piece of
        # speech is missing from the text.
        self.error: Optional[Exception] = None
        threading.Thread(target=self._worker, name="phrases", daemon=True).start()

    def warm_up(self) -> None:
        pass

    def begin(self, on_text: Callable[[str], None]) -> None:
        self._on_text = on_text
        self._texts = []
        self._errors = []
        self.unavailable = []
        self.error = None
        self._splitter = PhraseSplitter(self._q.put, pause=self._pause)

    def feed(self, pcm: bytes) -> None:
        self._splitter.feed(pcm)

    def end(self, timeout: float = 60.0) -> str:
        self._splitter.flush()
        done = threading.Event()
        self._q.put(done)
        done.wait(timeout)
        self._on_text = None
        if self._errors and not self._texts:
            raise self._errors[0]
        self.error = self._errors[0] if self._errors else None
        return " ".join(self._texts)

    def close(self) -> None:
        self._on_text = None
        self._q.put(None)

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            if isinstance(item, threading.Event):
                item.set()
                continue
            t0 = time.time()
            try:
                text = self._phrase(item)
            except Exception as e:
                self._errors.append(e)
                self._log(f"phrase failed: {e}")
                continue
            self._log(f"phrase {len(item) / (RATE * 2):.1f}s → {time.time() - t0:.1f}s")
            if text:
                piece = (" " if self._texts else "") + text
                self._texts.append(text)
                cb = self._on_text
                if cb:
                    cb(piece)

    def _phrase(self, pcm: bytes) -> str:
        context = " ".join(self._texts)[-300:]
        try:
            return self._transcribe(self._client, to_wav(pcm), self.model, prompt=context)
        except ModelUnavailable as e:
            if self.model == FALLBACK_MODEL:
                raise
            self._log(f"model unavailable ({e}); falling back to {FALLBACK_MODEL}")
            self.unavailable.append(self.model)
            self.model = FALLBACK_MODEL
            return self._transcribe(self._client, to_wav(pcm), self.model, prompt=context)
