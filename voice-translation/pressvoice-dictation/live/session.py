"""One live dictation from key press to the final text, whatever the mode.

The app hands over where typed text goes (on_text, called from background
threads) and gets a LiveResult when the dictation ends; the engine for the
chosen mode is kept between dictations so a press starts instantly."""
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from live.mic import RATE, MicStream

WHOLE, PHRASES = "whole", "phrases"
TEXT_MODES = (WHOLE, PHRASES)

# Shorter than this is an accidental key tap (same as pipeline.MIN_SECONDS).
_MIN_SECONDS = 0.4


def text_mode(cfg: dict) -> str:
    mode = cfg.get("text_mode")
    return mode if mode in TEXT_MODES else WHOLE


@dataclass
class LiveResult:
    text: str = ""                 # everything that was typed, stripped
    silent: bool = False           # the microphone gave only zeros (privacy block)
    error: Optional[Exception] = None
    unavailable: List[str] = field(default_factory=list)  # models replaced by the fallback
    model: str = ""                # the model that ended up doing the work


class ScriptedEngine:
    """Test stand-in for OpenAI (PRESSVOICE_FAKE_TRANSCRIPT): types the
    given text a word at a time while the key is held."""

    def __init__(self, text: str, pace: float = 0.15):
        self._words = text.split(" ")
        self._pace = pace
        self._thread: Optional[threading.Thread] = None
        self.model = "scripted"

    def warm_up(self) -> None:
        pass

    def begin(self, on_text: Callable[[str], None]) -> None:
        def run():
            for i, word in enumerate(self._words):
                time.sleep(self._pace)
                on_text(word if i == 0 else " " + word)
        self._thread = threading.Thread(target=run, name="scripted", daemon=True)
        self._thread.start()

    def feed(self, pcm: bytes) -> None:
        pass

    def end(self, timeout: float = 15.0) -> str:
        if self._thread:
            self._thread.join(timeout)
        return " ".join(self._words)

    def close(self) -> None:
        pass


class LiveDictation:
    def __init__(self, log: Callable[[str], None] = lambda m: None, mic=None):
        self._log = log
        self._mic = mic or MicStream()
        self._engine = None
        self._settings: tuple = ()
        self._turn_engine = None
        self._on_text: Optional[Callable[[str], None]] = None
        self._pieces: List[str] = []
        self._lock = threading.Lock()
        self._fed = 0
        self._heard = False

    @property
    def active(self) -> bool:
        """A live mode is on and usable (it needs an API key)."""
        return self._engine is not None

    @property
    def backend(self) -> str:
        return getattr(self._mic, "backend", "")

    def configure(self, api_key: str, mode: str, model: str,
                  fake_text: str = "") -> None:
        """Called at startup and after Settings are saved; keeps the engine
        (and its open connection) when nothing relevant changed."""
        settings = (api_key, mode, model if mode == PHRASES else "", fake_text)
        if settings == self._settings:
            return
        self._settings = settings
        old, self._engine = self._engine, None
        if old is not None and old is not self._turn_engine:
            old.close()  # one still in use is closed when its dictation ends
        if mode != PHRASES or not api_key:
            return
        if fake_text:
            self._engine = ScriptedEngine(fake_text)
        else:
            from live.phrases import PhraseTranscriber
            self._engine = PhraseTranscriber(api_key, model, log=self._log)
        self._log(f"live: {mode} mode, {self._engine.model}")

    def warm_up(self) -> None:
        engine = self._engine
        if engine is not None:
            engine.warm_up()

    # ── One dictation ────────────────────────────────────────────────────────

    def start(self, on_text: Callable[[str], None],
              device_index: int = -1, device_name: str = "") -> None:
        """Opens the microphone (blocking: call it off the main thread).
        Raises if it can't be opened; nothing is left running then."""
        engine = self._engine
        if engine is None:
            raise RuntimeError("live dictation is off")
        with self._lock:
            self._turn_engine = engine
            self._on_text = on_text
            self._pieces = []
        self._fed = 0
        self._heard = False
        engine.begin(self._text)
        try:
            self._mic.start(self._feed, device_index, device_name)
        except Exception:
            self._abandon()
            raise

    def stop_mic(self) -> None:
        """Blocking audio call: off the main thread."""
        self._mic.stop()

    def finish(self) -> LiveResult:
        """Waits for the last words (network; off the main thread)."""
        engine = self._turn_engine
        result = LiveResult(model=getattr(engine, "model", ""))
        if engine is None:
            return result
        final = ""
        try:
            final = engine.end()
        except Exception as e:
            result.error = e
        result.error = result.error or getattr(engine, "error", None)
        result.unavailable = list(getattr(engine, "unavailable", []))
        result.model = getattr(engine, "model", "")
        with self._lock:
            typed = "".join(self._pieces).strip()
        if not typed and final.strip():
            self._text(final)  # no partial text came: type the result at once
            typed = final.strip()
        with self._lock:
            self._on_text = None
            self._turn_engine = None
        result.text = typed
        result.silent = self._fed >= _MIN_SECONDS * RATE * 2 and not self._heard
        if engine is not self._engine:
            engine.close()  # Settings switched it while it was in use
        else:
            engine.warm_up()
        return result

    def cancel(self) -> None:
        """Drop the dictation: nothing more is typed. Blocking (off the main
        thread): it waits for the engine to wind down."""
        self.stop_mic()
        self._abandon()

    def _abandon(self) -> None:
        with self._lock:
            self._on_text = None
        try:
            self.finish()
        except Exception as e:
            self._log(f"live: abandon failed: {e}")

    # ── Audio and text (background threads) ──────────────────────────────────

    def _feed(self, pcm: bytes) -> None:
        self._fed += len(pcm)
        if not self._heard and pcm.count(0) != len(pcm):
            self._heard = True
        engine = self._turn_engine
        if engine is not None:
            engine.feed(pcm)

    def _text(self, piece: str) -> None:
        with self._lock:
            if not self._pieces:
                piece = piece.lstrip()
            if not piece or self._on_text is None:
                return
            self._pieces.append(piece)
            on_text = self._on_text
        on_text(piece)
