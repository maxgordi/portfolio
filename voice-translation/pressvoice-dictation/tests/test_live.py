"""Live dictation: phrase splitting, the phrase engine and one dictation
from start to finish."""
import time

import numpy as np
import pytest

from live import phrases
from live.mic import RATE, _halve
from live.session import PHRASES, WHOLE, LiveDictation, ScriptedEngine, text_mode
from transcribe import FALLBACK_MODEL, ModelUnavailable


def tone(seconds: float, level: float = 0.3) -> bytes:
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * 220 * t) * level * 32767).astype("<i2").tobytes()


def hush(seconds: float) -> bytes:
    rng = np.random.default_rng(1)
    return (rng.normal(0, 3, int(seconds * RATE))).astype("<i2").tobytes()


def chunks(pcm: bytes, ms: int = 20):
    step = RATE * 2 * ms // 1000
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]


# ── Phrase splitting ─────────────────────────────────────────────────────────

def test_splitter_cuts_at_pauses_and_ignores_clicks():
    got = []
    s = phrases.PhraseSplitter(got.append, pause=0.6)
    audio = hush(0.5) + tone(1.0) + hush(0.8) + tone(0.05) + hush(0.8) + tone(0.7) + hush(0.2)
    for c in chunks(audio):
        s.feed(c)
    assert len(got) == 1  # the 50 ms click is not a phrase
    s.flush()             # the key is released mid-pause: the last phrase still goes
    assert len(got) == 2
    first = len(got[0]) / (RATE * 2)
    assert 1.0 <= first <= 1.5  # the speech, ~200 ms before and after it


def test_splitter_cuts_very_long_phrases():
    got = []
    s = phrases.PhraseSplitter(got.append, max_len=2.0)
    for c in chunks(tone(5.0)):
        s.feed(c)
    assert len(got) == 2


# ── Phrase engine ────────────────────────────────────────────────────────────

class FakeFiles:
    """Stands in for transcribe_chunk; answers phrase N with word N."""

    def __init__(self, unavailable=(), fail_on=()):
        self.calls = []
        self.unavailable = set(unavailable)
        self.fail_on = set(fail_on)

    def __call__(self, client, wav, model, prompt=""):
        if model in self.unavailable:
            raise ModelUnavailable(model)
        n = len(self.calls) + 1
        self.calls.append((model, prompt))
        if n in self.fail_on:
            raise RuntimeError("network down")
        return ["Hello", "world.", "Again"][n - 1]


def speak(engine, *parts):
    for part in parts:
        for c in chunks(part):
            engine.feed(c)


def test_phrase_engine_types_each_phrase_with_context():
    fake = FakeFiles()
    e = phrases.PhraseTranscriber("sk-test", "gpt-transcribe", transcribe=fake)
    pieces = []
    e.begin(pieces.append)
    speak(e, hush(0.3), tone(0.8), hush(0.8), tone(0.8))
    assert e.end() == "Hello world."
    assert pieces == ["Hello", " world."]
    assert fake.calls == [("gpt-transcribe", ""), ("gpt-transcribe", "Hello")]
    e.close()


def test_phrase_engine_falls_back_from_an_unusable_model():
    fake = FakeFiles(unavailable={"gpt-new"})
    e = phrases.PhraseTranscriber("sk-test", "gpt-new", transcribe=fake)
    e.begin(lambda p: None)
    speak(e, tone(0.8), hush(0.8))
    assert e.end() == "Hello"
    assert e.unavailable == ["gpt-new"] and e.model == FALLBACK_MODEL
    e.close()


def test_phrase_engine_reports_a_lost_phrase():
    e = phrases.PhraseTranscriber("sk-test", "m", transcribe=FakeFiles(fail_on={1}))
    e.begin(lambda p: None)
    speak(e, tone(0.8), hush(0.8), tone(0.8))
    assert e.end() == "world."
    assert isinstance(e.error, RuntimeError)

    e.begin(lambda p: None)  # nothing at all made it: end() raises
    e._transcribe = FakeFiles(fail_on={1})
    speak(e, tone(0.8))
    with pytest.raises(RuntimeError):
        e.end()
    e.close()


# ── One dictation ────────────────────────────────────────────────────────────

class FakeMic:
    backend = "fake"

    def __init__(self, pcm=None, fail=None):
        self.pcm = tone(0.6) if pcm is None else pcm
        self.fail = fail
        self.log = []

    def start(self, on_chunk, device_index=-1, device_name=""):
        self.log.append("start")
        if self.fail:
            raise self.fail
        for c in chunks(self.pcm):
            on_chunk(c)

    def stop(self):
        self.log.append("stop")


def test_text_mode_defaults_to_whole():
    assert text_mode({}) == WHOLE
    assert text_mode({"text_mode": "bogus"}) == WHOLE
    assert text_mode({"text_mode": PHRASES}) == PHRASES
    assert text_mode({"text_mode": "words"}) == WHOLE  # removed after 1.0.4


def test_dictation_types_pieces_and_returns_the_text():
    live = LiveDictation(mic=FakeMic())
    assert not live.active
    live.configure("sk-test", PHRASES, "", fake_text="Привет from PressVoice")
    assert live.active
    pieces = []
    live.start(pieces.append)
    live.stop_mic()
    result = live.finish()
    assert "".join(pieces) == "Привет from PressVoice"
    assert result.text == "Привет from PressVoice"
    assert not result.silent and result.error is None


def test_dictation_needs_a_key_and_a_live_mode():
    live = LiveDictation(mic=FakeMic())
    live.configure("", PHRASES, "")
    assert not live.active
    live.configure("sk-test", WHOLE, "")
    assert not live.active


def test_dictation_notices_a_silent_microphone():
    live = LiveDictation(mic=FakeMic(pcm=bytes(RATE * 2)))
    live.configure("sk-test", PHRASES, "", fake_text="x")
    live.start(lambda p: None)
    assert live.finish().silent


def test_failed_microphone_leaves_nothing_running():
    mic = FakeMic(fail=OSError("no device"))
    live = LiveDictation(mic=mic)
    live.configure("sk-test", PHRASES, "", fake_text="a b")
    with pytest.raises(OSError):
        live.start(lambda p: None)
    assert live._turn_engine is None


def test_cancelled_dictation_types_nothing_more():
    live = LiveDictation(mic=FakeMic())
    live.configure("sk-test", PHRASES, "", fake_text="one two three four")
    pieces = []
    live.start(pieces.append)
    live.cancel()
    typed = len(pieces)
    time.sleep(0.4)
    assert len(pieces) == typed < 4


def test_whole_text_is_typed_when_no_pieces_came():
    class Quiet(ScriptedEngine):
        def begin(self, on_text):
            pass

    live = LiveDictation(mic=FakeMic())
    live.configure("sk-test", PHRASES, "", fake_text="x")
    live._engine = Quiet("all at once")
    pieces = []
    live.start(pieces.append)
    assert live.finish().text == "all at once"
    assert pieces == ["all at once"]


def test_settings_change_mid_dictation_closes_the_old_engine_afterwards():
    live = LiveDictation(mic=FakeMic())
    live.configure("sk-test", PHRASES, "", fake_text="a b")
    old = live._engine
    closed = []
    old.close = lambda: closed.append(True)
    live.start(lambda p: None)
    live.configure("sk-test", PHRASES, "", fake_text="c d")
    assert closed == []  # still in use
    assert live.finish().text == "a b"
    assert closed == [True]
    live.configure("sk-test", PHRASES, "", fake_text="c d")  # unchanged: kept
    assert live._engine is not old


def test_halving_48k_to_24k():
    x = np.array([100, 300, -50, -150], dtype="<i2").tobytes()
    assert np.frombuffer(_halve(x), dtype="<i2").tolist() == [200, -100]
