import pytest

import pipeline
from tests.conftest import make_wav
from transcribe import FALLBACK_MODEL, ModelUnavailable


@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setattr(pipeline, "make_client", lambda key: object())

    def fake_chunk(client, chunk, model):
        log.append(model)
        if model == "broken":
            raise ModelUnavailable("broken: gone")
        return f"part{len(log)}"
    monkeypatch.setattr(pipeline, "transcribe_chunk", fake_chunk)
    return log


def test_too_short_is_rejected(calls):
    with pytest.raises(pipeline.TooShort):
        pipeline.transcribe_wav("k", make_wav(0.2), "m")
    assert calls == []


def test_digital_silence_is_rejected(calls):
    with pytest.raises(pipeline.NoAudio):
        pipeline.transcribe_wav("k", make_wav(2, silent=True), "m")
    assert calls == []


def test_chunks_are_joined_with_progress(calls):
    status = []
    text = pipeline.transcribe_wav("k", make_wav(45), "m", on_status=status.append)
    assert text == "part1 part2 part3"
    assert status == ["chunk 1/3", "chunk 2/3", "chunk 3/3"]


def test_falls_back_once_when_model_unavailable(calls):
    notices = []
    text = pipeline.transcribe_wav("k", make_wav(45), "broken",
                                   on_notice=lambda t, m: notices.append(t))
    assert calls == ["broken", FALLBACK_MODEL, FALLBACK_MODEL, FALLBACK_MODEL]
    assert notices == ["Model unavailable"]
    assert text.count("part") == 3


def test_fallback_model_failure_is_raised(monkeypatch, calls):
    def always_gone(client, chunk, model):
        raise ModelUnavailable(model)
    monkeypatch.setattr(pipeline, "transcribe_chunk", always_gone)
    with pytest.raises(ModelUnavailable):
        pipeline.transcribe_wav("k", make_wav(1), FALLBACK_MODEL)


def test_style_is_applied(monkeypatch, calls):
    monkeypatch.setattr(pipeline, "apply_style", lambda c, p, t: f"<{p}>{t}")
    status = []
    text = pipeline.transcribe_wav("k", make_wav(1), "m", style_prompt=" formal ",
                                   on_status=status.append)
    assert text == "<formal>part1"
    assert status == ["style"]


def test_style_failure_keeps_raw_text(monkeypatch, calls):
    def boom(*a):
        raise RuntimeError("style down")
    monkeypatch.setattr(pipeline, "apply_style", boom)
    notices = []
    text = pipeline.transcribe_wav("k", make_wav(1), "m", style_prompt="x",
                                   on_notice=lambda t, m: notices.append(t))
    assert text == "part1"
    assert notices == ["Style failed"]
