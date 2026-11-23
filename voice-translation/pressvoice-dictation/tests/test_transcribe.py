from types import SimpleNamespace

import openai
import openai._exceptions as _openai_exceptions
import pytest

import transcribe
from tests.conftest import make_wav


# openai 1.x/2.x build on httpx, 3.x on httpx2; use whichever it has.
httpx = getattr(_openai_exceptions, "httpx2", None) or _openai_exceptions.httpx


def _api_error(cls, status: int, message: str = "boom", code=None):
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    body = {"error": {"message": message, "code": code}} if code else None
    return cls(message, response=httpx.Response(status, request=request), body=body)


class FakeTranscriptions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, model, file):
        self.calls.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def fake_client(*outcomes):
    tr = FakeTranscriptions(outcomes)
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=tr)), tr


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(transcribe.time, "sleep", lambda s: None)


def test_split_wav_into_20s_chunks():
    chunks = transcribe.split_wav(make_wav(45))
    assert [round(transcribe.wav_duration(c)) for c in chunks] == [20, 20, 5]


def test_wav_duration_handles_garbage():
    assert transcribe.wav_duration(b"not a wav") == 0.0
    assert transcribe.wav_duration(make_wav(1.5)) == pytest.approx(1.5)


def test_digital_silence_detection():
    assert transcribe.is_digital_silence(make_wav(1, silent=True))
    assert not transcribe.is_digital_silence(make_wav(1))
    assert not transcribe.is_digital_silence(b"garbage")


def test_list_transcription_models_filters_and_orders():
    ids = ["gpt-5", "tts-1", "gpt-4o-mini-tts", "gpt-4o-transcribe-diarize",
           "gpt-live-transcribe", "gpt-realtime-whisper", "whisper-1",
           "gpt-4o-mini-transcribe-2025-03-20", "gpt-4o-mini-transcribe",
           "gpt-transcribe", "gpt-audio"]
    client = SimpleNamespace(models=SimpleNamespace(
        list=lambda: [SimpleNamespace(id=i) for i in ids]))
    assert transcribe.list_transcription_models(client) == [
        "gpt-transcribe", "gpt-4o-mini-transcribe", "whisper-1",
        "gpt-4o-mini-transcribe-2025-03-20",
    ]


def test_model_choices():
    assert transcribe.model_choices({}) == transcribe.KNOWN_MODELS
    cfg = {"transcription_models": ["a-transcribe"], "transcription_model": "custom"}
    assert transcribe.model_choices(cfg) == ["custom", "a-transcribe"]


def test_transcribe_chunk_uses_given_model():
    client, tr = fake_client("  hello  ")
    assert transcribe.transcribe_chunk(client, make_wav(1), "gpt-transcribe") == "hello"
    assert tr.calls == ["gpt-transcribe"]


@pytest.mark.parametrize("error", [
    _api_error(openai.NotFoundError, 404, "The model `x` does not exist"),
    _api_error(openai.BadRequestError, 400, "invalid model", code="model_not_found"),
    _api_error(openai.PermissionDeniedError, 403, "no access to model"),
])
def test_unavailable_model_is_not_retried(error):
    client, tr = fake_client(error, "never")
    with pytest.raises(transcribe.ModelUnavailable):
        transcribe.transcribe_chunk(client, make_wav(1), "x")
    assert len(tr.calls) == 1


def test_bad_key_is_not_retried():
    client, tr = fake_client(_api_error(openai.AuthenticationError, 401), "never")
    with pytest.raises(openai.AuthenticationError):
        transcribe.transcribe_chunk(client, make_wav(1), "m")
    assert len(tr.calls) == 1


def test_server_errors_are_retried():
    err = _api_error(openai.InternalServerError, 500)
    client, tr = fake_client(err, err, "ok")
    assert transcribe.transcribe_chunk(client, make_wav(1), "m") == "ok"
    assert len(tr.calls) == 3


def test_friendly_error_messages():
    assert "Invalid API key" in transcribe.friendly_error(
        _api_error(openai.AuthenticationError, 401))
    assert "quota" in transcribe.friendly_error(
        _api_error(openai.RateLimitError, 429, "You exceeded your current quota"))
    assert transcribe.friendly_error(ValueError("x")) == "ValueError: x"
