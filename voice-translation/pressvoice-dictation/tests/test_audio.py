import sys

import pytest

import audio
from tests.conftest import make_wav


class FakeStream:
    def __init__(self, device=None, extra_settings=None, **kw):
        self.device = device
        self.extra = extra_settings

    def start(self):
        pass

    def abort(self, ignore_errors=True):
        pass

    def close(self, ignore_errors=True):
        pass


@pytest.fixture
def no_portaudio(monkeypatch):
    """Keep these tests off the real audio stack."""
    monkeypatch.setattr(audio, "_reset_portaudio", lambda force=False: None)
    monkeypatch.setattr(audio.time, "sleep", lambda s: None)
    monkeypatch.setattr(audio.AudioRecorder, "_resolve_device",
                        staticmethod(lambda i, n: None))


def test_windows_falls_back_to_other_audio_apis(monkeypatch, no_portaudio):
    opened = []

    def fake_stream(device=None, extra_settings=None, **kw):
        opened.append(device)
        if device != 7:  # MME (device None) refuses, WASAPI (7) works
            raise OSError("Unanticipated host error [MME error 1]")
        return FakeStream(device, extra_settings)

    monkeypatch.setattr(audio, "_IS_WINDOWS", True)
    monkeypatch.setattr(audio.sd, "InputStream", fake_stream)
    monkeypatch.setattr(audio, "_windows_fallback_inputs",
                        lambda: [(7, "wasapi-settings", "Windows WASAPI")])
    rec = audio.AudioRecorder()
    rec.start()
    assert opened == [None, None, None, 7]
    assert rec.backend == "Windows WASAPI"
    assert rec._stream.extra == "wasapi-settings"
    rec.stop()


def test_primary_error_is_reported_when_everything_fails(monkeypatch, no_portaudio):
    def fake_stream(device=None, **kw):
        raise OSError(f"fail {device}")

    monkeypatch.setattr(audio, "_IS_WINDOWS", True)
    monkeypatch.setattr(audio.sd, "InputStream", fake_stream)
    monkeypatch.setattr(audio, "_windows_fallback_inputs",
                        lambda: [(7, None, "Windows WASAPI"), (9, None, "Windows DirectSound")])
    with pytest.raises(OSError, match="fail None"):
        audio.AudioRecorder().start()


def test_no_fallbacks_outside_windows(monkeypatch, no_portaudio):
    monkeypatch.setattr(audio, "_IS_WINDOWS", False)
    monkeypatch.setattr(audio.sd, "InputStream",
                        lambda **kw: (_ for _ in ()).throw(OSError("nope")))
    monkeypatch.setattr(audio, "_windows_fallback_inputs",
                        lambda: pytest.fail("must not be used"))
    with pytest.raises(OSError):
        audio.AudioRecorder().start()


def test_wav_peak():
    assert audio.wav_peak(make_wav(1, silent=True)) == 0.0
    assert 0 < audio.wav_peak(make_wav(1)) < 0.001
    assert audio.wav_peak(b"junk") == 0.0


def test_file_recorder(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(make_wav(1))
    rec = audio.FileRecorder(str(path))
    rec.start()
    assert rec.stop() == path.read_bytes()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows registry")
def test_windows_microphone_block_reads_registry():
    from qtapp import platform_utils
    assert isinstance(platform_utils.windows_microphone_block(), str)
