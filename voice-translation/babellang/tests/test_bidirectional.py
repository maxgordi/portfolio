"""
Automated tests for bidirectional translation and voice mixing.
All tests are hardware-free: audio devices and API calls are mocked.
"""
from __future__ import annotations

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from babellang.config import AudioConfig, PipelineConfig, Settings
from babellang.pipeline import PipelineStatus, _run_vad_loop


# ─── A. OpenAITranslator language direction ───────────────────────────────────

def test_build_system_prompt_ru_en():
    from babellang.translation.openai_translator import _build_system_prompt
    prompt = _build_system_prompt("ru", "en")
    assert "Russian" in prompt
    assert "English" in prompt


def test_build_system_prompt_en_ru():
    from babellang.translation.openai_translator import _build_system_prompt
    prompt = _build_system_prompt("en", "ru")
    assert "English" in prompt
    assert "Russian" in prompt


def test_build_system_prompt_direction_order():
    from babellang.translation.openai_translator import _build_system_prompt
    prompt_ru_en = _build_system_prompt("ru", "en")
    ru_pos = prompt_ru_en.index("Russian")
    en_pos = prompt_ru_en.index("English")
    assert ru_pos < en_pos, "source language should appear before target"

    prompt_en_ru = _build_system_prompt("en", "ru")
    en_pos2 = prompt_en_ru.index("English")
    ru_pos2 = prompt_en_ru.index("Russian")
    assert en_pos2 < ru_pos2, "source language should appear before target"


@pytest.mark.asyncio
async def test_translator_stores_system_prompt_for_direction():
    """Translator stores the correct system prompt per direction."""
    from babellang.translation.openai_translator import OpenAITranslator

    cfg = PipelineConfig()
    tx_q: asyncio.Queue = asyncio.Queue()
    tok_q: asyncio.Queue = asyncio.Queue()

    tr_ru_en = OpenAITranslator(
        api_key="fake", transcript_queue=tx_q, token_queue=tok_q,
        config=cfg, source_lang="ru", target_lang="en",
    )
    tr_en_ru = OpenAITranslator(
        api_key="fake", transcript_queue=asyncio.Queue(), token_queue=asyncio.Queue(),
        config=cfg, source_lang="en", target_lang="ru",
    )

    assert "Russian" in tr_ru_en._system_prompt
    assert "English" in tr_ru_en._system_prompt
    assert tr_ru_en._system_prompt.index("Russian") < tr_ru_en._system_prompt.index("English")

    assert "English" in tr_en_ru._system_prompt
    assert "Russian" in tr_en_ru._system_prompt
    assert tr_en_ru._system_prompt.index("English") < tr_en_ru._system_prompt.index("Russian")


# ─── B. AudioPlayback passthrough mixing ─────────────────────────────────────

def _make_playback(original_ratio: float = 0.0):
    """Construct AudioPlayback without opening any hardware stream."""
    from babellang.audio.playback import AudioPlayback
    loop = asyncio.new_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    pb = AudioPlayback(device=None, sample_rate=44100, queue=q, loop=loop, original_ratio=original_ratio)
    return pb


def test_feed_passthrough_no_op_when_ratio_zero():
    pb = _make_playback(original_ratio=0.0)
    chunk = np.zeros(512, dtype=np.int16)
    pb.feed_passthrough(chunk)
    assert len(pb._passthrough_buffer) == 0, "passthrough buffer must stay empty when ratio=0.0"


def test_feed_passthrough_resamples_16k_to_44k():
    pb = _make_playback(original_ratio=0.5)
    chunk = np.zeros(512, dtype=np.int16)
    pb.feed_passthrough(chunk, src_sample_rate=16000)
    # Sinc resampler may produce 1 sample more than simple linear interpolation
    expected_approx = int(512 * 44100 / 16000)  # ~1411
    assert abs(len(pb._passthrough_buffer) - expected_approx) <= 2


def test_feed_passthrough_normalises_amplitude():
    """Sinc resampler may produce slight overshoot (Gibbs); clipping happens in _callback."""
    pb = _make_playback(original_ratio=1.0)
    chunk = np.full(512, 32767, dtype=np.int16)
    pb.feed_passthrough(chunk, src_sample_rate=16000)
    samples = list(pb._passthrough_buffer)
    # Overshoot from sinc filter is small (< 15%), clipped to [-1, 1] in _callback
    assert max(samples) <= 1.15
    assert min(samples) >= -1.15


def test_callback_mixing_blends_at_half_ratio():
    """TTS always at full volume; passthrough is additive. TTS=1.0, PT=-1.0, ratio=0.5 → 0.5."""
    pb = _make_playback(original_ratio=0.5)
    pb._duck_out_step = 0.0  # freeze duck ratio at 1.0 so it doesn't interfere
    pb._play_buffer.extend([1.0] * 1024)
    pb._passthrough_buffer.extend([-1.0] * 1024)

    outdata = np.zeros((1024, 1), dtype=np.float32)
    status = MagicMock()
    status.output_underflow = False
    pb._callback(outdata, 1024, None, status)

    # tts(1.0) + pt(-1.0) * ratio(0.5) * duck(1.0) = 0.5
    np.testing.assert_allclose(outdata[:, 0], 0.5, atol=1e-5)


def test_callback_pure_tts_when_ratio_zero():
    """With ratio=0.0, passthrough buffer is ignored entirely."""
    pb = _make_playback(original_ratio=0.0)
    pb._play_buffer.extend([0.5] * 512)
    pb._passthrough_buffer.extend([-1.0] * 512)  # should be ignored

    outdata = np.zeros((512, 1), dtype=np.float32)
    status = MagicMock()
    status.output_underflow = False
    pb._callback(outdata, 512, None, status)

    np.testing.assert_allclose(outdata[:, 0], 0.5, atol=1e-5)


def test_callback_clips_mixed_output():
    """Mixing should not produce values outside [-1.0, 1.0]."""
    pb = _make_playback(original_ratio=0.5)
    pb._play_buffer.extend([1.0] * 256)
    pb._passthrough_buffer.extend([1.0] * 256)  # both +1.0

    outdata = np.zeros((256, 1), dtype=np.float32)
    status = MagicMock()
    status.output_underflow = False
    pb._callback(outdata, 256, None, status)

    assert outdata.max() <= 1.0
    assert outdata.min() >= -1.0


def test_callback_stereo_duplication():
    """Mono signal is duplicated to channel 1 when channels=2."""
    pb = _make_playback(original_ratio=0.0)
    pb._play_buffer.extend([0.3] * 256)

    outdata = np.zeros((256, 2), dtype=np.float32)
    status = MagicMock()
    status.output_underflow = False
    pb._callback(outdata, 256, None, status)

    np.testing.assert_array_equal(outdata[:, 0], outdata[:, 1])


# ─── C. Settings serialization round-trip ────────────────────────────────────

def test_settings_bidirectional_round_trip():
    settings = Settings(
        pipeline=PipelineConfig(elevenlabs_voice_id="v123"),
        input_device=1,
        output_device=0,
        input_device_name="Built-in Mic",
        output_device_name="BlackHole 2ch",
        incoming_device=5,
        incoming_device_name="BlackHole 16ch",
        speaker_device=2,
        speaker_device_name="MacBook Pro Speakers",
        outgoing_original_ratio=0.25,
        incoming_original_ratio=0.1,
    )
    d = settings.to_dict()
    restored = Settings.from_dict(d)

    assert restored.incoming_device == 5
    assert restored.incoming_device_name == "BlackHole 16ch"
    assert restored.speaker_device == 2
    assert restored.speaker_device_name == "MacBook Pro Speakers"
    assert restored.outgoing_original_ratio == pytest.approx(0.25)
    assert restored.incoming_original_ratio == pytest.approx(0.1)
    assert restored.pipeline.elevenlabs_voice_id == "v123"


def test_settings_defaults_for_missing_incoming_fields():
    """Old settings.json without incoming fields should load with safe defaults."""
    data = {
        "input_device": 1,
        "output_device": 0,
        "input_device_name": "Mic",
        "output_device_name": "BlackHole 2ch",
        "elevenlabs_voice_id": "v1",
    }
    settings = Settings.from_dict(data)
    assert settings.incoming_device is None
    assert settings.speaker_device is None
    assert settings.outgoing_original_ratio == pytest.approx(0.0)
    assert settings.incoming_original_ratio == pytest.approx(0.0)


# ─── D. IncomingPipeline construction ────────────────────────────────────────

def _base_settings_with_incoming() -> Settings:
    return Settings(
        pipeline=PipelineConfig(elevenlabs_voice_id="v1"),
        input_device=1,
        output_device=0,
        incoming_device=5,
        speaker_device=2,
    )


_PATCH_TARGETS = [
    "babellang.pipeline.AudioCapture",
    "babellang.pipeline.AudioPlayback",
    "babellang.pipeline.SileroVAD",
    "babellang.pipeline.ElevenLabsTTS",
]


def test_incoming_pipeline_uses_en_language():
    """IncomingPipeline (classic mode) must configure OpenAISTT with language='en'."""
    from babellang.pipeline import IncomingPipeline

    settings = _base_settings_with_incoming()
    settings.stt_mode = "classic"  # OpenAISTT is only built in classic mode
    api_keys = {"openai": "oai", "elevenlabs": "el"}

    with patch("babellang.pipeline.AudioCapture"), \
         patch("babellang.pipeline.AudioPlayback"), \
         patch("babellang.pipeline.SileroVAD"), \
         patch("babellang.pipeline.ElevenLabsTTS"), \
         patch("babellang.pipeline.OpenAITranslator"), \
         patch("babellang.pipeline.OpenAISTT") as MockSTT:
        IncomingPipeline(settings=settings, api_keys=api_keys)
        config_arg = MockSTT.call_args.kwargs["config"]
        assert config_arg.stt_language == "en"


def test_incoming_pipeline_uses_en_ru_translator():
    """IncomingPipeline must wire OpenAITranslator with source=en, target=ru."""
    from babellang.pipeline import IncomingPipeline

    settings = _base_settings_with_incoming()
    settings.stt_mode = "classic"  # OpenAITranslator is only built in classic mode
    api_keys = {"openai": "oai", "elevenlabs": "el"}

    with patch("babellang.pipeline.AudioCapture"), \
         patch("babellang.pipeline.AudioPlayback"), \
         patch("babellang.pipeline.SileroVAD"), \
         patch("babellang.pipeline.ElevenLabsTTS"), \
         patch("babellang.pipeline.OpenAISTT"), \
         patch("babellang.pipeline.OpenAITranslator") as MockTranslator:
        IncomingPipeline(settings=settings, api_keys=api_keys)
        kwargs = MockTranslator.call_args.kwargs
        assert kwargs["source_lang"] == "en"
        assert kwargs["target_lang"] == "ru"


def test_incoming_pipeline_uses_speaker_device():
    """IncomingPipeline must route TTS output to settings.speaker_device."""
    from babellang.pipeline import IncomingPipeline

    settings = _base_settings_with_incoming()
    api_keys = {"openai": "oai", "elevenlabs": "el"}

    with patch("babellang.pipeline.AudioCapture"), \
         patch("babellang.pipeline.SileroVAD"), \
         patch("babellang.pipeline.ElevenLabsTTS"), \
         patch("babellang.pipeline.OpenAISTT"), \
         patch("babellang.pipeline.OpenAITranslator"), \
         patch("babellang.pipeline.AudioPlayback") as MockPlayback:
        IncomingPipeline(settings=settings, api_keys=api_keys)
        device_arg = MockPlayback.call_args.kwargs["device"]
        assert device_arg == settings.speaker_device


def test_outgoing_pipeline_uses_ru_en_defaults():
    """OutgoingPipeline must use ru→en direction by default."""
    from babellang.pipeline import OutgoingPipeline

    settings = Settings(
        pipeline=PipelineConfig(elevenlabs_voice_id="v1"),
        input_device=1,
        output_device=0,
        stt_mode="classic",  # OpenAITranslator is only built in classic mode
    )
    api_keys = {"openai": "oai", "elevenlabs": "el"}

    with patch("babellang.pipeline.AudioCapture"), \
         patch("babellang.pipeline.AudioPlayback"), \
         patch("babellang.pipeline.SileroVAD"), \
         patch("babellang.pipeline.ElevenLabsTTS"), \
         patch("babellang.pipeline.OpenAISTT"), \
         patch("babellang.pipeline.OpenAITranslator") as MockTranslator:
        OutgoingPipeline(settings=settings, api_keys=api_keys)
        kwargs = MockTranslator.call_args.kwargs
        assert kwargs["source_lang"] == "ru"
        assert kwargs["target_lang"] == "en"


# ─── E. VAD loop passthrough calls ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_vad_loop_calls_feed_passthrough_during_speech():
    """_run_vad_loop must call playback.feed_passthrough for each SPEAKING chunk."""
    cfg = AudioConfig(vad_speech_start_chunks=2, vad_speech_end_chunks=10)
    raw_q: asyncio.Queue = asyncio.Queue()
    stt_q: asyncio.Queue = asyncio.Queue()
    status = PipelineStatus()

    mock_playback = MagicMock()
    mock_vad = MagicMock()

    # 2 speech + 3 more speech (5 total SPEAKING frames before we cancel)
    # then silence — but we cancel before full silence threshold
    probs = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    prob_iter = iter(probs)

    async def fake_process(chunk):
        try:
            return next(prob_iter)
        except StopIteration:
            return 0.1

    mock_vad.process_chunk = fake_process

    for _ in range(len(probs) + 5):
        await raw_q.put(np.zeros(512, dtype=np.int16))

    task = asyncio.create_task(
        _run_vad_loop(raw_q, stt_q, mock_vad, cfg, status, mock_playback, passthrough_ratio=0.5)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert mock_playback.feed_passthrough.called, "feed_passthrough must be called during SPEAKING state"


@pytest.mark.asyncio
async def test_vad_loop_no_passthrough_when_ratio_zero():
    """With passthrough_ratio=0.0, feed_passthrough must never be called."""
    cfg = AudioConfig(vad_speech_start_chunks=2, vad_speech_end_chunks=10)
    raw_q: asyncio.Queue = asyncio.Queue()
    stt_q: asyncio.Queue = asyncio.Queue()
    status = PipelineStatus()

    mock_playback = MagicMock()
    probs = [0.9, 0.9, 0.9, 0.9, 0.9]
    prob_iter = iter(probs)

    async def fake_process(chunk):
        try:
            return next(prob_iter)
        except StopIteration:
            return 0.1

    mock_vad = MagicMock()
    mock_vad.process_chunk = fake_process

    for _ in range(len(probs) + 5):
        await raw_q.put(np.zeros(512, dtype=np.int16))

    task = asyncio.create_task(
        _run_vad_loop(raw_q, stt_q, mock_vad, cfg, status, mock_playback, passthrough_ratio=0.0)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    mock_playback.feed_passthrough.assert_not_called()


@pytest.mark.asyncio
async def test_vad_loop_emits_segment_with_passthrough():
    """_run_vad_loop still emits STT segments correctly even when passthrough is active."""
    cfg = AudioConfig(vad_speech_start_chunks=2, vad_speech_end_chunks=3)
    raw_q: asyncio.Queue = asyncio.Queue()
    stt_q: asyncio.Queue = asyncio.Queue()
    status = PipelineStatus()

    mock_playback = MagicMock()
    # 2 pre-speech (below) + 4 speech + 5 silence (enough to end)
    probs = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    prob_iter = iter(probs)

    async def fake_process(chunk):
        try:
            return next(prob_iter)
        except StopIteration:
            return 0.1

    mock_vad = MagicMock()
    mock_vad.process_chunk = fake_process

    for _ in range(len(probs) + 10):
        await raw_q.put(np.zeros(512, dtype=np.int16))

    task = asyncio.create_task(
        _run_vad_loop(raw_q, stt_q, mock_vad, cfg, status, mock_playback, passthrough_ratio=0.3)
    )
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert not stt_q.empty(), "speech segment should have been emitted"
    segment = await stt_q.get()
    assert isinstance(segment, np.ndarray)
    assert len(segment) > 0
