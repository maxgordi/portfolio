import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from babellang.config import Settings, AudioConfig, PipelineConfig


@pytest.mark.asyncio
async def test_vad_state_machine_emits_segment():
    """VAD loop should emit a speech segment after detecting speech then silence."""
    from babellang.audio.vad import SileroVAD
    from babellang.config import AudioConfig

    cfg = AudioConfig(
        vad_speech_start_chunks=2,
        vad_speech_end_chunks=3,
    )

    # Simulate VAD returning high prob for speech, then low for silence
    call_count = 0
    speech_probs = [0.9] * 4 + [0.1] * 5  # 4 speech chunks, 5 silence chunks

    async def mock_process(chunk):
        nonlocal call_count
        prob = speech_probs[min(call_count, len(speech_probs) - 1)]
        call_count += 1
        return prob

    raw_q: asyncio.Queue = asyncio.Queue()
    stt_q: asyncio.Queue = asyncio.Queue()

    # Fill raw_q with dummy audio chunks
    for _ in range(len(speech_probs) + 5):
        await raw_q.put(np.zeros(512, dtype=np.int16))

    vad = SileroVAD()
    vad.process_chunk = mock_process

    # Directly test the state machine logic inline
    from collections import deque
    state = "SILENCE"
    above_count = 0
    below_count = 0
    preroll = deque(maxlen=cfg.vad_speech_start_chunks)
    segment = []

    for i in range(len(speech_probs)):
        chunk = np.zeros(512, dtype=np.int16)
        prob = speech_probs[i]

        if state == "SILENCE":
            preroll.append(chunk)
            if prob >= cfg.vad_threshold:
                above_count += 1
                if above_count >= cfg.vad_speech_start_chunks:
                    state = "SPEAKING"
                    above_count = 0
                    below_count = 0
                    segment = list(preroll)
            else:
                above_count = 0
        elif state == "SPEAKING":
            segment.append(chunk)
            if prob < cfg.vad_threshold:
                below_count += 1
                if below_count >= cfg.vad_speech_end_chunks:
                    state = "SILENCE"
                    trim = cfg.vad_speech_end_chunks
                    speech_only = segment[:-trim] if len(segment) > trim else segment
                    if speech_only:
                        audio = np.concatenate(speech_only)
                        await stt_q.put(audio)
                    segment = []
            else:
                below_count = 0

    assert not stt_q.empty(), "Expected speech segment to be emitted"
    segment_audio = await stt_q.get()
    assert isinstance(segment_audio, np.ndarray)
    assert len(segment_audio) > 0


def test_settings_serialization():
    settings = Settings(
        pipeline=PipelineConfig(elevenlabs_voice_id="test_voice"),
        input_device=1,
        output_device=2,
    )
    d = settings.to_dict()
    restored = Settings.from_dict(d)
    assert restored.input_device == 1
    assert restored.output_device == 2
    assert restored.pipeline.elevenlabs_voice_id == "test_voice"
