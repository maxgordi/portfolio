import asyncio
import numpy as np
import pytest
from babellang.audio.vad import SileroVAD


@pytest.fixture(scope="module")
def vad():
    v = SileroVAD(threshold=0.5)
    asyncio.get_event_loop().run_until_complete(v.load())
    return v


@pytest.mark.asyncio
async def test_silence_is_not_speech(vad):
    silence = np.zeros(512, dtype=np.int16)
    prob = await vad.process_chunk(silence)
    assert prob < 0.3, f"Silence should have low VAD prob, got {prob:.4f}"


@pytest.mark.asyncio
async def test_chunk_size_512(vad):
    """Silero requires exactly 512 samples at 16kHz."""
    chunk = np.zeros(512, dtype=np.int16)
    prob = await vad.process_chunk(chunk)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


@pytest.mark.asyncio
async def test_repeated_chunks_stable(vad):
    """Processing the same chunk multiple times should give consistent results."""
    silence = np.zeros(512, dtype=np.int16)
    probs = [await vad.process_chunk(silence) for _ in range(5)]
    assert max(probs) - min(probs) < 0.01, "VAD should be deterministic for identical input"
