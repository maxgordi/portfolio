"""
Tests for the 2026-07 improvements: PTT auto-split, anti-echo capture gate,
TTS volume gain, and the numpy sample buffer.
All tests are hardware-free.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from babellang.config import AudioConfig
from babellang.pipeline import PipelineStatus, _run_ptt_loop, _run_vad_loop


class _FakePtt:
    def __init__(self) -> None:
        self.active = False


def _speech_vad(prob: float = 0.9):
    vad = MagicMock()

    async def fake_process(chunk):
        return prob

    vad.process_chunk = fake_process
    return vad


def _seq_vad(probs: list[float], tail: float = 0.1):
    vad = MagicMock()
    it = iter(probs)

    async def fake_process(chunk):
        try:
            return next(it)
        except StopIteration:
            return tail

    vad.process_chunk = fake_process
    return vad


async def _drain_markers(stt_q: asyncio.Queue) -> int:
    """Count False (finalize) markers currently in the queue."""
    finalizes = 0
    while not stt_q.empty():
        item = stt_q.get_nowait()
        if item is False:
            finalizes += 1
    return finalizes


# ─── PTT auto-split ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ptt_long_pause_splits_segment():
    """A long pause while the key is held must finalize a sub-segment."""
    cfg = AudioConfig(ptt_split_silence_chunks=5, ptt_split_min_chunks=3)
    raw_q: asyncio.Queue = asyncio.Queue()
    stt_q: asyncio.Queue = asyncio.Queue(maxsize=200)
    status = PipelineStatus()
    ptt = _FakePtt()
    ptt.active = True

    # 10 speech chunks, then 8 silence chunks → split must fire at silence #5
    vad = _seq_vad([0.9] * 10 + [0.1] * 8)
    for _ in range(18):
        await raw_q.put(np.zeros(512, dtype=np.int16))

    task = asyncio.create_task(_run_ptt_loop(raw_q, stt_q, ptt, status, vad=vad, cfg=cfg))
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert await _drain_markers(stt_q) >= 1, "long pause must finalize a sub-segment"


@pytest.mark.asyncio
async def test_ptt_no_split_without_enough_speech():
    """Silence-only hold must NOT emit finalize markers (nothing to transcribe)."""
    cfg = AudioConfig(ptt_split_silence_chunks=5, ptt_split_min_chunks=3)
    raw_q: asyncio.Queue = asyncio.Queue()
    stt_q: asyncio.Queue = asyncio.Queue(maxsize=200)
    status = PipelineStatus()
    ptt = _FakePtt()
    ptt.active = True

    vad = _speech_vad(prob=0.1)  # pure silence
    for _ in range(30):
        await raw_q.put(np.zeros(512, dtype=np.int16))

    task = asyncio.create_task(_run_ptt_loop(raw_q, stt_q, ptt, status, vad=vad, cfg=cfg))
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert await _drain_markers(stt_q) == 0, "no speech → no auto-split finalize"


@pytest.mark.asyncio
async def test_ptt_release_finalizes():
    """Key release must finalize the pending audio."""
    cfg = AudioConfig()
    raw_q: asyncio.Queue = asyncio.Queue()
    stt_q: asyncio.Queue = asyncio.Queue(maxsize=200)
    status = PipelineStatus()
    ptt = _FakePtt()
    ptt.active = True

    vad = _speech_vad(0.9)
    for _ in range(5):
        await raw_q.put(np.zeros(512, dtype=np.int16))

    task = asyncio.create_task(_run_ptt_loop(raw_q, stt_q, ptt, status, vad=vad, cfg=cfg))
    await asyncio.sleep(0.05)
    ptt.active = False
    await raw_q.put(np.zeros(512, dtype=np.int16))  # one more chunk to observe the release
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert await _drain_markers(stt_q) == 1, "release must send exactly one finalize"


# ─── Anti-echo capture gate ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_vad_gate_blocks_segment_start():
    """While capture_gate returns True, VAD must not start segments."""
    cfg = AudioConfig(vad_speech_start_chunks=2, vad_speech_end_chunks=3)
    raw_q: asyncio.Queue = asyncio.Queue()
    stt_q: asyncio.Queue = asyncio.Queue()
    status = PipelineStatus()

    vad = _speech_vad(0.9)  # constant loud speech (e.g. our own echo)
    for _ in range(20):
        await raw_q.put(np.zeros(512, dtype=np.int16))

    task = asyncio.create_task(_run_vad_loop(
        raw_q, stt_q, vad, cfg, status, MagicMock(), 0.0,
        capture_gate=lambda: True,
    ))
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert stt_q.empty(), "gated VAD must not emit any audio"
    assert status.state != "speaking"


@pytest.mark.asyncio
async def test_vad_gate_open_allows_speech():
    """With the gate open, the same audio must start a segment."""
    cfg = AudioConfig(vad_speech_start_chunks=2, vad_speech_end_chunks=3)
    raw_q: asyncio.Queue = asyncio.Queue()
    stt_q: asyncio.Queue = asyncio.Queue()
    status = PipelineStatus()

    vad = _speech_vad(0.9)
    for _ in range(20):
        await raw_q.put(np.zeros(512, dtype=np.int16))

    task = asyncio.create_task(_run_vad_loop(
        raw_q, stt_q, vad, cfg, status, MagicMock(), 0.0,
        capture_gate=lambda: False,
    ))
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert not stt_q.empty(), "open gate must let speech through"


# ─── TTS gain and sample buffer ───────────────────────────────────────────────

def _make_playback(**kwargs):
    from babellang.audio.playback import AudioPlayback
    loop = asyncio.new_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    return AudioPlayback(device=None, sample_rate=44100, queue=q, loop=loop, **kwargs)


def test_tts_gain_scales_output():
    pb = _make_playback(tts_gain=0.5)
    pb._play_buffer.extend([1.0] * 256)

    outdata = np.zeros((256, 1), dtype=np.float32)
    status = MagicMock()
    status.output_underflow = False
    pb._callback(outdata, 256, None, status)

    np.testing.assert_allclose(outdata[:, 0], 0.5, atol=1e-5)


def test_is_playing_reflects_buffer():
    pb = _make_playback()
    assert not pb.is_playing
    pb._play_buffer.extend([0.5] * 100)
    assert pb.is_playing
    # Drain fully via callback
    outdata = np.zeros((256, 1), dtype=np.float32)
    status = MagicMock()
    status.output_underflow = False
    pb._callback(outdata, 256, None, status)
    assert not pb.is_playing


def test_sample_buffer_partial_reads():
    from babellang.audio.playback import _SampleBuffer
    buf = _SampleBuffer()
    buf.write(np.arange(10, dtype=np.float32))
    out, active = buf.read(4)
    assert active
    np.testing.assert_array_equal(out, [0, 1, 2, 3])
    out, active = buf.read(4)
    np.testing.assert_array_equal(out, [4, 5, 6, 7])
    out, active = buf.read(4)  # only 2 left → zero-padded
    assert active
    np.testing.assert_array_equal(out, [8, 9, 0, 0])
    out, active = buf.read(4)
    assert not active
    np.testing.assert_array_equal(out, [0, 0, 0, 0])


def test_tts_active_within_hangover():
    pb = _make_playback()
    pb._play_buffer.extend([0.5] * 64)
    outdata = np.zeros((64, 1), dtype=np.float32)
    status = MagicMock()
    status.output_underflow = False
    pb._callback(outdata, 64, None, status)  # drains buffer, stamps last-active
    assert not pb.is_playing
    assert pb.tts_active_within(1.0), "hangover must hold shortly after playback ends"
    assert not pb.tts_active_within(0.0)
