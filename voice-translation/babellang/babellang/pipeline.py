from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from babellang.audio.capture import AudioCapture
from babellang.audio.direct_passthrough import DirectPassthrough
from babellang.audio.playback import AudioPlayback
from babellang.audio.vad import SileroVAD
from babellang.config import AudioConfig, PipelineConfig, Settings
from babellang.ptt import PttListener, PttState
from babellang.stt.audio_translator import OpenAIAudioTranslator
from babellang.stt.openai_stt import OpenAISTT
from babellang.translation.openai_translator import OpenAITranslator
from babellang.tts.elevenlabs import ElevenLabsTTS

logger = logging.getLogger(__name__)


@dataclass
class PipelineStatus:
    """Shared status updated by each component; read by CLI for display."""
    state: str = "idle"           # idle | capturing | speaking | translating | playing
    last_source_text: str = ""
    last_translated_text: str = ""
    last_latency_ms: float = 0.0
    vad_prob: float = 0.0
    speech_segments: int = 0
    utterance_start_time: float = 0.0
    tts_error: str = ""           # last TTS error message; cleared on success


async def _supervised(
    coro_factory: Callable,
    name: str,
    status: PipelineStatus,
    max_restarts: int = 5,
) -> None:
    """
    Run a pipeline stage; on unhandled exception log it, surface it in the GUI
    status, and restart the stage with a small backoff. Without this, a crashed
    task dies silently inside asyncio and the app looks alive while one channel
    is dead. More than max_restarts crashes within a minute → give up loudly.
    """
    restarts = 0
    window_start = time.monotonic()
    while True:
        try:
            await coro_factory()
            return  # clean exit (stop signal)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            now = time.monotonic()
            if now - window_start > 60.0:
                restarts = 0
                window_start = now
            restarts += 1
            if restarts > max_restarts:
                status.tts_error = f"Канал «{name}» остановлен после {max_restarts} сбоев подряд"
                logger.error("Task %s: giving up after %d restarts: %s", name, restarts, exc)
                return
            status.tts_error = f"Сбой «{name}» — перезапуск ({restarts}/{max_restarts})"
            logger.exception("Task %s crashed (restart %d/%d)", name, restarts, max_restarts)
            await asyncio.sleep(1.5)


async def _run_stream_watchdog(
    watched: Callable[[], list[tuple[str, object]]],
    status: PipelineStatus,
) -> None:
    """
    Detect dead PortAudio streams (Bluetooth headphones went to sleep, device
    unplugged) and try to revive them. A dead stream is otherwise completely
    silent — the app looks alive while one direction of audio is gone.
    """
    while True:
        await asyncio.sleep(2.0)
        for label, obj in watched():
            stream = getattr(obj, "_stream", None)
            if stream is None or stream.active:
                continue
            logger.warning("Watchdog: stream «%s» died, attempting restart", label)
            status.tts_error = f"⚠ Поток «{label}» остановился — переподключаю…"
            try:
                if asyncio.iscoroutinefunction(obj.stop):
                    await obj.stop()
                    await obj.start()
                else:
                    obj.stop()
                    obj.start()
                status.tts_error = ""
                logger.info("Watchdog: stream «%s» restarted OK", label)
            except Exception as exc:
                status.tts_error = (
                    f"⚠ Устройство «{label}» недоступно — включи его или нажми СТОП и СТАРТ"
                )
                logger.warning("Watchdog: restart of «%s» failed: %s", label, exc)
                await asyncio.sleep(3.0)  # don't hammer a missing device


def _chunk_to_vad_rate(chunk: np.ndarray, vad_samples: int = 512) -> np.ndarray:
    """
    Convert a native-rate 32ms chunk to the 512-sample/16kHz format Silero expects.
    Linear interpolation is fine here — this copy feeds ONLY voice-activity
    detection; the audio sent to STT stays at the native rate, untouched.
    """
    if len(chunk) == vad_samples:
        return chunk
    idx = np.linspace(0, len(chunk) - 1, vad_samples)
    return np.interp(idx, np.arange(len(chunk)), chunk.astype(np.float32)).astype(np.int16)


def _stt_q_put_nowait(stt_q: asyncio.Queue, item) -> None:
    """Non-blocking put. For control signals (False/None), evicts oldest audio chunk to make room."""
    try:
        stt_q.put_nowait(item)
    except asyncio.QueueFull:
        if item is not False and item is not None:
            return  # drop audio chunk silently — stale audio is worse than a gap
        # Control signal: evict oldest item to guarantee delivery
        try:
            stt_q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            stt_q.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("stt_q still full after eviction; dropping control signal %r", item)


async def _run_vad_loop(
    raw_audio_q: asyncio.Queue,
    stt_q: asyncio.Queue,
    vad: SileroVAD,
    cfg: AudioConfig,
    status: PipelineStatus,
    playback: AudioPlayback,
    passthrough_ratio: float,
    capture_gate: Optional[Callable[[], bool]] = None,
    label: str = "?",
    capture_rate: int = 16000,
) -> None:
    """
    VAD state machine shared by both OutgoingPipeline and IncomingPipeline.
    Streams native-rate 32ms int16 chunks directly to stt_q while in SPEAKING state
    (STT receives the audio bit-perfect; only the VAD copy is downsampled to 16kHz).
    stt_q protocol: np.ndarray (audio chunk) | False (send_finalize) | None (stop).
    Uses put_nowait — drops audio chunks when queue is full rather than blocking,
    which prevents stale audio accumulation during Deepgram reconnects.
    When passthrough_ratio > 0, feeds each SPEAKING chunk into playback.feed_passthrough().
    capture_gate: while it returns True, new segments are not started — used to
    ignore far-end echo of our own TTS coming back through the call.
    """
    state = "SILENCE"
    above_count = 0
    below_count = 0
    preroll: deque[np.ndarray] = deque(maxlen=cfg.vad_speech_start_chunks)
    speaking_start = 0.0

    while True:
        chunk = await raw_audio_q.get()
        prob = await vad.process_chunk(_chunk_to_vad_rate(chunk, cfg.chunk_samples))
        status.vad_prob = prob

        if state == "SILENCE":
            preroll.append(chunk)
            if capture_gate is not None and capture_gate():
                above_count = 0
                continue
            if prob >= cfg.vad_threshold:
                above_count += 1
                if above_count >= cfg.vad_speech_start_chunks:
                    state = "SPEAKING"
                    above_count = 0
                    below_count = 0
                    speaking_start = time.monotonic()
                    status.state = "speaking"
                    # Replay preroll so Deepgram gets audio context before speech onset
                    for pc in preroll:
                        _stt_q_put_nowait(stt_q, pc)
                    logger.info("VAD[%s]: *** SPEECH START (stt_q size=%d)", label, stt_q.qsize())
            else:
                above_count = 0

        elif state == "SPEAKING":
            # Non-blocking send — drop chunk if queue is full to avoid stale audio buildup
            _stt_q_put_nowait(stt_q, chunk)

            if passthrough_ratio > 0.0:
                playback.feed_passthrough(chunk, src_sample_rate=capture_rate)

            if prob < cfg.vad_threshold:
                below_count += 1
                if below_count >= cfg.vad_speech_end_chunks:
                    state = "SILENCE"
                    below_count = 0
                    preroll.clear()
                    status.state = "capturing"
                    status.speech_segments += 1
                    status.utterance_start_time = speaking_start
                    # Signal end of utterance → STT calls send_finalize()
                    _stt_q_put_nowait(stt_q, False)
                    duration = time.monotonic() - speaking_start
                    logger.info("VAD[%s]: *** SPEECH END (duration=%.2fs, stt_q size=%d)", label, duration, stt_q.qsize())
            else:
                below_count = 0

            # Force finalize if speaking too long — stay in SPEAKING for next sub-segment
            if (time.monotonic() - speaking_start) > cfg.max_segment_seconds:
                _stt_q_put_nowait(stt_q, False)
                speaking_start = time.monotonic()
                logger.warning("VAD[%s]: max segment length, force-finalizing", label)


async def _run_ptt_loop(
    raw_audio_q: asyncio.Queue,
    stt_q: asyncio.Queue,
    ptt_state: PttState,
    status: PipelineStatus,
    vad: Optional[SileroVAD] = None,
    cfg: Optional[AudioConfig] = None,
) -> None:
    """
    PTT with auto-split: record while the key is held, but finalize a sub-segment
    on every long pause (ptt_split_silence_chunks) and on max_segment_seconds —
    so a long monologue starts translating immediately instead of waiting for
    the key release. Short pauses between words stay inside one segment.
    """
    was_active = False
    silence_count = 0
    chunks_since_split = 0
    segment_start = 0.0

    while True:
        chunk = await raw_audio_q.get()
        active = ptt_state.active

        if active and not was_active:
            status.state = "speaking"
            silence_count = 0
            chunks_since_split = 0
            segment_start = time.monotonic()
            logger.info("PTT: recording started")

        if active:
            _stt_q_put_nowait(stt_q, chunk)
            chunks_since_split += 1

            if vad is not None and cfg is not None:
                prob = await vad.process_chunk(_chunk_to_vad_rate(chunk, cfg.chunk_samples))
                status.vad_prob = prob
                silence_count = silence_count + 1 if prob < cfg.vad_threshold else 0

                long_pause = (
                    silence_count >= cfg.ptt_split_silence_chunks
                    and chunks_since_split - silence_count >= cfg.ptt_split_min_chunks
                )
                too_long = (time.monotonic() - segment_start) > cfg.max_segment_seconds
                if long_pause or too_long:
                    _stt_q_put_nowait(stt_q, False)
                    status.speech_segments += 1
                    logger.info(
                        "PTT: auto-split (%s) after %d chunks",
                        "pause" if long_pause else "max length", chunks_since_split,
                    )
                    silence_count = 0
                    chunks_since_split = 0
                    segment_start = time.monotonic()

        if not active and was_active:
            status.state = "capturing"
            # Only finalize if the tail sub-segment actually contains audio
            if chunks_since_split > 0:
                _stt_q_put_nowait(stt_q, False)
            logger.info("PTT: key released — sending segment for transcription")

        was_active = active


class OutgoingPipeline:
    """
    Outgoing channel: Physical mic → BlackHole 2ch virtual device.
    Pipeline: AudioCapture → VAD → OpenAIAudioTranslator(ru→en) → ElevenLabsTTS → AudioPlayback

    Each stage communicates via asyncio.Queue.
    """

    def __init__(self, settings: Settings, api_keys: dict[str, str],
                 ptt_state: Optional[PttState] = None) -> None:
        self._settings = settings
        self._api_keys = api_keys
        self.status = PipelineStatus()

        loop = asyncio.get_event_loop()
        cfg = settings.audio
        my_lang = settings.my_language
        inter_lang = settings.interlocutor_language

        # Build outgoing PipelineConfig using selected languages
        pipe = PipelineConfig(
            stt_language=my_lang,
            openai_audio_model=settings.pipeline.openai_audio_model,
            elevenlabs_model=settings.pipeline.elevenlabs_model,
            elevenlabs_voice_id=settings.pipeline.elevenlabs_voice_id,
            elevenlabs_output_format=settings.pipeline.elevenlabs_output_format,
            tts_sample_rate=settings.pipeline.tts_sample_rate,
            stt_language_mode=settings.stt_language_mode,
        )

        # raw_audio_q: 32 chunks × 32ms = ~1s. AudioCapture drops with put_nowait if full.
        # stt_q: 50 chunks × 32ms = ~1.6s. Small prevents stale-audio buildup on reconnect.
        self._raw_audio_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=32)
        self._stt_q: asyncio.Queue = asyncio.Queue(maxsize=50)  # carries np.ndarray | False | None
        self._token_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=20)
        self._audio_out_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=200)

        self._capture = AudioCapture(
            device=settings.input_device,
            config=cfg,
            queue=self._raw_audio_q,
            loop=loop,
        )
        self._vad = SileroVAD(threshold=cfg.vad_threshold)
        self._mode = settings.stt_mode

        # PTT (push-to-talk) — only for outgoing pipeline, only in manual vad_mode.
        # ptt_state may be injected externally (GUI mode); otherwise created here (CLI mode).
        self._ptt_listener: Optional[PttListener] = None
        if ptt_state is not None:
            self._ptt_state: Optional[PttState] = ptt_state
        elif settings.vad_mode == "manual":
            self._ptt_state = PttState()
            self._ptt_listener = PttListener(settings.ptt_hotkey, self._ptt_state)
        else:
            self._ptt_state = None

        # Mode "audio": one API call — gpt-4o-audio-preview does STT + translation together
        self._audio_translator: Optional[OpenAIAudioTranslator] = None
        # Mode "classic": two API calls — OpenAISTT transcribes, OpenAITranslator translates
        self._transcript_q: Optional[asyncio.Queue] = None
        self._stt: Optional[OpenAISTT] = None
        self._translator: Optional[OpenAITranslator] = None

        if self._mode == "audio":
            self._audio_translator = OpenAIAudioTranslator(
                api_key=api_keys["openai"],
                stt_queue=self._stt_q,
                token_queue=self._token_q,
                config=pipe,
                input_sample_rate=self._capture.sample_rate,
                source_lang=my_lang,
                target_lang=inter_lang,
                on_transcript=lambda text: setattr(self.status, "last_source_text", text),
                on_translation=lambda text: setattr(self.status, "last_translated_text", text),
            )
        else:
            self._transcript_q = asyncio.Queue(maxsize=10)
            self._stt = OpenAISTT(
                api_key=api_keys["openai"],
                stt_queue=self._stt_q,
                transcript_queue=self._transcript_q,
                config=pipe,
                input_sample_rate=self._capture.sample_rate,
                on_transcript=lambda text: setattr(self.status, "last_source_text", text),
            )
            self._translator = OpenAITranslator(
                api_key=api_keys["openai"],
                transcript_queue=self._transcript_q,
                token_queue=self._token_q,
                config=pipe,
                on_translation=lambda text: setattr(self.status, "last_translated_text", text),
                source_lang=my_lang,
                target_lang=inter_lang,
            )

        self._tts = ElevenLabsTTS(
            api_key=api_keys["elevenlabs"],
            token_queue=self._token_q,
            audio_out_queue=self._audio_out_q,
            config=pipe,
            on_error=lambda msg: setattr(self.status, "tts_error", msg[:100]),
        )
        # TTS-only playback — DirectPassthrough handles the mic passthrough at native quality
        self._playback = AudioPlayback(
            device=settings.output_device,
            sample_rate=pipe.tts_sample_rate,
            queue=self._audio_out_q,
            loop=loop,
            original_ratio=0.0,
            tts_gain=settings.outgoing_tts_volume,
        )
        # High-quality direct mic → BlackHole passthrough (native sample rate, no degradation)
        self._direct_pt = DirectPassthrough(
            input_device=settings.input_device,
            output_device=settings.output_device,
            volume=settings.outgoing_original_ratio,
        )
        # While True, VAD won't start new segments (e.g. incoming TTS playing through speakers)
        self._capture_gate: Optional[Callable[[], bool]] = None
        self._tasks: list[asyncio.Task] = []

    def set_passthrough_volume(self, volume: float) -> None:
        """Called from asyncio via call_soon_threadsafe to update mic passthrough volume."""
        self._direct_pt.set_volume(volume)

    def set_tts_volume(self, volume: float) -> None:
        """Volume of the translated ElevenLabs voice sent to the interlocutor."""
        self._playback.set_tts_gain(volume)

    def set_capture_gate(self, gate: Callable[[], bool]) -> None:
        self._capture_gate = gate

    @property
    def tts_playing(self) -> bool:
        """True while translated audio is actually being sent to the interlocutor."""
        return self._playback.is_playing

    async def _duck_passthrough_loop(self) -> None:
        """Duck the live mic passthrough while translated TTS is playing to the interlocutor."""
        while True:
            self._direct_pt.set_ducked(self._playback.is_playing)
            await asyncio.sleep(0.05)

    def _watched_streams(self) -> list[tuple[str, object]]:
        return [
            ("микрофон", self._capture),
            ("вывод собеседнику", self._playback),
            ("живой голос", self._direct_pt),
        ]

    async def _watchdog_loop(self) -> None:
        await _run_stream_watchdog(self._watched_streams, self.status)

    async def start(self) -> None:
        logger.info("Loading outgoing VAD model...")
        await self._vad.load()
        logger.info("Starting outgoing audio streams...")
        self._direct_pt.start()
        await self._capture.start()
        await self._playback.start()
        st = self.status
        mode_tasks = (
            [asyncio.create_task(_supervised(self._audio_translator.run, "распознавание", st),
                                 name="audio_translator")]
            if self._mode == "audio"
            else [
                asyncio.create_task(_supervised(self._stt.run, "распознавание", st), name="stt"),
                asyncio.create_task(_supervised(self._translator.run, "перевод", st), name="translator"),
            ]
        )
        if self._ptt_listener:
            self._ptt_listener.start()
        input_loop = (
            asyncio.create_task(_supervised(self._ptt_loop, "запись (PTT)", st), name="ptt")
            if self._ptt_state is not None
            else asyncio.create_task(_supervised(self._vad_loop, "запись (VAD)", st), name="vad")
        )
        self._tasks = [
            input_loop,
            *mode_tasks,
            asyncio.create_task(_supervised(self._tts.run, "озвучка", st), name="tts"),
            asyncio.create_task(self._duck_passthrough_loop(), name="duck_passthrough"),
            asyncio.create_task(self._watchdog_loop(), name="outgoing_watchdog"),
        ]
        self.status.state = "capturing"
        logger.info("Outgoing pipeline started (mode=%s)", "ptt" if self._ptt_state else "vad")

    async def _vad_loop(self) -> None:
        # passthrough_ratio=0.0 — DirectPassthrough feeds the mic, not AudioPlayback
        await _run_vad_loop(
            self._raw_audio_q, self._stt_q, self._vad,
            self._settings.audio, self.status, self._playback,
            0.0,
            capture_gate=lambda: self._capture_gate() if self._capture_gate else False,
            label="OUT",
            capture_rate=self._capture.sample_rate,
        )

    async def _ptt_loop(self) -> None:
        await _run_ptt_loop(
            self._raw_audio_q, self._stt_q, self._ptt_state, self.status,
            vad=self._vad, cfg=self._settings.audio,
        )

    async def stop(self) -> None:
        logger.info("Stopping outgoing pipeline...")
        if self._ptt_listener:
            self._ptt_listener.stop()
        # Cancel tasks FIRST so the watchdog can't fight the shutdown by
        # "reviving" streams we are closing.
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        # Every stream must close even if a neighbour raises — a leaked open
        # stream crashes the process on the next PortAudio re-initialization.
        try:
            self._direct_pt.stop()
        except Exception as exc:
            logger.warning("Outgoing passthrough stop: %s", exc)
        try:
            await self._capture.stop()
        except Exception as exc:
            logger.warning("Outgoing capture stop: %s", exc)
        self._vad.shutdown()
        try:
            await self._playback.stop()
        except Exception as exc:
            logger.warning("Outgoing playback stop: %s", exc)
        self.status.state = "idle"
        logger.info("Outgoing pipeline stopped")


class IncomingPipeline:
    """
    Incoming channel: BlackHole 16ch (interlocutor audio) → user's real speakers.
    Pipeline: AudioCapture → VAD → OpenAIAudioTranslator(en→ru) → ElevenLabsTTS → AudioPlayback

    Mirror of OutgoingPipeline with reversed language direction and different devices.
    Only instantiated when settings.incoming_device is not None.

    transcript_filter: callable(text) -> True to drop an incoming utterance before
    translation — used to reject far-end echo of our own outgoing TTS.
    """

    def __init__(self, settings: Settings, api_keys: dict[str, str],
                 transcript_filter: Optional[Callable[[str], bool]] = None) -> None:
        self._settings = settings
        self._api_keys = api_keys
        self.status = PipelineStatus()

        loop = asyncio.get_event_loop()
        cfg = settings.audio

        inter_lang = settings.interlocutor_language
        my_lang = settings.my_language

        # Build incoming PipelineConfig (interlocutor's language → user's language)
        pipe = PipelineConfig(
            stt_language=inter_lang,
            openai_audio_model=settings.pipeline.openai_audio_model,
            elevenlabs_model=settings.pipeline.elevenlabs_model,
            elevenlabs_voice_id=settings.pipeline.elevenlabs_voice_id,
            elevenlabs_output_format=settings.pipeline.elevenlabs_output_format,
            tts_sample_rate=settings.pipeline.tts_sample_rate,
            stt_language_mode=settings.stt_language_mode,
        )

        self._raw_audio_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=32)
        self._stt_q: asyncio.Queue = asyncio.Queue(maxsize=50)  # carries np.ndarray | False | None
        self._token_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=20)
        self._audio_out_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=200)

        self._capture = AudioCapture(
            device=settings.incoming_device,
            config=cfg,
            queue=self._raw_audio_q,
            loop=loop,
        )
        self._vad = SileroVAD(threshold=cfg.vad_threshold)
        self._mode = settings.stt_mode

        self._audio_translator: Optional[OpenAIAudioTranslator] = None
        self._transcript_q: Optional[asyncio.Queue] = None
        self._stt: Optional[OpenAISTT] = None
        self._translator: Optional[OpenAITranslator] = None

        if self._mode == "audio":
            self._audio_translator = OpenAIAudioTranslator(
                api_key=api_keys["openai"],
                stt_queue=self._stt_q,
                token_queue=self._token_q,
                config=pipe,
                input_sample_rate=self._capture.sample_rate,
                source_lang=inter_lang,
                target_lang=my_lang,
                on_transcript=lambda text: setattr(self.status, "last_source_text", text),
                on_translation=lambda text: setattr(self.status, "last_translated_text", text),
                transcript_filter=transcript_filter,
            )
        else:
            self._transcript_q = asyncio.Queue(maxsize=10)
            self._stt = OpenAISTT(
                api_key=api_keys["openai"],
                stt_queue=self._stt_q,
                transcript_queue=self._transcript_q,
                config=pipe,
                input_sample_rate=self._capture.sample_rate,
                on_transcript=lambda text: setattr(self.status, "last_source_text", text),
                transcript_filter=transcript_filter,
            )
            self._translator = OpenAITranslator(
                api_key=api_keys["openai"],
                transcript_queue=self._transcript_q,
                token_queue=self._token_q,
                config=pipe,
                on_translation=lambda text: setattr(self.status, "last_translated_text", text),
                source_lang=inter_lang,
                target_lang=my_lang,
            )

        self._tts = ElevenLabsTTS(
            api_key=api_keys["elevenlabs"],
            token_queue=self._token_q,
            audio_out_queue=self._audio_out_q,
            config=pipe,
            on_error=lambda msg: setattr(self.status, "tts_error", msg[:100]),
        )
        self._playback = AudioPlayback(
            device=settings.speaker_device,
            sample_rate=pipe.tts_sample_rate,
            queue=self._audio_out_q,
            loop=loop,
            original_ratio=settings.incoming_original_ratio,
            tts_gain=settings.incoming_tts_volume,
        )
        # Anti-echo gate: while our outgoing TTS is playing into the call, the far end
        # may echo it back — don't start capturing "interlocutor speech" during that time
        self._capture_gate: Optional[Callable[[], bool]] = None
        self._tasks: list[asyncio.Task] = []

    def set_tts_volume(self, volume: float) -> None:
        """Volume of the translated ElevenLabs voice played to the user."""
        self._playback.set_tts_gain(volume)

    def set_capture_gate(self, gate: Callable[[], bool]) -> None:
        self._capture_gate = gate

    @property
    def tts_playing(self) -> bool:
        """True while translated audio is actually playing in the user's speakers."""
        return self._playback.is_playing

    def seconds_since_signal(self) -> float:
        """How long BlackHole 16ch has been silent — a large value means the call
        app is probably not routing its output here."""
        return time.time() - self._capture.last_signal_time

    def _watched_streams(self) -> list[tuple[str, object]]:
        return [
            ("звук собеседника", self._capture),
            ("твои динамики", self._playback),
        ]

    async def _watchdog_loop(self) -> None:
        await _run_stream_watchdog(self._watched_streams, self.status)

    async def start(self) -> None:
        logger.info("Loading incoming VAD model...")
        await self._vad.load()
        logger.info("Starting incoming audio streams...")
        await self._capture.start()
        await self._playback.start()
        st = self.status
        mode_tasks = (
            [asyncio.create_task(_supervised(self._audio_translator.run, "распознавание входящего", st),
                                 name="incoming_audio_translator")]
            if self._mode == "audio"
            else [
                asyncio.create_task(_supervised(self._stt.run, "распознавание входящего", st),
                                    name="incoming_stt"),
                asyncio.create_task(_supervised(self._translator.run, "перевод входящего", st),
                                    name="incoming_translator"),
            ]
        )
        self._tasks = [
            asyncio.create_task(_supervised(self._vad_loop, "захват собеседника", st),
                                name="incoming_vad"),
            *mode_tasks,
            asyncio.create_task(_supervised(self._tts.run, "озвучка входящего", st),
                                name="incoming_tts"),
            asyncio.create_task(self._watchdog_loop(), name="incoming_watchdog"),
        ]
        self.status.state = "capturing"
        logger.info("Incoming pipeline started")

    async def _vad_loop(self) -> None:
        await _run_vad_loop(
            self._raw_audio_q, self._stt_q, self._vad,
            self._settings.audio, self.status, self._playback,
            self._settings.incoming_original_ratio,
            capture_gate=lambda: self._capture_gate() if self._capture_gate else False,
            label="IN",
            capture_rate=self._capture.sample_rate,
        )

    async def stop(self) -> None:
        logger.info("Stopping incoming pipeline...")
        # Tasks first — the watchdog must not revive streams we are closing
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        try:
            await self._capture.stop()
        except Exception as exc:
            logger.warning("Incoming capture stop: %s", exc)
        self._vad.shutdown()
        try:
            await self._playback.stop()
        except Exception as exc:
            logger.warning("Incoming playback stop: %s", exc)
        self.status.state = "idle"
        logger.info("Incoming pipeline stopped")

