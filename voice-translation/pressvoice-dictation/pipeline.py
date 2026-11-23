"""Recorded WAV → final text. Shared by the macOS and the Qt (Windows/Linux)
apps; everything UI-related goes through the callbacks."""
import time
from typing import Callable, Optional

from i18n import t
from transcribe import (
    FALLBACK_MODEL,
    ModelUnavailable,
    apply_style,
    is_digital_silence,
    make_client,
    split_wav,
    transcribe_chunk,
    wav_duration,
)

# Shorter than this is an accidental key tap: sending it only invites the
# model to hallucinate a phrase out of silence.
MIN_SECONDS = 0.4


class TooShort(Exception):
    pass


class NoAudio(Exception):
    """The microphone delivered only zeros — almost always a privacy
    setting blocking it, not a quiet room."""


def transcribe_wav(
    api_key: str,
    wav: bytes,
    model: str,
    style_prompt: str = "",
    style_name: str = "",
    on_status: Optional[Callable[[str], None]] = None,
    on_notice: Optional[Callable[[str, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Returns the text to paste ("" if nothing was recognised).

    on_status(state) gets "chunk i/n" or "style"; on_notice(title, message)
    is for non-fatal problems the user should hear about."""
    on_status = on_status or (lambda s: None)
    on_notice = on_notice or (lambda t, m: None)
    log = log or (lambda m: None)

    if wav_duration(wav) < MIN_SECONDS:
        raise TooShort()
    if is_digital_silence(wav):
        raise NoAudio()

    client = make_client(api_key)
    chunks = split_wav(wav)
    total = len(chunks)
    model = model or FALLBACK_MODEL
    log(f"transcribe: {total} chunk(s), model={model}")

    parts = []
    for i, chunk in enumerate(chunks, 1):
        if total > 1:
            on_status(f"chunk {i}/{total}")
        log(f"transcribe: chunk {i}/{total} start")
        t0 = time.time()
        try:
            part = transcribe_chunk(client, chunk, model)
        except ModelUnavailable as e:
            if model == FALLBACK_MODEL:
                raise
            log(f"model unavailable ({e}); falling back to {FALLBACK_MODEL}")
            on_notice(
                t("notify.model_unavailable.title"),
                t("notify.model_unavailable.body", model=model, fallback=FALLBACK_MODEL),
            )
            model = FALLBACK_MODEL
            part = transcribe_chunk(client, chunk, model)
        log(f"transcribe: chunk {i}/{total} done in {time.time()-t0:.1f}s")
        parts.append(part)

    text = " ".join(p for p in parts if p)
    prompt = (style_prompt or "").strip()
    if not text or not prompt:
        return text

    # Run the transcript through the active style. Falls back to the raw
    # transcript on failure — never loses the text.
    on_status("style")
    log(f"style: applying '{style_name}'")
    t0 = time.time()
    try:
        styled = apply_style(client, prompt, text)
        log(f"style: done in {time.time()-t0:.1f}s")
        return styled or text
    except Exception as e:
        log(f"style error: {e}")
        on_notice(t("notify.style_failed.title"), t("notify.style_failed.body"))
        return text
