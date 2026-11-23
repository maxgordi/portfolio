from __future__ import annotations

import asyncio
import logging
import sys

import aiohttp
import sounddevice as sd
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from babellang.audio.capture import find_device_by_name, list_devices
from babellang.audio.devices import resolve_devices
from babellang.config import (
    LANGUAGE_NAMES,
    PipelineConfig,
    Settings,
    load_settings,
    require_api_key,
    save_settings,
    get_api_key,
)
from babellang.pipeline import IncomingPipeline, OutgoingPipeline, PipelineStatus

console = Console()
_log_file = logging.FileHandler("/tmp/babellang.log", mode="w")
_log_file.setLevel(logging.DEBUG)
_log_file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("babellang").setLevel(logging.DEBUG)
logging.getLogger("babellang").addHandler(_log_file)


# ─── Setup helpers ────────────────────────────────────────────────────────────

async def _check_blackhole_2ch() -> int:
    device_idx = find_device_by_name("BlackHole 2ch", require_output=True)
    if device_idx is None:
        device_idx = find_device_by_name("BlackHole", require_output=True)
    if device_idx is None:
        console.print("\n[bold red]BlackHole 2ch not found.[/bold red]")
        console.print("Install it:")
        console.print("  [bold cyan]brew install --cask blackhole-2ch[/bold cyan]")
        console.print("\nThen restart babellang.")
        sys.exit(1)
    return device_idx


async def _check_blackhole_16ch() -> int | None:
    """Returns device index or None if user opts to skip incoming pipeline."""
    device_idx = find_device_by_name("BlackHole 16ch", require_input=True)
    if device_idx is None:
        console.print("\n[yellow]BlackHole 16ch not found.[/yellow]")
        console.print("It is needed to capture the interlocutor's audio (incoming translation).")
        console.print("Install it:")
        console.print("  [bold cyan]brew install --cask blackhole-16ch[/bold cyan]")
        console.print("\nWithout it, incoming translation is disabled (outgoing still works).")
        if not Confirm.ask("Continue without incoming translation?", default=True):
            sys.exit(1)
        return None
    console.print(f"[green]✓ BlackHole 16ch found (device {device_idx})[/green]")
    return device_idx


async def _pick_voices(api_key: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": api_key},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    console.print(f"[red]ElevenLabs API error: {resp.status}[/red]")
                    return ""
                data = await resp.json()
    except Exception as exc:
        console.print(f"[red]Failed to fetch voices: {exc}[/red]")
        return ""

    all_voices = data.get("voices", [])
    if not all_voices:
        console.print("[yellow]No voices found in your ElevenLabs account.[/yellow]")
        return ""

    # Sort: cloned/generated voices first (user's own voices), then pre-made
    def _sort_key(v: dict) -> int:
        cat = v.get("category", "")
        return 0 if cat in ("cloned", "generated", "professional") else 1

    voices = sorted(all_voices, key=_sort_key)

    table = Table(title="ElevenLabs Voices", show_header=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Name")
    table.add_column("Type", width=12)
    table.add_column("Voice ID", style="dim")
    for i, v in enumerate(voices):
        cat = v.get("category", "?")
        cat_style = "bold green" if cat in ("cloned", "generated", "professional") else "dim"
        table.add_row(str(i), v.get("name", "?"), f"[{cat_style}]{cat}[/{cat_style}]", v.get("voice_id", "?"))
    console.print(table)

    _PREFERRED_VOICE_ID = "7ajqUfdTnjp4oZ6AK5Tg"
    default_idx = next(
        (i for i, v in enumerate(voices) if v.get("voice_id") == _PREFERRED_VOICE_ID),
        next(
            (i for i, v in enumerate(voices) if v.get("category") in ("cloned", "generated", "professional")),
            0,
        ),
    )
    idx = IntPrompt.ask("Select voice number", default=default_idx)
    idx = max(0, min(idx, len(voices) - 1))
    chosen = voices[idx]
    console.print(f"[green]Voice selected:[/green] {chosen['name']} (ID: {chosen['voice_id']})")
    return chosen["voice_id"]


def _pick_input_device() -> int:
    """Pick user's real microphone — excludes BlackHole virtual devices."""
    devices = list_devices()
    input_devices = [
        d for d in devices
        if d["max_input_channels"] > 0 and "blackhole" not in d["name"].lower()
    ]
    table = Table(title="Audio Input Devices (your real microphones)")
    table.add_column("Index", style="cyan")
    table.add_column("Name")
    table.add_column("Channels", justify="right")
    for d in input_devices:
        table.add_row(str(d["index"]), d["name"], str(d["max_input_channels"]))
    console.print(table)
    default_idx = input_devices[0]["index"] if input_devices else 0
    return IntPrompt.ask("Select microphone (input device index)", default=default_idx)


def _pick_speaker_device() -> int:
    """Pick user's real speakers — excludes BlackHole virtual devices."""
    devices = list_devices()
    output_devices = [
        d for d in devices
        if d["max_output_channels"] > 0 and "blackhole" not in d["name"].lower()
    ]
    table = Table(title="Audio Output Devices (your real speakers / headphones)")
    table.add_column("Index", style="cyan")
    table.add_column("Name")
    table.add_column("Channels", justify="right")
    for d in output_devices:
        table.add_row(str(d["index"]), d["name"], str(d["max_output_channels"]))
    console.print(table)
    default_idx = output_devices[0]["index"] if output_devices else 0
    return IntPrompt.ask("Select your speakers/headphones (output device index)", default=default_idx)


def _pick_languages() -> tuple[str, str]:
    langs = list(LANGUAGE_NAMES.items())

    table = Table(title="Available Languages / Доступные языки")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Language")
    table.add_column("Code", style="dim")
    for i, (code, name) in enumerate(langs):
        table.add_row(str(i), name, code)
    console.print(table)

    my_default = next((i for i, (c, _) in enumerate(langs) if c == "ru"), 0)
    my_idx = IntPrompt.ask(
        "На каком языке говорите ВЫ? / What language do YOU speak? (enter #)",
        default=my_default,
    )
    my_idx = max(0, min(my_idx, len(langs) - 1))
    my_code, my_name = langs[my_idx]
    console.print(f"[green]Ваш язык / Your language:[/green] {my_name} ({my_code})")

    inter_default = next((i for i, (c, _) in enumerate(langs) if c == "en"), 1)
    inter_idx = IntPrompt.ask(
        "На каком языке говорит СОБЕСЕДНИК? / What language does your INTERLOCUTOR speak? (enter #)",
        default=inter_default,
    )
    inter_idx = max(0, min(inter_idx, len(langs) - 1))
    inter_code, inter_name = langs[inter_idx]
    console.print(f"[green]Язык собеседника / Interlocutor language:[/green] {inter_name} ({inter_code})")

    return my_code, inter_code


def _pick_mix_ratios(my_lang: str = "ru", inter_lang: str = "en") -> tuple[float, float]:
    my_name = LANGUAGE_NAMES.get(my_lang, my_lang.upper())
    inter_name = LANGUAGE_NAMES.get(inter_lang, inter_lang.upper())
    console.print(
        "\n[bold]Voice Mixing Ratios[/bold]\n"
        "  [dim]0.0 = translated TTS only  |  1.0 = original voice only  |  0.3 = 30% original + 70% TTS[/dim]\n"
    )

    def _ask_ratio(label: str) -> float:
        while True:
            raw = Prompt.ask(label, default="0.0")
            try:
                return max(0.0, min(1.0, float(raw)))
            except ValueError:
                console.print("[red]Please enter a number between 0.0 and 1.0[/red]")

    out_ratio = _ask_ratio(
        f"Outgoing mix — how much of YOUR raw {my_name} voice participants also hear (alongside {inter_name} TTS)"
    )
    in_ratio = _ask_ratio(
        f"Incoming mix — how much of interlocutor's raw {inter_name} voice you also hear (alongside {my_name} TTS)"
    )
    return out_ratio, in_ratio


# ─── First-run / forced setup ─────────────────────────────────────────────────

async def setup_if_needed(settings: Settings) -> Settings:
    needs_full_setup = (
        not settings.pipeline.elevenlabs_voice_id
        or settings.input_device is None
        or settings.output_device is None
        or settings.speaker_device is None
    )
    needs_lang_setup = not settings.languages_configured

    if not needs_full_setup and not needs_lang_setup:
        return settings

    # ── Language-only setup (devices already configured) ──────────────────────
    if not needs_full_setup and needs_lang_setup:
        console.print(Panel(
            "[bold]Language Setup / Настройка языков[/bold]\n"
            "Choose your language and your interlocutor's language.",
            title="BabelLang — Language Setup",
        ))
        my_lang, inter_lang = _pick_languages()
        settings.my_language = my_lang
        settings.interlocutor_language = inter_lang
        settings.languages_configured = True
        save_settings(settings)
        console.print("\n[green]✓ Languages saved.[/green]")
        return settings

    # ── Full first-run setup ───────────────────────────────────────────────────
    console.print(Panel(
        "[bold]Welcome to BabelLang[/bold]\n"
        "Real-time bidirectional voice translation for video calls.\n\n"
        "Let's set up your API keys, languages, and audio devices.",
        title="First Run Setup",
    ))

    # Step 1 — API keys
    console.print("\n[bold]Step 1: API Keys[/bold] (stored securely in macOS Keychain)\n")
    oai_key = require_api_key("openai", "OpenAI API key")  # noqa: F841
    el_key = require_api_key("elevenlabs", "ElevenLabs API key")

    # Step 2 — Languages
    console.print("\n[bold]Step 2: Select Languages / Выберите языки[/bold]\n")
    my_lang, inter_lang = _pick_languages()

    # Step 3 — Voice
    console.print("\n[bold]Step 3: Select ElevenLabs Voice[/bold]\n")
    console.print("[dim]The eleven_flash_v2_5 model is multilingual — the same voice speaks in any language.[/dim]\n")
    voice_id = await _pick_voices(el_key)
    if not voice_id:
        console.print("[yellow]No voice selected. You can re-run setup later.[/yellow]")

    # Step 4 — BlackHole 2ch (outgoing: participants hear your translated voice)
    console.print("\n[bold]Step 4: Outgoing Virtual Device (BlackHole 2ch)[/bold]\n")
    output_device = await _check_blackhole_2ch()
    console.print(f"[green]✓ BlackHole 2ch found (device {output_device})[/green]")

    # Step 5 — Microphone
    console.print("\n[bold]Step 5: Select Your Microphone[/bold]\n")
    input_device = _pick_input_device()

    # Step 6 — BlackHole 16ch (incoming: capture interlocutor audio)
    console.print("\n[bold]Step 6: Incoming Virtual Device (BlackHole 16ch)[/bold]\n")
    incoming_device = await _check_blackhole_16ch()

    # Step 7 — Real speakers
    inter_name = LANGUAGE_NAMES.get(my_lang, my_lang.upper())
    console.print("\n[bold]Step 7: Select Your Speakers / Headphones[/bold]\n")
    console.print(f"[dim]This is where you will hear the translated {inter_name} TTS from your interlocutor.[/dim]\n")
    speaker_device = _pick_speaker_device()

    # Step 8 — Mix ratios
    console.print("\n[bold]Step 8: Voice Mixing[/bold]")
    outgoing_ratio, incoming_ratio = _pick_mix_ratios(my_lang, inter_lang)

    # Save
    new_pipeline_cfg = PipelineConfig(elevenlabs_voice_id=voice_id)
    new_settings = Settings(
        pipeline=new_pipeline_cfg,
        input_device=input_device,
        output_device=output_device,
        input_device_name=sd.query_devices(input_device)["name"],
        output_device_name=sd.query_devices(output_device)["name"],
        incoming_device=incoming_device,
        incoming_device_name=sd.query_devices(incoming_device)["name"] if incoming_device is not None else "",
        speaker_device=speaker_device,
        speaker_device_name=sd.query_devices(speaker_device)["name"],
        outgoing_original_ratio=outgoing_ratio,
        incoming_original_ratio=incoming_ratio,
        my_language=my_lang,
        interlocutor_language=inter_lang,
        languages_configured=True,
    )
    save_settings(new_settings)
    console.print("\n[green]✓ Setup complete! Settings saved.[/green]")
    _print_meet_instructions(incoming_device is not None)
    return new_settings


def _print_meet_instructions(has_incoming: bool) -> None:
    if has_incoming:
        console.print(
            "\n[bold cyan]In Google Meet / Zoom — set these two audio options:[/bold cyan]\n"
            "  Microphone → [bold]BlackHole 2ch[/bold]    (participants hear your translated English)\n"
            "  Speaker    → [bold]BlackHole 16ch[/bold]   (BabelLang captures interlocutor's English)\n"
            "  [dim]Your headphones/speakers play the translated Russian TTS automatically.[/dim]\n"
            "  [dim]Expect 3–8 sec translation delay.[/dim]"
        )
    else:
        console.print(
            "\n[bold cyan]In Google Meet / Zoom:[/bold cyan]\n"
            "  Microphone → [bold]BlackHole 2ch[/bold]    (participants hear your translated English)\n"
            "  Speaker    → [bold]your headphones / built-in speakers[/bold]  (NOT BlackHole)\n"
            "  [dim]Expect 3–8 sec translation delay.[/dim]"
        )


# ─── Device validation ────────────────────────────────────────────────────────

def _validate_devices(settings: Settings) -> Settings:
    """Re-resolve stored devices by name (indices shift when Bluetooth devices connect)."""
    res = resolve_devices(settings)
    if res.changed:
        save_settings(settings)
    for w in res.warnings:
        console.print(f"[yellow]{w}[/yellow]")
    if not res.ok:
        for e in res.errors:
            console.print(f"[red]✗ {e}[/red]")
        console.print("[red]Run [bold]babellang --setup[/bold] to reconfigure.[/red]")
        sys.exit(1)
    return settings


# ─── Rich live display ────────────────────────────────────────────────────────

def _build_display(
    out_status: PipelineStatus,
    in_status: PipelineStatus | None = None,
    my_lang: str = "?",
    inter_lang: str = "?",
) -> Panel:
    STATE_COLOR = {
        "idle": "dim", "capturing": "green",
        "speaking": "yellow", "translating": "cyan", "playing": "magenta",
    }
    my_tag = f"[{my_lang.upper()}]"
    inter_tag = f"[{inter_lang.upper()}]"

    def _status_line(status: PipelineStatus) -> Text:
        color = STATE_COLOR.get(status.state, "white")
        t = Text()
        t.append("Status: ", style="bold")
        t.append(status.state.upper(), style=f"bold {color}")
        t.append(f"   VAD: {status.vad_prob:.2f}")
        t.append(f"   Segments: {status.speech_segments}")
        if status.last_latency_ms > 0:
            t.append(f"   Last latency: {status.last_latency_ms:.0f}ms")
        if status.tts_error:
            t.append(f"\n  TTS ERROR: {status.tts_error}", style="bold red")
        return t

    content = Text()

    # Outgoing (MY LANG → INTER LANG)
    content.append(f"── Outgoing ({my_tag} → {inter_tag}) ──\n", style="bold dim")
    content.append_text(_status_line(out_status))
    content.append("\n\n")
    src = Text(f"{my_tag} ", style="bold yellow")
    src.append(out_status.last_source_text or "—", style="white")
    content.append_text(src)
    content.append("\n")
    trn = Text(f"{inter_tag} ", style="bold cyan")
    trn.append(out_status.last_translated_text or "—", style="white")
    content.append_text(trn)

    if in_status is not None:
        content.append(f"\n\n── Incoming ({inter_tag} → {my_tag}) ──\n", style="bold dim")
        content.append_text(_status_line(in_status))
        content.append("\n\n")
        src_in = Text(f"{inter_tag} ", style="bold cyan")
        src_in.append(in_status.last_source_text or "—", style="white")
        content.append_text(src_in)
        content.append("\n")
        trn_in = Text(f"{my_tag} ", style="bold yellow")
        trn_in.append(in_status.last_translated_text or "—", style="white")
        content.append_text(trn_in)

    return Panel(content, title="[bold]BabelLang[/bold] — Press Ctrl+C to stop")


# ─── Main ─────────────────────────────────────────────────────────────────────

def _load_api_keys() -> dict[str, str]:
    keys = {
        "openai": get_api_key("openai") or "",
        "elevenlabs": get_api_key("elevenlabs") or "",
    }
    missing = [k for k, v in keys.items() if not v]
    if missing:
        console.print(f"[red]Missing API keys: {', '.join(missing)}. Run setup first.[/red]")
        sys.exit(1)
    return keys


async def _async_main(force_setup: bool = False) -> None:
    settings = load_settings()
    if force_setup:
        settings = Settings()
    settings = await setup_if_needed(settings)
    settings = _validate_devices(settings)

    api_keys = _load_api_keys()

    my_lang = settings.my_language
    inter_lang = settings.interlocutor_language
    my_name = LANGUAGE_NAMES.get(my_lang, my_lang.upper())
    inter_name = LANGUAGE_NAMES.get(inter_lang, inter_lang.upper())

    in_name = settings.input_device_name or f"device {settings.input_device}"
    out_name = settings.output_device_name or f"device {settings.output_device}"
    spk_name = settings.speaker_device_name or f"device {settings.speaker_device}"
    console.print("\n[bold green]Starting translation pipeline...[/bold green]")
    console.print(f"  Languages         : [bold yellow]{my_name}[/bold yellow] → [bold cyan]{inter_name}[/bold cyan] (outgoing)")
    console.print(f"                      [bold cyan]{inter_name}[/bold cyan] → [bold yellow]{my_name}[/bold yellow] (incoming)")
    console.print(f"  Mic (input)       : [{settings.input_device}] {in_name}")
    console.print(f"  Out (BlackHole 2ch): [{settings.output_device}] {out_name}")
    console.print(f"  Speakers          : [{settings.speaker_device}] {spk_name}")
    if settings.incoming_device is not None:
        inc_name = settings.incoming_device_name or f"device {settings.incoming_device}"
        console.print(f"  Incoming (BH 16ch): [{settings.incoming_device}] {inc_name}")
    console.print(
        f"  Mix out/in        : {settings.outgoing_original_ratio:.0%} original"
        f" / {settings.incoming_original_ratio:.0%} original\n"
    )
    _print_meet_instructions(settings.incoming_device is not None)
    console.print()

    outgoing = OutgoingPipeline(settings=settings, api_keys=api_keys)
    incoming: IncomingPipeline | None = None
    if settings.incoming_device is not None and settings.speaker_device is not None:
        incoming = IncomingPipeline(settings=settings, api_keys=api_keys)
        # Anti-echo gates — see gui.py for rationale
        out_pb, in_pb = outgoing._playback, incoming._playback
        incoming.set_capture_gate(lambda: out_pb.tts_active_within(0.75))
        outgoing.set_capture_gate(lambda: in_pb.tts_active_within(0.3))

    stop_event = asyncio.Event()

    def _on_sigint() -> None:
        stop_event.set()

    loop = asyncio.get_event_loop()
    try:
        loop.add_signal_handler(__import__("signal").SIGINT, _on_sigint)
    except NotImplementedError:
        pass

    await outgoing.start()
    if incoming:
        await incoming.start()

    try:
        with Live(
            _build_display(outgoing.status, incoming.status if incoming else None, my_lang, inter_lang),
            console=console,
            refresh_per_second=4,
            transient=True,
        ) as live:
            while not stop_event.is_set():
                live.update(_build_display(
                    outgoing.status,
                    incoming.status if incoming else None,
                    my_lang,
                    inter_lang,
                ))
                await asyncio.sleep(0.25)
    finally:
        console.print("\n[yellow]Stopping...[/yellow]")
        await outgoing.stop()
        if incoming:
            await incoming.stop()
        console.print("[green]BabelLang stopped.[/green]")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BabelLang — real-time bidirectional voice translation")
    parser.add_argument("--setup", action="store_true", help="Re-run setup (change devices, voice, or mix ratios)")
    parser.add_argument("--gui", action="store_true", help="Launch graphical interface")
    args = parser.parse_args()

    if args.gui:
        from babellang.gui import launch_gui
        launch_gui()
        return

    try:
        asyncio.run(_async_main(force_setup=args.setup))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
