"""CLI: voiceset analyze / calibrate / build-suno / build-rvc / report / all."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from . import __version__
from .config import dump_config, load_config
from .workspace import Paths, resolve, setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Voice Dataset Builder: из сухой записи пения собирает набор 2:00 для Suno Voices и датасет для RVC.")
CONFIG = typer.Option(None, "--config", "-c", help="Свой config.yaml (перекрывает значения по умолчанию)")
log = logging.getLogger("voiceset")


def _cfg(config, **over):
    cfg = load_config(config)
    return cfg


def _start(out: Path, verbose: bool = False) -> Paths:
    p = resolve(out)
    setup_logging(p, verbose)
    return p


@app.callback()
def _main(version: bool = typer.Option(False, "--version", help="Показать версию", is_eager=True)):
    if version:
        typer.echo(f"voiceset {__version__}")
        raise typer.Exit()


@app.command()
def analyze(input_dir: Path = typer.Argument(..., help="Папка с сухими записями (WAV/FLAC)"),
            out: Path = typer.Option(Path("out"), "-o", "--out", help="Выходная папка"),
            labels: Path = typer.Option(None, "--labels", help="labels.yaml для калибровки подачи"),
            asr: bool = typer.Option(False, "--asr", help="Включить Whisper (слова фраз)"),
            config: Path = CONFIG, verbose: bool = typer.Option(False, "-v")):
    """Нарезка на фразы + все признаки → work/analysis.parquet."""
    from .analyze import analyze as run
    cfg = load_config(config, {"asr": {"enabled": True}} if asr else None)
    p = _start(out, verbose)
    dump_config(cfg, p.config_used)
    run(input_dir, p, cfg, labels)


@app.command()
def calibrate(work: Path = typer.Argument(...), labels: Path = typer.Option(..., "--labels"),
              config: Path = CONFIG):
    """Дообучить классификатор подачи по labels.yaml и пересчитать метки."""
    from .analyze import calibrate as run
    cfg = load_config(config)
    run(_start(work), cfg, labels)


@app.command("build-suno")
def build_suno(work: Path = typer.Argument(...), length: float = typer.Option(None, "--length", help="Длина набора, с (по умолчанию 120)"),
               seed: int = typer.Option(None, "--seed"), exclude: str = typer.Option(None, "--exclude", help="Номера фраз через запятую"),
               include: str = typer.Option(None, "--include", help="Номера фраз через запятую"),
               config: Path = CONFIG, verbose: bool = typer.Option(False, "-v")):
    """Собрать suno/voice_set_2min.wav ровно на 2:00."""
    from .suno import build_suno as run, parse_ids
    cfg = load_config(config)
    run(_start(work, verbose), cfg, length, seed, parse_ids(exclude), parse_ids(include))


@app.command("build-rvc")
def build_rvc(work: Path = typer.Argument(...), sr: int = typer.Option(None, "--sr", help="40000 или 48000"),
              combined: bool = typer.Option(False, "--combined", help="Ещё и один общий WAV"), config: Path = CONFIG):
    """Собрать чистый датасет для RVC (Applio)."""
    from .rvc import build_rvc as run
    from .workspace import load_table
    cfg = load_config(config)
    p = _start(work)
    run(load_table(p), p, cfg.rvc, sr, combined)


@app.command()
def report(work: Path = typer.Argument(...), config: Path = CONFIG):
    """Собрать report/report.html."""
    from .report import build_report
    cfg = load_config(config)
    build_report(_start(work), cfg)


@app.command("all")
def all_(input_dir: Path = typer.Argument(...), out: Path = typer.Option(Path("out"), "-o", "--out"),
         labels: Path = typer.Option(None, "--labels"), sr: int = typer.Option(None, "--sr"),
         length: float = typer.Option(None, "--length"), asr: bool = typer.Option(False, "--asr"),
         config: Path = CONFIG, verbose: bool = typer.Option(False, "-v")):
    """Весь пайплайн: анализ → Suno → RVC → отчёт."""
    from .analyze import analyze as run_an
    from .report import build_report
    from .rvc import build_rvc as run_rvc
    from .suno import build_suno as run_suno
    from .workspace import load_table
    cfg = load_config(config, {"asr": {"enabled": True}} if asr else None)
    p = _start(out, verbose)
    dump_config(cfg, p.config_used)
    run_an(input_dir, p, cfg, labels)
    try:
        run_suno(p, cfg, length, None, [], [])
    except ValueError as e:
        log.error("Набор для Suno не собран: %s", e)
    run_rvc(load_table(p), p, cfg.rvc, sr)
    build_report(p, cfg)
    log.info("Готово. Главный результат: %s", p.suno / "voice_set_2min.wav")


if __name__ == "__main__":
    app()
