"""Пути рабочей папки и чтение/запись таблицы анализа."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger("voiceset")


@dataclass
class Paths:
    root: Path

    @property
    def work(self): return self.root / "work"
    @property
    def parquet(self): return self.work / "analysis.parquet"
    @property
    def phrases(self): return self.root / "phrases"
    @property
    def suno(self): return self.root / "suno"
    @property
    def rvc(self): return self.root / "rvc"
    @property
    def report(self): return self.root / "report"
    @property
    def log(self): return self.root / "log.txt"
    @property
    def review(self): return self.root / "review.yaml"
    @property
    def config_used(self): return self.root / "config.used.yaml"


def resolve(path: str | Path) -> Paths:
    """Принимает out/ или out/work/ — возвращает корень выходной папки."""
    p = Path(path)
    if (p / "analysis.parquet").exists():
        return Paths(p.parent)
    return Paths(p)


def setup_logging(paths: Paths, verbose: bool = False) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("voiceset")
    lg.handlers.clear()
    lg.setLevel(logging.DEBUG)
    fh = logging.FileHandler(paths.log, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(fh)
    lg.addHandler(sh)


def load_table(paths: Paths) -> pd.DataFrame:
    if not paths.parquet.exists():
        raise FileNotFoundError(f"Не найден {paths.parquet}. Сначала запустите: voiceset analyze <папка> -o {paths.root}")
    df = pd.read_parquet(paths.parquet).set_index("id", drop=False)
    df.index.name = None
    return df


def save_table(paths: Paths, df: pd.DataFrame) -> None:
    paths.work.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(paths.parquet, index=False)
