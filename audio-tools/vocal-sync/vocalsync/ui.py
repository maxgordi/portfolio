"""
VocalSync — окно (наследник VoxMatch Простой).

Скиньте эталон в левую зону, дубли — в правую. Каждый дубль выравнивается по
времени под эталон, результат и весь пакет файлов (WAV, превью, карта, отчёт, график)
сохраняется в output/<имя>_<дата-время>/. Результат перетаскивается прямо в Cubase.
По желанию (⚙): подтяжка нот и комп из лучших фраз нескольких дублей.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import traceback

import numpy as np
import soundfile as sf
from PySide6.QtCore import QMimeData, QPointF, QRectF, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QDrag, QFont, QKeySequence, QPainter,
                           QPainterPath, QPen, QPolygonF, QShortcut)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QSlider,
    QSplitter, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from . import APP_NAME, __version__
from .config import MODE_NAMES, MODES, PRESETS, Settings

try:
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None

log = logging.getLogger("vocalsync.ui")
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(APP_DIR, "error.log")
AUDIO_EXT = {".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".m4a"}
AUDIO_FILTER = "Аудио (*.wav *.wave *.aif *.aiff *.flac *.ogg *.mp3 *.m4a)"
ACCENT = "#3fb6a8"
REF_COL = "#f0a040"
BEFORE_COL = "#7d868e"
BG = "#1b1e21"
PANEL = "#23272b"
REF_KEY = "__ref__"
SR_CHOICES = [(None, "как у эталона"), (44100, "44 100 Гц"), (48000, "48 000 Гц"), (96000, "96 000 Гц")]
ENGINES = [("auto", "авто (лучший доступный)"), ("cli", "Rubber Band CLI (R3)"),
           ("pylibrb", "Rubber Band pylibrb (R3)"), ("wsola", "WSOLA (запасной)")]

_pipeline = None
_comp = None


def pipeline():
    global _pipeline
    if _pipeline is None:
        from . import pipeline as p
        _pipeline = p
    return _pipeline


def comp_mod():
    global _comp
    if _comp is None:
        from . import comp as c
        _comp = c
    return _comp


def log_error(text: str):
    log.error(text)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
    except OSError:
        pass


def inbox_dir() -> str:
    return os.path.join(os.path.expanduser("~"), "Desktop", "VocalSync Inbox")


def score_color(v) -> QColor:
    if v is None:
        return QColor("#5a6168")
    if v >= 80:
        return QColor("#49c28a")
    if v >= 60:
        return QColor("#e3c25a")
    return QColor("#e0605a")


def fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


# ----------------------------------------------------------------------------
# Перетаскивание: файлы из Finder или куски прямо из Cubase (vst-xml) — из VoxMatch
# ----------------------------------------------------------------------------
def audio_paths(mime: QMimeData) -> list[str]:
    out = []
    if not mime.hasUrls():
        return out
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        p = url.toLocalFile()
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if os.path.splitext(name)[1].lower() in AUDIO_EXT:
                    out.append(os.path.join(p, name))
        elif os.path.splitext(p)[1].lower() in AUDIO_EXT:
            out.append(p)
        elif p.lower().endswith(".textclipping"):
            out.extend(cubase_to_files(read_textclipping(p))[0])
    return out


def mime_text(mime: QMimeData) -> str:
    txt = mime.text() if mime.hasText() else ""
    if "vst-xml" in txt:
        return txt
    for fmt in mime.formats():
        raw = bytes(mime.data(fmt))
        for enc in ("utf-8", "utf-16", "utf-16-le", "mac-roman"):
            try:
                t = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            if "vst-xml" in t:
                return t.replace("\x00", "")
    return txt


def parse_cubase_regions(text: str) -> list[dict]:
    import html
    import re
    if not text or "vst-xml" not in text:
        return []
    out = []
    for block in re.findall(r"<region\b.*?</region>", text, flags=re.S | re.I):
        fn = re.search(r"<filename>(.*?)</filename>", block, flags=re.S | re.I)
        st = re.search(r"<start>\s*([-\d.]+)\s*</start>", block, flags=re.I)
        en = re.search(r"<end>\s*([-\d.]+)\s*</end>", block, flags=re.I)
        qn = re.search(r"<projectTime[^>]*>\s*([-\d.]+)\s*</projectTime>", block, flags=re.I)
        if not fn:
            continue
        try:
            start = int(float(st.group(1))) if st else 0
            end = int(float(en.group(1))) if en else 0
        except ValueError:
            continue
        out.append({"filename": html.unescape(fn.group(1).strip()), "start": start, "end": end,
                    "qn": float(qn.group(1)) if qn else None})
    return out


def read_textclipping(path: str) -> str:
    import plistlib
    try:
        with open(path, "rb") as f:
            d = plistlib.load(f)
        u = d.get("UTI-Data", {})
        for key, enc in (("public.utf8-plain-text", "utf-8"), ("public.utf16-plain-text", "utf-16-le")):
            raw = u.get(key)
            if raw is not None:
                return raw.decode(enc, "replace") if isinstance(raw, bytes) else str(raw)
    except Exception:  # noqa: BLE001
        pass
    return ""


def _bar_label(qn):
    if qn is None:
        return ""
    bar = int(qn // 4) + 1
    beat = qn - (bar - 1) * 4 + 1
    b = f"{beat:.0f}" if abs(beat - round(beat)) < 1e-6 else f"{beat:.2f}".rstrip("0")
    return f"такт {bar}.{b}"


def cubase_to_files(text: str) -> tuple[list[str], list[str]]:
    regions = parse_cubase_regions(text)
    paths, errors = [], []
    if not regions:
        return paths, errors
    out_dir = os.path.join(inbox_dir(), "Куски из Cubase")
    os.makedirs(out_dir, exist_ok=True)
    for r in regions:
        fn = r["filename"]
        src = next((c for c in (fn, fn.replace("\\/", "/"), fn.replace("\\/", ":"))
                    if os.path.isfile(c)), None)
        name = os.path.basename(fn.replace("\\/", "/"))
        if src is None:
            errors.append(f"Файл не найден: {fn}")
            continue
        try:
            info = sf.info(src)
            a = max(0, min(r["start"], info.frames))
            b = info.frames if r["end"] <= 0 else max(a, min(r["end"], info.frames))
            if b - a < info.samplerate * 0.3:
                errors.append(f"«{name}»: кусок слишком короткий")
                continue
            data, sr = sf.read(src, start=a, stop=b, always_2d=True, dtype="float32")
            stem = os.path.splitext(name)[0]
            lab = _bar_label(r.get("qn"))
            base = f"{stem} — {lab}" if lab else f"{stem} — {a}"
            target = os.path.join(out_dir, base + ".wav")
            k = 2
            while os.path.exists(target):
                target = os.path.join(out_dir, f"{base} ({k}).wav")
                k += 1
            sub = info.subtype if info.subtype in ("PCM_16", "PCM_24", "PCM_32", "FLOAT") else "PCM_24"
            sf.write(target, data, sr, subtype=sub)
            paths.append(target)
        except Exception as exc:  # noqa: BLE001
            log_error(traceback.format_exc())
            errors.append(f"«{name}»: {exc}")
    return paths, errors


def dropped(mime: QMimeData) -> tuple[list[str], list[str]]:
    paths = audio_paths(mime)
    if paths:
        return paths, []
    return cubase_to_files(mime_text(mime))


def drag_files(widget, paths):
    if not paths:
        return
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
    d = QDrag(widget)
    d.setMimeData(mime)
    d.exec(Qt.CopyAction)


# ----------------------------------------------------------------------------
# Небольшие виджеты
# ----------------------------------------------------------------------------
class DropZone(QFrame):
    filesDropped = Signal(list)
    clicked = Signal()

    def __init__(self, title: str, hint: str, color: str):
        super().__init__()
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(58)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(1)
        self.title = QLabel(title)
        self.title.setObjectName("dropTitle")
        self.title.setStyleSheet(f"color: {color};")
        self.hint = QLabel(hint)
        self.hint.setObjectName("hint")
        lay.addWidget(self.title)
        lay.addWidget(self.hint)

    def set_hint(self, text: str):
        self.hint.setText(text)

    def _hover(self, on):
        self.setProperty("hover", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e):
        e.acceptProposedAction()
        self._hover(True)

    def dragLeaveEvent(self, e):
        self._hover(False)

    def dropEvent(self, e):
        self._hover(False)
        paths, errors = dropped(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self.filesDropped.emit(paths)
        if errors:
            QMessageBox.warning(self, APP_NAME, "\n".join(errors))
        elif not paths:
            log_error("Перетаскивание без файла, форматы: " + ", ".join(e.mimeData().formats()))
            QMessageBox.information(
                self, APP_NAME,
                "Не получилось взять звук из перетаскивания.\n\nВ Cubase: Аудио → Bounce, "
                "затем перетащите получившийся файл, или перетащите файл из Finder.")

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()


class TrackTable(QTableWidget):
    """Список дорожек. Принимает файлы; готовые результаты перетаскиваются в Cubase."""
    filesDropped = Signal(list)
    starClicked = Signal(int)

    COLS = ["★", "Дорожка", "Сдвиг", "Растяж.", "Атаки", "Статус"]

    def __init__(self, paths_for_drag):
        super().__init__(0, len(self.COLS))
        self._paths = paths_for_drag
        self.setHorizontalHeaderLabels(self.COLS)
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Fixed)
        self.setColumnWidth(0, 30)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in range(2, len(self.COLS)):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(30)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setShowGrid(False)
        self.setFocusPolicy(Qt.NoFocus)
        self.cellClicked.connect(lambda r, c: self.starClicked.emit(r) if c == 0 else None)

    def startDrag(self, actions):
        drag_files(self, self._paths())

    def dragEnterEvent(self, e):
        if e.source() is self:
            e.ignore()
        else:
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.source() is self:
            e.ignore()
        else:
            e.acceptProposedAction()

    def dropEvent(self, e):
        if e.source() is self:
            return
        paths, errors = dropped(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self.filesDropped.emit(paths)
        if errors:
            QMessageBox.warning(self, APP_NAME, "\n".join(errors))


class DragChip(QLabel):
    """Плашка «перетащите в Cubase»."""

    def __init__(self, text, make_paths):
        super().__init__(text)
        self.setObjectName("chip")
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.OpenHandCursor)
        self._make = make_paths
        self._press = None
        self.setToolTip("Зажмите и перетащите на дорожку Cubase")

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._press is None:
            return
        if (e.position().toPoint() - self._press).manhattanLength() < QApplication.startDragDistance():
            return
        self._press = None
        drag_files(self, self._make())


class MiniWave(QWidget):
    """Маленькая волна; для компа фразы подкрашены оценкой."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(220, 40)
        self.env = None
        self.blocks = []            # [(a, b, оценка, метка)] в долях 0..1

    def set_data(self, env, blocks=()):
        self.env, self.blocks = env, list(blocks)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#1a1d20"))
        if self.env is None or len(self.env) < 2:
            p.setPen(QColor("#6b737a"))
            p.drawText(self.rect(), Qt.AlignCenter, "появится после обработки")
            return
        for a, b, sc, _lab in self.blocks:
            c = score_color(sc)
            c.setAlpha(45)
            p.fillRect(QRectF(a * w, 0, (b - a) * w, h), c)
        m = len(self.env) // max(1, w)
        e = self.env[: m * w].reshape(w, m).max(axis=1) if m >= 1 else self.env
        e = np.clip(e / 0.5, 0, 1) * (h / 2 - 2)
        mid = h / 2
        xs = np.linspace(0, w, len(e))
        poly = QPolygonF([QPointF(float(x), float(mid - a)) for x, a in zip(xs, e)] +
                         [QPointF(float(x), float(mid + a)) for x, a in zip(xs[::-1], e[::-1])])
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ACCENT))
        p.drawPolygon(poly)
        p.setPen(QColor(255, 255, 255, 60))
        for a, _b, _sc, _lab in self.blocks[1:]:
            p.drawLine(int(a * w), 0, int(a * w), h)


class ResultCard(QFrame):
    """Карточка результата: выровненный dub (или комп). Перетаскивается в Cubase целиком."""

    def __init__(self, title, make_paths):
        super().__init__()
        self.setObjectName("final")
        self._make = make_paths
        self._press = None
        self.setCursor(Qt.OpenHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = QLabel(title)
        self.title.setObjectName("finalTitle")
        self.info = QLabel("Скиньте эталон и дубль — результат появится здесь.")
        self.info.setObjectName("hint")
        self.info.setWordWrap(True)
        col.addWidget(self.title)
        col.addWidget(self.info)
        lay.addLayout(col, 3)
        self.wave = MiniWave()
        self.wave.setMinimumSize(140, 40)
        lay.addWidget(self.wave, 2)
        btns = QGridLayout()
        btns.setSpacing(6)
        self.btn_play = QPushButton("▶ Слушать")
        self.chip = DragChip("⇱  В Cubase", make_paths)
        self.chip.setObjectName("bigchip")
        self.btn_export = QPushButton("💾 Экспорт WAV…")
        self.btn_folder = QPushButton("Папка")
        self.btn_plot = QPushButton("График")
        btns.addWidget(self.btn_play, 0, 0)
        btns.addWidget(self.chip, 0, 1, 1, 2)
        btns.addWidget(self.btn_export, 1, 0)
        btns.addWidget(self.btn_folder, 1, 1)
        btns.addWidget(self.btn_plot, 1, 2)
        lay.addLayout(btns)
        self.setToolTip("Зажмите карточку (или «В Cubase») и бросьте на дорожку Cubase — "
                        "файл ляжет с нуля, ровно под эталон.")

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._press is None:
            return
        if (e.position().toPoint() - self._press).manhattanLength() < QApplication.startDragDistance():
            return
        self._press = None
        drag_files(self, self._make())


class Slider:
    def __init__(self, grid, row, col, label, lo, hi, value, fmt, tip):
        self.fmt = fmt
        self.label = QLabel(label)
        self.s = QSlider(Qt.Horizontal)
        self.s.setRange(lo, hi)
        self.s.setValue(value)
        self.v = QLabel()
        self.v.setMinimumWidth(110)
        for w in (self.label, self.s, self.v):
            w.setToolTip(tip)
        grid.addWidget(self.label, row, col)
        grid.addWidget(self.s, row, col + 1)
        grid.addWidget(self.v, row, col + 2)
        self.s.valueChanged.connect(self._show)
        self._show(value)

    def _show(self, v):
        self.v.setText(self.fmt(v))

    def value(self):
        return self.s.value()

    def setValue(self, v):
        self.s.setValue(int(round(v)))

    def setEnabled(self, on: bool):
        for w in (self.label, self.s, self.v):
            w.setEnabled(on)


# ----------------------------------------------------------------------------
# Плеер: несколько источников на одной шкале, переключение без остановки, петля,
# стерео-превью (эталон слева, текущий источник справа)
# ----------------------------------------------------------------------------
class Player:
    def __init__(self):
        self.stream = None
        self.sources: dict[str, np.ndarray] = {}
        self.current = "result"
        self.sr = 48000
        self.pos = 0
        self.loop = None
        self.loop_on = False
        self.ref_under = False
        self.split_lr = False
        self.playing = False
        self.length = 0

    def set_sources(self, sources: dict, sr: int):
        self.sources = {k: v for k, v in sources.items() if v is not None}
        if sr != self.sr:
            self.close()
        self.sr = sr
        self.length = max((len(v) for v in self.sources.values()), default=0)
        self.pos = min(self.pos, self.length)

    def _cb(self, out, frames, _t, _s):
        out[:] = 0.0
        if not self.playing:
            return
        src = self.sources.get(self.current)
        ref = self.sources.get("ref") if (self.ref_under or self.split_lr) and self.current != "ref" else None
        filled = 0
        guard = 0
        while filled < frames and guard < 4:
            guard += 1
            a = self.pos
            end = self.length
            if self.loop_on and self.loop and self.loop[1] - self.loop[0] > 256:
                if a < self.loop[0] or a >= self.loop[1]:
                    a = self.loop[0]
                end = self.loop[1]
            k = min(frames - filled, max(0, end - a))
            if k <= 0:
                if self.loop_on and self.loop:
                    self.pos = self.loop[0]
                    continue
                self.playing = False
                self.pos = 0 if a >= self.length else a
                return
            if src is not None and a < len(src):
                seg = src[a:a + k]
                if self.split_lr:
                    out[filled:filled + len(seg), 1] += seg
                else:
                    out[filled:filled + len(seg), 0] += seg
                    out[filled:filled + len(seg), 1] += seg
            if ref is not None and a < len(ref):
                seg = ref[a:a + k]
                if self.split_lr:
                    out[filled:filled + len(seg), 0] += seg
                else:
                    out[filled:filled + len(seg), 0] += 0.45 * seg
                    out[filled:filled + len(seg), 1] += 0.45 * seg
            filled += k
            self.pos = a + k

    def ensure_stream(self):
        if sd is None:
            return False
        if self.stream is not None:
            return True
        try:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32",
                                          callback=self._cb, blocksize=512)
            self.stream.start()
            return True
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())
            self.stream = None
            return False

    def play(self):
        if not self.sources or not self.ensure_stream():
            return False
        self.playing = True
        return True

    def pause(self):
        self.playing = False

    def seek(self, seconds):
        self.pos = int(max(0, min(self.length, seconds * self.sr)))

    def seconds(self):
        return self.pos / self.sr if self.sr else 0.0

    def close(self):
        self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
        self.stream = None


# ----------------------------------------------------------------------------
# Таймлайн: лента «Комп», эталон + результат, дубль «до», кривая растяжения
# ----------------------------------------------------------------------------
def env_peaks(x: np.ndarray, step: int):
    if x is None or len(x) == 0:
        return np.zeros(1, np.float32)
    m = len(x) // step
    if m == 0:
        return np.array([np.max(np.abs(x))], np.float32)
    return np.abs(x[: m * step]).reshape(m, step).max(axis=1).astype(np.float32)


class Timeline(QWidget):
    seekRequested = Signal(float)
    loopChanged = Signal(object)
    choiceWheel = Signal(int, int)
    choiceMenu = Signal(int, object)
    phraseClicked = Signal(int)

    RULER_H = 18
    STRETCH_H = 56

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(340)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.NoFocus)
        self.comp_h = 0
        self.dur = 0.0
        self.v0, self.v1 = 0.0, 1.0
        self.res_env = 0.001
        self.ref_env = None
        self.after_env = None
        self.before_env = None
        self.stretch = None                  # (t (с), коэффициент), пределы
        self.limits = (0.67, 1.5)
        self.tiles = []
        self.comp_blocks = []
        self.problems = []
        self.playhead = 0.0
        self.loop = None
        self.active_phrase = -1
        self.after_label = "Результат"
        self.before_label = "Dub до (только сдвиг)"
        self._drag = None
        self._wheel_acc = 0
        self._hover = -1

    def set_comp_visible(self, on: bool):
        self.comp_h = 46 if on else 0
        self.update()

    def set_ref(self, x, sr):
        step = max(1, int(sr * self.res_env))
        self.ref_env = env_peaks(x, step)
        new_dur = len(x) / sr
        changed = abs(new_dur - self.dur) > 0.01
        self.dur = new_dur
        if changed or self.v1 <= self.v0 or self.v1 > self.dur or self.v1 - self.v0 < 0.2:
            self.v0, self.v1 = 0.0, max(0.5, self.dur)
        self.update()

    def set_take(self, after, before, sr, after_label, before_label, problems, stretch=None,
                 limits=None):
        step = max(1, int(sr * self.res_env))
        self.after_env = env_peaks(after, step) if after is not None else None
        self.before_env = env_peaks(before, step) if before is not None else None
        self.after_label, self.before_label = after_label, before_label
        self.problems = problems or []
        self.stretch = stretch
        if limits:
            self.limits = limits
        self.update()

    def set_comp(self, tiles, blocks):
        self.tiles = tiles
        self.comp_blocks = blocks
        self.update()

    def zoom_all(self):
        self.v0, self.v1 = 0.0, max(0.5, self.dur)
        self.update()

    def _x(self, t):
        return (t - self.v0) / max(1e-6, self.v1 - self.v0) * self.width()

    def _t(self, x):
        return self.v0 + x / max(1, self.width()) * (self.v1 - self.v0)

    def _lanes(self):
        h = self.height()
        top = self.RULER_H + self.comp_h + (6 if self.comp_h else 2)
        rest = h - top - self.STRETCH_H - 8
        main = int(rest * 0.64)
        before = rest - main - 4
        return ((self.RULER_H, self.comp_h), (top, main), (top + main + 4, before),
                (h - self.STRETCH_H - 2, self.STRETCH_H))

    def phrase_at(self, t):
        for i, (a, b) in enumerate(self.tiles):
            if a <= t < b:
                return i
        return -1

    def _wave(self, p, env, y0, h, color, alpha=255):
        if env is None or len(env) < 2:
            return
        w = self.width()
        i0 = max(0, int(self.v0 / self.res_env))
        i1 = min(len(env), int(np.ceil(self.v1 / self.res_env)) + 1)
        if i1 - i0 < 2:
            return
        seg = env[i0:i1]
        cols = max(1, w)
        if len(seg) > cols:
            m = len(seg) // cols
            seg = seg[: m * cols].reshape(cols, m).max(axis=1)
            xs = np.arange(cols) * (w / cols)
        else:
            xs = np.array([self._x((i0 + k) * self.res_env) for k in range(len(seg))])
        mid = y0 + h / 2
        amp = np.clip(seg / 0.5, 0, 1) * (h / 2 - 2)
        c = QColor(color)
        c.setAlpha(alpha)
        poly = QPolygonF([QPointF(float(x), float(mid - a)) for x, a in zip(xs, amp)] +
                         [QPointF(float(x), float(mid + a)) for x, a in zip(xs[::-1], amp[::-1])])
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawPolygon(poly)

    def _stretch_lane(self, p, y0, h):
        p.fillRect(QRectF(0, y0, self.width(), h), QColor(255, 255, 255, 5))
        if not self.stretch:
            self._tag(p, 6, y0 + 3, "Растяжение ×", "#8b949b")
            return
        t, k = self.stretch
        lo, hi = self.limits
        top, bot = max(hi, 1.0) * 1.05, min(lo, 1.0) * 0.95

        def y(v):
            return y0 + 4 + (top - v) / (top - bot) * (h - 8)

        for v, col in ((1.0, QColor(255, 255, 255, 40)), (lo, QColor(224, 96, 90, 90)),
                       (hi, QColor(224, 96, 90, 90))):
            p.setPen(QPen(col, 1, Qt.DashLine))
            p.drawLine(0, int(y(v)), self.width(), int(y(v)))
        i0 = max(0, int(np.searchsorted(t, self.v0)) - 1)
        i1 = min(len(t), int(np.searchsorted(t, self.v1)) + 1)
        if i1 - i0 < 2:
            return
        tt, kk = t[i0:i1], k[i0:i1]
        w = self.width()
        if len(tt) > 2 * w:
            m = len(tt) // w
            tt = tt[: m * w].reshape(w, m).mean(axis=1)
            kk = kk[: m * w].reshape(w, m)
            kk = np.where(np.abs(kk.max(axis=1) - 1) > np.abs(kk.min(axis=1) - 1),
                          kk.max(axis=1), kk.min(axis=1))
        path = QPainterPath()
        path.moveTo(self._x(tt[0]), y(kk[0]))
        for a, b in zip(tt[1:], kk[1:]):
            path.lineTo(self._x(a), y(b))
        p.setPen(QPen(QColor(ACCENT), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        self._tag(p, 6, y0 + 3, f"Растяжение × ({lo:g}–{hi:g})", "#8b949b")

    @staticmethod
    def _tag(p, x, y, text, color):
        """Подпись дорожки на тёмной плашке (поверх волн читается). Возвращает правый край."""
        fmw = p.fontMetrics()
        r = QRectF(x, y, fmw.horizontalAdvance(text) + 12, fmw.height() + 4)
        path = QPainterPath()
        path.addRoundedRect(r, 4, 4)
        p.fillPath(path, QColor(20, 23, 26, 215))
        p.setPen(QColor(color))
        p.drawText(r, Qt.AlignCenter, text)
        return r.right()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(PANEL))
        (cy, ch), (ay, ah), (by, bh), (sy, sh) = self._lanes()
        if self.dur <= 0:
            p.setPen(QColor("#7d868e"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Скиньте эталон и дубли — здесь появятся волны «до» и «после»")
            return
        p.setPen(QColor("#6b737a"))
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        span = self.v1 - self.v0
        step = next(s for s in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60) if span / s < 14)
        t = np.floor(self.v0 / step) * step
        while t <= self.v1:
            x = self._x(t)
            p.drawLine(int(x), self.RULER_H - 5, int(x), self.RULER_H)
            p.drawText(int(x) + 3, 12, fmt_t(t) if step >= 1 else f"{t:.2f}")
            t += step
        if self.loop:
            x0, x1 = self._x(self.loop[0]), self._x(self.loop[1])
            p.fillRect(QRectF(x0, 0, x1 - x0, h), QColor(63, 182, 168, 38))
            p.setPen(QPen(QColor(ACCENT), 1))
            p.drawLine(int(x0), 0, int(x0), h)
            p.drawLine(int(x1), 0, int(x1), h)
        if self.comp_h:
            for i, (a, b) in enumerate(self.tiles):
                x0, x1 = self._x(a), self._x(b)
                if x1 < 0 or x0 > w:
                    continue
                lab, sc, note, _r = self.comp_blocks[i] if i < len(self.comp_blocks) else ("", None, "", False)
                col = score_color(sc)
                col.setAlpha(210 if i == self.active_phrase else (170 if i == self._hover else 120))
                r = QRectF(x0 + 1, cy + 2, max(2.0, x1 - x0 - 2), ch - 4)
                path = QPainterPath()
                path.addRoundedRect(r, 6, 6)
                p.fillPath(path, col)
                if i == self.active_phrase:
                    p.setPen(QPen(QColor("#ffffff"), 1.5))
                    p.drawPath(path)
                if r.width() > 34:
                    p.setPen(QColor("#101314"))
                    f.setBold(True)
                    p.setFont(f)
                    txt = lab if sc is None else f"{lab}  {sc:.0f}"
                    p.drawText(r.adjusted(6, 2, -4, -ch / 2), Qt.AlignLeft | Qt.AlignVCenter, txt)
                    f.setBold(False)
                    p.setFont(f)
                    if note and r.width() > 90:
                        p.drawText(r.adjusted(6, ch / 2 - 4, -4, -2), Qt.AlignLeft | Qt.AlignVCenter, note)
        p.fillRect(QRectF(0, ay, w, ah), QColor(255, 255, 255, 6))
        p.fillRect(QRectF(0, by, w, bh), QColor(255, 255, 255, 4))
        for a, b in self.problems:
            x0, x1 = self._x(a), self._x(b)
            if x1 < 0 or x0 > w:
                continue
            p.fillRect(QRectF(x0, ay, max(2.0, x1 - x0), ah), QColor(224, 96, 90, 55))
            p.fillRect(QRectF(x0, ay, max(2.0, x1 - x0), 3), QColor(224, 96, 90, 220))
        self._wave(p, self.ref_env, ay, ah, REF_COL, 200)
        self._wave(p, self.after_env, ay, ah, ACCENT, 150)
        self._wave(p, self.ref_env, by, bh, REF_COL, 90)
        self._wave(p, self.before_env, by, bh, BEFORE_COL, 170)
        self._stretch_lane(p, sy, sh)
        x = self._tag(p, 6, ay + 3, "■ Эталон", REF_COL)
        self._tag(p, x + 4, ay + 3, f"■ {self.after_label}", ACCENT)
        self._tag(p, 6, by + 3, f"■ {self.before_label}", "#aab2b9")
        if self.comp_h:
            p.setPen(QPen(QColor(255, 255, 255, 30), 1, Qt.DashLine))
            for a, _b in self.tiles[1:]:
                x = self._x(a)
                p.drawLine(int(x), cy, int(x), h)
        x = self._x(self.playhead)
        p.setPen(QPen(QColor("#ffffff"), 1.5))
        p.drawLine(int(x), 0, int(x), h)

    def mousePressEvent(self, e):
        x, y = e.position().x(), e.position().y()
        (cy, ch), *_ = self._lanes()
        t = self._t(x)
        if e.button() == Qt.RightButton:
            i = self.phrase_at(t)
            if i >= 0 and self.comp_h:
                self.choiceMenu.emit(i, e.globalPosition().toPoint())
            return
        if e.button() != Qt.LeftButton:
            return
        if self.comp_h and cy <= y < cy + ch:
            i = self.phrase_at(t)
            if i >= 0:
                self.phraseClicked.emit(i)
            return
        self._drag = (x, t)

    def mouseMoveEvent(self, e):
        x, y = e.position().x(), e.position().y()
        (cy, ch), *_ = self._lanes()
        t = self._t(x)
        hv = self.phrase_at(t) if self.comp_h and cy <= y < cy + ch else -1
        if hv != self._hover:
            self._hover = hv
            if 0 <= hv < len(self.comp_blocks):
                lab, sc, note, _r = self.comp_blocks[hv]
                self.setToolTip(f"{lab}" + (f" · оценка {sc:.0f}" if sc is not None else "")
                                + (f"\n{note}" if note else "")
                                + "\nКолёсико — другой дубль, клик — слушать фразу по кругу, "
                                  "правый клик — список")
            else:
                self.setToolTip("Клик — перейти, протянуть — петля. Колёсико — масштаб, "
                                "Shift+колёсико — прокрутка. Двойной клик — весь файл.")
            self.update()
        if self._drag is not None and abs(x - self._drag[0]) > 4:
            a, b = sorted((self._drag[1], t))
            self.loop = (max(0.0, a), min(self.dur, b))
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drag is None:
            return
        x = e.position().x()
        if abs(x - self._drag[0]) <= 4:
            self.seekRequested.emit(max(0.0, min(self.dur, self._t(x))))
        else:
            self.loopChanged.emit(self.loop)
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        (cy, ch), *_ = self._lanes()
        if not (self.comp_h and cy <= e.position().y() < cy + ch):
            self.zoom_all()

    def leaveEvent(self, _e):
        self._hover = -1
        self.update()

    def show_range(self, a, b):
        pad = max(0.5, 0.3 * (b - a))
        self.v0, self.v1 = max(0.0, a - pad), min(max(0.5, self.dur), b + pad)
        self.update()

    def wheelEvent(self, e):
        x, y = e.position().x(), e.position().y()
        (cy, ch), *_ = self._lanes()
        d = e.angleDelta()
        if self.comp_h and cy <= y < cy + ch and abs(d.y()) >= abs(d.x()):
            i = self.phrase_at(self._t(x))
            if i < 0:
                return
            self._wheel_acc += d.y()
            while abs(self._wheel_acc) >= 120:
                step = 1 if self._wheel_acc > 0 else -1
                self._wheel_acc -= 120 * step
                self.choiceWheel.emit(i, step)
            return
        span = self.v1 - self.v0
        if e.modifiers() & Qt.ShiftModifier or abs(d.x()) > abs(d.y()):
            dx = d.x() if abs(d.x()) > abs(d.y()) else d.y()
            sh = -dx / 120.0 * span * 0.1
            self.v0, self.v1 = self.v0 + sh, self.v1 + sh
        else:
            t = self._t(x)
            k = 0.85 ** (d.y() / 120.0)
            new = min(max(0.15, span * k), max(0.5, self.dur))
            r = (t - self.v0) / span if span else 0.5
            self.v0 = t - r * new
            self.v1 = self.v0 + new
        if self.v0 < 0:
            self.v1 -= self.v0
            self.v0 = 0.0
        if self.v1 > self.dur:
            self.v0 = max(0.0, self.v0 - (self.v1 - self.dur))
            self.v1 = self.dur
        self.update()


# ----------------------------------------------------------------------------
# Фоновая обработка
# ----------------------------------------------------------------------------
class Worker(QThread):
    progress = Signal(str, float)
    takeDone = Signal(str, object)
    takeFailed = Signal(str, str)
    compReady = Signal(object)
    finishedAll = Signal()

    def __init__(self, session, guide, todo, all_dubs, settings, results, comp_on, out_root):
        super().__init__()
        self.session, self.guide, self.todo, self.all_dubs = session, guide, todo, all_dubs
        self.s, self.results, self.comp_on, self.out_root = settings, dict(results), comp_on, out_root
        self.stop = False

    def run(self):
        P = pipeline()
        try:
            total = max(1, len(self.todo))
            for i, path in enumerate(self.todo):
                if self.stop:
                    break
                name = os.path.basename(path)

                def prog(msg, f, i=i, name=name):
                    self.progress.emit(f"{i + 1}/{total} · {name}: {msg}", (i + f) / total * 0.95)
                try:
                    r = self.session.align(self.guide, path, self.s, prog, need_kind=self.comp_on)
                    P.export(r, self.out_root)
                    self.results[path] = r
                    self.takeDone.emit(path, r)
                except Exception as exc:  # noqa: BLE001
                    log_error(traceback.format_exc())
                    self.takeFailed.emit(path, str(exc))
            res = {p: r for p, r in self.results.items() if p in self.all_dubs}
            if not self.stop and self.comp_on and len(res) >= 2:
                self.progress.emit("Оцениваю фразы и собираю комп…", 0.96)
                gv = guide_view(self.session, self.guide, self.s, next(iter(res.values())))
                vc = comp_mod()
                tiles, sung = vc.phrases(gv)
                scores = vc.score_all(gv, res, tiles, sung)
                self.compReady.emit({"guide": self.guide, "tiles": tiles, "sung": sung,
                                     "scores": scores, "gv": gv})
        except Exception as exc:  # noqa: BLE001
            log_error(traceback.format_exc())
            self.takeFailed.emit("", str(exc))
        self.finishedAll.emit()


def guide_view(session, guide, s, res):
    G = session.track(guide, s.start, s.end, s.silence_db)
    return comp_mod().GuideView(fm=G.an.fm, n=G.an.n, loud=G.an.loud, db=G.an.db, sr=res.sr,
                                data=res.guide_y)


# ----------------------------------------------------------------------------
# Окно
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1360, 880)
        self.cfg = QSettings("MaxSync", "MaxSync")
        self.guide: str | None = None
        self.dubs: list[str] = []
        self.results: dict = {}
        self.status: dict = {}
        self.session = None
        self.worker = None
        self._old = []
        self.comp = None
        self.comp_audio = None
        self.comp_y = None
        self.audio_cache: dict = {}
        self.view_path = None
        self.player = Player()
        self._dirty_settings = False
        self._loading = False
        self._applying_preset = False
        self._pending_choices = None
        self.anchors_path = ""

        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(14, 12, 14, 10)
        lay.setSpacing(8)

        # верх: две зоны и кнопки
        top = QHBoxLayout()
        self.drop_g = DropZone("1 · ЭТАЛОН — вокал оригинала (стем Suno / сепаратор)",
                               "перетащите файл или кликните", REF_COL)
        self.drop_g.filesDropped.connect(self.add_guide_files)
        self.drop_g.clicked.connect(self.pick_guide)
        self.drop_d = DropZone("2 · ДУБЛИ — что подогнать (a cappella / каверы, можно несколько)",
                               "перетащите файлы или кликните", ACCENT)
        self.drop_d.filesDropped.connect(self.add_dub_files)
        self.drop_d.clicked.connect(self.pick_dubs)
        top.addWidget(self.drop_g, 1)
        top.addWidget(self.drop_d, 1)
        self.btn_go = QPushButton("▶  Выровнять")
        self.btn_go.setObjectName("go")
        self.btn_go.clicked.connect(self.run_all)
        top.addWidget(self.btn_go)
        self.btn_set = QToolButton()
        self.btn_set.setText("⚙ Настройки")
        self.btn_set.setCheckable(True)
        self.btn_set.toggled.connect(lambda on: self.set_panel.setVisible(on))
        top.addWidget(self.btn_set)
        self.btn_more = QToolButton()
        self.btn_more.setText("⋯")
        self.btn_more.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self)
        m.addAction("Открыть папку output", self.open_output)
        m.addAction("Открыть report.txt выбранного дубля", lambda: self.open_file("report_txt"))
        m.addAction("Открыть timemap.csv (можно править и подать как опорные точки)",
                    lambda: self.open_file("timemap"))
        m.addSeparator()
        m.addAction("Очистить всё", self.clear_all)
        self.btn_more.setMenu(m)
        top.addWidget(self.btn_more)
        lay.addLayout(top)

        self._build_settings(lay)

        # середина: список + отчёт | таймлайн
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        cap = QLabel("ДОРОЖКИ")
        cap.setObjectName("cap")
        ll.addWidget(cap)
        self.table = TrackTable(self.drag_selected)
        self.table.filesDropped.connect(self.add_dub_files)
        self.table.starClicked.connect(self.star_row)
        self.table.itemSelectionChanged.connect(self.on_select)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.table_menu)
        ll.addWidget(self.table, 2)
        legend = QLabel("★ — эталон (клик по звёздочке, чтобы сменить). Строку готового дубля "
                        "можно перетащить в Cubase.")
        legend.setObjectName("hint")
        legend.setWordWrap(True)
        ll.addWidget(legend)
        cap = QLabel("ОТЧЁТ")
        cap.setObjectName("cap")
        ll.addWidget(cap)
        self.report = QLabel("Здесь появятся сдвиг, темп, растяжение и качество до/после.")
        self.report.setObjectName("report")
        self.report.setWordWrap(True)
        self.report.setTextFormat(Qt.RichText)
        self.report.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        ll.addWidget(self.report)
        self.warn_list = QListWidget()
        self.warn_list.setObjectName("warns")
        self.warn_list.setWordWrap(True)
        self.warn_list.setToolTip("Клик — показать участок и слушать его по кругу")
        self.warn_list.itemClicked.connect(self.warning_clicked)
        ll.addWidget(self.warn_list, 1)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.card = ResultCard("РЕЗУЛЬТАТ · dub_aligned.wav", self.result_paths)
        self.card.btn_play.clicked.connect(self.play_result)
        self.card.btn_export.clicked.connect(self.export_wav)
        self.card.btn_folder.clicked.connect(self.show_result_folder)
        self.card.btn_plot.clicked.connect(lambda: self.open_file("alignment_png"))
        rl.addWidget(self.card)
        self.comp_card = ResultCard("КОМП · собран из лучших фраз", self.export_comp_paths)
        self.comp_card.btn_play.clicked.connect(self.play_final)
        self.comp_card.btn_export.clicked.connect(self.export_comp_as)
        self.comp_card.btn_folder.clicked.connect(self.show_comp_file)
        self.comp_card.btn_plot.setText("↺ Лучшие")
        self.comp_card.btn_plot.setToolTip("Отменить ваши замены: в каждой фразе снова лучший вариант")
        self.comp_card.btn_plot.clicked.connect(self.reset_choices)
        rl.addWidget(self.comp_card)
        self.tl = Timeline()
        self.tl.seekRequested.connect(self.seek)
        self.tl.loopChanged.connect(self.on_loop)
        self.tl.choiceWheel.connect(self.wheel_choice)
        self.tl.choiceMenu.connect(self.choice_menu)
        self.tl.phraseClicked.connect(self.loop_phrase)
        rl.addWidget(self.tl, 1)

        tr = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("play")
        self.btn_play.clicked.connect(self.toggle_play)
        tr.addWidget(self.btn_play)
        self.src_group = QButtonGroup(self)
        self.src_btns = {}
        for i, (key, text) in enumerate((("ref", "1 Эталон"), ("before", "2 Dub до"),
                                         ("result", "3 Результат"), ("comp", "4 Комп"))):
            bt = QPushButton(text)
            bt.setCheckable(True)
            bt.setObjectName("seg")
            self.src_group.addButton(bt, i)
            bt.clicked.connect(lambda _=False, k=key: self.set_source(k))
            tr.addWidget(bt)
            self.src_btns[key] = bt
        self.src_btns["result"].setChecked(True)
        self.player.current = "result"
        self.chk_under = QCheckBox("+ эталон (E)")
        self.chk_under.setToolTip("Подмешать эталон тише, чтобы слышать совпадение")
        self.chk_under.toggled.connect(lambda on: setattr(self.player, "ref_under", on))
        tr.addWidget(self.chk_under)
        self.chk_lr = QCheckBox("L/R (S)")
        self.chk_lr.setToolTip("Стерео как preview_stereo.wav: эталон в левом канале, выбранный источник — в правом")
        self.chk_lr.toggled.connect(lambda on: setattr(self.player, "split_lr", on))
        tr.addWidget(self.chk_lr)
        self.chk_loop = QCheckBox("Петля (L)")
        self.chk_loop.toggled.connect(self.on_loop_toggle)
        tr.addWidget(self.chk_loop)
        tr.addStretch(1)
        self.time_lbl = QLabel("0:00.0")
        self.time_lbl.setObjectName("time")
        tr.addWidget(self.time_lbl)
        rl.addLayout(tr)
        split.addWidget(right)
        split.setSizes([540, 820])
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)
        self.split = split
        QTimer.singleShot(0, lambda: self.split.setSizes([540, max(600, self.width() - 570)]))

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFormat("Скиньте эталон и дубли — дальше всё само.")
        self.bar.setTextVisible(True)
        lay.addWidget(self.bar)
        keys = QLabel("Пробел — играть/пауза · 1 эталон · 2 dub до · 3 результат · 4 комп · "
                      "↑/↓ — вариант фразы (комп) · ←/→ — соседняя фраза · L — петля · "
                      "E — эталон тише · S — стерео L/R")
        keys.setObjectName("hint")
        lay.addWidget(keys)

        for key, fn in ((Qt.Key_Space, self.toggle_play), (Qt.Key_1, lambda: self.set_source("ref")),
                        (Qt.Key_2, lambda: self.set_source("before")),
                        (Qt.Key_3, lambda: self.set_source("result")),
                        (Qt.Key_4, lambda: self.set_source("comp")),
                        (Qt.Key_Up, lambda: self.step_choice(+1)),
                        (Qt.Key_Down, lambda: self.step_choice(-1)),
                        (Qt.Key_Left, lambda: self.step_phrase(-1)),
                        (Qt.Key_Right, lambda: self.step_phrase(+1)),
                        (Qt.Key_L, lambda: self.chk_loop.toggle()),
                        (Qt.Key_E, lambda: self.chk_under.toggle()),
                        (Qt.Key_S, lambda: self.chk_lr.toggle())):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(fn)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)
        self.autorun = QTimer(self)
        self.autorun.setSingleShot(True)
        self.autorun.timeout.connect(self.run_all)
        self.xfade_timer = QTimer(self)
        self.xfade_timer.setSingleShot(True)
        self.xfade_timer.timeout.connect(lambda: (self.rebuild_comp_audio(), self.save_settings()))

        self.load_settings()
        self.connect_settings()
        self.update_comp_visibility()
        QTimer.singleShot(0, self.restore)

    # ------------------------------------------------------------------ панель настроек
    def _build_settings(self, lay):
        self.set_panel = QFrame()
        self.set_panel.setObjectName("panel")
        g = QGridLayout(self.set_panel)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(4, 1)
        g.setHorizontalSpacing(10)
        d = Settings()
        g.addWidget(QLabel("Пресет"), 0, 0)
        self.cb_preset = QComboBox()
        self.cb_preset.addItem("— свои настройки —")
        for name, desc, _v in PRESETS:
            self.cb_preset.addItem(name)
            self.cb_preset.setItemData(self.cb_preset.count() - 1, desc, Qt.ToolTipRole)
        g.addWidget(self.cb_preset, 0, 1, 1, 2)
        g.addWidget(QLabel("Режим"), 0, 3)
        mode_row = QHBoxLayout()
        self.cb_mode = QComboBox()
        for m_ in MODES:
            self.cb_mode.addItem(MODE_NAMES[m_], m_)
        self.cb_mode.setToolTip("Авто — сдвиг, темп и каждый слог по эталону.\nТолько сдвиг + темп — "
                                "для равномерного дрейфа.\nПо опорным точкам — CSV с парами "
                                "«время в эталоне; время в dub» для проблемных фраз.")
        mode_row.addWidget(self.cb_mode, 1)
        self.btn_anchors = QPushButton("Опорные точки (CSV)…")
        self.btn_anchors.clicked.connect(self.pick_anchors)
        mode_row.addWidget(self.btn_anchors)
        g.addLayout(mode_row, 0, 4, 1, 2)

        self.sl_tight = Slider(g, 1, 0, "Tightness", 0, 100, int(d.tightness),
                               lambda v: f"{v} % · " + ("каждый слог" if v >= 95 else
                                                        "почти точно" if v >= 75 else
                                                        "мягко" if v >= 30 else "сдвиг+темп"),
                               "Смесь глобальной карты (сдвиг + постоянный темп) и точной карты DTW.\n"
                               "100 % — максимально точно; 90 % — остаётся 10 % каждого отклонения.")
        g.addWidget(QLabel("Растяжение, ×"), 1, 3)
        sr_row = QHBoxLayout()
        self.sp_smin = QDoubleSpinBox()
        self.sp_smin.setRange(0.3, 0.99)
        self.sp_smin.setSingleStep(0.01)
        self.sp_smax = QDoubleSpinBox()
        self.sp_smax.setRange(1.01, 3.0)
        self.sp_smax.setSingleStep(0.05)
        for sp, tip in ((self.sp_smin, "мин. (сжатие)"), (self.sp_smax, "макс. (растяжение)")):
            sp.setToolTip(f"Предел локального коэффициента растяжения: {tip}. По умолчанию 0,67–1,5.")
            sr_row.addWidget(sp)
        sr_row.addStretch(1)
        g.addLayout(sr_row, 1, 4, 1, 2)
        self.sl_smooth = Slider(g, 2, 0, "Сглаживание", 0, 300, int(d.smooth_ms), lambda v: f"{v} мс",
                                "Сглаживание карты DTW: больше — спокойнее, меньше — точнее по слогам.")
        self.sl_rate = Slider(g, 2, 3, "Плавность растяжения", 20, 500, int(d.rate_smooth_ms),
                              lambda v: f"{v} мс",
                              "За сколько мс коэффициент растяжения может смениться. Больше — меньше "
                              "«плавания» звука, полезно перед RVC.")
        self.sl_key = Slider(g, 3, 0, "Опорные точки, шаг", 10, 300, int(d.keyframe_ms),
                             lambda v: f"{v} мс + онсеты",
                             "Карта прореживается до опорных точек: сетка с этим шагом в звучащих "
                             "местах, плюс атаки слогов эталона и края пауз.")
        g.addWidget(QLabel("Фрагмент, с"), 3, 3)
        fr_row = QHBoxLayout()
        self.sp_start = QDoubleSpinBox()
        self.sp_end = QDoubleSpinBox()
        for sp, tip in ((self.sp_start, "начало"), (self.sp_end, "конец")):
            sp.setRange(0.0, 7200.0)
            sp.setDecimals(1)
            sp.setSpecialValueText("весь")
            sp.setToolTip(f"Обработать только фрагмент эталона ({tip}), например один припев для "
                          "теста. 0 — весь файл. Результат — длиной с фрагмент.")
            fr_row.addWidget(sp)
        fr_row.addStretch(1)
        g.addLayout(fr_row, 3, 4, 1, 2)
        self.sl_wc = Slider(g, 4, 0, "Вес: chroma (ноты)", 0, 200, int(d.w_chroma * 100), lambda v: f"{v} %",
                            "Насколько учитывать ноты. 0 — если тональность другая.")
        self.sl_wm = Slider(g, 4, 3, "Вес: MFCC (тембр)", 0, 200, int(d.w_mfcc * 100), lambda v: f"{v} %",
                            "Насколько учитывать фонемы/тембр.")
        self.sl_wo = Slider(g, 5, 0, "Вес: онсеты (атаки)", 0, 200, int(d.w_onset * 100), lambda v: f"{v} %",
                            "Насколько учитывать атаки слогов.")
        self.sl_we = Slider(g, 5, 3, "Вес: энергия", 0, 200, int(d.w_energy * 100), lambda v: f"{v} %",
                            "Насколько учитывать громкость.")
        g.addWidget(QLabel("Выход"), 6, 0)
        out_row = QHBoxLayout()
        self.cb_fmt = QComboBox()
        self.cb_fmt.addItem("WAV 24 бит", "PCM_24")
        self.cb_fmt.addItem("WAV float32", "FLOAT")
        self.cb_sr = QComboBox()
        for v, t in SR_CHOICES:
            self.cb_sr.addItem(t, v)
        out_row.addWidget(self.cb_fmt)
        out_row.addWidget(self.cb_sr)
        out_row.addStretch(1)
        g.addLayout(out_row, 6, 1, 1, 2)
        g.addWidget(QLabel("Движок"), 6, 3)
        self.cb_engine = QComboBox()
        for v, t in ENGINES:
            self.cb_engine.addItem(t, v)
        g.addWidget(self.cb_engine, 6, 4, 1, 2)
        self.sl_pitch = Slider(g, 7, 0, "Питч к эталону", 0, 100, 0,
                               lambda v: "не трогать" if v == 0 else f"{v} %",
                               "Подтянуть ноты к эталону (из VoxMatch). Для RVC обычно не нужно — 0.")
        self.sl_xfade = Slider(g, 7, 3, "Кроссфейд компа", 10, 200, 60, lambda v: f"{v} мс",
                               "Плавность перехода между фразами из разных дублей в компе.")
        hb = QHBoxLayout()
        self.chk_comp = QCheckBox("Комп из лучших фраз (если дублей ≥ 2)")
        self.chk_form = QCheckBox("Сохранять тембр при питче")
        self.chk_harm = QCheckBox("Бэки в гармонию — ноты не трогать")
        self.chk_auto = QCheckBox("Сразу обрабатывать, как только скинули")
        for wdg in (self.chk_comp, self.chk_form, self.chk_harm, self.chk_auto):
            hb.addWidget(wdg)
        hb.addStretch(1)
        b = QPushButton("Стандартные")
        b.clicked.connect(self.reset_defaults)
        hb.addWidget(b)
        self.btn_apply = QPushButton("Применить ко всем")
        self.btn_apply.clicked.connect(lambda: self.run_all(force=True))
        hb.addWidget(self.btn_apply)
        g.addLayout(hb, 8, 0, 1, 6)
        self.engine_lbl = QLabel("")
        self.engine_lbl.setObjectName("hint")
        g.addWidget(self.engine_lbl, 9, 0, 1, 6)
        self.set_panel.setVisible(False)
        lay.addWidget(self.set_panel)
        self.update_engine_label()

    def update_engine_label(self):
        from . import warp
        av = warp.available_engines()
        txt = "Доступные движки растяжения: " + ", ".join(av)
        if "cli" not in av and "pylibrb" not in av:
            txt += " — ВНИМАНИЕ: Rubber Band не найден, качество хуже (см. README)."
        txt += f"  ·  Результаты: {pipeline().default_out_root()}"
        self.engine_lbl.setText(txt)

    def _setting_widgets(self):
        sliders = [self.sl_tight, self.sl_smooth, self.sl_rate, self.sl_key, self.sl_wc, self.sl_wm,
                   self.sl_wo, self.sl_we, self.sl_pitch]
        spins = [self.sp_smin, self.sp_smax, self.sp_start, self.sp_end]
        combos = [self.cb_mode, self.cb_fmt, self.cb_sr, self.cb_engine]
        checks = [self.chk_form, self.chk_harm]
        return sliders, spins, combos, checks

    def connect_settings(self):
        sliders, spins, combos, checks = self._setting_widgets()
        for w in sliders:
            w.s.valueChanged.connect(self.settings_changed)
        for w in spins:
            w.valueChanged.connect(self.settings_changed)
        for w in combos:
            w.currentIndexChanged.connect(self.settings_changed)
        for w in checks:
            w.toggled.connect(self.settings_changed)
        self.cb_preset.activated.connect(self.apply_preset)
        self.chk_comp.toggled.connect(self.comp_toggled)
        self.chk_auto.toggled.connect(lambda _v: self.save_settings())
        self.sl_xfade.s.valueChanged.connect(lambda _v: self.xfade_timer.start(250))

    def settings(self) -> Settings:
        st, en = self.sp_start.value(), self.sp_end.value()
        return Settings(
            mode=self.cb_mode.currentData(), tightness=float(self.sl_tight.value()),
            stretch_min=round(self.sp_smin.value(), 3), stretch_max=round(self.sp_smax.value(), 3),
            smooth_ms=float(self.sl_smooth.value()), rate_smooth_ms=float(self.sl_rate.value()),
            keyframe_ms=float(self.sl_key.value()),
            w_chroma=self.sl_wc.value() / 100.0, w_mfcc=self.sl_wm.value() / 100.0,
            w_onset=self.sl_wo.value() / 100.0, w_energy=self.sl_we.value() / 100.0,
            start=st if st > 0 else None, end=en if en > 0 else None,
            anchors_path=self.anchors_path if self.cb_mode.currentData() == "anchors" else "",
            out_sr=self.cb_sr.currentData(), out_format=self.cb_fmt.currentData(),
            engine=self.cb_engine.currentData(), pitch=self.sl_pitch.value(),
            formants=self.chk_form.isChecked(), harmony_auto=self.chk_harm.isChecked())

    def apply_values(self, s: Settings):
        self._loading = True
        try:
            self.cb_mode.setCurrentIndex(max(0, self.cb_mode.findData(s.mode)))
            self.sl_tight.setValue(s.tightness)
            self.sp_smin.setValue(s.stretch_min)
            self.sp_smax.setValue(s.stretch_max)
            self.sl_smooth.setValue(s.smooth_ms)
            self.sl_rate.setValue(s.rate_smooth_ms)
            self.sl_key.setValue(s.keyframe_ms)
            self.sl_wc.setValue(s.w_chroma * 100)
            self.sl_wm.setValue(s.w_mfcc * 100)
            self.sl_wo.setValue(s.w_onset * 100)
            self.sl_we.setValue(s.w_energy * 100)
            self.sp_start.setValue(s.start or 0.0)
            self.sp_end.setValue(s.end or 0.0)
            self.cb_fmt.setCurrentIndex(max(0, self.cb_fmt.findData(s.out_format)))
            self.cb_sr.setCurrentIndex(max(0, self.cb_sr.findData(s.out_sr)))
            self.cb_engine.setCurrentIndex(max(0, self.cb_engine.findData(s.engine)))
            self.sl_pitch.setValue(s.pitch)
            self.chk_form.setChecked(s.formants)
            self.chk_harm.setChecked(s.harmony_auto)
        finally:
            self._loading = False

    def apply_preset(self, idx):
        if idx <= 0:
            return
        name, desc, vals = PRESETS[idx - 1]
        cur = self.settings()
        keep = {k: getattr(cur, k) for k in ("start", "end", "out_sr", "out_format", "engine",
                                             "pitch", "formants", "harmony_auto")}
        s = Settings.from_dict({**Settings().to_dict(), **vals, **keep})
        self.apply_values(s)
        self._applying_preset = True
        try:
            self.settings_changed()
        finally:
            self._applying_preset = False
        self.cb_preset.setCurrentIndex(idx)
        self.bar.setFormat(f"Пресет «{name}»: {desc}")

    def load_settings(self):
        try:
            d = json.loads(str(self.cfg.value("settings", "") or "{}"))
        except ValueError:
            d = {}
        self.apply_values(Settings.from_dict({**Settings().to_dict(), **d}))
        self.anchors_path = str(self.cfg.value("anchors", "") or "")
        self.update_anchor_btn()
        self.chk_comp.setChecked(str(self.cfg.value("comp", "false")).lower() == "true")
        self.chk_auto.setChecked(str(self.cfg.value("auto", "true")).lower() == "true")
        self.sl_xfade.setValue(int(self.cfg.value("xfade", 60)))
        self.cb_preset.setCurrentIndex(int(self.cfg.value("preset", 1)))

    def save_settings(self):
        c = self.cfg
        d = self.settings().to_dict()
        d.pop("anchors", None)
        c.setValue("settings", json.dumps(d, ensure_ascii=False))
        c.setValue("anchors", self.anchors_path)
        c.setValue("comp", self.chk_comp.isChecked())
        c.setValue("auto", self.chk_auto.isChecked())
        c.setValue("xfade", self.sl_xfade.value())
        c.setValue("preset", self.cb_preset.currentIndex())
        sess = {"guide": self.guide, "dubs": self.dubs,
                "choices": self._choices_by_name() if self.comp else None}
        c.setValue("session", json.dumps(sess, ensure_ascii=False))

    def reset_defaults(self):
        self.apply_values(Settings())
        self._applying_preset = True
        try:
            self.settings_changed()
        finally:
            self._applying_preset = False
        self.cb_preset.setCurrentIndex(1)

    def settings_changed(self, *_):
        if self._loading:
            return
        if not self._applying_preset:
            self.cb_preset.setCurrentIndex(0)
        if self.results:
            self._dirty_settings = True
            self.btn_apply.setText("Применить ко всем •")
        self.save_settings()

    def pick_anchors(self):
        start = os.path.dirname(self.anchors_path) if self.anchors_path else APP_DIR
        p, _ = QFileDialog.getOpenFileName(self, "Опорные точки", start, "CSV (*.csv *.txt)")
        if not p:
            return
        try:
            pts = pipeline().read_anchors(p)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"Не удалось прочитать CSV: {exc}")
            return
        if not pts:
            QMessageBox.warning(self, APP_NAME, "В файле нет пар «время в эталоне; время в dub».")
            return
        self.anchors_path = p
        self.update_anchor_btn()
        self.cb_mode.setCurrentIndex(self.cb_mode.findData("anchors"))
        self.bar.setFormat(f"Опорных точек: {len(pts)} из {os.path.basename(p)}")
        self.settings_changed()

    def update_anchor_btn(self):
        self.btn_anchors.setText(f"Точки: {os.path.basename(self.anchors_path)}" if self.anchors_path
                                 else "Опорные точки (CSV)…")
        self.btn_anchors.setToolTip(self.anchors_path or "CSV: время в эталоне; время в dub")

    def comp_toggled(self, _on):
        self.update_comp_visibility()
        self.save_settings()
        if self.chk_comp.isChecked() and len(self.results) >= 2 and not self.comp:
            self.results.clear()
            for p in self.dubs:
                self.status[p] = "ждёт"
            self.refresh_table()
            if self.chk_auto.isChecked():
                self.autorun.start(300)

    def update_comp_visibility(self):
        on = self.chk_comp.isChecked()
        self.comp_card.setVisible(on)
        self.tl.set_comp_visible(on)
        self.src_btns["comp"].setVisible(on)
        self.sl_xfade.setEnabled(on)

    def restore(self):
        try:
            sess = json.loads(str(self.cfg.value("session", "") or "{}"))
        except ValueError:
            sess = {}
        g = sess.get("guide")
        dubs = [p for p in sess.get("dubs") or [] if os.path.isfile(p)]
        if g and os.path.isfile(g):
            self.set_guide(g, rerun=False)
        if dubs:
            self.add_dub_files(dubs, autorun=False)
        self._pending_choices = sess.get("choices")
        if self.guide or self.dubs:
            self.bar.setFormat("Прошлая сессия открыта. Нажмите «Выровнять».")

    # ------------------------------------------------------------------ дорожки
    def _start_dir(self):
        return inbox_dir() if os.path.isdir(inbox_dir()) else os.path.expanduser("~")

    def pick_guide(self):
        p, _ = QFileDialog.getOpenFileName(self, "Эталон", self._start_dir(), AUDIO_FILTER)
        if p:
            self.add_guide_files([p])

    def pick_dubs(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Дубли", self._start_dir(), AUDIO_FILTER)
        if ps:
            self.add_dub_files(ps)

    @staticmethod
    def _own_output(p):
        return os.path.basename(p) in ("dub_aligned.wav", "preview_stereo.wav", "preview_mix.wav") \
            or "КОМП" in os.path.basename(p)

    def add_guide_files(self, paths):
        """Первый файл — эталон, остальные (если бросили несколько) — дубли."""
        paths = [os.path.abspath(p) for p in paths if not self._own_output(p)]
        if not paths:
            return
        self.set_guide(paths[0])
        if len(paths) > 1:
            self.add_dub_files(paths[1:])

    def set_guide(self, path, rerun=True):
        path = os.path.abspath(path)
        if path == self.guide:
            return
        old = self.guide
        if path in self.dubs:
            self.dubs.remove(path)
        if old and old not in self.dubs and rerun:
            self.dubs.insert(0, old)
        self.guide = path
        self.results.clear()
        self.comp = None
        self.comp_audio = None
        self.audio_cache.clear()
        for p in self.dubs:
            self.status[p] = "ждёт"
        self.status[path] = "эталон"
        self.drop_g.set_hint(f"★ {os.path.basename(path)}")
        self.refresh_table()
        self.load_ref_view()
        self.save_settings()
        if rerun and self.dubs and self.chk_auto.isChecked():
            self.autorun.start(600)

    def add_dub_files(self, paths, autorun=True):
        new = 0
        for p in paths:
            p = os.path.abspath(p)
            if p in self.dubs or p == self.guide or self._own_output(p):
                continue
            self.dubs.append(p)
            self.status[p] = "ждёт"
            new += 1
        if not new:
            return
        self.drop_d.set_hint(f"дублей: {len(self.dubs)}")
        self.refresh_table()
        self.save_settings()
        if autorun and self.chk_auto.isChecked() and self.guide:
            self.bar.setFormat(f"Добавлено: {new}. Начну через секунду…")
            self.autorun.start(1200)
        elif not self.guide:
            self.bar.setFormat("Теперь скиньте эталон в левую зону.")

    @property
    def paths(self):
        return ([self.guide] if self.guide else []) + self.dubs

    def label(self, path):
        if path == self.guide or path == REF_KEY:
            return "Э"
        return f"Д{self.dubs.index(path) + 1}" if path in self.dubs else "?"

    def refresh_table(self):
        t = self.table
        sel = self.view_path
        t.blockSignals(True)
        t.setRowCount(len(self.paths))
        avg = self.take_avg_scores()
        for r, p in enumerate(self.paths):
            is_ref = p == self.guide
            res = self.results.get(p)
            m = res.metrics if res else {}
            oe = (m.get("onset_error_ms") or {}).get("after", {}) if res else {}
            st = self.status.get(p, "")
            items = [
                ("★" if is_ref else "☆", REF_COL if is_ref else "#5f676e"),
                (f"{self.label(p)}  {os.path.basename(p)}", None),
                (f"{m['offset_s']:+.2f} с" if res else "", None),
                (f"{m['stretch_min']:.2f}–{m['stretch_max']:.2f}" if res else "", None),
                (f"{oe['median']:.0f} мс" if res and oe.get("median") is not None else "", None),
                (st, None),
            ]
            for c, (txt, col) in enumerate(items):
                it = QTableWidgetItem(txt)
                if col:
                    it.setForeground(QColor(col))
                if c == 1:
                    tip = p + (f"\nтемп dub {m['tempo_percent']:+.2f} %" if res else "")
                    if p in avg:
                        tip += f"\nсредняя оценка фраз в компе: {avg[p]:.0f}"
                    if res:
                        tip += "\n" + "\n".join(w["text"] for w in res.warnings[:8])
                    it.setToolTip(tip)
                if c == 0:
                    it.setTextAlignment(Qt.AlignCenter)
                    it.setToolTip("Сделать эталоном")
                if c == 5 and res and res.warnings:
                    it.setForeground(QColor("#e3c25a"))
                t.setItem(r, c, it)
        t.blockSignals(False)
        if sel in self.paths:
            t.selectRow(self.paths.index(sel))
        elif self.paths and not t.selectedItems():
            t.selectRow(min(1, len(self.paths) - 1))

    def take_avg_scores(self):
        if not self.comp or not self.comp.get("scores"):
            return {}
        out = {}
        w = self.comp.get("weights")
        for k, rows in self.comp["scores"].items():
            if k == REF_KEY:
                continue
            vals = np.array([v for v, _n in rows])
            ww = w if w is not None and len(w) == len(vals) else np.ones(len(vals))
            out[k] = float(np.sum(vals * ww) / max(1e-9, np.sum(ww)))
        return out

    def star_row(self, row):
        if 0 <= row < len(self.paths):
            self.set_guide(self.paths[row])

    def table_menu(self, pt):
        r = self.table.rowAt(pt.y())
        if r < 0:
            return
        p = self.paths[r]
        m = QMenu(self)
        m.addAction("★ Сделать эталоном", lambda: self.set_guide(p))
        res = self.results.get(p)
        if res and res.out_dir:
            m.addAction("Открыть папку результата", lambda: self.reveal(res.out_dir))
            m.addAction("Открыть отчёт", lambda: self.open_path(res.files.get("report_txt")))
            m.addAction("Открыть график", lambda: self.open_path(res.files.get("alignment_png")))
        m.addAction("Показать исходник в Finder", lambda: self.reveal(p))
        m.addSeparator()
        m.addAction("Убрать из списка", lambda: self.remove_path(p))
        m.exec(self.table.viewport().mapToGlobal(pt))

    def remove_path(self, p):
        if self.worker and self.worker.isRunning():
            return
        if p == self.guide:
            self.guide = None
            self.results.clear()
            self.comp = None
            self.drop_g.set_hint("перетащите файл или кликните")
        else:
            self.dubs.remove(p)
            self.results.pop(p, None)
        self.status.pop(p, None)
        self.refresh_table()
        self.save_settings()

    def clear_all(self):
        if self.worker and self.worker.isRunning():
            return
        self.player.pause()
        self.guide = None
        self.dubs.clear()
        self.results.clear()
        self.status.clear()
        self.comp = None
        self.comp_audio = None
        self.audio_cache.clear()
        self.view_path = None
        self.drop_g.set_hint("перетащите файл или кликните")
        self.drop_d.set_hint("перетащите файлы или кликните")
        self.refresh_table()
        self.tl.dur = 0
        self.tl.set_comp([], [])
        self.update_card()
        self.update_report()
        self.player.set_sources({}, 48000)
        self.bar.setValue(0)
        self.bar.setFormat("Скиньте эталон и дубли — дальше всё само.")
        self.save_settings()

    def drag_selected(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        out = []
        for r in rows:
            p = self.paths[r]
            res = self.results.get(p)
            if res and os.path.isfile(res.out_path):
                out.append(res.out_path)
            elif p == self.guide:
                out.append(p)
        return out

    # ------------------------------------------------------------------ обработка
    def run_all(self, force=False):
        if self.worker and self.worker.isRunning():
            if force is True or self.sender() is self.btn_go:
                self.worker.stop = True
                self.bar.setFormat("Останавливаю после текущего дубля…")
            return
        if not self.guide or not self.dubs:
            self.bar.setFormat("Нужны эталон (левая зона) и хотя бы один дубль (правая зона).")
            return
        if force is True or self._dirty_settings:
            self.results.clear()
            self.comp = None
            self._dirty_settings = False
            self.btn_apply.setText("Применить ко всем")
        todo = [p for p in self.dubs if p not in self.results]
        if not todo and (self.comp or not self.chk_comp.isChecked() or len(self.dubs) < 2):
            self.bar.setFormat("Всё уже выровнено. Поменяли настройки — «Применить ко всем».")
            return
        self.player.pause()
        self.save_settings()
        if self.session is None:
            self.bar.setFormat("Загружаю движок…")
            QApplication.processEvents()
            self.session = pipeline().Session()
        for p in todo:
            self.status[p] = "в очереди"
        self.refresh_table()
        self.worker = Worker(self.session, self.guide, todo, list(self.dubs), self.settings(),
                             self.results, self.chk_comp.isChecked(), pipeline().default_out_root())
        self.worker.progress.connect(self.on_progress)
        self.worker.takeDone.connect(self.on_take_done)
        self.worker.takeFailed.connect(self.on_take_failed)
        self.worker.compReady.connect(self.on_comp_ready)
        self.worker.finishedAll.connect(self.on_finished)
        self.btn_go.setText("■  Стоп")
        self.worker.start()

    def on_progress(self, msg, f):
        self.bar.setValue(int(f * 1000))
        self.bar.setFormat(msg)

    def on_take_done(self, path, res):
        self.results[path] = res
        for k in [k for k in self.audio_cache if isinstance(k, tuple) and k[1] == path]:
            self.audio_cache.pop(k, None)
        nw = len(res.warnings)
        self.status[path] = f"✓ {res.seconds:.0f} с" + (f" ⚠{nw}" if nw else "")
        self.refresh_table()
        if self.view_path in (None, path, self.guide) or self.view_path not in self.results:
            self.view_path = path
            self.table.blockSignals(True)
            self.table.selectRow(self.paths.index(path))
            self.table.blockSignals(False)
            self.show_take(path)

    def on_take_failed(self, path, msg):
        if path:
            self.status[path] = "ошибка"
            self.refresh_table()
        QMessageBox.warning(self, APP_NAME, (os.path.basename(path) + ":\n" if path else "") + msg)

    def on_finished(self):
        self.btn_go.setText("▶  Выровнять")
        self.bar.setValue(1000)
        n = len(self.results)
        if n:
            nw = sum(len(r.warnings) for r in self.results.values())
            self.bar.setFormat(f"Готово: выровнено {n}. Файлы — в папке output"
                               + (f" · предупреждений: {nw} (см. отчёт слева)" if nw else "") + ".")
        self._old = [x for x in self._old if x.isRunning()] + [self.worker]

    # ------------------------------------------------------------------ комп
    def on_comp_ready(self, d):
        if d["guide"] != self.guide:
            return
        tiles, scores, sung = d["tiles"], d["scores"], d["sung"]
        cands = [p for p in self.dubs if p in self.results and p in scores
                 and self.results[p].kind == "унисон"]
        extra = [p for p in self.dubs if p in self.results and p in scores and p not in cands]
        if not cands:
            self.comp = None
            self.update_comp_view()
            self.bar.setFormat("Для компа нет дублей в унисон (бэки в гармонию не берутся).")
            return
        w = np.array([sum(max(0, min(b, sb) - max(a, sa)) for sa, sb in sung) for a, b in tiles],
                     dtype=float)
        choices = comp_mod().default_choices(scores, cands, len(tiles))
        pend = getattr(self, "_pending_choices", None)
        if pend and len(pend) == len(tiles):
            for i, name in enumerate(pend):
                k = next((p for p in cands if os.path.basename(p) == name), None)
                if k:
                    choices[i] = k
        self._pending_choices = None
        self.comp = {"tiles": tiles, "scores": scores, "choices": choices, "auto": list(choices),
                     "cands": cands + extra, "gv": d["gv"], "weights": w}
        self.rebuild_comp_audio()
        self.refresh_table()
        self.update_comp_view()
        self.save_settings()

    def _choices_by_name(self):
        return [os.path.basename(k) for k in self.comp["choices"]]

    def rebuild_comp_audio(self):
        if not self.comp:
            return
        audio = {k: self.results[k].y for k in set(self.comp["choices"]) if k in self.results}
        if not audio:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            y = comp_mod().build_comp(self.comp["gv"], audio, self.comp["tiles"],
                                      self.comp["choices"], fade_ms=float(self.sl_xfade.value()))
        finally:
            QApplication.restoreOverrideCursor()
        self.comp_y = y
        self.comp_audio = y.mean(axis=0).astype(np.float32)
        self.update_player_sources()
        try:
            self.export_comp_paths()
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())
        self.update_comp_card()

    def update_comp_view(self):
        if not self.comp:
            self.tl.set_comp([], [])
            self.update_comp_card()
            return
        fm = self.comp["gv"].fm / 1000.0
        tiles = [(a * fm, b * fm) for a, b in self.comp["tiles"]]
        blocks = []
        for i, k in enumerate(self.comp["choices"]):
            sc, note = self.comp["scores"][k][i]
            blocks.append((self.label(k), sc, note, False))
        self.tl.set_comp(tiles, blocks)

    def set_choice(self, i, key, play=True):
        if not self.comp or self.comp["choices"][i] == key:
            return
        self.comp["choices"][i] = key
        self.rebuild_comp_audio()
        self.update_comp_view()
        self.save_settings()
        sc, note = self.comp["scores"][key][i]
        self.bar.setFormat(f"Фраза {i + 1}: {self.label(key)} — {os.path.basename(key)} · оценка {sc:.0f}"
                           + (f" · {note}" if note else ""))
        if play:
            self.set_source("comp")
            self.loop_phrase(i, restart=True)

    def wheel_choice(self, i, step):
        if not self.comp:
            return
        order = comp_mod().ranked(self.comp["scores"], self.comp["cands"], i)
        cur = self.comp["choices"][i]
        j = order.index(cur) if cur in order else 0
        self.set_choice(i, order[max(0, min(len(order) - 1, j - step))])

    def choice_menu(self, i, gpos):
        if not self.comp:
            return
        m = QMenu(self)
        for k in comp_mod().ranked(self.comp["scores"], self.comp["cands"], i):
            sc, note = self.comp["scores"][k][i]
            a = QAction(f"{self.label(k)}  {sc:.0f}   {os.path.basename(k)}" + (f"  — {note}" if note else ""), m)
            a.setCheckable(True)
            a.setChecked(k == self.comp["choices"][i])
            a.triggered.connect(lambda _=False, k=k: self.set_choice(i, k))
            m.addAction(a)
        m.exec(gpos)

    def reset_choices(self):
        if self.comp:
            self.comp["choices"] = list(self.comp["auto"])
            self.rebuild_comp_audio()
            self.update_comp_view()
            self.save_settings()

    def current_phrase(self):
        return self.tl.phrase_at(self.player.seconds()) if self.comp else -1

    def step_choice(self, d):
        i = self.current_phrase()
        if i >= 0:
            self.wheel_choice(i, d)

    def step_phrase(self, d):
        if not self.comp:
            return
        i = self.current_phrase()
        self.loop_phrase(max(0, min(len(self.comp["tiles"]) - 1, (i if i >= 0 else 0) + d)))

    def loop_phrase(self, i, restart=True):
        if not self.comp:
            return
        a, b = self.tl.tiles[i]
        self.tl.loop = (a, b)
        self.tl.active_phrase = i
        self.on_loop((a, b))
        if not self.chk_loop.isChecked():
            self.chk_loop.setChecked(True)
        if restart:
            self.player.seek(a)
        self.player.play()

    def comp_path(self):
        base = os.path.splitext(os.path.basename(self.guide))[0]
        d = pipeline().default_out_root()
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{base} — КОМП.wav")

    def export_comp_paths(self):
        if not self.comp or self.comp_y is None:
            return []
        from . import io as mio
        p = self.comp_path()
        mio.write_audio(p, self.comp_y, self.comp["gv"].sr, self.cb_fmt.currentData())
        return [p]

    def export_comp_as(self):
        if not self.comp or self.comp_y is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Сохранить комп", self.comp_path(), "WAV (*.wav)")
        if p:
            from . import io as mio
            mio.write_audio(p, self.comp_y, self.comp["gv"].sr, self.cb_fmt.currentData())
            self.bar.setFormat(f"Комп сохранён: {p}")

    def show_comp_file(self):
        ps = self.export_comp_paths()
        if ps:
            self.reveal(ps[0])

    def update_comp_card(self):
        fc = self.comp_card
        if not self.comp or self.comp_audio is None:
            fc.info.setText("Нужно минимум два дубля в унисон. Колёсико на фразе — другой дубль.")
            fc.wave.set_data(None, [])
            return
        from collections import Counter
        cnt = Counter(self.label(k) for k in self.comp["choices"])
        parts = ", ".join(f"{k} ×{v}" for k, v in sorted(cnt.items()))
        n = len(self.comp["tiles"])
        changed = sum(1 for a, b in zip(self.comp["choices"], self.comp["auto"]) if a != b)
        sc = [self.comp["scores"][k][i][0] for i, k in enumerate(self.comp["choices"])]
        fc.info.setText(f"{n} фраз: {parts} · средняя оценка {float(np.mean(sc)):.0f}"
                        + (f" · ваших замен: {changed}" if changed else "")
                        + f"\n{os.path.basename(self.comp_path())}")
        total = self.comp["tiles"][-1][1] if self.comp["tiles"] else 1
        blocks = [(a / total, b / total, sc[i], self.label(self.comp["choices"][i]))
                  for i, (a, b) in enumerate(self.comp["tiles"])]
        fc.wave.set_data(env_peaks(self.comp_audio, max(1, len(self.comp_audio) // 4000)), blocks)

    def play_final(self):
        if self.comp_audio is None:
            return
        self.chk_loop.setChecked(False)
        self.set_source("comp")
        self.player.seek(0)
        self.player.play()

    # ------------------------------------------------------------------ результат
    def current_result(self):
        return self.results.get(self.view_path)

    def result_paths(self):
        r = self.current_result()
        return [r.out_path] if r and os.path.isfile(r.out_path) else []

    def play_result(self):
        if not self.current_result():
            return
        self.chk_loop.setChecked(False)
        self.set_source("result")
        self.player.seek(0)
        self.player.play()

    def export_wav(self):
        r = self.current_result()
        if not r:
            self.bar.setFormat("Сначала выровняйте дубль.")
            return
        stem = os.path.splitext(os.path.basename(r.dub_path))[0]
        start = os.path.join(os.path.dirname(r.dub_path), f"{stem}_синхрон.wav")
        p, _ = QFileDialog.getSaveFileName(self, "Экспорт выровненного dub", start, "WAV (*.wav)")
        if not p:
            return
        if not p.lower().endswith(".wav"):
            p += ".wav"
        shutil.copyfile(r.out_path, p)
        self.bar.setFormat(f"Сохранено: {p}")

    def show_result_folder(self):
        r = self.current_result()
        if r and r.out_path:
            self.reveal(r.out_path)

    def open_file(self, key):
        r = self.current_result()
        if r and r.files.get(key):
            self.open_path(r.files[key])
        else:
            self.bar.setFormat("Нет файла — выберите выровненный дубль.")

    def open_path(self, p):
        if p and os.path.exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def reveal(self, p):
        import subprocess
        try:
            subprocess.Popen(["open", "-R", p])
        except Exception:  # noqa: BLE001
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(p)))

    def open_output(self):
        d = pipeline().default_out_root()
        os.makedirs(d, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def update_card(self):
        r = self.current_result()
        c = self.card
        if not r:
            c.title.setText("РЕЗУЛЬТАТ · dub_aligned.wav")
            c.info.setText("Скиньте эталон и дубль — результат появится здесь.")
            c.wave.set_data(None)
            return
        m = r.metrics
        c.title.setText(f"РЕЗУЛЬТАТ · {self.label(r.dub_path)} {os.path.basename(r.dub_path)}")
        c.info.setText(f"{os.path.basename(r.out_dir)}/dub_aligned.wav · "
                       f"{m['out_sr']} Гц · {'float32' if r.settings.out_format == 'FLOAT' else '24 бит'}"
                       f" · {m['channels']} кан. · длина = эталону · движок {r.engine}")
        mono = r.y.mean(axis=0)
        c.wave.set_data(env_peaks(mono, max(1, len(mono) // 4000)))

    def update_report(self):
        r = self.current_result()
        self.warn_list.clear()
        if not r:
            self.report.setText("Здесь появятся сдвиг, темп, растяжение и качество до/после.")
            return
        m = r.metrics
        dr = m["dtw_resid_ms"]
        oe = m["onset_error_ms"]

        def on(d):
            return "—" if d["median"] is None else f"{d['median']:.0f} мс ({d['matched'] * 100:.0f} %)"
        off = m["offset_s"]
        tp = m["tempo_percent"]
        self.report.setText(
            f"<b>Сдвиг</b> {off:+.3f} с (dub {'позже' if off > 0 else 'раньше'}) · "
            f"<b>темп</b> {'медленнее' if tp < 0 else 'быстрее'} на {abs(tp):.2f} %<br>"
            f"<b>Растяжение</b> ср. {m['stretch_mean']:.3f}, {m['stretch_min']:.2f}–{m['stretch_max']:.2f}×, "
            f"за 100 мс до {m['stretch_change_max_100ms']:.3f}<br>"
            f"<b>Расхождение по DTW</b>: как есть {dr['as_is']:.0f} → сдвиг {dr['shift_only']:.0f} → "
            f"сдвиг+темп {dr['global']:.0f} → <b>после {dr['after']:.0f} мс</b><br>"
            f"<b>Атаки</b> (медиана, доля совпавших): сдвиг {on(oe['shift_only'])} → "
            f"<b>после {on(oe['after'])}</b><br>"
            f"<span style='color:#8b949b'>{r.seconds:.1f} с · {m['keyframes']} опорных точек · "
            f"{os.path.basename(r.out_dir)}</span>")
        for w in r.warnings:
            it = QListWidgetItem(("🎼 " if w["kind"] == "key" else "⚠ ") + w["text"])
            it.setData(Qt.UserRole, (w.get("t0"), w.get("t1")))
            self.warn_list.addItem(it)
        for n in r.notes:
            it = QListWidgetItem("· " + n)
            it.setForeground(QColor("#8b949b"))
            self.warn_list.addItem(it)
        if not r.warnings and not r.notes:
            self.warn_list.addItem(QListWidgetItem("✓ Предупреждений нет"))

    def warning_clicked(self, it):
        a, b = it.data(Qt.UserRole) or (None, None)
        r = self.current_result()
        if a is None or r is None:
            return
        a, b = a - r.t0, b - r.t0
        self.tl.show_range(a, b)
        self.tl.loop = (a, b)
        self.on_loop((a, b))
        self.set_source("result")
        self.player.play()

    # ------------------------------------------------------------------ просмотр
    def audio_for(self, key):
        if key in self.audio_cache:
            return self.audio_cache[key]
        val = None
        if key == "ref":
            r = next(iter(self.results.values()), None)
            if r is not None:
                val = r.guide_y.mean(axis=0).astype(np.float32)
        elif isinstance(key, tuple) and key[0] == "after":
            r = self.results.get(key[1])
            if r is not None:
                val = r.y.mean(axis=0).astype(np.float32)
        elif isinstance(key, tuple) and key[0] == "before":
            r = self.results.get(key[1])
            if r is not None and self.session:
                from . import io as mio
                D = self.session.track(key[1], None, None, r.settings.silence_db)
                x = mio.resample(D.data.mean(axis=0), D.sr, r.sr)
                n = r.y.shape[1]
                lag = int(round(r.line.offset * r.sr))
                out = np.zeros(n, np.float32)
                a, b = max(0, lag), min(len(x), lag + n)
                if b > a:
                    out[a - lag:b - lag] = x[a:b]
                val = out
        self.audio_cache[key] = val
        return val

    def load_ref_view(self):
        if not self.guide:
            return
        try:
            s = self.settings()
            data, sr = sf.read(self.guide, always_2d=True, dtype="float32")
            a = int((s.start or 0) * sr)
            b = data.shape[0] if not s.end else int(s.end * sr)
            self.tl.set_ref(data[a:b].mean(axis=1), sr)
            self.tl.set_take(None, None, sr, "Результат", "Dub до (только сдвиг)", [])
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())

    def on_select(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows:
            return
        p = self.paths[rows[0]]
        if p in self.results:
            self.view_path = p
            self.show_take(p)
        elif p == self.guide:
            self.load_ref_view()

    def show_take(self, path):
        r = self.results.get(path)
        if r is None:
            return
        self.tl.set_ref(r.guide_y.mean(axis=0), r.sr)
        probs = [(w["t0"] - r.t0, w["t1"] - r.t0) for w in r.warnings if w.get("t0") is not None]
        n = len(r.tm.pos)
        t = np.arange(n) * r.fm / 1000.0
        self.tl.set_take(self.audio_for(("after", path)), self.audio_for(("before", path)), r.sr,
                         f"{self.label(path)} после выравнивания",
                         f"{self.label(path)} до (только сдвиг {r.line.offset:+.2f} с)", probs,
                         (t, r.tm.stretch), (r.settings.stretch_min, r.settings.stretch_max))
        self.update_player_sources()
        self.update_card()
        self.update_report()

    def update_player_sources(self):
        r = self.current_result() or next(iter(self.results.values()), None)
        if r is None:
            return
        p = self.view_path if self.view_path in self.results else None
        src = {"ref": self.audio_for("ref"),
               "result": self.audio_for(("after", p)) if p else None,
               "before": self.audio_for(("before", p)) if p else None,
               "comp": self.comp_audio}
        self.player.set_sources(src, r.sr)

    # ------------------------------------------------------------------ плеер
    def set_source(self, key):
        self.player.current = key
        b = self.src_btns.get(key)
        if b:
            b.setChecked(True)
        if self.player.sources.get(key) is None:
            names = {"ref": "эталона", "before": "дубля", "result": "результата", "comp": "компа"}
            self.bar.setFormat(f"Пока нет {names.get(key, '')} — выберите дорожку или дождитесь обработки.")

    def toggle_play(self):
        if self.player.playing:
            self.player.pause()
        else:
            if not self.player.sources:
                self.update_player_sources()
            if not self.player.play():
                self.bar.setFormat("Нечего играть или не открылся звуковой выход.")

    def seek(self, t):
        self.player.seek(t)
        if self.chk_loop.isChecked() and self.tl.loop and not (self.tl.loop[0] <= t < self.tl.loop[1]):
            self.chk_loop.setChecked(False)
        self.tl.active_phrase = self.tl.phrase_at(t) if self.comp else -1

    def on_loop(self, loop):
        if loop and loop[1] - loop[0] > 0.05:
            sr = self.player.sr
            self.player.loop = (int(loop[0] * sr), int(loop[1] * sr))
            if not self.chk_loop.isChecked():
                self.chk_loop.setChecked(True)
            self.player.seek(loop[0])
        else:
            self.player.loop = None

    def on_loop_toggle(self, on):
        self.player.loop_on = on
        if not on:
            self.tl.loop = None
            self.tl.update()

    def tick(self):
        t = self.player.seconds()
        # перерисовка только когда что-то сдвинулось: рисование волн держит GIL и
        # иначе тормозит фоновый расчёт
        if abs(t - self.tl.playhead) > 1e-4 or self.player.playing:
            self.tl.playhead = t
            if self.comp and self.player.playing:
                self.tl.active_phrase = self.tl.phrase_at(t)
            self.tl.update()
        dur = self.player.length / self.player.sr if self.player.sr else 0
        self.time_lbl.setText(f"{fmt_t(t)} / {fmt_t(dur)}")
        self.btn_play.setText("⏸" if self.player.playing else "▶")

    def closeEvent(self, e):
        self.save_settings()
        self.player.close()
        running = [w for w in [self.worker] + self._old if w is not None and w.isRunning()]
        for w in running:
            w.stop = True
        for w in running:
            w.wait(8000)
        try:
            from .world import shutdown
            shutdown()
        except Exception:  # noqa: BLE001
            pass
        e.accept()
        if any(w.isRunning() for w in running):
            os._exit(0)


STYLE = f"""
QWidget {{ background: {BG}; color: #e6e6e6; font-size: 13px; }}
QFrame#drop {{ border: 2px dashed #4a5056; border-radius: 10px; background: {PANEL}; }}
QFrame#drop[hover="true"] {{ border-color: {ACCENT}; background: #22312f; }}
QFrame#drop QLabel {{ background: transparent; }}
QLabel#dropTitle {{ font-size: 13px; font-weight: 700; }}
QLabel#hint {{ color: #8b949b; background: transparent; font-size: 12px; }}
QLabel#cap {{ color: #8b949b; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel#time {{ font-family: Menlo, monospace; font-size: 13px; color: #cfd5da; }}
QLabel#report {{ background: {PANEL}; border-radius: 8px; padding: 8px; font-size: 12px; }}
QListWidget#warns {{ background: {PANEL}; border: none; border-radius: 8px; font-size: 12px; }}
QListWidget#warns::item {{ padding: 4px; border-bottom: 1px solid #2c3136; }}
QListWidget#warns::item:selected {{ background: #2f4a47; }}
QLabel#chip {{ background: #2c3a39; border: 1px solid {ACCENT}; border-radius: 6px; padding: 6px 12px; color: {ACCENT}; font-weight: 600; }}
QFrame#panel {{ background: {PANEL}; border-radius: 10px; }}
QFrame#final {{ background: #1f2b2a; border: 1px solid #2f6f68; border-radius: 10px; }}
QFrame#final QLabel {{ background: transparent; }}
QLabel#finalTitle {{ color: {ACCENT}; font-size: 13px; font-weight: 800; letter-spacing: 1px; }}
QFrame#final QLabel#bigchip {{ background: {ACCENT}; color: #0e1a19; border-radius: 6px; padding: 7px 12px; font-weight: 800; }}
QFrame#panel QLabel, QFrame#panel QCheckBox {{ background: transparent; }}
QPushButton, QToolButton {{ background: #2d3236; border: 1px solid #3a4046; border-radius: 6px; padding: 6px 11px; }}
QPushButton:hover, QToolButton:hover {{ background: #363c41; }}
QToolButton:checked {{ background: #2f4a47; border-color: {ACCENT}; }}
QPushButton#go {{ background: {ACCENT}; color: #0e1a19; font-size: 14px; font-weight: 700; padding: 10px 18px; border: none; }}
QPushButton#go:hover {{ background: #52c7b9; }}
QPushButton#play {{ background: {ACCENT}; color: #0e1a19; font-size: 16px; font-weight: 700; min-width: 44px; border: none; }}
QPushButton#seg {{ padding: 6px 10px; }}
QPushButton#seg:checked {{ background: #2f4a47; border-color: {ACCENT}; color: #ffffff; }}
QProgressBar {{ background: {PANEL}; border: none; border-radius: 4px; height: 20px; text-align: center; color: #cfd5da; }}
QProgressBar::chunk {{ background: #2f6f68; border-radius: 4px; }}
QTableWidget {{ background: {PANEL}; border: none; border-radius: 8px; }}
QHeaderView::section {{ background: #2a2e33; border: none; padding: 5px; color: #8b949b; }}
QTableWidget::item:selected {{ background: #2f4a47; color: #ffffff; }}
QSlider::groove:horizontal {{ height: 4px; background: #3c4248; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QComboBox, QDoubleSpinBox {{ background: #2d3236; border: 1px solid #3a4046; border-radius: 6px; padding: 4px 8px; }}
QMenu {{ background: #2a2e33; border: 1px solid #3a4046; }}
QMenu::item:selected {{ background: #2f4a47; }}
QSplitter::handle {{ background: {BG}; width: 8px; }}
QToolTip {{ background: #2a2e33; color: #e6e6e6; border: 1px solid #3a4046; }}
"""


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
