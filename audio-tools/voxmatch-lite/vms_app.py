"""
VoxMatch Простой — окно (версия 2).

Скиньте все дорожки разом → программа сама выберет эталон (или отметьте ★),
определит, где дабл в унисон, а где бэк в гармонию, выровняет всё и соберёт
комп из лучших фраз. Колёсиком на фразе в ленте «Комп» листаются варианты из
других дублей — от лучшего к худшему.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np
import soundfile as sf
from PySide6.QtCore import QMimeData, QPointF, QRectF, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QDrag, QFont, QKeySequence, QPainter,
                           QPainterPath, QPen, QPolygonF, QShortcut)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMenu, QMessageBox, QProgressBar,
    QPushButton, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
    QWidget, QButtonGroup)

try:
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(APP_DIR, "error.log")
AUDIO_EXT = {".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}
AUDIO_FILTER = "Аудио (*.wav *.wave *.aif *.aiff *.flac *.ogg *.mp3)"
ACCENT = "#3fb6a8"
REF_COL = "#f0a040"
BG = "#1b1e21"
PANEL = "#23272b"
DEFAULTS = {"xfade": 60, "timing": 100, "tightness": 70, "stretch": 1, "pitch": 100, "formants": True,
            "harmony": True, "auto": True}
STRETCH = [(1.5, "1,5×"), (2.0, "2×"), (3.0, "3×")]
REF_KEY = "__ref__"

_engine = None
_comp = None


def engine():
    global _engine
    if _engine is None:
        import vms_engine
        _engine = vms_engine
    return _engine


def comp_mod():
    global _comp
    if _comp is None:
        import vms_comp
        _comp = vms_comp
    return _comp


def log_error(text: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
    except OSError:
        pass


def inbox_dir() -> str:
    return os.path.join(os.path.expanduser("~"), "Desktop", "VoxMatch Входящие")


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
# Перетаскивание: файлы из Finder или куски прямо из Cubase (vst-xml)
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

    def __init__(self):
        super().__init__()
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(54)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 6, 16, 6)
        self.title = QLabel("⬇  Перетащите сюда все дорожки разом — эталон и дубли (из Cubase или Finder)")
        self.title.setObjectName("dropTitle")
        self.hint = QLabel("или кликните, чтобы выбрать файлы")
        self.hint.setObjectName("hint")
        lay.addWidget(self.title)
        lay.addStretch(1)
        lay.addWidget(self.hint)

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
            QMessageBox.warning(self, "VoxMatch Простой", "\n".join(errors))
        elif not paths:
            log_error("Перетаскивание без файла, форматы: " + ", ".join(e.mimeData().formats()))
            QMessageBox.information(
                self, "VoxMatch Простой",
                "Не получилось взять звук из перетаскивания.\n\nВ Cubase: Аудио → Bounce, "
                "затем перетащите получившийся файл, или перетащите файл из Finder.")

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()


class TrackTable(QTableWidget):
    """Список дорожек. Принимает файлы; готовые результаты перетаскиваются в Cubase."""
    filesDropped = Signal(list)
    starClicked = Signal(int)

    COLS = ["★", "Дорожка", "Тип", "Оценка", "Статус"]

    def __init__(self, paths_for_drag):
        super().__init__(0, len(self.COLS))
        self._paths = paths_for_drag
        self.setHorizontalHeaderLabels(self.COLS)
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Fixed)
        self.setColumnWidth(0, 30)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in (2, 3, 4):
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
            QMessageBox.warning(self, "VoxMatch Простой", "\n".join(errors))


class DragChip(QLabel):
    """Плашка «перетащите в Cubase»: перед перетаскиванием сохраняет файл."""

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
    """Маленькая волна финала: каждая фраза подкрашена дублем, из которого взята."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(260, 44)
        self.env = None
        self.blocks = []            # [(a, b, оценка, метка)] в долях 0..1

    def set_data(self, env, blocks):
        self.env, self.blocks = env, blocks
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


class FinalCard(QFrame):
    """Карточка «Финал»: собранный вокал. Его можно перетащить в Cubase целиком."""

    def __init__(self, make_paths):
        super().__init__()
        self.setObjectName("final")
        self._make = make_paths
        self._press = None
        self.setCursor(Qt.OpenHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(12)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = QLabel("ФИНАЛ · собранный вокал")
        self.title.setObjectName("finalTitle")
        self.info = QLabel("Скиньте дорожки — финал соберётся сам из лучших фраз.")
        self.info.setObjectName("hint")
        self.info.setWordWrap(True)
        col.addWidget(self.title)
        col.addWidget(self.info)
        lay.addLayout(col, 2)
        self.wave = MiniWave()
        lay.addWidget(self.wave, 3)
        self.btn_play = QPushButton("▶ Слушать финал")
        self.chip = DragChip("⇱  Перетащить в Cubase", make_paths)
        self.chip.setObjectName("bigchip")
        self.btn_show = QPushButton("Показать файл")
        lay.addWidget(self.btn_play)
        lay.addWidget(self.chip)
        lay.addWidget(self.btn_show)
        self.setToolTip("Это итоговая дорожка. Зажмите карточку (или кнопку «Перетащить») и "
                        "бросьте на дорожку Cubase. Файл сохраняется сам после каждой правки.")

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
        self.v.setMinimumWidth(120)
        for w in (self.label, self.s, self.v):
            w.setToolTip(tip)
        grid.addWidget(self.label, row, col)
        grid.addWidget(self.s, row, col + 1)
        grid.addWidget(self.v, row, col + 2)
        self.s.valueChanged.connect(self._show)
        self._show(value)

    def _show(self, v):
        self.v.setText(self.fmt(v))

    def setEnabled(self, on: bool):
        for w in (self.label, self.s, self.v):
            w.setEnabled(on)


# ----------------------------------------------------------------------------
# Плеер: несколько источников на одной шкале, переключение без остановки, петля
# ----------------------------------------------------------------------------
class Player:
    def __init__(self):
        self.stream = None
        self.sources: dict[str, np.ndarray] = {}
        self.current = "result"
        self.sr = 48000
        self.pos = 0
        self.loop = None          # (a, b) в сэмплах
        self.loop_on = False
        self.ref_under = False
        self.playing = False
        self.length = 0

    def set_sources(self, sources: dict, sr: int):
        self.sources = {k: v for k, v in sources.items() if v is not None}
        self.sr = sr
        self.length = max((len(v) for v in self.sources.values()), default=0)
        self.pos = min(self.pos, self.length)

    def _cb(self, out, frames, _t, _s):
        out[:, 0] = 0.0
        if not self.playing:
            return
        src = self.sources.get(self.current)
        ref = self.sources.get("ref") if self.ref_under and self.current != "ref" else None
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
                out[filled:filled + len(seg), 0] += seg
            if ref is not None and a < len(ref):
                seg = ref[a:a + k]
                out[filled:filled + len(seg), 0] += 0.45 * seg
            filled += k
            self.pos = a + k

    def ensure_stream(self):
        if sd is None:
            return False
        if self.stream is not None and getattr(self.stream, "samplerate", self.sr) == self.sr:
            return True
        self.close()
        try:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32",
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
# Таймлайн: лента «Комп», волны эталона/результата и «до», петля, пометки
# ----------------------------------------------------------------------------
def env_peaks(x: np.ndarray, step: int):
    """Огибающая (макс по модулю) с шагом step сэмплов."""
    if x is None or len(x) == 0:
        return np.zeros(1, np.float32)
    m = len(x) // step
    if m == 0:
        return np.array([np.max(np.abs(x))], np.float32)
    return np.abs(x[: m * step]).reshape(m, step).max(axis=1).astype(np.float32)


class Timeline(QWidget):
    seekRequested = Signal(float)
    loopChanged = Signal(object)              # (a, b) секунд или None
    choiceWheel = Signal(int, int)            # фраза, шаг (+1 лучше / −1 хуже)
    choiceMenu = Signal(int, object)          # фраза, глобальная точка
    phraseClicked = Signal(int)

    COMP_H = 46
    RULER_H = 18

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(330)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.NoFocus)
        self.dur = 0.0
        self.v0, self.v1 = 0.0, 1.0
        self.res_env = 0.001                  # шаг огибающих, секунд
        self.ref_env = None
        self.after_env = None
        self.before_env = None
        self.tiles = []                       # [(a, b)] секунд
        self.comp_blocks = []                 # [(метка, оценка, пояснение, ref?)]
        self.problems = []                    # [(a, b)] секунд
        self.playhead = 0.0
        self.loop = None
        self.active_phrase = -1
        self.after_label = "Финал (собранный из фраз)"
        self.before_label = "Дубль до"
        self._drag = None
        self._wheel_acc = 0
        self._hover = -1

    # --- данные ---
    def set_ref(self, x, sr):
        step = max(1, int(sr * self.res_env))
        self.ref_env = env_peaks(x, step)
        self.dur = len(x) / sr
        if self.v1 <= self.v0 or self.v1 > self.dur or self.v1 - self.v0 < 0.2:
            self.v0, self.v1 = 0.0, max(0.5, self.dur)
        self.update()

    def set_take(self, after, before, sr, after_label, before_label, problems):
        step = max(1, int(sr * self.res_env))
        self.after_env = env_peaks(after, step) if after is not None else None
        self.before_env = env_peaks(before, step) if before is not None else None
        self.after_label, self.before_label = after_label, before_label
        self.problems = problems or []
        self.update()

    def set_comp(self, tiles, blocks):
        self.tiles = tiles
        self.comp_blocks = blocks
        self.update()

    def zoom_all(self):
        self.v0, self.v1 = 0.0, max(0.5, self.dur)
        self.update()

    # --- геометрия ---
    def _x(self, t):
        return (t - self.v0) / max(1e-6, self.v1 - self.v0) * self.width()

    def _t(self, x):
        return self.v0 + x / max(1, self.width()) * (self.v1 - self.v0)

    def _lanes(self):
        h = self.height()
        top = self.RULER_H + self.COMP_H + 6
        rest = h - top - 4
        return (self.RULER_H, self.COMP_H), (top, rest), (h, 0)

    def phrase_at(self, t):
        for i, (a, b) in enumerate(self.tiles):
            if a <= t < b:
                return i
        return -1

    # --- рисование ---
    def _wave(self, p, env, y0, h, color, alpha=255, fill=True):
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
        peak = 0.5
        mid = y0 + h / 2
        amp = np.clip(seg / peak, 0, 1) * (h / 2 - 2)
        c = QColor(color)
        c.setAlpha(alpha)
        poly = QPolygonF([QPointF(float(x), float(mid - a)) for x, a in zip(xs, amp)] +
                         [QPointF(float(x), float(mid + a)) for x, a in zip(xs[::-1], amp[::-1])])
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawPolygon(poly)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(PANEL))
        (cy, ch), (ay, ah), (by, bh) = self._lanes()
        if self.dur <= 0:
            p.setPen(QColor("#7d868e"))
            p.drawText(self.rect(), Qt.AlignCenter, "Скиньте дорожки — здесь появятся волны и комп")
            return
        # линейка
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
        # петля
        if self.loop:
            x0, x1 = self._x(self.loop[0]), self._x(self.loop[1])
            p.fillRect(QRectF(x0, 0, x1 - x0, h), QColor(63, 182, 168, 38))
            p.setPen(QPen(QColor(ACCENT), 1))
            p.drawLine(int(x0), 0, int(x0), h)
            p.drawLine(int(x1), 0, int(x1), h)
        # лента «Комп»
        p.setPen(QColor("#9aa3ab"))
        for i, (a, b) in enumerate(self.tiles):
            x0, x1 = self._x(a), self._x(b)
            if x1 < 0 or x0 > w:
                continue
            lab, sc, note, is_ref = self.comp_blocks[i] if i < len(self.comp_blocks) else ("", None, "", False)
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
                    p.drawText(r.adjusted(6, ch / 2 - 4, -4, -2), Qt.AlignLeft | Qt.AlignVCenter,
                               note)
        # подписи дорожек
        p.setPen(QColor(REF_COL))
        p.drawText(8, ay + 15, "■ Эталон")
        p.setPen(QColor(ACCENT))
        p.drawText(80, ay + 15, f"■ {self.after_label}")
        p.fillRect(QRectF(0, ay, w, ah), QColor(255, 255, 255, 6))
        # пометки «послушать»
        for a, b in self.problems:
            x0, x1 = self._x(a), self._x(b)
            if x1 < 0 or x0 > w:
                continue
            p.fillRect(QRectF(x0, ay, max(2.0, x1 - x0), ah), QColor(224, 96, 90, 55))
            p.fillRect(QRectF(x0, ay, max(2.0, x1 - x0), 3), QColor(224, 96, 90, 220))
        # волны
        self._wave(p, self.ref_env, ay, ah, REF_COL, 200)
        self._wave(p, self.after_env, ay, ah, ACCENT, 150)
        # границы фраз
        p.setPen(QPen(QColor(255, 255, 255, 30), 1, Qt.DashLine))
        for a, _b in self.tiles[1:]:
            x = self._x(a)
            p.drawLine(int(x), cy, int(x), h)
        # курсор
        x = self._x(self.playhead)
        p.setPen(QPen(QColor("#ffffff"), 1.5))
        p.drawLine(int(x), 0, int(x), h)

    # --- мышь ---
    def mousePressEvent(self, e):
        x, y = e.position().x(), e.position().y()
        (cy, ch), _a, _b = self._lanes()
        t = self._t(x)
        if e.button() == Qt.RightButton:
            i = self.phrase_at(t)
            if i >= 0:
                self.choiceMenu.emit(i, e.globalPosition().toPoint())
            return
        if e.button() != Qt.LeftButton:
            return
        if cy <= y < cy + ch:
            i = self.phrase_at(t)
            if i >= 0:
                self.phraseClicked.emit(i)
            return
        self._drag = (x, t)

    def mouseMoveEvent(self, e):
        x, y = e.position().x(), e.position().y()
        (cy, ch), _a, _b = self._lanes()
        t = self._t(x)
        hv = self.phrase_at(t) if cy <= y < cy + ch else -1
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
        (cy, ch), _a, _b = self._lanes()
        if not (cy <= e.position().y() < cy + ch):
            self.zoom_all()

    def leaveEvent(self, _e):
        self._hover = -1
        self.update()

    def wheelEvent(self, e):
        x, y = e.position().x(), e.position().y()
        (cy, ch), _a, _b = self._lanes()
        d = e.angleDelta()
        if cy <= y < cy + ch and abs(d.y()) >= abs(d.x()):
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
    refChosen = Signal(str)
    takeDone = Signal(str, object)
    takeFailed = Signal(str, str)
    compReady = Signal(object)
    finishedAll = Signal()

    def __init__(self, session, ref, paths, todo, settings, results):
        super().__init__()
        self.session, self.ref, self.paths = session, ref, paths
        self.todo, self.s, self.results = todo, settings, dict(results)
        self.stop = False

    def run(self):
        ve, vc = engine(), comp_mod()
        try:
            ref = self.ref
            if not ref:
                self.progress.emit("Слушаю все дорожки, выбираю эталон…", 0.02)
                sr0 = sf.info(self.paths[0]).samplerate
                qt = [ve.quick_track(p, sr0) for p in self.paths]
                ref = self.paths[vc.suggest_reference(qt)]
                self.refChosen.emit(ref)
                self.todo = [p for p in self.paths if p != ref]
            todo = [p for p in self.todo if p != ref]
            total = max(1, len(todo))
            for i, path in enumerate(todo):
                if self.stop:
                    break
                name = os.path.basename(path)

                def prog(msg, f, i=i, name=name):
                    self.progress.emit(f"{i + 1}/{total} · {name}: {msg}", (i + f) / total * 0.95)
                try:
                    r = self.session.process(ref, path, self.s, progress=prog)
                    self.results[path] = r
                    self.takeDone.emit(path, r)
                except Exception as exc:  # noqa: BLE001
                    log_error(traceback.format_exc())
                    self.takeFailed.emit(path, str(exc))
            if not self.stop:
                self.progress.emit("Оцениваю фразы и собираю комп…", 0.96)
                R = self.session.track(ref)
                res = {p: r for p, r in self.results.items() if p in self.paths and p != ref}
                tiles, sung = vc.phrases(R)
                scores = vc.score_all(R, res, tiles, sung) if res else {}
                self.compReady.emit({"ref": ref, "tiles": tiles, "sung": sung, "scores": scores,
                                     "fm": R.fm, "sr": R.sr})
        except Exception as exc:  # noqa: BLE001
            log_error(traceback.format_exc())
            self.takeFailed.emit("", str(exc))
        self.finishedAll.emit()


# ----------------------------------------------------------------------------
# Окно
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoxMatch Простой")
        self.resize(1320, 820)
        self.cfg = QSettings("VoxMatch", "VoxMatchSimple2")
        self.paths: list[str] = []
        self.ref: str | None = None
        self.results: dict = {}
        self.status: dict = {}
        self.session = None
        self.worker = None
        self._old = []
        self.comp = None                       # dict: tiles, scores, choices, candidates
        self.comp_audio = None
        self.audio_cache: dict = {}
        self.view_path = None
        self.player = Player()
        self._pending_choices = None
        self._pending_tp = None
        self._dirty_settings = False

        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(14, 12, 14, 10)
        lay.setSpacing(8)

        # верх: зона и кнопки
        top = QHBoxLayout()
        self.drop = DropZone()
        self.drop.filesDropped.connect(self.add_files)
        self.drop.clicked.connect(self.pick_files)
        top.addWidget(self.drop, 1)
        self.cb_mode = QComboBox()
        self.cb_mode.addItem("Режим: Глобальный")
        self.cb_mode.addItem("Режим: Гибкий — по фразам")
        self.cb_mode.setToolTip("Глобальный — все дубли обрабатываются целиком общими настройками "
                                "(бэки, стерео).\nГибкий — дубли только анализируются и режутся "
                                "на фразы; тайминг и питч вы задаёте сами для каждой фразы "
                                "(основной вокал, «хирургия»).")
        top.addWidget(self.cb_mode)
        self.btn_go = QPushButton("▶  Выровнять всё")
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
        m.addAction("Открыть папку с результатами", self.open_folder)
        m.addAction("Сохранить каждый выровненный дубль… (уже лежат в папке)", self.open_folder)
        m.addSeparator()
        m.addAction("Очистить всё", self.clear_all)
        self.btn_more.setMenu(m)
        top.addWidget(self.btn_more)
        lay.addLayout(top)

        # настройки (скрыты)
        self.set_panel = QFrame()
        self.set_panel.setObjectName("panel")
        g = QGridLayout(self.set_panel)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(4, 1)
        self.sl_timing = Slider(g, 0, 0, "Тайминг", 0, 100, DEFAULTS["timing"],
                                lambda v: "не трогать" if v == 0 else f"{v} %",
                                "Насколько подогнать дубли по времени под эталон.")
        self.sl_tight = Slider(g, 1, 0, "Жёсткость", 0, 100, DEFAULTS["tightness"],
                               lambda v: f"{v} % · " + ("каждый слог" if v >= 80 else
                                                        "по словам" if v >= 40 else "по фразам"),
                               "Мягко — двигаются фразы целиком. Жёстко — каждый слог.")
        self.sl_pitch = Slider(g, 0, 3, "Питч", 0, 100, DEFAULTS["pitch"],
                               lambda v: "не трогать" if v == 0 else f"{v} %",
                               "Насколько подтянуть ноты к эталону. Вибрато дубля остаётся.")
        g.addWidget(QLabel("Макс. растяжение"), 1, 3)
        self.cb_stretch = QComboBox()
        for _v, t in STRETCH:
            self.cb_stretch.addItem(t)
        g.addWidget(self.cb_stretch, 1, 4)
        self.chk_form = QCheckBox("Сохранять тембр (форманты)")
        self.chk_harm = QCheckBox("Бэки в гармонию — ноты не трогать (авто)")
        self.chk_harm.setToolTip("Если дубль поёт другие ноты (терция, квинта…), программа "
                                 "выровняет только тайминг, чтобы гармония не стала унисоном.")
        self.chk_auto = QCheckBox("Сразу обрабатывать, как только скинули")
        hb = QHBoxLayout()
        for wdg in (self.chk_form, self.chk_harm, self.chk_auto):
            hb.addWidget(wdg)
        hb.addStretch(1)
        b = QPushButton("Стандартные")
        b.clicked.connect(self.reset_defaults)
        hb.addWidget(b)
        self.btn_apply = QPushButton("Применить ко всем")
        self.btn_apply.clicked.connect(lambda: self.run_all(force=True))
        hb.addWidget(self.btn_apply)
        self.sl_xfade = Slider(g, 2, 0, "Кроссфейд", 10, 200, DEFAULTS["xfade"],
                               lambda v: f"{v} мс",
                               "Плавность перехода между фразами из разных дублей "
                               "(как кроссфейд X в Cubase). Меняется сразу, без пересчёта.")
        g.addLayout(hb, 3, 0, 1, 6)
        self.set_panel.setVisible(False)
        lay.addWidget(self.set_panel)

        # середина: список | таймлайн
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        cap = QLabel("ДОРОЖКИ")
        cap.setObjectName("cap")
        ll.addWidget(cap)
        self.table = TrackTable(self.drag_selected)
        self.table.filesDropped.connect(self.add_files)
        self.table.starClicked.connect(self.star_row)
        self.table.itemSelectionChanged.connect(self.on_select)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.table_menu)
        ll.addWidget(self.table, 1)
        self.legend = QLabel("★ — эталон (клик по звёздочке, чтобы сменить).  "
                             "Строку можно перетащить в Cubase.")
        self.legend.setObjectName("hint")
        self.legend.setWordWrap(True)
        ll.addWidget(self.legend)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        hdr = QHBoxLayout()
        cap2 = QLabel("КОМП: ИЗ КАКОГО ДУБЛЯ КАЖДАЯ ФРАЗА  ·  колёсико на фразе — другой дубль")
        cap2.setObjectName("cap")
        hdr.addWidget(cap2)
        hdr.addStretch(1)
        self.btn_best = QPushButton("↺ Сбросить на лучшие")
        self.btn_best.setToolTip("Отменить ваши замены: в каждой фразе снова лучший по оценке вариант")
        self.btn_best.clicked.connect(self.reset_choices)
        hdr.addWidget(self.btn_best)
        rl.addLayout(hdr)
        self.final = FinalCard(self.export_comp_paths)
        self.final.btn_play.clicked.connect(self.play_final)
        self.final.btn_show.clicked.connect(self.show_final_file)
        rl.addWidget(self.final)
        self.tl = Timeline()
        self.tl.seekRequested.connect(self.seek)
        self.tl.loopChanged.connect(self.on_loop)
        self.tl.choiceWheel.connect(self.wheel_choice)
        self.tl.choiceMenu.connect(self.choice_menu)
        self.tl.phraseClicked.connect(self.loop_phrase)
        rl.addWidget(self.tl, 1)
        self.ph = QFrame()
        self.ph.setObjectName("panel")
        pg = QGridLayout(self.ph)
        pg.setContentsMargins(12, 8, 12, 8)
        pg.setColumnStretch(1, 1)
        self.ph_title = QLabel("Кликните фразу в ленте сверху")
        self.ph_title.setObjectName("finalTitle")
        pg.addWidget(self.ph_title, 0, 0, 1, 2)
        pg.addWidget(QLabel("Дубль"), 0, 2, Qt.AlignRight)
        self.ph_take = QComboBox()
        self.ph_take.setMinimumWidth(320)
        pg.addWidget(self.ph_take, 0, 3, 1, 2)
        self.ph_t = Slider(pg, 1, 0, "Тайминг к эталону", 0, 100, 0,
                           lambda v: "как спето" if v == 0 else f"{v} %",
                           "Только для этой фразы: насколько подогнать по времени под эталон.")
        self.ph_p = Slider(pg, 2, 0, "Питч к эталону", 0, 100, 0,
                           lambda v: "как спето" if v == 0 else f"{v} %",
                           "Только для этой фразы: насколько подтянуть ноты к эталону.")
        bx = QVBoxLayout()
        self.ph_all = QPushButton("Эти настройки → на все фразы")
        self.ph_reset = QPushButton("Сброс фразы (как спето)")
        bx.addWidget(self.ph_all)
        bx.addWidget(self.ph_reset)
        pg.addLayout(bx, 1, 3, 2, 2)
        self.ph_hint = QLabel("")
        self.ph_hint.setObjectName("hint")
        pg.addWidget(self.ph_hint, 3, 0, 1, 5)
        rl.addWidget(self.ph)
        self.sel_phrase = -1
        self._ph_updating = False
        self.ph_timer = QTimer(self)
        self.ph_timer.setSingleShot(True)
        self.ph_timer.timeout.connect(self.apply_phrase_edit)
        self.ph_t.s.valueChanged.connect(self.phrase_slider_changed)
        self.ph_p.s.valueChanged.connect(self.phrase_slider_changed)
        self.ph_take.activated.connect(self.phrase_take_changed)
        self.ph_all.clicked.connect(self.phrase_to_all)
        self.ph_reset.clicked.connect(self.phrase_reset)

        # транспорт
        tr = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("play")
        self.btn_play.clicked.connect(self.toggle_play)
        tr.addWidget(self.btn_play)
        self.src_group = QButtonGroup(self)
        self.src_btns = {}
        for i, (key, text) in enumerate((("ref", "1 Эталон"), ("result", "2 Дубль из списка"),
                                         ("comp", "3 Финал"))):
            bt = QPushButton(text)
            bt.setCheckable(True)
            bt.setObjectName("seg")
            self.src_group.addButton(bt, i)
            bt.clicked.connect(lambda _=False, k=key: self.set_source(k))
            tr.addWidget(bt)
            self.src_btns[key] = bt
        self.src_btns["comp"].setChecked(True)
        self.player.current = "comp"
        self.chk_under = QCheckBox("+ эталон тише")
        self.chk_under.setToolTip("Подмешать эталон, чтобы слышать, совпадает ли (E)")
        self.chk_under.toggled.connect(lambda on: setattr(self.player, "ref_under", on))
        tr.addWidget(self.chk_under)
        self.chk_loop = QCheckBox("Петля (L)")
        self.chk_loop.toggled.connect(self.on_loop_toggle)
        tr.addWidget(self.chk_loop)
        tr.addStretch(1)
        self.time_lbl = QLabel("0:00.0")
        self.time_lbl.setObjectName("time")
        tr.addWidget(self.time_lbl)
        rl.addLayout(tr)
        split.addWidget(right)
        split.setSizes([380, 940])
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)

        # низ
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFormat("Скиньте дорожки — дальше всё само.")
        self.bar.setTextVisible(True)
        lay.addWidget(self.bar)
        keys = QLabel("Пробел — играть/пауза · 1 эталон · 2 дубль · 3 финал · ↑/↓ — лучший/худший вариант "
                      "фразы под курсором · ←/→ — соседняя фраза · L — петля · E — эталон тише")
        keys.setObjectName("hint")
        lay.addWidget(keys)

        for key, fn in ((Qt.Key_Space, self.toggle_play), (Qt.Key_1, lambda: self.set_source("ref")),
                        (Qt.Key_2, lambda: self.set_source("result")),
                        (Qt.Key_3, lambda: self.set_source("comp")),
                        (Qt.Key_Up, lambda: self.step_choice(+1)),
                        (Qt.Key_Down, lambda: self.step_choice(-1)),
                        (Qt.Key_Left, lambda: self.step_phrase(-1)),
                        (Qt.Key_Right, lambda: self.step_phrase(+1)),
                        (Qt.Key_L, lambda: self.chk_loop.toggle()),
                        (Qt.Key_E, lambda: self.chk_under.toggle())):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(fn)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.autosave_final)
        self.final_saved = None
        self.autorun = QTimer(self)
        self.autorun.setSingleShot(True)
        self.autorun.timeout.connect(self.run_all)

        self.load_settings()
        for wdg in (self.sl_timing.s, self.sl_tight.s, self.sl_pitch.s):
            wdg.valueChanged.connect(self.settings_changed)
        self.cb_stretch.currentIndexChanged.connect(self.settings_changed)
        self.chk_form.toggled.connect(self.settings_changed)
        self.chk_harm.toggled.connect(self.settings_changed)
        self.cb_mode.currentIndexChanged.connect(self.mode_changed)
        self.xfade_timer = QTimer(self)
        self.xfade_timer.setSingleShot(True)
        self.xfade_timer.timeout.connect(lambda: (self.rebuild_comp_audio(), self.save_settings()))
        self.sl_xfade.s.valueChanged.connect(lambda _v: self.xfade_timer.start(250))
        QTimer.singleShot(0, self.restore)
        QTimer.singleShot(0, self.update_phrase_panel)

    # ------------------------------------------------------------------ настройки
    def settings(self):
        s = engine().Settings()
        s.timing = self.sl_timing.s.value()
        s.tightness = self.sl_tight.s.value()
        s.pitch = self.sl_pitch.s.value()
        s.max_stretch = STRETCH[self.cb_stretch.currentIndex()][0]
        s.formants = self.chk_form.isChecked()
        s.harmony_auto = self.chk_harm.isChecked()
        s.flexible = self.flexible()
        return s

    def flexible(self):
        return self.cb_mode.currentIndex() == 1

    def mode_changed(self, _i):
        self.save_settings()
        self.update_phrase_panel()
        if self.results:
            self.results.clear()
            self.comp = None
            self.comp_audio = None
            self.audio_cache.clear()
            self.update_comp_view()
            self.update_final_card()
            for p in self.paths:
                if p != self.ref:
                    self.status[p] = "ждёт"
            self.refresh_table()
            self.bar.setFormat("Режим сменён — пересчитываю…" if self.chk_auto.isChecked()
                               else "Режим сменён — нажмите «Выровнять всё».")
            if self.chk_auto.isChecked():
                self.autorun.start(300)

    def _b(self, key):
        return str(self.cfg.value(key, DEFAULTS[key])).lower() in ("true", "1")

    def load_settings(self):
        c = self.cfg
        self.sl_timing.s.setValue(int(c.value("timing", DEFAULTS["timing"])))
        self.sl_tight.s.setValue(int(c.value("tightness", DEFAULTS["tightness"])))
        self.sl_pitch.s.setValue(int(c.value("pitch", DEFAULTS["pitch"])))
        self.cb_stretch.setCurrentIndex(int(c.value("stretch", DEFAULTS["stretch"])))
        self.chk_form.setChecked(self._b("formants"))
        self.chk_harm.setChecked(self._b("harmony"))
        self.chk_auto.setChecked(self._b("auto"))
        self.sl_xfade.s.setValue(int(self.cfg.value("xfade", DEFAULTS["xfade"])))
        self.cb_mode.setCurrentIndex(int(self.cfg.value("mode", 0)))

    def save_settings(self):
        c = self.cfg
        c.setValue("timing", self.sl_timing.s.value())
        c.setValue("tightness", self.sl_tight.s.value())
        c.setValue("pitch", self.sl_pitch.s.value())
        c.setValue("stretch", self.cb_stretch.currentIndex())
        c.setValue("formants", self.chk_form.isChecked())
        c.setValue("harmony", self.chk_harm.isChecked())
        c.setValue("auto", self.chk_auto.isChecked())
        c.setValue("xfade", self.sl_xfade.s.value())
        c.setValue("mode", self.cb_mode.currentIndex())
        sess = {"paths": self.paths, "ref": self.ref,
                "choices": self._choices_by_name() if self.comp else None,
                "tp": [self.comp["t"], self.comp["p"]] if self.comp else None}
        c.setValue("session", json.dumps(sess, ensure_ascii=False))

    def reset_defaults(self):
        self.sl_timing.s.setValue(DEFAULTS["timing"])
        self.sl_tight.s.setValue(DEFAULTS["tightness"])
        self.sl_pitch.s.setValue(DEFAULTS["pitch"])
        self.cb_stretch.setCurrentIndex(DEFAULTS["stretch"])
        self.chk_form.setChecked(DEFAULTS["formants"])
        self.chk_harm.setChecked(DEFAULTS["harmony"])
        self.sl_xfade.s.setValue(DEFAULTS["xfade"])

    def settings_changed(self, *_):
        if self.results:
            self._dirty_settings = True
            self.btn_apply.setText("Применить ко всем •")
        self.save_settings()

    def restore(self):
        try:
            sess = json.loads(str(self.cfg.value("session", "") or "{}"))
        except ValueError:
            sess = {}
        paths = [p for p in sess.get("paths") or [] if os.path.isfile(p)]
        if not paths:
            return
        self._pending_choices = sess.get("choices")
        self._pending_tp = sess.get("tp")
        ref = sess.get("ref")
        self.add_files(paths, autorun=False)
        if ref and ref in self.paths:
            self.set_reference(ref, rerun=False)
        self.bar.setFormat("Прошлая сессия открыта.")
        if self.chk_auto.isChecked():
            self.autorun.start(300)

    # ------------------------------------------------------------------ дорожки
    def pick_files(self):
        d = inbox_dir() if os.path.isdir(inbox_dir()) else os.path.expanduser("~")
        ps, _ = QFileDialog.getOpenFileNames(self, "Дорожки", d, AUDIO_FILTER)
        if ps:
            self.add_files(ps)

    def add_files(self, paths, autorun=True):
        new = 0
        for p in paths:
            p = os.path.abspath(p)
            if p in self.paths:
                continue
            if "VoxMatch Простой" in p and ("_выровнен" in p or "КОМП" in p):
                continue                                   # свои результаты не берём
            self.paths.append(p)
            self.status[p] = "ждёт"
            new += 1
        if not new:
            return
        self.refresh_table()
        self.save_settings()
        if autorun and self.chk_auto.isChecked():
            self.bar.setFormat(f"Добавлено: {new}. Начну через секунду…")
            self.autorun.start(1200)

    def label(self, path):
        if path == self.ref or path == REF_KEY:
            return "Э"
        others = [p for p in self.paths if p != self.ref]
        return f"Д{others.index(path) + 1}" if path in others else "?"

    def refresh_table(self):
        t = self.table
        sel = self.view_path
        t.blockSignals(True)
        t.setRowCount(len(self.paths))
        avg = self.take_avg_scores()
        for r, p in enumerate(self.paths):
            is_ref = p == self.ref
            res = self.results.get(p)
            items = [
                ("★" if is_ref else "☆", REF_COL if is_ref else "#5f676e"),
                (f"{self.label(p)}  {os.path.basename(p)}", None),
                ("эталон" if is_ref else (res.kind if res else "—"), None),
                (f"{avg[p]:.0f}" if p in avg else "", None),
                (self.status.get(p, ""), None),
            ]
            for c, (txt, col) in enumerate(items):
                it = QTableWidgetItem(txt)
                if col:
                    it.setForeground(QColor(col))
                if c == 1:
                    it.setToolTip(p + ("\n" + "\n".join(res.notes) if res and res.notes else ""))
                if c == 3 and p in avg:
                    it.setForeground(score_color(avg[p]))
                    it.setTextAlignment(Qt.AlignCenter)
                if c == 0:
                    it.setTextAlignment(Qt.AlignCenter)
                    it.setToolTip("Сделать эталоном")
                if c == 2 and res and res.kind == "гармония":
                    it.setForeground(QColor("#b48ce0"))
                if c == 2 and res and res.kind == "копия эталона":
                    it.setForeground(QColor("#7d868e"))
                t.setItem(r, c, it)
        t.blockSignals(False)
        if sel in self.paths:
            t.selectRow(self.paths.index(sel))
        elif self.paths and not t.selectedItems():
            t.selectRow(0)

    def take_avg_scores(self):
        if not self.comp or not self.comp.get("scores"):
            return {}
        out = {}
        sc = self.comp["scores"]
        sung_w = self.comp.get("weights")
        for k, rows in sc.items():
            if k == REF_KEY:
                continue
            vals = np.array([v for v, _n in rows])
            w = sung_w if sung_w is not None and len(sung_w) == len(vals) else np.ones(len(vals))
            path = self.ref if k == REF_KEY else k
            if path:
                out[path] = float(np.sum(vals * w) / max(1e-9, np.sum(w)))
        return out

    def star_row(self, row):
        if 0 <= row < len(self.paths):
            self.set_reference(self.paths[row])

    def set_reference(self, path, rerun=True):
        if path == self.ref:
            return
        self.ref = path
        self.results.clear()
        self.comp = None
        self.comp_audio = None
        self.audio_cache.clear()
        for p in self.paths:
            self.status[p] = "эталон" if p == path else "ждёт"
        self.refresh_table()
        self.load_ref_view()
        self.save_settings()
        if rerun and self.chk_auto.isChecked():
            self.autorun.start(400)

    def table_menu(self, pt):
        r = self.table.rowAt(pt.y())
        if r < 0:
            return
        p = self.paths[r]
        m = QMenu(self)
        m.addAction("★ Сделать эталоном", lambda: self.set_reference(p))
        m.addAction("Показать в Finder", lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(os.path.dirname(p))))
        m.addSeparator()
        m.addAction("Убрать из списка", lambda: self.remove_path(p))
        m.exec(self.table.viewport().mapToGlobal(pt))

    def remove_path(self, p):
        if self.worker and self.worker.isRunning():
            return
        self.paths.remove(p)
        self.results.pop(p, None)
        self.status.pop(p, None)
        if p == self.ref:
            self.ref = None
            self.results.clear()
            self.comp = None
        elif self.comp:
            self.rebuild_scores()
        self.refresh_table()
        self.save_settings()

    def clear_all(self):
        if self.worker and self.worker.isRunning():
            return
        self.player.pause()
        self.paths.clear()
        self.results.clear()
        self.status.clear()
        self.ref = None
        self.comp = None
        self.comp_audio = None
        self.audio_cache.clear()
        self.view_path = None
        self.refresh_table()
        self.tl.dur = 0
        self.tl.set_comp([], [])
        self.update_final_card()
        self.player.set_sources({}, 48000)
        self.bar.setValue(0)
        self.bar.setFormat("Скиньте дорожки — дальше всё само.")
        self.save_settings()

    def drag_selected(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        out = []
        for r in rows:
            p = self.paths[r]
            res = self.results.get(p)
            if res and os.path.isfile(res.out_path):
                out.append(res.out_path)
            elif p == self.ref:
                out.append(p)
        return out

    def open_folder(self):
        target = None
        for res in self.results.values():
            target = os.path.dirname(res.out_path)
        if not target and self.paths:
            target = os.path.dirname(self.paths[0])
        if target and os.path.isdir(target):
            QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    # ------------------------------------------------------------------ обработка
    def run_all(self, force=False):
        if self.worker and self.worker.isRunning():
            if force is True or self.sender() is self.btn_go:
                self.worker.stop = True
                self.bar.setFormat("Останавливаю после текущего дубля…")
            return
        if len(self.paths) < 2:
            if self.paths and not self.ref:
                self.set_reference(self.paths[0], rerun=False)
            self.bar.setFormat("Нужно минимум две дорожки: эталон и дубль.")
            return
        if force is True or self._dirty_settings:
            self.results.clear()
            self._dirty_settings = False
            self.btn_apply.setText("Применить ко всем")
        todo = [p for p in self.paths if p != self.ref and p not in self.results]
        if self.ref and not todo and self.comp:
            self.bar.setFormat("Всё уже выровнено.")
            return
        self.player.pause()
        self.save_settings()
        if self.session is None:
            self.bar.setFormat("Загружаю движок…")
            QApplication.processEvents()
            self.session = engine().Session()
        for p in todo:
            self.status[p] = "в очереди"
        self.refresh_table()
        self.worker = Worker(self.session, self.ref, list(self.paths), todo, self.settings(),
                             self.results)
        self.worker.progress.connect(self.on_progress)
        self.worker.refChosen.connect(self.on_ref_chosen)
        self.worker.takeDone.connect(self.on_take_done)
        self.worker.takeFailed.connect(self.on_take_failed)
        self.worker.compReady.connect(self.on_comp_ready)
        self.worker.finishedAll.connect(self.on_finished)
        self.btn_go.setText("■  Стоп")
        self.worker.start()

    def on_progress(self, msg, f):
        self.bar.setValue(int(f * 1000))
        self.bar.setFormat(msg)

    def on_ref_chosen(self, ref):
        self.ref = ref
        for p in self.paths:
            self.status[p] = "эталон" if p == ref else "в очереди"
        self.refresh_table()
        self.load_ref_view()
        self.bar.setFormat(f"Эталоном выбран «{os.path.basename(ref)}» — самый типичный дубль. "
                           "Можно сменить звёздочкой.")

    def on_take_done(self, path, res):
        self.results[path] = res
        self.audio_cache.pop(("after", path), None)
        self.status[path] = f"готово · {res.seconds:.0f} с"
        self.refresh_table()
        if self.view_path in (None, path) or self.view_path == self.ref:
            self.view_path = path
            self.show_take(path)

    def on_take_failed(self, path, msg):
        if path:
            self.status[path] = "ошибка"
            self.refresh_table()
        QMessageBox.warning(self, "VoxMatch Простой",
                            (os.path.basename(path) + ":\n" if path else "") + msg)

    def on_comp_ready(self, d):
        if d["ref"] != self.ref:
            return
        tiles = d["tiles"]
        scores = d["scores"]
        cands = [p for p in self.paths if p in self.results
                 and self.results[p].kind == "унисон"]
        extra = [p for p in self.paths if p in self.results and p not in cands and p in scores]
        sung = d["sung"]
        w = np.array([sum(max(0, min(b, sb) - max(a, sa)) for sa, sb in sung) for a, b in tiles],
                     dtype=float)
        cands = [c for c in cands if c in scores]
        if not cands:
            self.comp = None
            self.comp_audio = None
            self.update_comp_view()
            self.update_final_card()
            self.refresh_table()
            self.bar.setFormat("Нет дублей в унисон для финала: эталон в финал не берётся, "
                               "бэки в гармонию и копии эталона тоже.")
            return
        choices = comp_mod().default_choices(scores, cands, len(tiles))
        if self._pending_choices and len(self._pending_choices) == len(tiles):
            for i, name in enumerate(self._pending_choices):
                k = next((p for p in cands if os.path.basename(p) == name), None)
                if k:
                    choices[i] = k
        self._pending_choices = None
        nt = len(tiles)
        tv, pv = [0] * nt, [0] * nt
        tp = self._pending_tp
        if tp and len(tp) == 2 and len(tp[0]) == nt and len(tp[1]) == nt:
            tv, pv = [int(x) for x in tp[0]], [int(x) for x in tp[1]]
        self._pending_tp = None
        self.comp = {"tiles": tiles, "scores": scores, "choices": choices, "auto": list(choices),
                     "cands": [c for c in cands if c in scores] + extra, "fm": d["fm"], "sr": d["sr"],
                     "weights": w, "t": tv, "p": pv, "cache": {}}
        self.sel_phrase = -1
        self.rebuild_comp_audio()
        self.refresh_table()
        self.update_comp_view()
        self.save_settings()

    def rebuild_scores(self):
        if not self.ref or not self.session:
            return
        R = self.session.track(self.ref)
        res = {p: r for p, r in self.results.items() if p in self.paths}
        vc = comp_mod()
        tiles, sung = vc.phrases(R)
        scores = vc.score_all(R, res, tiles, sung) if res else {}
        self.on_comp_ready({"ref": self.ref, "tiles": tiles, "sung": sung, "scores": scores,
                            "fm": R.fm, "sr": R.sr})

    def on_finished(self):
        self.btn_go.setText("▶  Выровнять всё")
        self.bar.setValue(1000)
        n = len(self.results)
        if self.comp:
            self.bar.setFormat(f"Готово: {n} дубл. выровнено, комп собран из лучших фраз. "
                               "Колёсико на фразе — другой вариант.")
        w = self.worker
        self._old = [x for x in self._old if x.isRunning()] + [w]

    # ------------------------------------------------------------------ комп
    def _choices_by_name(self):
        return [REF_KEY if k == REF_KEY else os.path.basename(k) for k in self.comp["choices"]]

    def audio_for(self, key):
        """Звук на шкале эталона (моно float32) для плеера и сборки."""
        if key in self.audio_cache:
            return self.audio_cache[key]
        val = None
        if key == "ref" and self.ref and self.session:
            R = self.session.track(self.ref)
            val = R.data.mean(axis=0).astype(np.float32)
        elif isinstance(key, tuple) and key[0] == "after":
            r = self.results.get(key[1])
            if r is not None:
                val = r.y.mean(axis=0).astype(np.float32)
        elif isinstance(key, tuple) and key[0] == "before":
            r = self.results.get(key[1])
            if r is not None and self.session:
                R = self.session.track(self.ref)
                D = self.session.track(key[1], R.sr)
                x = D.data.mean(axis=0).astype(np.float32)
                lag = int(r.lag_frames) * R.hop                # грубо подвинуть к эталону
                n = R.data.shape[1]
                out = np.zeros(n, np.float32)
                if lag >= 0:
                    seg = x[lag:lag + n]
                    out[:len(seg)] = seg
                else:
                    seg = x[: n + lag]
                    out[-lag:-lag + len(seg)] = seg
                val = out
        self.audio_cache[key] = val
        return val

    def phrase_piece(self, i):
        """Звук фразы i: в глобальном режиме — кусок обработанного дубля,
        в гибком — эта фраза с её собственными таймингом и питчем."""
        c = self.comp
        k = c["choices"][i]
        r = self.results[k]
        if not self.flexible():
            return 0, r.y
        a, b = c["tiles"][i]
        t, p = c["t"][i], c["p"][i]
        key = (k, a, b, t, p, self.chk_form.isChecked())
        hit = c["cache"].get(key)
        if hit is not None:
            return hit
        R = self.session.track(self.ref)
        D = self.session.track(k, R.sr)
        piece = engine().render_phrase(D, R.n, r.lag_frames, r.pos_full, r.corr_c, a, b,
                                       t, p, self.chk_form.isChecked())
        if len(c["cache"]) > 200:
            c["cache"].clear()
        c["cache"][key] = piece
        return piece

    def rebuild_comp_audio(self):
        if not self.comp or not self.session:
            return
        R = self.session.track(self.ref)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pieces = [self.phrase_piece(i) for i in range(len(self.comp["tiles"]))]
            y = comp_mod().build_from_pieces(R, self.comp["tiles"], pieces,
                                             fade_ms=float(self.sl_xfade.s.value()))
        finally:
            QApplication.restoreOverrideCursor()
        self.comp_y = y
        self.comp_audio = y.mean(axis=0).astype(np.float32)
        self.update_player_sources()
        self.refresh_waves()
        self.final_saved = None
        self.save_timer.start(700)
        self.update_final_card()

    def refresh_waves(self):
        """Оранжевая — эталон, зелёная поверх — финал, собранный из фраз."""
        if not self.ref or not self.session:
            return
        R = self.session.track(self.ref)
        if self.tl.ref_env is None or abs(self.tl.dur - R.data.shape[1] / R.sr) > 0.01:
            self.tl.set_ref(R.data.mean(axis=0), R.sr)
        probs = []
        if self.comp:
            fm = R.fm / 1000.0
            for i, (a, b) in enumerate(self.comp["tiles"]):
                r = self.results.get(self.comp["choices"][i])
                if r is None or r.dissim is None:
                    continue
                ve = engine()
                bad = ve._smooth(((r.dissim[a:b] > 1.6) & R.loud[a:b]).astype(np.float64),
                                 int(round(80.0 / R.fm))) > 0.5          # не то спето / не тот звук
                for pa, pb in ve._segments(bad):
                    probs.append(((a + pa) * fm, (a + pb) * fm))
        self.tl.set_take(self.comp_audio, None, R.sr, "Финал (собран из фраз)", "", probs)

    def update_comp_view(self):
        if not self.comp:
            self.tl.set_comp([], [])
            return
        fm = self.comp["fm"] / 1000.0
        tiles = [(a * fm, b * fm) for a, b in self.comp["tiles"]]
        blocks = []
        flex = self.flexible()
        for i, k in enumerate(self.comp["choices"]):
            sc, note = self.comp["scores"][k][i]
            lab = self.label(k)
            if flex:
                t, p = self.comp["t"][i], self.comp["p"][i]
                lab += f" · T{t} P{p}" if (t or p) else " · как спето"
            blocks.append((lab, sc, note, False))
        self.tl.set_comp(tiles, blocks)
        self.tl.active_phrase = self.sel_phrase

    def set_choice(self, i, key, play=True):
        if not self.comp or self.comp["choices"][i] == key:
            return
        self.comp["choices"][i] = key
        self.rebuild_comp_audio()
        self.update_comp_view()
        self.save_settings()
        lab = self.label(key)
        sc, note = self.comp["scores"][key][i]
        self.bar.setFormat(f"Фраза {i + 1}: {lab} — {os.path.basename(self.ref if key == REF_KEY else key)}"
                           f" · оценка {sc:.0f}" + (f" · {note}" if note else ""))
        if key != REF_KEY and key in self.paths:                # показать волны этого дубля
            self.view_path = key
            self.table.blockSignals(True)
            self.table.selectRow(self.paths.index(key))
            self.table.blockSignals(False)
            self.update_player_sources()
        if play:
            self.set_source("comp")
            self.loop_phrase(i, restart=True)

    def wheel_choice(self, i, step):
        if not self.comp:
            return
        order = comp_mod().ranked(self.comp["scores"], self.comp["cands"], i)
        cur = self.comp["choices"][i]
        j = order.index(cur) if cur in order else 0
        j = max(0, min(len(order) - 1, j - step))        # вверх — к лучшему
        self.set_choice(i, order[j])

    def choice_menu(self, i, gpos):
        if not self.comp:
            return
        m = QMenu(self)
        order = comp_mod().ranked(self.comp["scores"], self.comp["cands"], i)
        for k in order:
            sc, note = self.comp["scores"][k][i]
            name = os.path.basename(self.ref if k == REF_KEY else k)
            a = QAction(f"{self.label(k)}  {sc:.0f}   {name}" + (f"  — {note}" if note else ""), m)
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
        if not self.comp:
            return -1
        return self.tl.phrase_at(self.player.seconds())

    def step_choice(self, d):
        i = self.current_phrase()
        if i >= 0:
            self.wheel_choice(i, d)

    def step_phrase(self, d):
        if not self.comp:
            return
        i = self.current_phrase()
        j = max(0, min(len(self.comp["tiles"]) - 1, (i if i >= 0 else 0) + d))
        self.loop_phrase(j, restart=True)

    def loop_phrase(self, i, restart=True):
        if not self.comp:
            return
        a, b = self.tl.tiles[i]
        self.tl.loop = (a, b)
        self.tl.active_phrase = i
        self.sel_phrase = i
        self.update_phrase_panel()
        self.on_loop((a, b))
        if not self.chk_loop.isChecked():
            self.chk_loop.setChecked(True)
        if restart:
            self.player.seek(a)
        self.player.play()

    # ------------------------------------------------------------------ панель фразы
    def update_phrase_panel(self):
        i = self.sel_phrase
        flex = self.flexible()
        ok = bool(self.comp) and 0 <= i < len(self.comp["tiles"])
        self._ph_updating = True
        try:
            for w in (self.ph_t, self.ph_p):
                w.setEnabled(ok and flex)
            self.ph_all.setEnabled(ok and flex)
            self.ph_reset.setEnabled(ok and flex)
            self.ph_take.setEnabled(ok)
            self.ph_take.clear()
            if not ok:
                self.ph_title.setText("Кликните фразу в ленте сверху")
                self.ph_hint.setText("" if flex else
                                     "Глобальный режим: тайминг и питч общие (⚙ Настройки). "
                                     "Чтобы править каждую фразу отдельно — «Режим: Гибкий».")
                self.ph_t.s.setValue(0)
                self.ph_p.s.setValue(0)
                return
            fm = self.comp["fm"] / 1000.0
            a, b = self.comp["tiles"][i]
            self.ph_title.setText(f"ФРАЗА {i + 1} из {len(self.comp['tiles'])}  ·  "
                                  f"{fmt_t(a * fm)} – {fmt_t(b * fm)}")
            order = comp_mod().ranked(self.comp["scores"], self.comp["cands"], i)
            for k in order:
                sc, note = self.comp["scores"][k][i]
                kd = self.results[k].kind if k in self.results else ""
                tag = f"  [{kd}]" if kd and kd != "унисон" else ""
                self.ph_take.addItem(f"{self.label(k)}  ·  {sc:.0f}  ·  {os.path.basename(k)}{tag}"
                                     + (f"  —  {note}" if note else ""), k)
            cur = self.comp["choices"][i]
            self.ph_take.setCurrentIndex(max(0, order.index(cur) if cur in order else 0))
            self.ph_t.s.setValue(int(self.comp["t"][i]))
            self.ph_p.s.setValue(int(self.comp["p"][i]))
            self.ph_hint.setText(
                "Ползунки меняют только эту фразу — она пересчитается и зазвучит по кругу."
                if flex else "Глобальный режим: здесь можно сменить дубль. Тайминг и питч — общие "
                             "(⚙). Для правки по фразам — «Режим: Гибкий».")
        finally:
            self._ph_updating = False

    def phrase_slider_changed(self, _v):
        if self._ph_updating or not self.comp or self.sel_phrase < 0 or not self.flexible():
            return
        i = self.sel_phrase
        self.comp["t"][i] = self.ph_t.s.value()
        self.comp["p"][i] = self.ph_p.s.value()
        self.update_comp_view()
        self.ph_timer.start(350)

    def apply_phrase_edit(self):
        if not self.comp:
            return
        self.rebuild_comp_audio()
        self.save_settings()
        if self.sel_phrase >= 0:
            self.set_source("comp")
            if not self.player.playing:
                self.loop_phrase(self.sel_phrase, restart=True)

    def phrase_take_changed(self, idx):
        if self._ph_updating or not self.comp or self.sel_phrase < 0:
            return
        k = self.ph_take.itemData(idx)
        if k:
            self.set_choice(self.sel_phrase, k)

    def phrase_to_all(self):
        if not self.comp or self.sel_phrase < 0:
            return
        t, p = self.comp["t"][self.sel_phrase], self.comp["p"][self.sel_phrase]
        n = len(self.comp["tiles"])
        self.comp["t"], self.comp["p"] = [t] * n, [p] * n
        self.update_comp_view()
        self.bar.setFormat(f"Тайминг {t} % и питч {p} % — на все фразы. Пересчитываю…")
        QApplication.processEvents()
        self.rebuild_comp_audio()
        self.save_settings()

    def phrase_reset(self):
        if not self.comp or self.sel_phrase < 0:
            return
        self.ph_t.s.setValue(0)
        self.ph_p.s.setValue(0)

    def comp_path(self):
        base = os.path.splitext(os.path.basename(self.ref))[0]
        d = os.path.join(os.path.dirname(self.ref), "VoxMatch Простой")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{base} — КОМП.wav")

    def export_comp_paths(self):
        if not self.comp:
            return []
        p = self.comp_path()
        sf.write(p, self.comp_y.T, self.comp["sr"], subtype="PCM_24")
        return [p]

    def autosave_final(self):
        try:
            if self.comp:
                self.export_comp_paths()
                self.final_saved = time.strftime("%H:%M:%S")
                self.update_final_card()
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())

    def update_final_card(self):
        fc = self.final
        if not self.comp or self.comp_audio is None:
            fc.info.setText("Скиньте дорожки — финал соберётся сам из лучших фраз.")
            fc.wave.set_data(None, [])
            return
        from collections import Counter
        cnt = Counter(self.label(k) for k in self.comp["choices"])
        parts = ", ".join(f"{k} ×{v}" for k, v in sorted(cnt.items()))
        n = len(self.comp["tiles"])
        changed = sum(1 for a, b in zip(self.comp["choices"], self.comp["auto"]) if a != b)
        sc = [self.comp["scores"][k][i][0] for i, k in enumerate(self.comp["choices"])]

        avg = float(np.mean(sc)) if sc else 0
        saved = (f"сохранён {self.final_saved}" if self.final_saved else "сохраняю…")
        name = os.path.basename(self.comp_path()) if self.ref else ""
        fc.info.setText(f"{n} фраз: {parts} · средняя оценка {avg:.0f}"
                        + (f" · ваших замен: {changed}" if changed else "")
                        + f"\n{saved} · {name}")
        total = self.comp["tiles"][-1][1] if self.comp["tiles"] else 1
        blocks = [(a / total, b / total, sc[i], self.label(self.comp["choices"][i]))
                  for i, (a, b) in enumerate(self.comp["tiles"])]
        step = max(1, len(self.comp_audio) // 4000)
        fc.wave.set_data(env_peaks(self.comp_audio, step), blocks)

    def play_final(self):
        if self.comp_audio is None:
            return
        self.chk_loop.setChecked(False)
        self.set_source("comp")
        self.player.seek(0)
        self.player.play()

    def show_final_file(self):
        if not self.comp:
            return
        p = self.export_comp_paths()[0]
        import subprocess
        try:
            subprocess.Popen(["open", "-R", p])
        except Exception:  # noqa: BLE001
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(p)))

    def export_comp(self):
        ps = self.export_comp_paths()
        if ps:
            self.bar.setFormat(f"Комп сохранён: {ps[0]}")
        else:
            self.bar.setFormat("Комп появится после выравнивания.")

    # ------------------------------------------------------------------ просмотр
    def load_ref_view(self):
        if not self.ref:
            return
        try:
            data, sr = sf.read(self.ref, always_2d=True, dtype="float32")
            self.tl.set_ref(data.mean(axis=1), sr)
            self.tl.set_take(None, None, sr, "Финал (собран из фраз)", "", [])
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())

    def on_select(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows:
            return
        p = self.paths[rows[0]]
        self.view_path = p
        if p in self.results:
            self.show_take(p)
        elif p == self.ref:
            self.load_ref_view()

    def show_take(self, path):
        r = self.results.get(path)
        if r is None or not self.session:
            return
        R = self.session.track(self.ref)
        if self.tl.ref_env is None or abs(self.tl.dur - R.data.shape[1] / R.sr) > 0.01:
            self.tl.set_ref(R.data.mean(axis=0), R.sr)
        if self.comp and self.comp_audio is not None:      # волны — всегда финал
            self.refresh_waves()
        else:                                               # пока финала нет — этот дубль
            fm = R.fm / 1000.0
            probs = ([(a * fm, b * fm) for a, b in engine()._segments(r.problem)]
                     if r.problem is not None else [])
            self.tl.set_take(self.audio_for(("after", path)), None, R.sr,
                             f"{self.label(path)} после выравнивания", "", probs)
        self.update_player_sources()

    def update_player_sources(self):
        if not self.ref or not self.session:
            return
        R = self.session.track(self.ref)
        p = self.view_path if self.view_path in self.results else None
        src = {"ref": self.audio_for("ref"),
               "result": self.audio_for(("after", p)) if p else None,
               "before": self.audio_for(("before", p)) if p else None,
               "comp": self.comp_audio}
        self.player.set_sources(src, R.sr)

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
            if _engine is not None:
                _engine.shutdown_workers()
        except Exception:  # noqa: BLE001
            pass
        e.accept()
        if any(w.isRunning() for w in running):     # не ждём долгий расчёт — просто выходим
            os._exit(0)


STYLE = f"""
QWidget {{ background: {BG}; color: #e6e6e6; font-size: 13px; }}
QFrame#drop {{ border: 2px dashed #4a5056; border-radius: 10px; background: {PANEL}; }}
QFrame#drop[hover="true"] {{ border-color: {ACCENT}; background: #22312f; }}
QLabel#dropTitle {{ font-size: 14px; font-weight: 600; background: transparent; }}
QLabel#hint {{ color: #8b949b; background: transparent; font-size: 12px; }}
QLabel#cap {{ color: #8b949b; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel#time {{ font-family: Menlo, monospace; font-size: 13px; color: #cfd5da; }}
QLabel#chip {{ background: #2c3a39; border: 1px solid {ACCENT}; border-radius: 6px; padding: 6px 12px; color: {ACCENT}; font-weight: 600; }}
QFrame#panel {{ background: {PANEL}; border-radius: 10px; }}
QFrame#final {{ background: #1f2b2a; border: 1px solid #2f6f68; border-radius: 10px; }}
QFrame#final QLabel {{ background: transparent; }}
QLabel#finalTitle {{ color: {ACCENT}; font-size: 13px; font-weight: 800; letter-spacing: 1px; }}
QFrame#final QLabel#bigchip {{ background: {ACCENT}; color: #0e1a19; border-radius: 6px; padding: 9px 14px; font-weight: 800; }}
QFrame#panel QLabel, QFrame#panel QCheckBox {{ background: transparent; }}
QPushButton, QToolButton {{ background: #2d3236; border: 1px solid #3a4046; border-radius: 6px; padding: 7px 12px; }}
QPushButton:hover, QToolButton:hover {{ background: #363c41; }}
QToolButton:checked {{ background: #2f4a47; border-color: {ACCENT}; }}
QPushButton#go {{ background: {ACCENT}; color: #0e1a19; font-size: 14px; font-weight: 700; padding: 10px 18px; border: none; }}
QPushButton#go:hover {{ background: #52c7b9; }}
QPushButton#play {{ background: {ACCENT}; color: #0e1a19; font-size: 16px; font-weight: 700; min-width: 44px; border: none; }}
QPushButton#seg {{ padding: 6px 10px; }}
QPushButton#seg:checked {{ background: #2f4a47; border-color: {ACCENT}; color: #ffffff; }}
QProgressBar {{ background: {PANEL}; border: none; border-radius: 4px; height: 20px; text-align: center; color: #cfd5da; }}
QProgressBar::chunk {{ background: #2f6f68; border-radius: 4px; }}
QTableWidget {{ background: {PANEL}; border: none; border-radius: 8px; alternate-background-color: #262a2e; }}
QHeaderView::section {{ background: #2a2e33; border: none; padding: 5px; color: #8b949b; }}
QTableWidget::item:selected {{ background: #2f4a47; color: #ffffff; }}
QSlider::groove:horizontal {{ height: 4px; background: #3c4248; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QComboBox {{ background: #2d3236; border: 1px solid #3a4046; border-radius: 6px; padding: 4px 8px; }}
QMenu {{ background: #2a2e33; border: 1px solid #3a4046; }}
QMenu::item:selected {{ background: #2f4a47; }}
QSplitter::handle {{ background: {BG}; width: 8px; }}
QToolTip {{ background: #2a2e33; color: #e6e6e6; border: 1px solid #3a4046; }}
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:  # noqa: BLE001
        log_error(traceback.format_exc())
        raise
