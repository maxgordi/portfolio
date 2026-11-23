#!/usr/bin/env python3
"""
Букви — окно ручной правки проглоченных букв (Suno → Kits).

Программа (diction_fix.py) сама находит проглоченные согласные и вставляет их, а здесь каждую
вставку можно послушать и подправить руками: включить/выключить, сделать громче или тише, шире
или уже, плавнее, сдвинуть, выбрать другого донора или режим (вставка / поверх / замена).
Результат собирается заново из оригинала за пару секунд и сохраняется в WAV + MP3.

Запуск:  .venv/bin/python bukvi_app.py   (или двойной клик по start_mac.command)
"""
from __future__ import annotations

import copy
import re
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QMimeData, QProcess, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDrag, QFont, QKeySequence, QPainter, QPalette, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QSlider, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

try:
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
import diction_fix as dfx  # noqa: E402  (лёгкий импорт: нейросеть грузится только при проверке)

LOG_PATH = os.path.join(APP_DIR, "bukvi_error.log")
AUDIO_EXT = {".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".m4a"}
AUDIO_FILTER = "Аудио (*.wav *.wave *.aif *.aiff *.flac *.ogg *.mp3 *.m4a)"

ACCENT = "#3fb6a8"
COL_ORIG, COL_OUT = "#e8a33d", ACCENT
COL_ON, COL_OFF, COL_FAIL = "#3fb6a8", "#e8a33d", "#e05d5d"

MODES = [("insert", "Вставка — раздвинуть слово"),
         ("overlay", "Поверх — подмешать к оригиналу"),
         ("replace", "Замена — приглушить оригинал")]
MODE_SHORT = {"insert": "вставка", "overlay": "поверх", "replace": "замена"}
EDITABLE = ("enabled", "mode", "donor", "gain_offset_db", "width_ms", "fade_ms", "shift_ms", "pos")


def log_error(text: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
    except OSError:
        pass


def mark_word(word: str, i: int) -> str:
    return word[:i] + word[i].upper() + word[i + 1:]


def fmt_time(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def column_minmax(x: np.ndarray, cols: int):
    """Огибающая для рисования: минимум и максимум в каждом столбце."""
    cols = max(1, cols)
    if len(x) == 0:
        z = np.zeros(cols, dtype=np.float32)
        return z, z
    step = max(1, len(x) // cols)
    n = (len(x) // step) * step
    if n == 0:
        return x[:1].repeat(cols), x[:1].repeat(cols)
    r = x[:n].reshape(-1, step)
    return r.min(axis=1), r.max(axis=1)


# ---------------------------------------------------------------- плеер

class Player:
    """Две дорожки одной длины (оригинал и результат) на одной шкале: переключение A/B на лету."""

    def __init__(self):
        self.sr = 44100
        self.tracks = {}
        self.track = "out"
        self.pos = 0
        self.end = None
        self.playing = False
        self._stream = None

    @property
    def length(self) -> int:
        return max((len(t) for t in self.tracks.values()), default=0)

    def set_track(self, name, data, sr):
        self.sr = sr
        self.tracks[name] = np.ascontiguousarray(data, dtype=np.float32)

    def _callback(self, outdata, frames, _time, _status):
        t = self.tracks.get(self.track)
        pos = self.pos
        stop = self.end if self.end is not None else self.length
        n = max(0, min(frames, stop - pos))
        outdata[:] = 0
        if t is not None and n:
            seg = t[pos:pos + n]
            outdata[:len(seg), :seg.shape[1]] = seg
        self.pos = pos + frames
        if self.pos >= stop:
            self.pos = min(stop, self.length)
            self.playing = False
            raise sd.CallbackStop

    def play(self, start=None, end=None):
        if sd is None or not self.tracks:
            return
        self._close()
        if start is not None:
            self.pos = int(max(0, start) * self.sr)
        self.end = int(end * self.sr) if end is not None else None
        if self.pos >= (self.end or self.length):
            self.pos = 0
        ch = next(iter(self.tracks.values())).shape[1]
        self._stream = sd.OutputStream(samplerate=self.sr, channels=ch, dtype="float32", blocksize=1024,
                                       callback=self._callback)
        self.playing = True
        self._stream.start()

    def pause(self):
        self.playing = False
        self._close()

    def _close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    def seek(self, seconds):
        self.pos = int(max(0.0, min(seconds * self.sr, self.length)))

    @property
    def time(self) -> float:
        return self.pos / self.sr if self.sr else 0.0


# ---------------------------------------------------------------- рисование

class Overview(QWidget):
    """Вся песня: волна результата и метки всех найденных букв. Клик по метке — выбрать, по волне — перемотать."""
    markerClicked = Signal(int)
    seekRequested = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(96)
        self._data = None
        self._dur = 0.0
        self._marks = []  # (время, цвет, индекс правки)
        self._sel = None
        self._cursor = None
        self._pix = None

    def set_audio(self, data, sr):
        self._data = data.mean(axis=1) if data is not None else None
        self._dur = len(data) / sr if data is not None else 0.0
        self._pix = None
        self.update()

    def set_marks(self, marks, sel):
        self._marks, self._sel = marks, sel
        self.update()

    def set_cursor(self, t):
        self._cursor = t
        self.update()

    def _x(self, t):
        return 8 + (self.width() - 16) * t / max(self._dur, 1e-6)

    def resizeEvent(self, e):
        self._pix = None
        super().resizeEvent(e)

    def mousePressEvent(self, e):
        if self._dur <= 0:
            return
        x = e.position().x()
        near = [(abs(self._x(t) - x), i) for t, _, i in self._marks]
        if near and min(near)[0] <= 6 and e.position().y() < 22:
            self.markerClicked.emit(min(near)[1])
            return
        self.seekRequested.emit(max(0.0, min(1.0, (x - 8) / max(1.0, self.width() - 16))) * self._dur)

    def _render(self):
        w, h = max(1, self.width()), max(1, self.height())
        pix = QPixmap(w, h)
        pix.fill(QColor("#1a1d22"))
        if self._data is not None:
            p = QPainter(pix)
            lo, hi = column_minmax(self._data, w - 16)
            scale = (h - 30) / 2 / max(float(np.abs(self._data).max()), 1e-6)
            mid = 22 + (h - 22) / 2
            p.setPen(QPen(QColor(COL_OUT), 1))
            for c in range(len(lo)):
                p.drawLine(8 + c, int(mid - hi[c] * scale), 8 + c, int(mid - lo[c] * scale))
            p.end()
        return pix

    def paintEvent(self, e):
        if self._pix is None or self._pix.size() != self.size():
            self._pix = self._render()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._pix)
        for t, color, i in self._marks:
            x = int(self._x(t))
            sel = i == self._sel
            p.setPen(QPen(QColor("#ffffff" if sel else color), 3 if sel else 2))
            p.drawLine(x, 4, x, 20)
            if sel:
                p.setPen(QPen(QColor(255, 255, 255, 70), 1))
                p.drawLine(x, 20, x, self.height())
        if self._cursor is not None and self._dur > 0:
            x = int(self._x(self._cursor))
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawLine(x, 0, x, self.height())
        p.end()


class ZoomView(QWidget):
    """Крупно: слово с соседями — оригинал сверху, результат снизу; подсвечено место буквы.
    Клик — послушать результат с этого места."""
    playFrom = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(170)
        self._orig = self._out = None
        self._sr = 44100
        self._a = self._b = 0.0
        self._hl = None
        self._cursor = None
        self._pix = None

    def set_view(self, orig, out, sr, a, b, hl):
        self._orig, self._out, self._sr, self._a, self._b, self._hl = orig, out, sr, a, b, hl
        self._pix = None
        self.update()

    def set_cursor(self, t):
        self._cursor = t
        self.update()

    def _x(self, t):
        return 70 + (self.width() - 78) * (t - self._a) / max(self._b - self._a, 1e-6)

    def resizeEvent(self, e):
        self._pix = None
        super().resizeEvent(e)

    def mousePressEvent(self, e):
        if self._b > self._a:
            frac = (e.position().x() - 70) / max(1.0, self.width() - 78)
            self.playFrom.emit(self._a + max(0.0, min(1.0, frac)) * (self._b - self._a))

    def _render(self):
        w, h = max(1, self.width()), max(1, self.height())
        pix = QPixmap(w, h)
        pix.fill(QColor("#1a1d22"))
        p = QPainter(pix)
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        row_h = h / 2
        if self._hl:
            xa, xb = self._x(self._hl[0]), self._x(self._hl[1])
            p.fillRect(int(xa), 0, max(2, int(xb - xa)), h, QColor(63, 182, 168, 40))
        peak = 1e-6
        for x in (self._orig, self._out):
            if x is not None:
                a, b = int(self._a * self._sr), int(self._b * self._sr)
                peak = max(peak, float(np.abs(x[a:b]).max()) if b > a else 0.0)
        for r, (name, x, color) in enumerate((("Оригинал", self._orig, COL_ORIG), ("Результат", self._out, COL_OUT))):
            y0 = r * row_h
            mid = y0 + row_h / 2
            p.setPen(QColor("#8a94a1"))
            p.drawText(8, int(mid + 4), name)
            if x is None:
                continue
            a, b = int(self._a * self._sr), int(self._b * self._sr)
            seg = x[max(0, a):b].mean(axis=1)
            lo, hi = column_minmax(seg, w - 78)
            scale = (row_h / 2 - 4) / peak
            p.setPen(QPen(QColor(color), 1))
            for c in range(len(lo)):
                p.drawLine(70 + c, int(mid - hi[c] * scale), 70 + c, int(mid - lo[c] * scale))
        p.setPen(QColor("#2b3038"))
        p.drawLine(0, int(row_h), w, int(row_h))
        p.end()
        return pix

    def paintEvent(self, e):
        if self._pix is None or self._pix.size() != self.size():
            self._pix = self._render()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._pix)
        if self._cursor is not None and self._a <= self._cursor <= self._b:
            x = int(self._x(self._cursor))
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawLine(x, 0, x, self.height())
        p.end()


class DropArea(QFrame):
    """Зона, куда можно бросить вокал, текст или проект."""
    dropped = Signal(list)

    def __init__(self, text):
        super().__init__()
        self.setObjectName("dropArea")
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        self.label = QLabel(text)
        self.label.setObjectName("hint")
        self.label.setWordWrap(True)
        lay.addWidget(self.label)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setProperty("hover", True)
            self.style().polish(self)

    def dragLeaveEvent(self, e):
        self.setProperty("hover", False)
        self.style().polish(self)

    def dropEvent(self, e):
        self.setProperty("hover", False)
        self.style().polish(self)
        self.dropped.emit([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])


class DragChip(QLabel):
    """Плашка, которую можно перетащить в Cubase — несёт сохранённый результат."""

    def __init__(self, paths_for_drag):
        super().__init__("⇱  Перетащите результат в Cubase")
        self.setObjectName("chip")
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.OpenHandCursor)
        self._paths_for_drag = paths_for_drag
        self._press = None

    def mousePressEvent(self, e):
        self._press = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._press is None:
            return
        if (e.position().toPoint() - self._press).manhattanLength() < QApplication.startDragDistance():
            return
        self._press = None
        paths = self._paths_for_drag()
        if not paths:
            return
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class SliderRow:
    """Ползунок с подписью и значением; scale — во сколько раз значение мельче шага ползунка."""

    def __init__(self, grid, row, title, lo, hi, unit, tip, scale=1, fmt="{:+g}"):
        self.scale, self.fmt, self.unit = scale, fmt, unit
        lab = QLabel(title)
        lab.setToolTip(tip)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(lo * scale), int(hi * scale))
        self.slider.setToolTip(tip)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(64)
        self.slider.valueChanged.connect(self._show)
        grid.addWidget(lab, row, 0)
        grid.addWidget(self.slider, row, 1)
        grid.addWidget(self.value_label, row, 2)

    def _show(self, v):
        self.value_label.setText(self.fmt.format(v / self.scale) + " " + self.unit)

    def value(self):
        return self.slider.value() / self.scale

    def set(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * self.scale)))
        self.slider.blockSignals(False)
        self._show(self.slider.value())


# ---------------------------------------------------------------- фоновые задачи

class RenderWorker(QThread):
    """Собирает исправленную дорожку из оригинала по проекту (с ручными правками)."""
    done = Signal(int, object, object)

    def __init__(self, gen, audio, proj):
        super().__init__()
        self.gen, self.audio, self.proj = gen, audio, proj

    def run(self):
        try:
            buf, notes = dfx.render_project(self.audio, self.proj)
            self.done.emit(self.gen, buf, notes)
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())
            self.done.emit(self.gen, None, traceback.format_exc())


_MODEL = {}


class CheckWorker(QThread):
    """«Слушает» нейросетью слово с соседями в текущем результате: насколько слышна буква и соседние."""
    done = Signal(int, object)

    def __init__(self, idx, buf, proj):
        super().__init__()
        self.idx, self.buf, self.proj = idx, buf, proj

    def run(self):
        try:
            mid = self.proj.get("model", dfx.DEFAULT_MODEL)
            if mid not in _MODEL:
                _MODEL[mid] = dfx.LetterModel(mid)
            F = self.proj["fixes"][self.idx]
            words = [tuple(w) for w in self.proj["words"]]
            heard = dfx.listen_words(_MODEL[mid], self.buf, self.proj["sr"], words, F["ws"], F["t_lo"], F["t_hi"])
            self.done.emit(self.idx, heard)
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())
            self.done.emit(self.idx, traceback.format_exc())


# ---------------------------------------------------------------- окно

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Букви — ручная правка букв")
        self.settings_store = QSettings("Bukvi", "Bukvi")
        self.vocal_path = None
        self.lyrics_path = None
        self.proj = None
        self.proj_path = None
        self.audio = None
        self.out = None
        self.notes = {}
        self.sel = None
        self.checks = {}  # индекс правки → результат проверки моделью
        self.render_gen = 0
        self.rendered_gen = -1
        self.workers = []
        self.export_paths = []
        self.player = Player()
        self.proc = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 10)

        # --- шапка: файлы и автоматический поиск
        top = QHBoxLayout()
        title = QLabel("Букви")
        title.setObjectName("appTitle")
        top.addWidget(title)
        self.drop = DropArea("Перетащите сюда вокал (WAV/MP3) и текст песни (.txt) — или готовый проект (*_project.json)")
        self.drop.dropped.connect(self.on_files)
        self.drop.setMaximumHeight(90)
        top.addWidget(self.drop, 1)
        col = QVBoxLayout()
        b_vocal = QPushButton("Вокал…")
        b_vocal.clicked.connect(lambda: self.pick("vocal"))
        b_lyr = QPushButton("Текст…")
        b_lyr.clicked.connect(lambda: self.pick("lyrics"))
        b_proj = QPushButton("Открыть проект…")
        b_proj.clicked.connect(lambda: self.pick("project"))
        for b in (b_vocal, b_lyr, b_proj):
            col.addWidget(b)
        top.addLayout(col)
        auto = QGroupBox("Автоматический поиск")
        ag = QGridLayout(auto)
        self.auto_mode = QComboBox()
        self.auto_mode.addItems(["Вставка (как раньше)", "Всё поверх оригинала", "Вставка, а «т» поверх"])
        ag.addWidget(QLabel("Режим:"), 0, 0)
        ag.addWidget(self.auto_mode, 0, 1)
        self.auto_thr = QDoubleSpinBox()
        self.auto_thr.setRange(0.1, 0.95)
        self.auto_thr.setSingleStep(0.05)
        self.auto_thr.setValue(0.5)
        self.auto_thr.setToolTip("Буква слабее этой уверенности считается проглоченной. Больше — найдёт больше мест.")
        ag.addWidget(QLabel("Порог:"), 1, 0)
        ag.addWidget(self.auto_thr, 1, 1)
        self.b_auto = QPushButton("Найти буквы автоматически")
        self.b_auto.setObjectName("primarySmall")
        self.b_auto.clicked.connect(self.run_auto)
        ag.addWidget(self.b_auto, 2, 0, 1, 2)
        top.addWidget(auto)
        root.addLayout(top)
        self.files_label = QLabel("Файлы не выбраны")
        self.files_label.setObjectName("hint")
        root.addWidget(self.files_label)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # --- вся песня
        self.overview = Overview()
        self.overview.markerClicked.connect(self.select_fix)
        self.overview.seekRequested.connect(self.seek)
        root.addWidget(self.overview)

        # --- список букв и редактор
        split = QSplitter(Qt.Horizontal)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Вкл", "Время", "Слово", "Буква", "Было", "Стало", "Режим", "Статус"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        for c in range(7):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(7, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.on_table_select)
        self.table.cellClicked.connect(self.on_cell_clicked)
        split.addWidget(self.table)

        ed = QWidget()
        el = QVBoxLayout(ed)
        el.setContentsMargins(8, 0, 0, 0)
        self.ed_title = QLabel("Выберите букву в списке или метку на волне")
        self.ed_title.setObjectName("sectionTitle")
        el.addWidget(self.ed_title)
        self.zoom = ZoomView()
        self.zoom.playFrom.connect(lambda t: self.play_word("out", t))
        el.addWidget(self.zoom)
        pb = QHBoxLayout()
        self.b_play_orig = QPushButton("▶ Оригинал слова")
        self.b_play_orig.clicked.connect(lambda: self.play_word("orig"))
        self.b_play_out = QPushButton("▶ Результат слова")
        self.b_play_out.setObjectName("primarySmall")
        self.b_play_out.clicked.connect(lambda: self.play_word("out"))
        self.b_check = QPushButton("Проверить нейросетью")
        self.b_check.setToolTip("Нейросеть «послушает» слово с соседями в текущем результате:\n"
                                "насколько чётко слышна буква и не испортились ли соседние.")
        self.b_check.clicked.connect(self.check_selected)
        for b in (self.b_play_orig, self.b_play_out, self.b_check):
            pb.addWidget(b)
        el.addLayout(pb)

        box = QGroupBox("Настройка буквы")
        g = QGridLayout(box)
        self.c_enabled = QCheckBox("Вставлять эту букву")
        self.c_enabled.toggled.connect(lambda v: self.edit("enabled", v))
        g.addWidget(self.c_enabled, 0, 0, 1, 3)
        g.addWidget(QLabel("Режим"), 1, 0)
        self.c_mode = QComboBox()
        for key, name in MODES:
            self.c_mode.addItem(name, key)
        self.c_mode.setToolTip("Вставка — слово раздвигается, время забирается из соседней гласной.\n"
                               "Поверх — буква плавно подмешивается к оригиналу, ничего не сдвигается.\n"
                               "Замена — оригинал под буквой приглушается.")
        self.c_mode.currentIndexChanged.connect(lambda _: self.edit("mode", self.c_mode.currentData()))
        g.addWidget(self.c_mode, 1, 1, 1, 2)
        g.addWidget(QLabel("Донор"), 2, 0)
        self.c_donor = QComboBox()
        self.c_donor.setToolTip("Откуда взята чёткая буква. Разные доноры звучат по-разному — пробуйте.")
        self.c_donor.currentIndexChanged.connect(lambda i: self.edit("donor", i) if i >= 0 else None)
        g.addWidget(self.c_donor, 2, 1, 1, 2)
        self.s_gain = SliderRow(g, 3, "Громкость", -18, 9, "дБ",
                                "Громкость вставки относительно автоматической (0 = как подобрала программа).", 2)
        self.s_width = SliderRow(g, 4, "Ширина", -30, 60, "мс",
                                 "Шире — берётся больше звука вокруг буквы-донора; уже — только сама буква.")
        self.s_fade = SliderRow(g, 5, "Плавность", 10, 80, "мс",
                                "Длина плавных переходов на стыках (как X в Cubase). Больше — мягче.", fmt="{:g}")
        self.s_shift = SliderRow(g, 6, "Сдвиг", -80, 80, "мс", "Сдвинуть вставку раньше (−) или позже (+).")
        self.s_pos = SliderRow(g, 7, "Место в букве", 0, 100, "%",
                               "Куда ставить: 0% — к началу проглоченной буквы, 100% — к концу.", fmt="{:g}")
        self.s_gain.slider.valueChanged.connect(lambda _: self.edit("gain_offset_db", self.s_gain.value()))
        self.s_width.slider.valueChanged.connect(lambda _: self.edit("width_ms", self.s_width.value()))
        self.s_fade.slider.valueChanged.connect(lambda _: self.edit("fade_ms", self.s_fade.value()))
        self.s_shift.slider.valueChanged.connect(lambda _: self.edit("shift_ms", self.s_shift.value()))
        self.s_pos.slider.valueChanged.connect(lambda _: self.edit("pos", self.s_pos.value() / 100))
        self.b_reset = QPushButton("↺ Вернуть как было автоматически")
        self.b_reset.clicked.connect(self.reset_selected)
        g.addWidget(self.b_reset, 8, 0, 1, 3)
        el.addWidget(box)
        self.ed_info = QLabel("")
        self.ed_info.setObjectName("hint")
        self.ed_info.setWordWrap(True)
        el.addWidget(self.ed_info)
        el.addStretch(1)
        split.addWidget(ed)
        split.setSizes([620, 560])
        root.addWidget(split, 1)

        # --- низ: плеер и сохранение
        bot = QHBoxLayout()
        self.b_play = QPushButton("▶")
        self.b_play.setObjectName("playBtn")
        self.b_play.setToolTip("Играть / пауза всей песни (пробел)")
        self.b_play.clicked.connect(self.toggle_play)
        bot.addWidget(self.b_play)
        bot.addWidget(QLabel("Слушать:"))
        self.r_out = QRadioButton("Результат")
        self.r_orig = QRadioButton("Оригинал")
        self.r_out.setChecked(True)
        self.r_out.setToolTip("Переключать можно прямо во время воспроизведения (клавиша A)")
        self.r_out.toggled.connect(lambda v: setattr(self.player, "track", "out" if v else "orig"))
        bot.addWidget(self.r_out)
        bot.addWidget(self.r_orig)
        self.time_label = QLabel("0:00.00")
        bot.addWidget(self.time_label)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        bot.addWidget(self.status, 1)
        self.b_save = QPushButton("Сохранить WAV + MP3")
        self.b_save.setObjectName("primary")
        self.b_save.clicked.connect(self.export)
        bot.addWidget(self.b_save)
        self.chip = DragChip(self.paths_for_drag)
        bot.addWidget(self.chip)
        root.addLayout(bot)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence(Qt.Key_A), self, activated=lambda: (self.r_orig if self.r_out.isChecked()
                                                                    else self.r_out).setChecked(True))
        self.render_timer = QTimer(self, singleShot=True, interval=250, timeout=self.start_render)
        self.save_timer = QTimer(self, singleShot=True, interval=800, timeout=self.save_project)
        self.ui_timer = QTimer(self, interval=40, timeout=self.tick)
        self.ui_timer.start()
        self.set_editor_enabled(False)
        geo = self.settings_store.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    # ----- файлы

    def pick(self, what):
        last = self.settings_store.value("last_dir", str(Path.home() / "Desktop"))
        if what == "vocal":
            p, _ = QFileDialog.getOpenFileName(self, "Вокал", last, AUDIO_FILTER)
        elif what == "lyrics":
            p, _ = QFileDialog.getOpenFileName(self, "Текст песни", last, "Текст (*.txt)")
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Проект", last, "Проект Букви (*_project.json *.json)")
        if p:
            self.on_files([p])

    def on_files(self, paths):
        for p in paths:
            ext = Path(p).suffix.lower()
            self.settings_store.setValue("last_dir", str(Path(p).parent))
            if ext == ".json":
                self.load_project(p)
                return
            if ext in AUDIO_EXT:
                self.vocal_path = p
            elif ext == ".txt":
                self.lyrics_path = p
        self.update_files_label()
        if self.vocal_path and not self.proj:
            # уже обрабатывали этот вокал? — предложим открыть проект
            cand = sorted(Path(self.vocal_path).parent.glob(f"**/{Path(self.vocal_path).stem}_project.json"),
                          key=lambda x: -x.stat().st_mtime)
            if cand and QMessageBox.question(self, "Букви", f"Для этого вокала уже есть проект:\n{cand[0]}\n\n"
                                             "Открыть его?") == QMessageBox.Yes:
                self.load_project(str(cand[0]))

    def update_files_label(self):
        parts = [f"Вокал: {Path(self.vocal_path).name}" if self.vocal_path else "Вокал: —",
                 f"Текст: {Path(self.lyrics_path).name}" if self.lyrics_path else "Текст: —"]
        if self.proj_path:
            parts.append(f"Проект: {self.proj_path}")
        self.files_label.setText("    ".join(parts))

    # ----- автоматический поиск (отдельным процессом: нейросеть не тормозит окно)

    def run_auto(self):
        if not (self.vocal_path and self.lyrics_path):
            QMessageBox.information(self, "Букви", "Нужны вокал и текст песни: перетащите их в окно.")
            return
        out_dir = Path(self.vocal_path).parent / "out"
        args = [os.path.join(APP_DIR, "diction_fix.py"), self.vocal_path, self.lyrics_path, "--out-dir",
                str(out_dir), "--threshold", f"{self.auto_thr.value():.2f}"]
        mode = self.auto_mode.currentIndex()
        if mode == 1:
            args += ["--mode", "overlay"]
        elif mode == 2:
            args += ["--overlay-letters", "т"]
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_proc_output)
        self.proc.finished.connect(lambda code, _st: self.on_proc_done(code, out_dir))
        self.b_auto.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("Ищу проглоченные буквы… (3 минуты песни ≈ 5 минут)")
        self.proc.start(sys.executable, args)

    def on_proc_output(self):
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            line = line.strip()
            if not line or "Warning" in line or "warn" in line or "Loading" in line:
                continue
            m = re.match(r"\[(\d+)/(\d+)\]", line)
            if m:
                self.progress.setRange(0, int(m.group(2)))
                self.progress.setValue(int(m.group(1)))
            self.status.setText(line[:140])

    def on_proc_done(self, code, out_dir):
        self.b_auto.setEnabled(True)
        self.progress.setVisible(False)
        pj = out_dir / f"{Path(self.vocal_path).stem}_project.json"
        if code != 0 or not pj.exists():
            QMessageBox.warning(self, "Букви", "Автоматический поиск не удался. Последняя строка:\n"
                                + self.status.text())
            return
        self.load_project(str(pj))

    # ----- проект

    def load_project(self, path):
        try:
            proj = json.loads(Path(path).read_text(encoding="utf-8"))
            audio, sr, *_ = dfx.load_audio(proj["vocal"])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Букви", f"Не получилось открыть проект:\n{e}")
            return
        for F in proj["fixes"]:
            F.setdefault("width_ms", 0)
            F.setdefault("shift_ms", 0)
            F.setdefault("fade_ms", None)
            F.setdefault("gain_offset_db", 0.0)
            F.setdefault("auto", {k: F.get(k) for k in EDITABLE})  # чтобы можно было вернуть как было
        self.player.pause()
        self.proj, self.proj_path, self.audio = proj, path, audio
        self.vocal_path, self.lyrics_path = proj["vocal"], proj.get("lyrics")
        self.out, self.notes, self.checks, self.export_paths = None, {}, {}, []
        self.player.set_track("orig", audio, sr)
        self.player.set_track("out", audio, sr)
        self.overview.set_audio(audio, sr)
        self.update_files_label()
        self.fill_table()
        self.sel = None
        if proj["fixes"]:
            self.select_fix(min(range(len(proj["fixes"])), key=self.fix_time))
        self.schedule_render(now=True)

    def save_project(self):
        if self.proj and self.proj_path:
            Path(self.proj_path).write_text(json.dumps(self.proj, ensure_ascii=False, indent=1), encoding="utf-8")

    def fix_time(self, i):
        return self.proj["letters"][self.proj["fixes"][i]["letter"]]["start"]

    # ----- список

    def ordered(self):
        return sorted(range(len(self.proj["fixes"])), key=self.fix_time)

    def fill_table(self):
        self.table.blockSignals(True)
        order = self.ordered()
        self.table.setRowCount(len(order))
        for row, i in enumerate(order):
            self.fill_row(row, i)
        self.table.blockSignals(False)
        self.update_marks()

    def fill_row(self, row, i):
        F = self.proj["fixes"][i]
        L = self.proj["letters"][F["letter"]]
        chk = self.checks.get(i)
        # после ручной правки старая оценка уже неверна — до проверки нейросетью показываем пусто
        after = chk if isinstance(chk, float) else (None if self.is_edited(F) else F.get("after"))
        if self.notes.get(i, "") is None and F["enabled"]:
            status = "нет места для вставки — попробуйте «Поверх»"
        elif F["enabled"]:
            status = "вставлено" + (" (правлено)" if self.is_edited(F) else "")
        else:
            status = F["status"] if not self.is_edited(F) else "выключено"
        cells = ["✓" if F["enabled"] else "", fmt_time(L["start"]), mark_word(L["word"], L["char_idx"]), L["char"],
                 f"{F['before']:.2f}", f"{after:.2f}" if after is not None else "", MODE_SHORT[F["mode"]], status]
        for c, text in enumerate(cells):
            it = QTableWidgetItem(text)
            it.setData(Qt.UserRole, i)
            if c == 0:
                it.setTextAlignment(Qt.AlignCenter)
                it.setForeground(QColor(COL_ON))
            if c == 7:
                it.setToolTip(F["status"] + (f"\n{F['note']}" if F.get("note") else ""))
            self.table.setItem(row, c, it)

    def row_of(self, i):
        return self.ordered().index(i)

    def is_edited(self, F):
        return any(F.get(k) != F["auto"].get(k) for k in EDITABLE)

    def update_marks(self):
        marks = []
        for i, F in enumerate(self.proj["fixes"]):
            color = COL_ON if F["enabled"] else COL_OFF
            if F["enabled"] and self.notes.get(i, "") is None:
                color = COL_FAIL
            marks.append((self.fix_time(i), color, i))
        self.overview.set_marks(marks, self.sel)

    def on_table_select(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            i = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
            if i != self.sel:
                self.select_fix(i)

    def on_cell_clicked(self, row, col):
        if col == 0 and self.proj:
            i = self.table.item(row, 0).data(Qt.UserRole)
            self.select_fix(i)
            self.c_enabled.setChecked(not self.proj["fixes"][i]["enabled"])

    # ----- редактор

    def set_editor_enabled(self, on):
        for w in (self.c_enabled, self.c_mode, self.c_donor, self.b_reset, self.b_play_orig, self.b_play_out,
                  self.b_check, self.s_gain.slider, self.s_width.slider, self.s_fade.slider, self.s_shift.slider,
                  self.s_pos.slider):
            w.setEnabled(on)

    def select_fix(self, i):
        self.sel = i
        F = self.proj["fixes"][i]
        L = self.proj["letters"][F["letter"]]
        self.table.blockSignals(True)
        self.table.selectRow(self.row_of(i))
        self.table.blockSignals(False)
        self.ed_title.setText(f"{fmt_time(L['start'])}   «{mark_word(L['word'], L['char_idx'])}» — буква «{L['char']}»")
        has_donor = bool(F["donors"])
        self.set_editor_enabled(has_donor)
        self.b_play_orig.setEnabled(True)
        self.c_enabled.blockSignals(True)
        self.c_enabled.setChecked(F["enabled"])
        self.c_enabled.blockSignals(False)
        self.c_mode.blockSignals(True)
        self.c_mode.setCurrentIndex([k for k, _ in MODES].index(F["mode"]))
        self.c_mode.blockSignals(False)
        self.c_donor.blockSignals(True)
        self.c_donor.clear()
        for D in F["donors"]:
            self.c_donor.addItem(D["label"] + (f"  (чёткость {D['conf']:.2f})" if D.get("conf") else ""))
        if has_donor:
            self.c_donor.setCurrentIndex(F["donor"])
        self.c_donor.blockSignals(False)
        self.s_gain.set(F["gain_offset_db"])
        self.s_width.set(F["width_ms"])
        self.s_fade.set(F["fade_ms"] or self.proj["xfade_ms"])
        self.s_shift.set(F["shift_ms"])
        self.s_pos.set(F["pos"] * 100)
        self.update_info()
        self.update_zoom()
        self.update_marks()

    def update_info(self):
        F = self.proj["fixes"][self.sel]
        lines = [f"Автоматически: {F['status']}"]
        if F.get("note"):
            lines.append(f"Как вставлено: {F['note']}")
        if not F["donors"]:
            lines.append("Для этой буквы в песне нет чёткого донора. Запишите её сами (файл «буква.wav») и запустите "
                         "автоматический поиск с папкой своих букв.")
        chk = self.checks.get(self.sel)
        if isinstance(chk, float):
            lines.append(f"Проверка нейросетью: буква слышна на {chk:.2f}" + ("" if chk >= 0.5 else " — слабо"))
        elif isinstance(chk, str):
            lines.append(chk)
        self.ed_info.setText("\n".join(lines))

    def word_window(self):
        F = self.proj["fixes"][self.sel]
        return max(0.0, F["t_lo"] - 0.15), min(len(self.audio) / self.proj["sr"], F["t_hi"] + 0.15)

    def update_zoom(self):
        if self.sel is None:
            return
        F = self.proj["fixes"][self.sel]
        L = self.proj["letters"][F["letter"]]
        a, b = self.word_window()
        sh = F["shift_ms"] / 1000
        out = self.out if self.out is not None else None
        # подсвечиваем место буквы с запасом: сама метка модели — всего 10–40 мс
        self.zoom.set_view(self.audio, out, self.proj["sr"], a, b, (L["lo"] + sh - 0.04, L["hi"] + sh + 0.04))

    def edit(self, key, value):
        if self.sel is None or not self.proj:
            return
        F = self.proj["fixes"][self.sel]
        if key == "fade_ms" and abs(value - self.proj["xfade_ms"]) < 1e-6 and F["fade_ms"] is None:
            return
        if F.get(key) == value:
            return
        F[key] = value
        if key == "donor":
            F["gain_offset_db"] = 0.0  # у другого донора своя подобранная громкость
            self.s_gain.set(0.0)
        if key != "enabled" and not F["enabled"] and key in ("mode", "donor", "gain_offset_db", "width_ms",
                                                              "fade_ms", "shift_ms", "pos"):
            F["enabled"] = True  # крутите ползунок — значит, хотите эту букву слышать
            self.c_enabled.blockSignals(True)
            self.c_enabled.setChecked(True)
            self.c_enabled.blockSignals(False)
        self.checks.pop(self.sel, None)
        self.fill_row(self.row_of(self.sel), self.sel)
        self.update_marks()
        self.update_info()
        self.schedule_render()
        self.save_timer.start()

    def reset_selected(self):
        if self.sel is None:
            return
        F = self.proj["fixes"][self.sel]
        for k in EDITABLE:
            F[k] = F["auto"].get(k)
        self.checks.pop(self.sel, None)
        self.select_fix(self.sel)
        self.fill_row(self.row_of(self.sel), self.sel)
        self.schedule_render()
        self.save_timer.start()

    # ----- сборка результата

    def schedule_render(self, now=False):
        self.render_gen += 1
        self.status.setText("Пересобираю…")
        if now:
            self.start_render()
        else:
            self.render_timer.start()

    def start_render(self):
        if self.proj is None:
            return
        w = RenderWorker(self.render_gen, self.audio, copy.deepcopy(self.proj))
        w.done.connect(self.on_rendered)
        w.finished.connect(lambda: self.workers.remove(w) if w in self.workers else None)
        self.workers.append(w)
        w.start()

    def on_rendered(self, gen, buf, notes):
        if buf is None:
            self.status.setText("Ошибка сборки — подробности в bukvi_error.log")
            return
        if gen < self.rendered_gen:
            return
        self.rendered_gen = gen
        peak = float(np.abs(buf).max())
        if peak > 0.999:
            buf = buf * (0.999 / peak)
        self.out, self.notes = buf, notes
        self.player.set_track("out", buf, self.proj["sr"])
        self.overview.set_audio(buf, self.proj["sr"])
        self.fill_table()
        self.update_zoom()
        on = sum(1 for F in self.proj["fixes"] if F["enabled"])
        failed = sum(1 for i, n in notes.items() if n is None)
        self.status.setText(f"Готово: вставлено {on - failed} из {len(self.proj['fixes'])}"
                            + (f", без места: {failed}" if failed else "")
                            + ("" if gen == self.render_gen else "  (пересобираю…)"))

    def check_selected(self):
        if self.sel is None or self.out is None:
            return
        self.b_check.setEnabled(False)
        self.checks[self.sel] = "Нейросеть слушает… (первый раз — загрузка, ~20 с)"
        self.update_info()
        w = CheckWorker(self.sel, self.out.copy(), self.proj)
        w.done.connect(self.on_checked)
        w.finished.connect(lambda: self.workers.remove(w) if w in self.workers else None)
        self.workers.append(w)
        w.start()

    def on_checked(self, i, heard):
        self.b_check.setEnabled(True)
        F = self.proj["fixes"][i]
        L = self.proj["letters"][F["letter"]]
        if isinstance(heard, str) or not heard:
            self.checks[i] = "Проверка не удалась — подробности в bukvi_error.log"
        else:
            self.checks[i] = heard.get((L["word_idx"], L["char_idx"]), 0.0)
            weak = [f"«{self.proj['words'][wi][1][ci]}» в «{self.proj['words'][wi][1]}» {c:.2f}"
                    for (wi, ci), c in heard.items() if (wi, ci) != (L["word_idx"], L["char_idx"])
                    and self.proj["words"][wi][1][ci] in dfx.CONSONANTS and c < 0.35]
            if weak:
                self.checks[i] = float(self.checks[i])
                self.status.setText("Рядом слабо слышны: " + ", ".join(weak))
        self.fill_row(self.row_of(i), i)
        if i == self.sel:
            self.update_info()

    # ----- звук

    def play_word(self, track, start=None):
        if self.sel is None:
            return
        (self.r_out if track == "out" else self.r_orig).setChecked(True)
        a, b = self.word_window()
        self.player.track = track
        self.player.play(start if start is not None else a, b)

    def toggle_play(self):
        if self.player.playing:
            self.player.pause()
        else:
            self.player.end = None
            self.player.play()

    def seek(self, t):
        self.player.seek(t)
        if self.player.playing:
            self.player.play()

    def tick(self):
        t = self.player.time
        self.time_label.setText(fmt_time(t))
        self.b_play.setText("⏸" if self.player.playing else "▶")
        self.overview.set_cursor(t)
        self.zoom.set_cursor(t if self.player.playing else None)

    # ----- сохранение

    def export(self):
        if self.proj is None:
            return []
        if self.rendered_gen != self.render_gen or self.out is None:
            # дособираем синхронно, чтобы сохранить ровно то, что настроено
            buf, notes = dfx.render_project(self.audio, self.proj)
            self.on_rendered(self.render_gen, buf, notes)
        stem = Path(self.proj["vocal"]).stem
        out_dir = Path(self.proj_path).parent
        wav = out_dir / f"{stem}_fixed_edit.wav"
        mp3 = out_dir / f"{stem}_fixed_edit.mp3"
        sf.write(str(wav), self.out, self.proj["sr"], subtype="PCM_24")
        paths = [str(wav)]
        try:
            sf.write(str(mp3), self.out, self.proj["sr"], format="MP3", subtype="MPEG_LAYER_III",
                     compression_level=0.0)
            paths.append(str(mp3))
        except Exception as e:  # noqa: BLE001
            log_error(f"MP3: {e}")
        self.save_project()
        self.export_paths = paths
        self.status.setText("Сохранено: " + "  ·  ".join(Path(p).name for p in paths) + f"  (папка {out_dir})")
        return paths

    def paths_for_drag(self):
        paths = self.export()
        return paths[:1]

    def closeEvent(self, e):
        self.player.pause()
        self.save_project()
        self.settings_store.setValue("geometry", self.saveGeometry())
        super().closeEvent(e)


STYLE = f"""
QWidget {{ background: #1e2126; color: #e6e8eb; font-size: 10pt; }}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QLabel#appTitle {{ font-size: 17pt; font-weight: 700; color: #ffffff; padding-right: 10px; }}
QLabel#hint {{ color: #8a94a1; }}
QLabel#sectionTitle {{ color: #ffffff; font-weight: 700; font-size: 12pt; }}
QGroupBox {{ border: 1px solid #343a42; border-radius: 8px; margin-top: 14px; padding: 10px 10px 8px 10px;
            font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #9fb3c8; }}
#dropArea {{ border: 2px dashed #4a5563; border-radius: 10px; background: #23272e; }}
#dropArea[hover="true"] {{ border-color: {ACCENT}; background: #1f3332; }}
QPushButton {{ background: #2b3038; border: 1px solid #3a414b; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ background: #343a44; }}
QPushButton:pressed {{ background: #262a31; }}
QPushButton:disabled {{ color: #5f6772; border-color: #30353d; }}
QPushButton#primary {{ background: #2f9e91; border: none; color: #ffffff; font-weight: 700;
                       padding: 9px 22px; font-size: 11pt; }}
QPushButton#primary:hover {{ background: #36b3a4; }}
QPushButton#primarySmall {{ background: #2f9e91; border: none; color: #ffffff; font-weight: 700; padding: 6px 14px; }}
QPushButton#primarySmall:hover {{ background: #36b3a4; }}
QPushButton#primarySmall:disabled {{ background: #2b3038; color: #5f6772; }}
QPushButton#playBtn {{ background: #2f9e91; border: none; color: #ffffff; font-size: 13pt;
                       min-width: 44px; padding: 4px 10px; }}
QPushButton#playBtn:hover {{ background: #36b3a4; }}
QSlider::groove:horizontal {{ height: 4px; background: #3a414b; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::sub-page:horizontal:disabled {{ background: #4a5563; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; background: #e6e8eb; }}
QSlider::handle:horizontal:disabled {{ background: #5f6772; }}
QProgressBar {{ border: 1px solid #3a414b; border-radius: 5px; background: #23272e; height: 12px; }}
QProgressBar::chunk {{ background: #2f9e91; border-radius: 4px; }}
QTableWidget {{ background: #23272e; alternate-background-color: #262b32; border: 1px solid #343a42;
               border-radius: 6px; selection-background-color: #2c4a47; selection-color: #ffffff; }}
QHeaderView::section {{ background: #2b3038; color: #9fb3c8; border: none; padding: 5px; font-weight: 600; }}
QComboBox, QDoubleSpinBox {{ background: #2b3038; border: 1px solid #3a414b; border-radius: 5px; padding: 3px 6px; }}
QComboBox QAbstractItemView {{ background: #2b3038; selection-background-color: #2c4a47; }}
QLabel#chip {{ border: 1px dashed {ACCENT}; border-radius: 6px; padding: 7px 10px; color: {ACCENT};
               font-weight: 600; background: #1f2b2b; }}
QSplitter::handle {{ background: #2b3038; width: 3px; }}
QToolTip {{ background: #2b3038; color: #e6e8eb; border: 1px solid #3a414b; padding: 4px; }}
"""


def main():
    def excepthook(t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        log_error(text)
        try:
            QMessageBox.critical(None, "Букви", f"Непредвиденная ошибка:\n{v}\n\nПодробности записаны в {LOG_PATH}")
        except Exception:  # noqa: BLE001
            print(text, file=sys.stderr)

    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    app.setApplicationName("Букви")
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.Window, "#1e2126"), (QPalette.Base, "#23272e"),
                        (QPalette.Text, "#e6e8eb"), (QPalette.WindowText, "#e6e8eb"),
                        (QPalette.Button, "#2b3038"), (QPalette.ButtonText, "#e6e8eb"),
                        (QPalette.Highlight, "#2c4a47"), (QPalette.HighlightedText, "#ffffff")):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
    f = QFont()
    f.setPointSize(10)
    app.setFont(f)
    w = MainWindow()
    if w.settings_store.value("geometry") is None:
        w.resize(1280, 860)
    w.show()
    for p in sys.argv[1:]:
        w.on_files([p])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
