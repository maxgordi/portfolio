"""
VoxMatch — окно программы.

Перетащите эталонный вокал и дубли (из проводника/Finder или из Cubase),
нажмите «Выровнять» — получите дубли, подогнанные под эталон по таймингу,
питчу и, по желанию, динамике. Результат можно перетащить обратно в Cubase.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
import traceback

import numpy as np
import soundfile as sf
from PySide6.QtCore import QMimeData, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QDrag, QFont, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
    QInputDialog, QMenu,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy, QTabWidget, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)

try:
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)
from voxmatch_presets import (SECTION_TEMPLATES, SECTION_TITLES, GENRE_PRESETS,  # noqa: E402
                              genre_values)
LOG_PATH = os.path.join(APP_DIR, "voxmatch_error.log")
PRESETS_PATH = os.path.join(APP_DIR, "presets.json")
SESSIONS_PATH = os.path.join(APP_DIR, "sessions.json")
MAX_SESSIONS = 15

BUILTIN_DESC = {
    "Стандарт": "Стандартные настройки программы: тайминг и питч в ноль, своё вибрато, вдохи "
                "на месте без обработки, мягкая очистка.",
    "Дабл — ровно в ноль": "Дабл максимально совпадает с лидом: быстрая подтяжка нот, вибрато "
                           "копируется с эталона.",
    "Дабл — естественно": "Дабл как хорошо спетый вживую: допуски по таймингу и питчу, живые "
                          "начала и окончания нот, мягкое растяжение.",
    "Бэк / гармония — только тайминг": "Только тайминг под лид, высота как спета — интервалы "
                                       "гармонии сохраняются.",
    "Толстый дабл (+детюн)": "Дабл в ноль, но чуть выше (+8 ц), чуть позади (+10 мс) и темнее — "
                             "для ширины.",
    "Фразы целиком — без растяжения": "Фразы сдвигаются под эталон целиком, внутри ничего не "
                                      "растягивается. Самый естественный звук.",
    "Только тюнинг — без тайминга": "Тайминг как спет, ноты подтягиваются к эталону.",
    "Бережно к вдохам и хвостам": "Большая защита окончаний фраз и плавные стыки вдохов — если "
                                  "где-то обрываются хвосты.",
}

# Встроенные пресеты: только то, что отличается от стандартных настроек
BUILTIN_PRESETS = {
    "Стандарт": {},
    "Дабл — ровно в ноль": {"timing": 100, "pitch": 100, "smooth": 15, "vibrato": False,
                             "maxst": 3},
    "Дабл — естественно": {"timing": 100, "pitch": 80, "smooth": 60, "vibrato": True,
                            "ttol": 15, "tsmooth": 30, "tlimit_on": True, "tlimit": 15,
                            "endings": True, "ptol": 10, "patt": 60, "prel": 60},
    "Бэк / гармония — только тайминг": {"timing": 100, "pitch": 0, "ttol": 10, "tsmooth": 20},
    "Толстый дабл (+детюн)": {"timing": 100, "pitch": 100, "ttol": 10, "toffset": 10,
                              "detune": 8, "formant": -1},
    "Фразы целиком — без растяжения": {"tmode": 1, "pitch": 100},
    "Только тюнинг — без тайминга": {"timing": 0, "pitch": 100},
    "Бережно к вдохам и хвостам": {"breath_mode": 0, "breath_sens": 60, "breath_min": 60,
                                   "breath_pre": 20, "breath_post": 20, "tail": 200,
                                   "bxfade": 20, "breath_noclean": True},
}
AUDIO_EXT = {".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}
AUDIO_FILTER = "Аудио (*.wav *.wave *.aif *.aiff *.flac *.ogg *.mp3)"

ACCENT = "#3fb6a8"
COL_REF, COL_DUB, COL_OUT = "#5aa9e6", "#e8a33d", ACCENT

MODES = [  # название, тайминг, питч
    ("Дабл — тайминг и питч", 100, 100),
    ("Гармония — только тайминг", 100, 0),
    ("Только питч", 0, 100),
    ("Свои настройки", None, None),
]

_engine = None


def engine():
    """Ядро грузится лениво: окно открывается сразу, тяжёлые библиотеки — в фоне."""
    global _engine
    if _engine is None:
        import voxmatch_engine
        _engine = voxmatch_engine
    return _engine


def default_inbox() -> str:
    return os.path.join(os.path.expanduser("~"), "Desktop", "VoxMatch Входящие")


def log_error(text: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
    except OSError:
        pass


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
    return out


# ----------------------------------------------------------------------------
# Куски, перетащенные прямо из Cubase
# ----------------------------------------------------------------------------
# Cubase при перетаскивании события отдаёт текст vst-xml: путь к исходному файлу
# и границы куска в сэмплах этого файла. По ним вырезаем кусок сами.
CUT_DIR = None      # куда сохранять вырезанные куски (задаёт главное окно)
CUT_QN: dict = {}   # вырезанный файл → позиция в проекте (в четвертях)


def parse_cubase_regions(text: str) -> list[dict]:
    """Разбор текста vst-xml от Cubase. Регулярными выражениями — так надёжнее,
    чем строгим XML-парсером (в путях бывают символы, которые ломают XML)."""
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


def mime_text(mime: QMimeData) -> str:
    """Текст из перетаскивания в любой кодировке."""
    txt = mime.text() if mime.hasText() else ""
    if "vst-xml" in txt:
        return txt
    for fmt in mime.formats():
        raw = bytes(mime.data(fmt))
        if not raw:
            continue
        for enc in ("utf-8", "utf-16", "utf-16-le", "mac-roman"):
            try:
                t = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            if "vst-xml" in t:
                return t.replace("\x00", "")
    return txt


def _resolve_cubase_path(fn: str):
    for cand in (fn, fn.replace("\\/", "/"), fn.replace("\\/", ":")):
        if os.path.isfile(cand):
            return cand
    return None


def _bar_label(qn):
    """Позиция в проекте словами (для размера 4/4)."""
    if qn is None:
        return ""
    bar = int(qn // 4) + 1
    beat = qn - (bar - 1) * 4 + 1
    beat_s = f"{beat:.0f}" if abs(beat - round(beat)) < 1e-6 else f"{beat:.2f}".rstrip("0")
    return f"такт {bar}.{beat_s}"


def cut_cubase_regions(regions: list[dict], out_dir: str) -> tuple[list[str], list[str]]:
    """Вырезает куски в WAV. Возвращает (пути, ошибки)."""
    paths, errors = [], []
    os.makedirs(out_dir, exist_ok=True)
    for r in regions:
        src = _resolve_cubase_path(r["filename"])
        name = os.path.basename(r["filename"].replace("\\/", "/"))
        if src is None:
            errors.append(f"Файл не найден: {r['filename']}")
            continue
        try:
            info = sf.info(src)
            start = max(0, min(r["start"], info.frames))
            end = info.frames if r["end"] <= 0 else max(start, min(r["end"], info.frames))
            if end - start < info.samplerate * 0.2:
                errors.append(f"«{name}»: кусок слишком короткий")
                continue
            data, sr = sf.read(src, start=start, stop=end, always_2d=True, dtype="float32")
            stem = os.path.splitext(name)[0]
            label = _bar_label(r.get("qn"))
            base = f"{stem} — {label}" if label else f"{stem} — {start}"
            target = os.path.join(out_dir, base + ".wav")
            n = 2
            while os.path.exists(target):
                target = os.path.join(out_dir, f"{base} ({n}).wav")
                n += 1
            subtype = info.subtype if info.subtype in ("PCM_16", "PCM_24", "PCM_32", "FLOAT") else "PCM_24"
            sf.write(target, data, sr, subtype=subtype)
            CUT_QN[os.path.abspath(target)] = r.get("qn")
            paths.append(target)
        except Exception as exc:  # noqa: BLE001
            log_error(traceback.format_exc())
            errors.append(f"«{name}»: {exc}")
    return paths, errors


def has_audio(mime: QMimeData) -> bool:
    return bool(audio_paths(mime)) or "vst-xml" in mime_text(mime)


def dropped_audio(mime: QMimeData) -> tuple[list[str], list[str]]:
    """Файлы из перетаскивания: обычные аудиофайлы или куски из Cubase."""
    paths = audio_paths(mime)
    if paths:
        return paths, []
    text = mime_text(mime)
    regions = parse_cubase_regions(text)
    if regions:
        return cut_cubase_regions(regions, CUT_DIR or os.path.join(default_inbox(), "Куски из Cubase"))
    if text:
        log_error("Не распознан текст перетаскивания: " + repr(text[:600]))
    return [], []


def read_textclipping(path: str) -> str:
    import plistlib
    try:
        with open(path, "rb") as f:
            d = plistlib.load(f)
        u = d.get("UTI-Data", {})
        raw = u.get("public.utf8-plain-text")
        if raw is not None:
            return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        raw = u.get("public.utf16-plain-text")
        if raw is not None:
            return raw.decode("utf-16-le", "replace") if isinstance(raw, bytes) else str(raw)
    except Exception:  # noqa: BLE001
        pass
    return ""


def read_mono(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return data.mean(axis=1), sr


def file_info(path: str) -> str:
    try:
        i = sf.info(path)
        ch = "моно" if i.channels == 1 else ("стерео" if i.channels == 2 else f"{i.channels} кан.")
        return f"{i.duration:.1f} с · {i.samplerate} Гц · {ch}"
    except Exception:  # noqa: BLE001
        return "не удалось прочитать файл"


# ----------------------------------------------------------------------------
# Виджеты
# ----------------------------------------------------------------------------
class DropArea(QFrame):
    """Зона, куда перетаскивают файлы. По клику открывает выбор файла."""
    filesDropped = Signal(list)
    clicked = Signal()
    noFileDropped = Signal(list)    # перетащили что-то без готового файла (например, из Cubase)

    def __init__(self, title: str, hint: str, parent=None):
        super().__init__(parent)
        self.setObjectName("dropArea")
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(86)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        self.title = QLabel(title)
        self.title.setObjectName("dropTitle")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setWordWrap(True)
        self.hint = QLabel(hint)
        self.hint.setObjectName("hint")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)
        lay.addWidget(self.title)
        lay.addWidget(self.hint)

    def _set_prop(self, name, value):
        self.setProperty(name, value)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_filled(self, filled: bool):
        self._set_prop("filled", filled)

    def dragEnterEvent(self, e):
        if has_audio(e.mimeData()) or e.mimeData().formats():
            e.acceptProposedAction()
            self._set_prop("hover", True)
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self._set_prop("hover", False)

    def dropEvent(self, e):
        self._set_prop("hover", False)
        paths, errors = dropped_audio(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self.filesDropped.emit(paths)
        if errors:
            QMessageBox.warning(self, "VoxMatch", "\n".join(errors))
        elif not paths:
            self.noFileDropped.emit(list(e.mimeData().formats()))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()


class DubTable(QTableWidget):
    """Таблица дублей. Принимает файлы и отдаёт готовые результаты перетаскиванием."""
    filesDropped = Signal(list)

    def __init__(self, paths_for_drag):
        super().__init__(0, 5)
        self._paths_for_drag = paths_for_drag
        self.setHorizontalHeaderLabels(["Дубль", "Оценка", "Тайминг, мс", "Питч, центов", "Результат"])
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)

    def startDrag(self, actions):
        paths = self._paths_for_drag()
        if not paths:
            return
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)

    def dragEnterEvent(self, e):
        if e.source() is self:
            e.ignore()
        elif has_audio(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.source() is not self and has_audio(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        if e.source() is self:
            return
        paths, errors = dropped_audio(e.mimeData())
        if paths:
            e.acceptProposedAction()
            self.filesDropped.emit(paths)
        if errors:
            QMessageBox.warning(self, "VoxMatch", "\n".join(errors))


class DragChip(QLabel):
    """Плашка, которую можно перетащить в Cubase — несёт выбранные результаты."""

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
        paths = self._paths_for_drag()
        self._press = None
        if not paths:
            return
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class SliderRow:
    """Ползунок с подписью значения."""

    def __init__(self, grid: QGridLayout, row: int, label: str, lo: int, hi: int, value: int,
                 suffix: str, tip: str = "", step: int = 1, fmt=None):
        self.fmt = fmt
        self.label = QLabel(label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setSingleStep(step)
        self.slider.setPageStep(step * 10)
        self.slider.setValue(value)
        self.value = QLabel()
        self.value.setMinimumWidth(64)
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.suffix = suffix
        for w in (self.label, self.slider, self.value):
            w.setToolTip(tip)
        grid.addWidget(self.label, row, 0)
        grid.addWidget(self.slider, row, 1)
        grid.addWidget(self.value, row, 2)
        self.slider.valueChanged.connect(self._show)
        self._show(value)

    def _show(self, v):
        self.value.setText(self.fmt(v) if self.fmt else f"{v} {self.suffix}")

    def setEnabled(self, on: bool):
        for w in (self.label, self.slider, self.value):
            w.setEnabled(on)


# ----------------------------------------------------------------------------
# Фоновые задачи
# ----------------------------------------------------------------------------
class Loader(QThread):
    done = Signal(str)

    def run(self):
        try:
            engine()
            self.done.emit("")
        except Exception as exc:  # noqa: BLE001
            log_error(traceback.format_exc())
            self.done.emit(str(exc))


class Worker(QThread):
    progress = Signal(int, str)
    rowStarted = Signal(str)
    rowDone = Signal(str, object)
    rowFailed = Signal(str, str)

    def __init__(self, session, ref: str, dubs: list[str], settings):
        super().__init__()
        self.session, self.ref, self.dubs, self.settings = session, ref, dubs, settings
        self.cancelled = False

    def run(self):
        n = len(self.dubs)
        for k, dub in enumerate(self.dubs):
            if self.cancelled:
                break
            self.rowStarted.emit(dub)

            def prog(msg, f, k=k):
                self.progress.emit(int((k + f) / n * 100), f"[{k + 1}/{n}] {msg}")

            try:
                r = self.session.process(self.ref, dub, self.settings, progress=prog)
                self.rowDone.emit(dub, r)
            except Exception as exc:  # noqa: BLE001
                log_error(f"{dub}\n{traceback.format_exc()}")
                self.rowFailed.emit(dub, str(exc))
        self.progress.emit(100, "")


# ----------------------------------------------------------------------------
# Главное окно
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoxMatch — подгонка дублей под эталон")
        self.settings_store = QSettings("VoxMatch", "VoxMatch")
        self.ref_path: str | None = None
        self.dubs: list[str] = []
        self.results: dict[str, object] = {}
        self.session = None
        self.worker: Worker | None = None
        self.comp = None                 # собранный комп (словарь из make_comp)
        self._comp_busy = False
        self._comp_pending = None
        self._then_comp = False
        self._retired: list = []
        self._comp_busy = False
        self._restore_choices = None      # выбор фраз компа из прошлой сессии
        self._sess_restoring = False
        self._sess_timer = QTimer(self)
        self._sess_timer.setSingleShot(True)
        self._sess_timer.setInterval(600)
        self._sess_timer.timeout.connect(self._sess_save)
        self.out_dir = ""
        self.engine_ready = False
        self._play_offset = 0.0

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        head = QHBoxLayout()
        title = QLabel("VoxMatch")
        title.setObjectName("appTitle")
        sub = QLabel("подгонка дублей вокала под эталон: тайминг, питч, динамика")
        sub.setObjectName("hint")
        head.addWidget(title)
        head.addSpacing(10)
        head.addWidget(sub, 1, Qt.AlignBottom)
        root.addLayout(head)
        root.addWidget(self._build_inbox())

        cols = QHBoxLayout()
        cols.setSpacing(12)
        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(self._build_ref())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_w = self._build_settings()
        for c in settings_w.findChildren(QComboBox):     # длинные пункты не расширяют панель
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(6)
        scroll.setWidget(settings_w)
        left.addWidget(scroll, 1)
        right = QVBoxLayout()
        right.setSpacing(10)
        right.addWidget(self._build_dubs(), 2)
        right.addWidget(self._build_preview(), 3)
        cols.addLayout(left, 2)
        cols.addLayout(right, 3)
        root.addLayout(cols, 1)

        root.addLayout(self._build_bottom())

        self._load_settings()
        self._refresh_buttons()

        self.auto_timer = QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.setInterval(700)
        self.auto_timer.timeout.connect(self._auto_rerun)

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(50)
        self.play_timer.timeout.connect(self._update_playhead)

        self._inbox_known: dict = {}
        self._inbox_pending: dict = {}
        self._inbox_queue: list[str] = []
        self._set_inbox(self.settings_store.value("inbox_dir", "") or default_inbox())
        self.inbox_timer = QTimer(self)
        self.inbox_timer.setInterval(600)
        self.inbox_timer.timeout.connect(self._poll_inbox)
        self.inbox_timer.start()

        self.status.setText("Загружаю модули обработки…")
        self.loader = Loader()
        self.loader.done.connect(self._engine_loaded)
        self.loader.start()

    # ---------------------------------------------------------------- папка «Входящие»
    def _build_inbox(self):
        bar = QFrame()
        bar.setObjectName("inbox")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(10)
        icon = QLabel("⇣  Из Cubase:")
        icon.setObjectName("inboxTitle")
        icon.setToolTip("Перетащите кусок из Cubase в эту папку (в Finder, на Рабочем столе или "
                        "в боковой панели Finder) — VoxMatch сразу его подхватит.")
        self.btn_inbox_open = QPushButton()
        self.btn_inbox_open.setObjectName("inboxPath")
        self.btn_inbox_open.setCursor(Qt.PointingHandCursor)
        self.btn_inbox_open.setToolTip("Открыть папку в Finder. Удобно держать её открытой рядом с Cubase.")
        self.btn_inbox_open.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.inbox_dir)))
        nxt = QLabel("следующий файл →")
        nxt.setObjectName("hint")
        self.rb_ref = QRadioButton("Эталон")
        self.rb_dub = QRadioButton("Дубль")
        self.rb_ref.setChecked(True)
        self.rb_ref.setToolTip("Следующий файл из папки станет эталоном, потом переключится на «Дубль».")
        self.cb_inbox_run = QCheckBox("сразу выравнивать")
        self.cb_inbox_run.setToolTip("Как только появился дубль и есть эталон — сразу обработать.")
        self.cb_inbox_run.setChecked(True)
        btn_change = QPushButton("Другая папка…")
        btn_change.clicked.connect(self._pick_inbox)
        lay.addWidget(icon)
        lay.addWidget(self.btn_inbox_open, 1)
        lay.addWidget(nxt)
        lay.addWidget(self.rb_ref)
        lay.addWidget(self.rb_dub)
        lay.addSpacing(6)
        lay.addWidget(self.cb_inbox_run)
        lay.addWidget(btn_change)
        self.btn_sessions = QPushButton("🕘  Сессии")
        self.btn_sessions.setToolTip("Последние сессии: эталон, дубли и выбор фраз в компе.\n"
                                     "Прошлая сессия открывается сама при запуске.")
        self.btn_sessions.clicked.connect(self._sessions_menu)
        lay.addWidget(self.btn_sessions)
        return bar

    def _set_inbox(self, d):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            d = default_inbox()
            os.makedirs(d, exist_ok=True)
        self.inbox_dir = d
        global CUT_DIR
        CUT_DIR = os.path.join(d, "Куски из Cubase")
        self.btn_inbox_open.setText("📂  " + d.replace(os.path.expanduser("~"), "~"))
        self._inbox_known = {}                             # старые файлы не трогаем
        for p in self._inbox_files():
            try:
                st = os.stat(p)
                self._inbox_known[p] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
        self._inbox_pending.clear()

    def _no_file_dropped(self, formats):
        log_error("Перетаскивание без файла, форматы: " + ", ".join(formats))
        QMessageBox.information(
            self, "VoxMatch",
            "Cubase не отдаёт файл напрямую в другие программы.\n\n"
            "Перетащите кусок из Cubase в папку «Входящие» (кнопка 📂 вверху окна откроет её "
            "в Finder) — VoxMatch подхватит его сам за секунду.")

    def _pick_inbox(self):
        d = QFileDialog.getExistingDirectory(self, "Папка «Входящие»", self.inbox_dir)
        if d:
            self._set_inbox(d)

    def _inbox_files(self):
        try:
            names = os.listdir(self.inbox_dir)
        except OSError:
            return []
        return [os.path.join(self.inbox_dir, n) for n in names
                if not n.startswith(".")
                and os.path.splitext(n)[1].lower() in AUDIO_EXT | {".textclipping"}]

    def _poll_inbox(self):
        """Новый или перезаписанный файл берём, когда он перестал меняться ~2 секунды."""
        for p in self._inbox_files():
            try:
                st = os.stat(p)
            except OSError:
                continue
            sig = (st.st_size, st.st_mtime_ns)
            if self._inbox_known.get(p) == sig:
                continue
            prev, count = self._inbox_pending.get(p, (None, 0))
            count = count + 1 if prev == sig else 0
            self._inbox_pending[p] = (sig, count)
            if st.st_size == 0 or count < 3:
                continue                            # ещё пишется
            self._inbox_pending.pop(p, None)
            if p.lower().endswith(".textclipping"):
                self._inbox_known[p] = sig
                self._inbox_clipping(p)
                continue
            try:
                sf.info(p)
            except Exception:  # noqa: BLE001
                continue
            self._inbox_known[p] = sig
            self._inbox_arrived(p)

    def _inbox_clipping(self, p):
        """В папку бросили кусок из Cubase — Finder сохранил его как текстовую вырезку."""
        regions = parse_cubase_regions(read_textclipping(p))
        if not regions:
            return
        paths, errors = cut_cubase_regions(regions, CUT_DIR)
        try:                                        # вырезку убираем, чтобы не мешала
            done = os.path.join(CUT_DIR, "_вырезки")
            os.makedirs(done, exist_ok=True)
            target = os.path.join(done, f"{int(time.time() * 1000)}.textClipping")
            os.replace(p, target)
            self._inbox_known.pop(p, None)
        except OSError:
            pass
        for q in paths:
            self._inbox_arrived(q)
        if errors:
            QMessageBox.warning(self, "VoxMatch", "\n".join(errors))

    def _inbox_arrived(self, p):
        name = os.path.basename(p)
        if p in self.results:                       # файл перезаписали — старый результат неактуален
            self.results.pop(p, None)
        if self.ref_path and os.path.abspath(p) == os.path.abspath(self.ref_path):
            self._set_ref(p)                        # эталон перезаписан — обновить
            self.status.setText(f"Эталон обновлён: {name}")
            return
        if p in self.dubs:
            self._rebuild_table()
        if self.rb_ref.isChecked():
            self._set_ref(p)
            self.rb_dub.setChecked(True)
            self.status.setText(f"Эталон из Cubase: {name}. Теперь кидайте дубли.")
            return
        self._add_dubs([p])
        if p in self.dubs:
            self.table.clearSelection()
            self.table.selectRow(self.dubs.index(p))
        self.status.setText(f"Дубль из Cubase: {name}")
        if self.cb_inbox_run.isChecked() and self.ref_path:
            self._inbox_queue.append(p)
            self._run_inbox_queue()

    def _run_inbox_queue(self):
        if not self._inbox_queue or not self.engine_ready:
            return
        if self.worker is not None and self.worker.isRunning():
            return                                  # запустится после текущей обработки
        batch, self._inbox_queue = [d for d in self._inbox_queue if d in self.dubs], []
        self._start(batch)

    # ---------------------------------------------------------------- панели
    def _build_ref(self):
        box = QGroupBox("1. Эталон")
        lay = QVBoxLayout(box)
        self.ref_drop = DropArea("Перетащите сюда эталонный вокал",
                                 "или нажмите, чтобы выбрать файл · WAV, AIFF, FLAC, MP3")
        self.ref_drop.setMinimumHeight(96)
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.ref_drop.filesDropped.connect(self._ref_dropped)
        self.ref_drop.noFileDropped.connect(self._no_file_dropped)
        self.ref_drop.clicked.connect(self._pick_ref)
        lay.addWidget(self.ref_drop, 1)
        row = QHBoxLayout()
        self.btn_ref_clear = QPushButton("Убрать")
        self.btn_ref_clear.clicked.connect(lambda: self._set_ref(None))
        hint = QLabel("Сухой, одноголосый, без реверба.")
        hint.setObjectName("hint")
        row.addWidget(hint, 1)
        row.addWidget(self.btn_ref_clear)
        lay.addLayout(row)
        return box

    def _build_dubs(self):
        box = QGroupBox("2. Дубли и комп")
        outer = QVBoxLayout(box)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        self.dub_drop = DropArea("Перетащите сюда дубли",
                                 "можно сразу несколько файлов или папку")
        self.dub_drop.setMinimumHeight(64)
        self.dub_drop.filesDropped.connect(self._add_dubs)
        self.dub_drop.noFileDropped.connect(self._no_file_dropped)
        self.dub_drop.clicked.connect(self._pick_dubs)
        lay.addWidget(self.dub_drop)
        self.table = DubTable(self._selected_outputs)
        self.table.filesDropped.connect(self._add_dubs)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda *_: self._play("out"))
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.btn_remove = QPushButton("Убрать выбранные")
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self._clear_dubs)
        row.addStretch(1)
        row.addWidget(self.btn_remove)
        row.addWidget(self.btn_clear)
        lay.addLayout(row)
        self.tabs.addTab(page, "Дубли")

        cpage = QWidget()
        cl = QVBoxLayout(cpage)
        cl.setContentsMargins(0, 8, 0, 0)
        top = QHBoxLayout()
        self.btn_comp = QPushButton("★  Собрать комп")
        self.btn_comp.setObjectName("primarySmall")
        self.btn_comp.setToolTip("Из каждой фразы берётся лучший дубль (по оценке), стыки — в паузах.")
        self.btn_comp.clicked.connect(lambda: self._build_comp(reset=True))
        self.lbl_comp = QLabel("Сначала выровняйте дубли, потом соберите комп.")
        self.lbl_comp.setObjectName("hint")
        self.lbl_comp.setWordWrap(True)
        top.addWidget(self.btn_comp)
        top.addWidget(self.lbl_comp, 1)
        cl.addLayout(top)
        self.comp_table = QTableWidget(0, 4)
        self.comp_table.setHorizontalHeaderLabels(["Фраза", "Время", "Из дубля", "Оценка"])
        h = self.comp_table.horizontalHeader()
        for c in (0, 1, 3):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        self.comp_table.verticalHeader().setVisible(False)
        self.comp_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.comp_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.comp_table.setShowGrid(False)
        self.comp_table.cellClicked.connect(self._comp_row_clicked)
        cl.addWidget(self.comp_table, 1)
        hint = QLabel("Поменяйте дубль у любой фразы — комп пересоберётся сразу. Клик по фразе — "
                      "перемотка к ней.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        cl.addWidget(hint)
        self.tabs.addTab(cpage, "★ Комп")
        self.tabs.currentChanged.connect(lambda *_: self._selection_changed())
        return box

    def _build_settings(self):
        box = QGroupBox("3. Настройки")
        g = QGridLayout(box)
        g.setColumnStretch(1, 1)
        g.setVerticalSpacing(7)
        r = 0

        g.addWidget(QLabel("Пресет"), r, 0)
        prow = QHBoxLayout()
        self.cmb_preset = QComboBox()
        self.cmb_preset.setToolTip("Выберите пресет — все настройки применятся сразу.")
        self.cmb_preset.activated.connect(self._preset_chosen)
        btn_save = QPushButton("Сохранить…")
        btn_save.setToolTip("Сохранить все текущие настройки как пресет под своим именем.")
        btn_save.clicked.connect(self._preset_save)
        self.btn_preset_del = QPushButton("Удалить")
        self.btn_preset_del.clicked.connect(self._preset_delete)
        prow.addWidget(self.cmb_preset, 1)
        prow.addWidget(btn_save)
        prow.addWidget(self.btn_preset_del)
        g.addLayout(prow, r, 1, 1, 2)
        r += 1
        self.preset_desc = QLabel("")
        self.preset_desc.setObjectName("tplDesc")
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setVisible(False)
        g.addWidget(self.preset_desc, r, 0, 1, 3)
        r += 1
        self.btn_defaults = QPushButton("↺  Стандартные настройки")
        self.btn_defaults.setToolTip("Вернуть все настройки обработки к стандартным значениям.\n"
                                     "Папки для результатов и «Входящие» не меняются.")
        self.btn_defaults.clicked.connect(self._reset_defaults)
        g.addWidget(self.btn_defaults, r, 0, 1, 3)
        r += 1
        self.btn_adv = QPushButton("▸  Расширенные настройки")
        self.btn_adv.setObjectName("advToggle")
        self.btn_adv.setCheckable(True)
        self.btn_adv.setToolTip("Показать все тонкие настройки (миллисекунды, допуски, детали "
                                "вдохов и очистки).\nОбычно хватает шаблонов и пресетов.")
        self.btn_adv.toggled.connect(self._show_advanced)
        g.addWidget(self.btn_adv, r, 0, 1, 3)
        r += 1

        g.addWidget(QLabel("Режим"), r, 0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([m[0] for m in MODES])
        self.cmb_mode.currentIndexChanged.connect(self._mode_changed)
        g.addWidget(self.cmb_mode, r, 1, 1, 2)
        r += 1

        self.s_timing = SliderRow(g, r, "Тайминг", 0, 100, 100, "%",
                                  "Насколько подтягивать тайминг к эталону. 0 — не трогать.")
        r += 1
        self.s_pitch = SliderRow(g, r, "Питч", 0, 100, 100, "%",
                                 "Насколько подтягивать высоту к эталону. 0 — не трогать.")
        r += 1
        self.s_smooth = SliderRow(g, r, "Плавность питча", 0, 200, 30, "мс",
                                  "Меньше — жёстче и точнее, больше — мягче и естественнее.", 5)
        r += 1
        self.cb_vibrato = QCheckBox("Сохранить своё вибрато дубля")
        self.cb_vibrato.setChecked(True)
        self.cb_vibrato.setToolTip("Вкл: от эталона берётся нота, вибрато и живость остаются свои.\n"
                                   "Выкл: копируется вся линия питча эталона.")
        g.addWidget(self.cb_vibrato, r, 0, 1, 3)
        r += 1

        g.addWidget(QLabel("Предел коррекции"), r, 0)
        self.sp_maxst = QSpinBox()
        self.sp_maxst.setRange(1, 12)
        self.sp_maxst.setValue(3)
        self.sp_maxst.setSuffix(" полутона")
        self.sp_maxst.setToolTip("Если дубль отличается сильнее — считается, что спета другая нота, "
                                 "и это место не трогается.")
        g.addWidget(self.sp_maxst, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("Макс. сдвиг по времени"), r, 0)
        self.sp_shift = QSpinBox()
        self.sp_shift.setRange(100, 3000)
        self.sp_shift.setSingleStep(50)
        self.sp_shift.setValue(600)
        self.sp_shift.setSuffix(" мс")
        self.sp_shift.setToolTip("Насколько далеко программа ищет совпадение. "
                                 "Увеличьте, если дубль сильно опаздывает или спешит.")
        g.addWidget(self.sp_shift, r, 1, 1, 2)
        r += 1

        self.cb_same = QCheckBox("Файлы вырезаны с одной позиции (одинаковое начало)")
        self.cb_same.setChecked(True)
        self.cb_same.setToolTip("Выключите, если резали эталон и дубль в разных местах — "
                                "программа сама найдёт смещение.")
        g.addWidget(self.cb_same, r, 0, 1, 3)
        r += 1

        # ---------------- Тайминг подробно
        lt = QFrame()
        lt.setObjectName("sep")
        lt.setFrameShape(QFrame.HLine)
        g.addWidget(lt, r, 0, 1, 3)
        r += 1
        lab = QLabel("Тайминг")
        lab.setObjectName("sectionTitle")
        g.addWidget(lab, r, 0, 1, 3)
        r += 1
        r = self._template_row(g, r, "timing")
        g.addWidget(QLabel("Режим тайминга"), r, 0)
        self.cmb_tmode = QComboBox()
        self.cmb_tmode.addItems(["По слогам — точно", "Фразы целиком — только сдвиг"])
        self.cmb_tmode.setToolTip("По слогам: каждый слог встаёт под эталон (растягивание внутри фраз).\n"
                                  "Фразы целиком: фраза сдвигается под эталон одним куском, внутри "
                                  "ничего не растягивается —\nмаксимально естественно, но слоги внутри "
                                  "остаются как спеты.")
        g.addWidget(self.cmb_tmode, r, 1, 1, 2)
        r += 1
        self.s_ttol = SliderRow(g, r, "Допуск", 0, 60, 0, "мс",
                                "Расхождения меньше допуска не исправляются, большие — подтягиваются до "
                                "допуска.\nОставляет живость: 10–20 мс почти не слышно, но дабл не «клонирован».")
        r += 1
        self.s_tsmooth = SliderRow(g, r, "Сглаживание", 0, 200, 0, "мс",
                                   "Плавность растяжения. Больше — мягче, без «рывков» на слогах, "
                                   "но менее точно.", 5)
        r += 1
        self.s_tflex = SliderRow(g, r, "Гибкость", 0, 100, 50, "%",
                                 "Насколько смело программа двигает отдельные слоги.\n"
                                 "Меньше — осторожнее (ближе к сдвигу целиком), больше — точнее по "
                                 "каждому слогу, но выше риск ошибок.", 5)
        r += 1
        self.cb_tlimit = QCheckBox("Ограничить растяжение/сжатие")
        self.cb_tlimit.setToolTip("Не даёт растягивать или сжимать звук сильнее заданного — "
                                  "меньше «резины» и артефактов.")
        g.addWidget(self.cb_tlimit, r, 0, 1, 3)
        r += 1
        self.s_tlimit = SliderRow(g, r, "Не больше чем", 11, 40, 20, "×",
                                  "1.2× — очень бережно, 2× — обычно, 4× — почти без ограничения.", 1,
                                  fmt=lambda v: f"{v / 10:.1f}×")
        r += 1
        self.s_toffset = SliderRow(g, r, "Сдвиг раньше/позже", -100, 100, 0, "мс",
                                   "Весь результат чуть позже (+) или раньше (−) эталона.\n"
                                   "+10…20 мс — дабл «сидит» за лидом, мягче; − — напористее.", 1,
                                   fmt=lambda v: f"{v:+d} мс")
        r += 1
        self.cb_endings = QCheckBox("Не растягивать окончания фраз")
        self.cb_endings.setToolTip("Окончание каждой фразы остаётся с естественной длиной, "
                                   "без растягивания/сжатия.")
        g.addWidget(self.cb_endings, r, 0, 1, 3)
        r += 1
        self.s_ending = SliderRow(g, r, "Длина окончания", 50, 500, 150, "мс", "", 10)
        r += 1

        # ---------------- Питч подробно
        lp = QFrame()
        lp.setObjectName("sep")
        lp.setFrameShape(QFrame.HLine)
        g.addWidget(lp, r, 0, 1, 3)
        r += 1
        lab = QLabel("Питч")
        lab.setObjectName("sectionTitle")
        g.addWidget(lab, r, 0, 1, 3)
        r += 1
        r = self._template_row(g, r, "pitch")
        self.s_ptol = SliderRow(g, r, "Допуск", 0, 50, 0, "ц",
                                "Отклонения меньше допуска (в центах) не исправляются.\n"
                                "5–15 центов — естественно, без «автотюна».")
        r += 1
        self.s_patt = SliderRow(g, r, "Атака ноты", 0, 200, 0, "мс",
                                "В начале ноты коррекция нарастает за это время — сохраняется живой "
                                "«подъезд» к ноте.", 5)
        r += 1
        self.s_prel = SliderRow(g, r, "Отпускание", 0, 200, 0, "мс",
                                "В конце ноты коррекция плавно уходит — живое окончание.", 5)
        r += 1
        self.s_vib = SliderRow(g, r, "Своё вибрато", 0, 200, 100, "%",
                               "100% — как спето, 0% — ровная нота, 150–200% — ярче.", 5)
        r += 1
        self.s_detune = SliderRow(g, r, "Подстройка", -50, 50, 0, "ц",
                                  "Общий сдвиг высоты в центах. Для толстого дабла: +5…+10 на одном, "
                                  "−5…−10 на другом.", 1, fmt=lambda v: f"{v:+d} ц")
        r += 1
        self.s_formant = SliderRow(g, r, "Форманты (тембр)", -12, 12, 0, "",
                                   "Тембр темнее (−) или светлее (+) без изменения высоты.\n"
                                   "Хорошо разводит даблы между собой: ±0.5…1.", 1,
                                   fmt=lambda v: f"{v / 2:+.1f} пт")
        r += 1
        self.cb_tlimit.toggled.connect(self.s_tlimit.setEnabled)
        self.cb_endings.toggled.connect(self.s_ending.setEnabled)

        line = QFrame()
        line.setObjectName("sep")
        line.setFrameShape(QFrame.HLine)
        g.addWidget(line, r, 0, 1, 3)
        r += 1

        lab = QLabel("Вдохи")
        lab.setObjectName("sectionTitle")
        g.addWidget(lab, r, 0, 1, 3)
        r += 1
        r = self._template_row(g, r, "breath")
        self.cb_breath = QCheckBox("Не трогать вдохи и паузы (брать из оригинала)")
        self.cb_breath.setChecked(True)
        self.cb_breath.setToolTip("Через обработку питча идут только спетые ноты. Вдохи, паузы, шумы "
                                  "и согласные берутся\nиз вашей исходной записи и только сдвигаются "
                                  "по времени. Убирает «жужжание» и призвуки на вдохах.")
        self.cb_breath.toggled.connect(self._breath_toggled)
        g.addWidget(self.cb_breath, r, 0, 1, 3)
        r += 1
        g.addWidget(QLabel("Вдохи"), r, 0)
        self.cmb_breath_mode = QComboBox()
        self.cmb_breath_mode.addItems(["Оригинал на своём месте — не трогать",
                                       "Сдвигать к началу фразы"])
        self.cmb_breath_mode.setToolTip(
            "Оригинал на своём месте: вдох распознаётся, но не обрабатывается совсем — стоит там же,\n"
            "где был в дубле, без растяжения, без питча, без сдвига.\n"
            "Сдвигать к началу фразы: тот же оригинальный вдох, но переносится так, чтобы\n"
            "закончиться ровно перед выровненной фразой.")
        g.addWidget(self.cmb_breath_mode, r, 1, 1, 2)
        r += 1
        self.s_breath_sens = SliderRow(g, r, "Чувствительность", 0, 100, 50, "%",
                                       "Больше — вдохом считаются и более громкие шумы.\n"
                                       "Меньше — только явные тихие вдохи. Вдохи видны на волне "
                                       "«После» серыми полосами.", 5)
        r += 1
        self.s_breath_min = SliderRow(g, r, "Мин. длина вдоха", 30, 400, 90, "мс",
                                      "Шумы короче этого вдохом не считаются (это могут быть согласные).", 10)
        r += 1
        self.s_breath_pre = SliderRow(g, r, "Запас до вдоха", 0, 100, 10, "мс",
                                      "Сколько захватить перед найденным вдохом.", 5)
        r += 1
        self.s_breath_post = SliderRow(g, r, "Запас после вдоха", 0, 100, 10, "мс",
                                       "Сколько захватить после найденного вдоха.", 5)
        r += 1
        self.s_tail = SliderRow(g, r, "Защита окончаний фраз", 0, 400, 120, "мс",
                                "Столько после конца ноты — это затухание голоса, а не вдох.\n"
                                "Окончание остаётся с фразой. Если обрываются хвосты — увеличьте.", 10)
        r += 1
        self.s_bxfade = SliderRow(g, r, "Плавность стыков вдоха", 2, 50, 10, "мс",
                                  "Длина кроссфейда на краях вдоха. Щелчки на стыках — увеличьте.", 1)
        r += 1
        self.s_breath_gain = SliderRow(g, r, "Громкость вдохов", 0, 150, 100, "%",
                                       "100% — как в дубле, 0% — убрать вдохи совсем.", 5)
        r += 1
        self.cb_breath_noclean = QCheckBox("Очистка не трогает вдохи")
        self.cb_breath_noclean.setToolTip("Деэссер, тишина в паузах, громкость фраз и щелчки — "
                                          "мимо вдохов. Вдох остаётся как в записи.")
        g.addWidget(self.cb_breath_noclean, r, 0, 1, 3)
        r += 1

        line1 = QFrame()
        line1.setObjectName("sep")
        line1.setFrameShape(QFrame.HLine)
        g.addWidget(line1, r, 0, 1, 3)
        r += 1

        lab = QLabel("Динамика")
        lab.setObjectName("sectionTitle")
        g.addWidget(lab, r, 0, 1, 3)
        r += 1
        r = self._template_row(g, r, "dyn")
        self.cb_dyn = QCheckBox("Динамика как у эталона (компрессия)")
        self.cb_dyn.setToolTip("Громкость дубля ведётся так же, как у эталона: та же сжатость, "
                               "те же акценты и спады.")
        self.cb_dyn.toggled.connect(self._dyn_toggled)
        g.addWidget(self.cb_dyn, r, 0, 1, 3)
        r += 1
        self.s_dyn_amount = SliderRow(g, r, "Сила", 0, 100, 100, "%",
                                      "100% — громкость повторяет эталон полностью.")
        r += 1
        self.s_dyn_speed = SliderRow(g, r, "Скорость", 10, 300, 50, "мс",
                                     "Как атака и релиз компрессора: меньше — быстрее "
                                     "и точнее, больше — мягче.", 5)
        r += 1

        lineC = QFrame()
        lineC.setObjectName("sep")
        lineC.setFrameShape(QFrame.HLine)
        g.addWidget(lineC, r, 0, 1, 3)
        r += 1
        lab = QLabel("Очистка (после выравнивания и в компе)")
        lab.setObjectName("sectionTitle")
        g.addWidget(lab, r, 0, 1, 3)
        r += 1
        r = self._template_row(g, r, "clean")
        self.cb_clicks = QCheckBox("Убирать щелчки губ (в паузах и на вдохах)")
        self.cb_clicks.setToolTip("Короткие чмоки и щелчки до 7 мс. В пении и на согласных не трогает.")
        g.addWidget(self.cb_clicks, r, 0, 1, 3)
        r += 1
        self.cb_deess = QCheckBox("Деэссер (шипящие «с», «ш», «ц»)")
        g.addWidget(self.cb_deess, r, 0, 1, 3)
        r += 1
        self.s_deess = SliderRow(g, r, "Сила деэссера", 0, 12, 5, "дБ",
                                 "На сколько приглушать резкие шипящие. 3–6 дБ — обычно хватает.")
        r += 1
        self.cb_level = QCheckBox("Ровная громкость фраз")
        self.cb_level.setToolTip("Как клип-гейн: тихие фразы чуть поднимает, громкие — опускает. "
                                 "Внутри фразы ничего не меняет.")
        g.addWidget(self.cb_level, r, 0, 1, 3)
        r += 1
        self.s_level = SliderRow(g, r, "Насколько ровно", 0, 100, 70, "%",
                                 "100% — все фразы одной громкости, 50% — наполовину. Не больше ±8 дБ.", 5)
        r += 1
        self.cb_gate = QCheckBox("Тише в паузах (приглушить шум и гул)")
        self.cb_gate.setToolTip("Шум и гул между фразами приглушаются на 30 дБ (не в ноль — воздух "
                                "остаётся).\nНе трогает вдохи, окончания фраз (300 мс после пения) и "
                                "подготовку к фразе.")
        g.addWidget(self.cb_gate, r, 0, 1, 3)
        r += 1
        self.s_gate = SliderRow(g, r, "Тише пения на", 30, 70, 50, "дБ",
                                "Меньше — убирает больше (осторожно с тихими окончаниями фраз).", 1)
        r += 1
        for cb, sl in ((self.cb_deess, (self.s_deess,)), (self.cb_level, (self.s_level,)),
                       (self.cb_gate, (self.s_gate,))):
            cb.toggled.connect(lambda on, sl=sl: [w.setEnabled(on) for w in sl])

        line2 = QFrame()
        line2.setObjectName("sep")
        line2.setFrameShape(QFrame.HLine)
        g.addWidget(line2, r, 0, 1, 3)
        r += 1

        g.addWidget(QLabel("Анализ"), r, 0)
        self.cmb_quality = QComboBox()
        self.cmb_quality.addItems(["Точный", "Быстрый"])
        self.cmb_quality.setToolTip("Быстрый — примерно вдвое быстрее, но питч определяется грубее.")
        g.addWidget(self.cmb_quality, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("Формат"), r, 0)
        self.cmb_fmt = QComboBox()
        self.cmb_fmt.addItems(["WAV 24 бит", "WAV 32 бит float"])
        g.addWidget(self.cmb_fmt, r, 1, 1, 2)
        r += 1

        g.addWidget(QLabel("Папка"), r, 0)
        folder_row = QHBoxLayout()
        self.lbl_out = QLabel()
        self.lbl_out.setObjectName("hint")
        self.lbl_out.setMinimumWidth(40)
        self.lbl_out.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        btn_out = QPushButton("Выбрать…")
        btn_out.clicked.connect(self._pick_out)
        btn_out_reset = QPushButton("Сброс")
        btn_out_reset.clicked.connect(lambda: self._set_out(""))
        folder_row.addWidget(self.lbl_out, 1)
        folder_row.addWidget(btn_out)
        folder_row.addWidget(btn_out_reset)
        g.addLayout(folder_row, r, 1, 1, 2)
        r += 1

        self.cb_auto = QCheckBox("Пересчитывать выбранный дубль при изменении настроек")
        self.cb_auto.setChecked(True)
        g.addWidget(self.cb_auto, r, 0, 1, 3)
        r += 1
        g.setRowStretch(r, 1)

        for w in (self.s_timing, self.s_pitch):
            w.slider.valueChanged.connect(self._sliders_moved)
        for w in (self.s_smooth, self.s_dyn_amount, self.s_dyn_speed, self.s_breath_sens,
                  self.s_breath_gain, self.s_deess, self.s_level, self.s_gate, self.s_breath_min,
                  self.s_breath_pre, self.s_breath_post, self.s_tail, self.s_bxfade,
                  self.s_ttol, self.s_tsmooth, self.s_tflex, self.s_tlimit, self.s_toffset,
                  self.s_ending, self.s_ptol, self.s_patt, self.s_prel, self.s_vib, self.s_detune,
                  self.s_formant):
            w.slider.valueChanged.connect(self._settings_changed)
        for w in (self.cb_vibrato, self.cb_same, self.cb_dyn, self.cb_breath, self.cb_clicks,
                  self.cb_deess, self.cb_level, self.cb_gate, self.cb_breath_noclean,
                  self.cb_tlimit, self.cb_endings):
            w.toggled.connect(self._settings_changed)
        for w in (self.sp_maxst, self.sp_shift):
            w.valueChanged.connect(self._settings_changed)
        self.cmb_quality.currentIndexChanged.connect(self._settings_changed)
        self.cmb_breath_mode.currentIndexChanged.connect(self._settings_changed)
        self.cmb_tmode.currentIndexChanged.connect(self._settings_changed)
        self._collect_advanced(g)
        return box

    def _collect_advanced(self, g):
        """Что прятать под «Расширенные»: тонкие регуляторы. На виду остаются пресеты,
        шаблоны разделов, режим, тайминг/питч %, вдохи вкл/громкость, формат и папка."""
        rows = [self.s_smooth, self.s_ttol, self.s_tsmooth, self.s_tflex, self.s_tlimit,
                self.s_toffset, self.s_ending, self.s_ptol, self.s_patt, self.s_prel, self.s_vib,
                self.s_detune, self.s_formant, self.s_breath_sens, self.s_breath_min,
                self.s_breath_pre, self.s_breath_post, self.s_tail, self.s_bxfade, self.s_deess,
                self.s_level, self.s_gate, self.s_dyn_amount, self.s_dyn_speed]
        widgets = [self.cb_vibrato, self.sp_maxst, self.sp_shift, self.cmb_tmode, self.cb_tlimit,
                   self.cb_endings, self.cmb_breath_mode, self.cb_breath_noclean, self.cb_clicks,
                   self.cb_deess, self.cb_level, self.cb_gate, self.cb_dyn, self.cmb_quality]
        adv = []
        for sr in rows:
            adv += [sr.label, sr.slider, sr.value]
        for w in widgets:
            adv.append(w)
            idx = g.indexOf(w)
            if idx >= 0:
                row, col, _rs, cs = g.getItemPosition(idx)
                if col == 1:                                   # подпись слева в той же строке
                    item = g.itemAtPosition(row, 0)
                    if item is not None and item.widget() is not None:
                        adv.append(item.widget())
        self._adv_widgets = adv

    def _show_advanced(self, on):
        for w in getattr(self, "_adv_widgets", []):
            w.setVisible(on)
        self.btn_adv.setText(("▾  Скрыть расширенные настройки" if on else "▸  Расширенные настройки"))
        if not getattr(self, "_loading", False):
            self.settings_store.setValue("advanced", on)

    def _build_preview(self):
        box = QGroupBox("4. Прослушивание")
        lay = QVBoxLayout(box)
        self.wave = WaveView()
        self.wave.setCursor(Qt.PointingHandCursor)
        self.wave.seekRequested.connect(self._seek)
        lay.addWidget(self.wave, 1)
        self._peak_cache: dict = {}

        self.lbl_stats = QLabel("")
        self.lbl_stats.setObjectName("hint")
        self.lbl_stats.setMinimumHeight(34)
        self.lbl_stats.setWordWrap(True)
        lay.addWidget(self.lbl_stats)

        self.player = Player()
        prow = QHBoxLayout()
        self.btn_toggle = QPushButton("▶")
        self.btn_toggle.setObjectName("playBtn")
        self.btn_toggle.setToolTip("Играть / пауза (пробел)")
        self.btn_toggle.clicked.connect(self._toggle_play)
        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.setRange(0, 1000)
        self.pos_slider.sliderMoved.connect(lambda v: self._seek_frac(v / 1000.0))
        self.pos_slider.sliderPressed.connect(lambda: self._seek_frac(self.pos_slider.value() / 1000.0))
        self.lbl_time = QLabel("0:00.0 / 0:00.0")
        self.lbl_time.setObjectName("hint")
        self.lbl_time.setMinimumWidth(110)
        prow.addWidget(self.btn_toggle)
        prow.addWidget(self.pos_slider, 1)
        prow.addWidget(self.lbl_time)
        lay.addLayout(prow)

        trow = QHBoxLayout()
        lab = QLabel("Слушать:")
        lab.setObjectName("hint")
        trow.addWidget(lab)
        self.cb_listen = {}
        for key, text, color in (("ref", "Эталон", COL_REF), ("dub", "Дубль до", COL_DUB),
                                 ("out", "Результат", COL_OUT)):
            cb = QCheckBox(text)
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 600; }}")
            cb.setChecked(self.player.on[key])
            cb.toggled.connect(lambda on, k=key: self._listen_toggled(k, on))
            self.cb_listen[key] = cb
            trow.addWidget(cb)
        trow.addStretch(1)
        lay.addLayout(trow)
        if sd is None:
            self.btn_toggle.setEnabled(False)
            self.btn_toggle.setToolTip("Для прослушивания установите sounddevice")

        row2 = QHBoxLayout()
        self.chip = DragChip(self._selected_outputs)
        self.btn_folder = QPushButton("Открыть папку с результатами")
        self.btn_folder.clicked.connect(self._open_folder)
        row2.addWidget(self.chip, 1)
        row2.addWidget(self.btn_folder)
        lay.addLayout(row2)
        return box

    def _build_bottom(self):
        row = QHBoxLayout()
        self.btn_all = QPushButton("★  Всё сразу")
        self.btn_all.setObjectName("primary")
        self.btn_all.setToolTip("Выровнять все дубли → почистить → собрать комп из лучших фраз → "
                                "сохранить один готовый файл.")
        self.btn_all.clicked.connect(self._run_everything)
        self.btn_run = QPushButton("Выровнять")
        self.btn_run.clicked.connect(self._run_all)
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.clicked.connect(self._cancel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setMinimumWidth(320)
        row.addWidget(self.btn_all)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.progress, 1)
        row.addWidget(self.status, 1)
        return row

    # ---------------------------------------------------------------- файлы
    def _last_dir(self):
        return self.settings_store.value("last_dir", os.path.expanduser("~"))

    def _remember_dir(self, path):
        self.settings_store.setValue("last_dir", os.path.dirname(path))

    def _pick_ref(self):
        p, _ = QFileDialog.getOpenFileName(self, "Эталонный вокал", self._last_dir(), AUDIO_FILTER)
        if p:
            self._set_ref(p)

    def _pick_dubs(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Дубли", self._last_dir(), AUDIO_FILTER)
        if ps:
            self._add_dubs(ps)

    def _ref_dropped(self, paths):
        self._set_ref(paths[0])
        if len(paths) > 1:
            self._add_dubs(paths[1:])
            self.status.setText("Первый файл — эталон, остальные добавлены как дубли.")

    def _reset_comp(self):
        self.comp = None
        self.comp_table.setRowCount(0)
        self.lbl_comp.setText("Сначала выровняйте дубли, потом соберите комп.")

    def _set_ref(self, path):
        self._sess_touch()
        self._stop()
        self._reset_comp()
        self.ref_path = path
        if path:
            self._remember_dir(path)
            self.ref_drop.title.setText(os.path.basename(path))
            self.ref_drop.hint.setText(file_info(path) + " · нажмите, чтобы заменить")
            self.ref_drop.set_filled(True)
            self.rb_dub.setChecked(True)            # следующий кусок из Cubase — дубль
            self.dubs = [d for d in self.dubs if os.path.abspath(d) != os.path.abspath(path)]
        else:
            self.ref_drop.title.setText("Перетащите сюда эталонный вокал")
            self.ref_drop.hint.setText("или нажмите, чтобы выбрать файл · WAV, AIFF, FLAC, MP3")
            self.ref_drop.set_filled(False)
            self.rb_ref.setChecked(True)
        self.results.clear()
        self._rebuild_table()
        self._refresh_buttons()

    def _check_positions(self):
        """Если эталон и дубль взяты из Cubase с разных мест — включить поиск смещения."""
        if not self.ref_path:
            return
        rq = CUT_QN.get(os.path.abspath(self.ref_path))
        if rq is None:
            return
        for d in self.dubs:
            dq = CUT_QN.get(os.path.abspath(d))
            if dq is not None and abs(dq - rq) > 1e-6 and self.cb_same.isChecked():
                self.cb_same.setChecked(False)
                self.status.setText("Куски взяты с разных мест проекта — смещение найду сам.")
                return

    def _add_dubs(self, paths):
        self._sess_touch()
        added = 0
        for p in paths:
            if self.ref_path and os.path.abspath(p) == os.path.abspath(self.ref_path):
                continue
            if p not in self.dubs:
                self.dubs.append(p)
                added += 1
        if paths:
            self._remember_dir(paths[0])
        self._rebuild_table()
        if added and not self.table.selectedItems():
            self.table.selectRow(0)
        self._check_positions()
        self._refresh_buttons()

    def _remove_selected(self):
        self._sess_touch()
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            d = self.dubs.pop(r)
            self.results.pop(d, None)
        self._rebuild_table()
        self._refresh_buttons()

    def _clear_dubs(self):
        self._sess_touch()
        self._stop()
        self._reset_comp()
        self.dubs.clear()
        self.results.clear()
        self._rebuild_table()
        self._refresh_buttons()

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Папка для результатов",
                                             self.out_dir or self._last_dir())
        if d:
            self._set_out(d)

    def _set_out(self, d):
        self.out_dir = d
        if d:
            parts = os.path.normpath(d).split(os.sep)
            short = os.sep.join(parts[-2:]) if len(parts) > 2 else d
            self.lbl_out.setText(("…" + os.sep + short) if len(parts) > 2 else short)
            self.lbl_out.setToolTip(d)
        else:
            self.lbl_out.setText("рядом с дублем, папка VoxMatch")
            self.lbl_out.setToolTip("Результат сохраняется в папку VoxMatch рядом с файлом дубля")

    # ---------------------------------------------------------------- таблица
    def _rebuild_table(self):
        selected = self._selected_dubs()
        self.table.setRowCount(len(self.dubs))
        for row, d in enumerate(self.dubs):
            self._fill_row(row, d)
        for row, d in enumerate(self.dubs):
            if d in selected:
                self.table.selectRow(row)
        self.dub_drop.set_filled(bool(self.dubs))
        self._selection_changed()

    def _fill_row(self, row, dub, status=None):
        r = self.results.get(dub)
        name = QTableWidgetItem(os.path.basename(dub))
        name.setToolTip(dub + "\n" + file_info(dub))
        if r is not None:
            t = QTableWidgetItem(f"{r.timing_before_ms:.0f} → {r.timing_after_ms:.0f}")
            pt = ("—" if np.isnan(r.pitch_before_c)
                  else f"{r.pitch_before_c:.0f} → {r.pitch_after_c:.0f}")
            p = QTableWidgetItem(pt)
            out = QTableWidgetItem(os.path.basename(r.out_path))
            out.setToolTip(r.out_path + "\nПеретащите строку в Cubase")
            out.setForeground(QColor(ACCENT))
            t.setToolTip("Среднее отклонение тайминга от эталона: до → после")
            p.setToolTip("Среднее отклонение основной ноты от эталона: до → после")
            sc = r.score.total if r.score is not None else float("nan")
            score = QTableWidgetItem("—" if np.isnan(sc) else f"{sc:.0f}")
            score.setForeground(QColor(score_color(sc)))
            score.setToolTip("Насколько хорошо дубль спет ДО обработки: ноты, тайминг, все ли слова, "
                             "нет ли перегруза.\nЧем выше — тем меньше правок и тем естественнее звук.")
        else:
            t, p, score = QTableWidgetItem(""), QTableWidgetItem(""), QTableWidgetItem("")
            out = QTableWidgetItem(status or "ожидает")
            out.setForeground(QColor("#8a94a1"))
        for c in (t, p, score):
            c.setTextAlignment(Qt.AlignCenter)
        for c, item in enumerate((name, score, t, p, out)):
            self.table.setItem(row, c, item)

    def _set_status_for(self, dub, text, color="#8a94a1"):
        if dub in self.dubs:
            row = self.dubs.index(dub)
            item = QTableWidgetItem(text)
            item.setForeground(QColor(color))
            self.table.setItem(row, 4, item)

    def _selected_dubs(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectedItems()})
        return [self.dubs[r] for r in rows if r < len(self.dubs)]

    def _current_dub(self):
        sel = self._selected_dubs()
        return sel[0] if sel else None

    def _comp_view(self) -> bool:
        return self.tabs.currentIndex() == 1

    def _selected_outputs(self) -> list[str]:
        if self._comp_view():
            p = self.comp["out_path"] if self.comp else None
            return [p] if p and os.path.exists(p) else []
        out = []
        for d in self._selected_dubs() or ([self.dubs[0]] if len(self.dubs) == 1 else []):
            r = self.results.get(d)
            if r is not None and os.path.exists(r.out_path):
                out.append(r.out_path)
        return out

    def _selection_changed(self):
        self._update_plot()
        self._load_player()

    # ---------------------------------------------------------------- настройки
    def _mode_changed(self, idx):
        _, t, p = MODES[idx]
        if t is None:
            return
        for w, v in ((self.s_timing, t), (self.s_pitch, p)):
            w.slider.blockSignals(True)
            w.slider.setValue(v)
            w._show(v)
            w.slider.blockSignals(False)
        self._settings_changed()

    def _sliders_moved(self):
        t, p = self.s_timing.slider.value(), self.s_pitch.slider.value()
        idx = next((i for i, m in enumerate(MODES) if m[1] == t and m[2] == p), len(MODES) - 1)
        self.cmb_mode.blockSignals(True)
        self.cmb_mode.setCurrentIndex(idx)
        self.cmb_mode.blockSignals(False)
        self._settings_changed()

    def _breath_toggled(self, on):
        for w in (self.s_breath_sens, self.s_breath_gain, self.s_breath_min, self.s_breath_pre,
                  self.s_breath_post, self.s_tail, self.s_bxfade):
            w.setEnabled(on)
        self.cmb_breath_mode.setEnabled(on)
        self.cb_breath_noclean.setEnabled(on)

    def _dyn_toggled(self, on):
        self.s_dyn_amount.setEnabled(on)
        self.s_dyn_speed.setEnabled(on)

    def _settings_changed(self, *_):
        if getattr(self, "_loading", False):
            return
        self._save_settings()
        if self.cb_auto.isChecked() and self.results and self._current_dub() in self.results:
            self.auto_timer.start()

    def current_settings(self):
        ve = engine()
        return ve.Settings(
            timing=self.s_timing.slider.value() / 100.0,
            pitch=self.s_pitch.slider.value() / 100.0,
            keep_vibrato=self.cb_vibrato.isChecked(),
            pitch_smooth_ms=float(self.s_smooth.slider.value()),
            max_correction_st=float(self.sp_maxst.value()),
            max_shift_ms=float(self.sp_shift.value()),
            same_start=self.cb_same.isChecked(),
            quality="fast" if self.cmb_quality.currentIndex() == 1 else "accurate",
            dyn_match=self.cb_dyn.isChecked(),
            dyn_amount=self.s_dyn_amount.slider.value() / 100.0,
            dyn_speed_ms=float(self.s_dyn_speed.slider.value()),
            protect_breath=self.cb_breath.isChecked(),
            breath_sens=float(self.s_breath_sens.slider.value()),
            breath_gain=self.s_breath_gain.slider.value() / 100.0,
            breath_mode="original" if self.cmb_breath_mode.currentIndex() == 0 else "phrase",
            breath_min_ms=float(self.s_breath_min.slider.value()),
            breath_pre_ms=float(self.s_breath_pre.slider.value()),
            breath_post_ms=float(self.s_breath_post.slider.value()),
            tail_ms=float(self.s_tail.slider.value()),
            breath_xfade_ms=float(self.s_bxfade.slider.value()),
            breath_no_clean=self.cb_breath_noclean.isChecked(),
            timing_mode="syllables" if self.cmb_tmode.currentIndex() == 0 else "phrases",
            timing_tol_ms=float(self.s_ttol.slider.value()),
            timing_smooth_ms=float(self.s_tsmooth.slider.value()),
            timing_flex=float(self.s_tflex.slider.value()),
            max_stretch=self.s_tlimit.slider.value() / 10.0 if self.cb_tlimit.isChecked() else 0.0,
            timing_offset_ms=float(self.s_toffset.slider.value()),
            keep_endings=self.cb_endings.isChecked(),
            ending_ms=float(self.s_ending.slider.value()),
            pitch_tol_c=float(self.s_ptol.slider.value()),
            pitch_attack_ms=float(self.s_patt.slider.value()),
            pitch_release_ms=float(self.s_prel.slider.value()),
            vibrato_amount=self.s_vib.slider.value() / 100.0,
            detune_c=float(self.s_detune.slider.value()),
            formant_st=self.s_formant.slider.value() / 2.0,
            clean_clicks=self.cb_clicks.isChecked(),
            clean_deess=self.cb_deess.isChecked(),
            deess_db=float(self.s_deess.slider.value()),
            clean_level=self.cb_level.isChecked(),
            level_strength=self.s_level.slider.value() / 100.0,
            clean_gate=self.cb_gate.isChecked(),
            gate_db=float(self.s_gate.slider.value()),
            out_dir=self.out_dir,
            subtype="FLOAT" if self.cmb_fmt.currentIndex() == 1 else "PCM_24",
        )

    DEFAULTS = {
        "timing": 100, "pitch": 100, "smooth": 30, "vibrato": True, "maxst": 3, "shift": 600,
        "same": True, "breath": True, "breath_sens": 50, "breath_gain": 100,
        "dyn": False, "dyn_amount": 100, "dyn_speed": 50, "quality": 0, "fmt": 0, "auto": True,
        "clicks": True, "deess": True, "deess_db": 5, "level": True, "level_str": 70,
        "gate": False, "gate_db": 50,
        "breath_mode": 0, "breath_min": 90, "breath_pre": 10, "breath_post": 10, "tail": 120,
        "bxfade": 10, "breath_noclean": True,
        "tmode": 0, "ttol": 0, "tsmooth": 0, "tflex": 50, "tlimit_on": False, "tlimit": 20,
        "toffset": 0, "endings": False, "ending": 150,
        "ptol": 0, "patt": 0, "prel": 0, "vib": 100, "detune": 0, "formant": 0,
    }

    def _controls(self):
        """Все настройки: ключ → (прочитать, записать). Одно место для сохранения,
        стандартных значений и пресетов."""
        def sl(w):
            return (lambda: w.slider.value(), lambda v: w.slider.setValue(int(v)))

        def cb(w):
            return (lambda: w.isChecked(), lambda v: w.setChecked(bool(v)))

        def cm(w):
            return (lambda: w.currentIndex(), lambda v: w.setCurrentIndex(int(v)))

        def sp(w):
            return (lambda: w.value(), lambda v: w.setValue(int(v)))

        return {
            "timing": sl(self.s_timing), "pitch": sl(self.s_pitch), "smooth": sl(self.s_smooth),
            "vibrato": cb(self.cb_vibrato), "maxst": sp(self.sp_maxst), "shift": sp(self.sp_shift),
            "same": cb(self.cb_same), "breath": cb(self.cb_breath),
            "breath_sens": sl(self.s_breath_sens), "breath_gain": sl(self.s_breath_gain),
            "breath_mode": cm(self.cmb_breath_mode), "breath_min": sl(self.s_breath_min),
            "breath_pre": sl(self.s_breath_pre), "breath_post": sl(self.s_breath_post),
            "tail": sl(self.s_tail), "bxfade": sl(self.s_bxfade),
            "breath_noclean": cb(self.cb_breath_noclean),
            "dyn": cb(self.cb_dyn), "dyn_amount": sl(self.s_dyn_amount),
            "dyn_speed": sl(self.s_dyn_speed), "quality": cm(self.cmb_quality),
            "fmt": cm(self.cmb_fmt), "auto": cb(self.cb_auto),
            "clicks": cb(self.cb_clicks), "deess": cb(self.cb_deess), "deess_db": sl(self.s_deess),
            "level": cb(self.cb_level), "level_str": sl(self.s_level),
            "gate": cb(self.cb_gate), "gate_db": sl(self.s_gate),
            "tmode": cm(self.cmb_tmode), "ttol": sl(self.s_ttol), "tsmooth": sl(self.s_tsmooth),
            "tflex": sl(self.s_tflex), "tlimit_on": cb(self.cb_tlimit), "tlimit": sl(self.s_tlimit),
            "toffset": sl(self.s_toffset), "endings": cb(self.cb_endings),
            "ending": sl(self.s_ending), "ptol": sl(self.s_ptol), "patt": sl(self.s_patt),
            "prel": sl(self.s_prel), "vib": sl(self.s_vib), "detune": sl(self.s_detune),
            "formant": sl(self.s_formant),
        }

    def _collect_values(self) -> dict:
        return {k: getter() for k, (getter, _setter) in self._controls().items()}

    def _apply_values(self, v):
        self._loading = True
        ctl = self._controls()
        for k, default in self.DEFAULTS.items():
            ctl[k][1](v.get(k, default))
        self.s_deess.setEnabled(self.cb_deess.isChecked())
        self.s_level.setEnabled(self.cb_level.isChecked())
        self.s_gate.setEnabled(self.cb_gate.isChecked())
        self._breath_toggled(self.cb_breath.isChecked())
        self._dyn_toggled(self.cb_dyn.isChecked())
        self.s_tlimit.setEnabled(self.cb_tlimit.isChecked())
        self.s_ending.setEnabled(self.cb_endings.isChecked())
        self._sliders_moved()
        self._loading = False

    # ---------------------------------------------------------------- сессии
    def _sessions(self) -> list:
        try:
            import json
            with open(SESSIONS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return [d for d in data if isinstance(d, dict) and d.get("ref")]
        except (OSError, ValueError):
            return []

    def _sess_touch(self):
        if not self._sess_restoring:
            self._sess_timer.start()

    def _sess_save(self):
        """Запомнить текущую сессию (эталон, дубли, выбор фраз компа) первой в списке."""
        if not self.ref_path:
            return
        import json
        cur = {
            "ref": self.ref_path,
            "dubs": list(self.dubs),
            "comp_choices": list(self.comp["choices"]) if self.comp else None,
            "tab": self.tabs.currentIndex(),
            "time": time.strftime("%d.%m %H:%M"),
        }
        sessions = [d for d in self._sessions()
                    if os.path.abspath(d["ref"]) != os.path.abspath(self.ref_path)]
        sessions.insert(0, cur)
        try:
            tmp = SESSIONS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(sessions[:MAX_SESSIONS], f, ensure_ascii=False, indent=1)
            os.replace(tmp, SESSIONS_PATH)
        except OSError:
            pass

    def _sessions_menu(self):
        menu = QMenu(self)
        act_new = menu.addAction("＋  Новая сессия (очистить всё)")
        act_new.triggered.connect(self._sess_new)
        sessions = self._sessions()
        if sessions:
            menu.addSeparator()
        for d in sessions:
            name = os.path.basename(d["ref"])
            n = len(d.get("dubs") or [])
            missing = not os.path.exists(d["ref"])
            label = f"{d.get('time', '')}   {name}  ·  дублей: {n}" + ("   (файл не найден)" if missing else "")
            a = menu.addAction(label)
            a.setEnabled(not missing)
            a.triggered.connect(lambda _c=False, d=d: self._sess_open(d))
        menu.exec(self.btn_sessions.mapToGlobal(self.btn_sessions.rect().bottomLeft()))

    def _sess_new(self):
        self._sess_save()
        self._sess_restoring = True
        self._set_ref(None)
        self._clear_dubs()
        self._sess_restoring = False
        self.tabs.setCurrentIndex(0)
        self.status.setText("Новая сессия. Прошлая сохранена в «🕘 Сессии».")

    def _sess_open(self, d, auto=False):
        """Открыть сессию: файлы, затем выровнять заново и пересобрать комп с прежним выбором."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "VoxMatch", "Дождитесь окончания обработки.")
            return
        if self.ref_path and not auto:
            self._sess_save()
        ref = d.get("ref")
        if not ref or not os.path.exists(ref):
            if not auto:
                QMessageBox.warning(self, "VoxMatch", f"Эталон не найден:\n{ref}")
            return
        dubs = [p for p in d.get("dubs") or [] if os.path.exists(p)]
        lost = len(d.get("dubs") or []) - len(dubs)
        self._sess_restoring = True
        self._clear_dubs()
        self._set_ref(ref)
        self._add_dubs(dubs)
        self._sess_restoring = False
        self.tabs.setCurrentIndex(0)
        msg = f"{'Открыта прошлая сессия' if auto else 'Сессия открыта'}: {os.path.basename(ref)}, " \
              f"дублей: {len(dubs)}"
        if lost:
            msg += f" (не найдено файлов: {lost})"
        self.status.setText(msg)
        if dubs and self.engine_ready:
            choices = d.get("comp_choices")
            if choices and all(c in dubs for c in choices):
                self._restore_choices = list(choices)
                self._then_comp = True
            self._start(list(dubs))
            if self.worker is None:
                self._then_comp = False
            else:
                self.status.setText(msg + " — выравниваю заново…")

    # ---------------------------------------------------------------- шаблоны разделов
    def _template_row(self, g, r, section):
        """Строка «Шаблон: [список]» и объяснение под ней."""
        if not hasattr(self, "tpl_combo"):
            self.tpl_combo, self.tpl_desc = {}, {}
        g.addWidget(QLabel("Шаблон"), r, 0)
        cmb = QComboBox()
        cmb.addItem("— выберите шаблон —", None)
        for i, (name, desc, _vals) in enumerate(SECTION_TEMPLATES[section]):
            cmb.addItem(name, i)
            cmb.setItemData(i + 1, desc, Qt.ToolTipRole)
        cmb.setToolTip(SECTION_TITLES[section] + ": выберите — применятся настройки только этого "
                       "раздела.\nНаведите на пункт списка, чтобы прочитать объяснение.")
        cmb.activated.connect(lambda _i, sec=section: self._apply_template(sec))
        g.addWidget(cmb, r, 1, 1, 2)
        r += 1
        desc = QLabel("")
        desc.setObjectName("tplDesc")
        desc.setWordWrap(True)
        desc.setVisible(False)
        g.addWidget(desc, r, 0, 1, 3)
        r += 1
        self.tpl_combo[section] = cmb
        self.tpl_desc[section] = desc
        return r

    def _apply_template(self, section):
        cmb = self.tpl_combo[section]
        i = cmb.currentData()
        if i is None:
            return
        name, desc, vals = SECTION_TEMPLATES[section][i]
        merged = self._collect_values()
        merged.update(vals)
        self._apply_values(merged)
        self.tpl_desc[section].setText(desc)
        self.tpl_desc[section].setVisible(True)
        self.settings_store.setValue(f"tpl_{section}", i)
        self.status.setText(f"{SECTION_TITLES[section]}: «{name}» применён.")
        self._save_settings()
        self._settings_changed()

    def _restore_templates(self):
        for section, cmb in self.tpl_combo.items():
            v = self.settings_store.value(f"tpl_{section}")
            try:
                i = int(v)
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(SECTION_TEMPLATES[section]):
                cmb.setCurrentIndex(i + 1)
                self.tpl_desc[section].setText(SECTION_TEMPLATES[section][i][1])
                self.tpl_desc[section].setVisible(True)

    def _clear_templates(self):
        for section, cmb in self.tpl_combo.items():
            cmb.setCurrentIndex(0)
            self.tpl_desc[section].setVisible(False)
            self.settings_store.remove(f"tpl_{section}")

    # ---------------------------------------------------------------- пресеты
    def _user_presets(self) -> dict:
        try:
            import json
            with open(PRESETS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            return {}

    def _write_presets(self, presets: dict):
        import json
        tmp = PRESETS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PRESETS_PATH)

    def _refresh_presets(self, select: str | None = None):
        cur = select or self.cmb_preset.currentData()
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.clear()
        for name in BUILTIN_PRESETS:
            self.cmb_preset.addItem(f"★ {name}", "builtin:" + name)
            self.cmb_preset.setItemData(self.cmb_preset.count() - 1, BUILTIN_DESC.get(name, ""),
                                        Qt.ToolTipRole)
        self.cmb_preset.insertSeparator(self.cmb_preset.count())
        for i, (name, desc, _sec, _extra) in enumerate(GENRE_PRESETS):
            self.cmb_preset.addItem(f"♪ {name}", f"genre:{i}")
            self.cmb_preset.setItemData(self.cmb_preset.count() - 1, desc, Qt.ToolTipRole)
        users = sorted(self._user_presets())
        if users:
            self.cmb_preset.insertSeparator(self.cmb_preset.count())
        for name in users:
            self.cmb_preset.addItem(name, "user:" + name)
        if isinstance(cur, (list, tuple)):
            cur = f"{cur[0]}:{cur[1]}"
        idx = self.cmb_preset.findData(cur) if cur else -1
        self.cmb_preset.setCurrentIndex(max(0, idx))
        self.cmb_preset.blockSignals(False)
        data = self._preset_key()
        self.btn_preset_del.setEnabled(bool(data) and data[0] == "user")
        self._show_preset_desc(data)

    def _show_preset_desc(self, data):
        text = ""
        if data:
            kind, name = data
            if kind == "builtin":
                text = BUILTIN_DESC.get(name, "")
            elif kind == "genre":
                try:
                    text = GENRE_PRESETS[int(name)][1]
                except (ValueError, IndexError):
                    text = ""
            elif kind == "user":
                text = "Ваш пресет."
        self.preset_desc.setText(text)
        self.preset_desc.setVisible(bool(text))

    def _preset_key(self):
        d = self.cmb_preset.currentData()
        if not d or ":" not in str(d):
            return None
        kind, name = str(d).split(":", 1)
        return kind, name

    def _preset_chosen(self, _i=None):
        data = self._preset_key()
        if not data:
            return
        kind, name = data
        vals = dict(self.DEFAULTS)
        if kind == "builtin":
            vals.update(BUILTIN_PRESETS[name])
        elif kind == "genre":
            vals.update(genre_values(int(name)))
        else:
            vals.update(self._user_presets().get(name, {}))
        self._apply_values(vals)
        self._clear_templates()
        if kind == "genre":                       # показать, из каких шаблонов собран жанр
            gname, _d, sections, _x = GENRE_PRESETS[int(name)]
            for sec, tname in sections.items():
                names = [t[0] for t in SECTION_TEMPLATES[sec]]
                if tname in names:
                    i = names.index(tname)
                    self.tpl_combo[sec].setCurrentIndex(i + 1)
                    self.tpl_desc[sec].setText(SECTION_TEMPLATES[sec][i][1])
                    self.tpl_desc[sec].setVisible(True)
                    self.settings_store.setValue(f"tpl_{sec}", i)
            name = gname
        self._show_preset_desc(data)
        self.settings_store.setValue("preset", list(data))
        self.btn_preset_del.setEnabled(kind == "user")
        self._save_settings()
        self.status.setText(f"Пресет «{name}» применён.")
        self._settings_changed()

    def _preset_save(self):
        data = self._preset_key()
        suggest = data[1] if data and data[0] == "user" else ""
        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Название пресета:", text=suggest)
        name = (name or "").strip()
        if not ok or not name:
            return
        presets = self._user_presets()
        if name in presets and QMessageBox.question(
                self, "VoxMatch", f"Пресет «{name}» уже есть. Заменить?") != QMessageBox.Yes:
            return
        presets[name] = self._collect_values()
        try:
            self._write_presets(presets)
        except OSError as exc:
            QMessageBox.warning(self, "VoxMatch", f"Не удалось сохранить пресет:\n{exc}")
            return
        self._refresh_presets(("user", name))
        self.settings_store.setValue("preset", ["user", name])
        self.status.setText(f"Пресет «{name}» сохранён.")

    def _preset_delete(self):
        data = self._preset_key()
        if not data or data[0] != "user":
            return
        if QMessageBox.question(self, "VoxMatch", f"Удалить пресет «{data[1]}»?") != QMessageBox.Yes:
            return
        presets = self._user_presets()
        presets.pop(data[1], None)
        try:
            self._write_presets(presets)
        except OSError as exc:
            QMessageBox.warning(self, "VoxMatch", f"Не удалось удалить пресет:\n{exc}")
        self._refresh_presets(("builtin", "Стандарт"))

    def _reset_defaults(self):
        self._apply_values(self.DEFAULTS)
        self._clear_templates()
        self._refresh_presets(("builtin", "Стандарт"))
        self.settings_store.setValue("preset", ["builtin", "Стандарт"])
        self._save_settings()
        self.status.setText("Настройки сброшены к стандартным.")
        self._settings_changed()

    def _load_settings(self):
        st = self.settings_store

        def val(key, default, typ=int):
            v = st.value(key, default)
            try:
                return typ(v) if typ is not bool else (v in (True, "true", "1", 1))
            except (TypeError, ValueError):
                return default

        vals = {}
        for key, default in self.DEFAULTS.items():
            vals[key] = val(key, default, bool if isinstance(default, bool) else int)
        self._apply_values(vals)
        self._restore_templates()
        adv = st.value("advanced", False)
        adv = adv in (True, "true", "1", 1)
        self.btn_adv.setChecked(adv)
        self._show_advanced(adv)
        saved = st.value("preset")
        self._refresh_presets(tuple(saved) if isinstance(saved, (list, tuple)) and len(saved) == 2
                              else ("builtin", "Стандарт"))
        self._set_out(st.value("out_dir", "") or "")
        geo = st.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

    def _save_settings(self):
        st = self.settings_store
        for k, v in self._collect_values().items():
            st.setValue(k, v)
        st.setValue("out_dir", self.out_dir)
        st.setValue("geometry", self.saveGeometry())
        st.setValue("inbox_dir", self.inbox_dir)

    # ---------------------------------------------------------------- обработка
    def _engine_loaded(self, error):
        if error:
            self.status.setText("Ошибка загрузки модулей — см. voxmatch_error.log")
            QMessageBox.critical(self, "VoxMatch", f"Не удалось загрузить модули обработки:\n{error}")
            return
        self.engine_ready = True
        self.session = engine().Session()
        self.status.setText("Готово. Кидайте куски из Cubase в папку «Входящие» или прямо сюда.")
        self._refresh_buttons()
        QTimer.singleShot(0, self._run_inbox_queue)
        if not self.ref_path and not self.dubs:
            sessions = self._sessions()
            if sessions:
                QTimer.singleShot(50, lambda: self._sess_open(sessions[0], auto=True))

    def _refresh_buttons(self):
        busy = self.worker is not None and self.worker.isRunning()
        can = bool(self.ref_path and self.dubs and self.engine_ready and not busy)
        self.btn_run.setEnabled(can)
        self.btn_all.setEnabled(can and not self._comp_busy)
        self.btn_comp.setEnabled(bool(self.results) and not busy and not self._comp_busy)
        self.btn_run.setText("Выровнять" if len(self.dubs) <= 1 else f"Выровнять все ({len(self.dubs)})")
        self.btn_cancel.setEnabled(busy)
        has_out = bool(self._selected_outputs())
        self.chip.setEnabled(has_out)
        self.btn_folder.setEnabled(bool(self.results))
        if sd is not None:
            self.btn_toggle.setEnabled(bool(self.player.tracks))

    def _start(self, dubs):
        if not dubs or not self.ref_path or not self.engine_ready:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        self._retire_worker()
        for p in [self.ref_path] + dubs:
            if not os.path.exists(p):
                QMessageBox.warning(self, "VoxMatch", f"Файл не найден:\n{p}")
                return
        self.worker = Worker(self.session, self.ref_path, list(dubs), self.current_settings())
        self.worker.progress.connect(self._on_progress)
        self.worker.rowStarted.connect(lambda d: self._set_status_for(d, "обработка…", ACCENT))
        self.worker.rowDone.connect(self._on_row_done)
        self.worker.rowFailed.connect(self._on_row_failed)
        self.worker.finished.connect(self._on_finished)
        self._errors = []
        self._t0 = time.time()
        self.worker.start()
        self._refresh_buttons()

    def _retire_worker(self):
        """Старую фоновую задачу нельзя удалять, пока её поток не завершился полностью,
        иначе Qt аварийно закрывает программу. Держим ссылку, удаляем позже."""
        if self.worker is not None:
            self._retired.append(self.worker)
            self.worker = None
        self._retired = [w for w in self._retired if not w.isFinished()] + \
            [w for w in self._retired if w.isFinished()][-2:]

    def _run_all(self):
        self._start(list(self.dubs))

    def _auto_rerun(self):
        cur = self._current_dub()
        if cur is None or cur not in self.results:
            return
        if self.worker is not None and self.worker.isRunning():
            self.auto_timer.start()      # дождёмся окончания текущей обработки
            return
        self._start([cur])

    def _cancel(self):
        if self.worker is not None:
            self.worker.cancelled = True
            self.status.setText("Останавливаю после текущего дубля…")

    def _on_progress(self, value, text):
        self.progress.setValue(value)
        if text:
            self.status.setText(text)

    def _on_row_done(self, dub, r):
        self.results[dub] = r
        if dub in self.dubs:
            self._fill_row(self.dubs.index(dub), dub)
        if dub == self._current_dub():
            self._update_plot()
            self._load_player()
        self._refresh_buttons()

    def _on_row_failed(self, dub, msg):
        self._errors.append(f"{os.path.basename(dub)}: {msg}")
        self._set_status_for(dub, "ошибка", "#e5534b")

    def _on_finished(self):
        dt = time.time() - self._t0
        if self._errors:
            self.status.setText("Готово с ошибками — подробности в окне.")
            QMessageBox.warning(self, "VoxMatch", "Не все дубли обработаны:\n\n" +
                                "\n".join(self._errors))
        elif self.worker is not None and self.worker.cancelled:
            self.status.setText("Остановлено.")
        else:
            self.status.setText(f"Готово за {dt:.1f} с. Перетащите результат в Cubase.")
        self.progress.setValue(100)
        cancelled = self.worker is not None and self.worker.cancelled
        self._retire_worker()
        self._refresh_buttons()
        if self._then_comp and not cancelled and self.results:
            self._then_comp = False
            self._build_comp(reset=True, switch=True)
        elif self.comp is not None and self.results and not cancelled:
            self._build_comp(reset=False)            # дубли пересчитаны — обновить комп
        self._then_comp = False
        QTimer.singleShot(0, self._run_inbox_queue)

    # ---------------------------------------------------------------- комп
    def _run_everything(self):
        """Всё сразу: выровнять все дубли, почистить, собрать комп, сохранить файл."""
        if not (self.ref_path and self.dubs):
            return
        todo = [d for d in self.dubs if d not in self.results]
        if not todo and self.results:
            self._build_comp(reset=True, switch=True)
            return
        self._then_comp = True
        self._start(list(self.dubs))
        if self.worker is None:
            self._then_comp = False

    def _build_comp(self, reset=False, switch=False):
        if not self.results or not self.ref_path or self.session is None:
            return
        if self.worker is not None and self.worker.isRunning():
            self._then_comp = True
            return
        choices = None if reset or self.comp is None else list(self.comp["choices"])
        if self._restore_choices:                    # выбор фраз из прошлой сессии
            choices = [c for c in self._restore_choices]
            self._restore_choices = None
            if any(c not in self.results for c in choices):
                choices = None
        if self._comp_busy:
            self._comp_pending = (choices, switch)
            return
        self._comp_busy = True
        self.status.setText("Собираю комп…")
        results = {d: r for d, r in self.results.items() if d in self.dubs}
        ref, s, ses = self.ref_path, self.current_settings(), self.session
        mk = engine().make_comp
        w = FuncWorker(lambda: mk(ses, ref, results, s, choices))
        w.done.connect(lambda c, sw=switch: self._comp_done(c, sw))
        w.failed.connect(self._comp_failed)
        self._retired.append(w)
        w.start()
        self._refresh_buttons()

    def _comp_done(self, c, switch):
        self._comp_busy = False
        self.comp = c
        self._sess_touch()
        self._fill_comp_table()
        n_auto = sum(1 for a, b in zip(c["choices"], c["auto"]) if a == b)
        self.lbl_comp.setText(f"Готово: {os.path.basename(c['out_path'])} · лучших фраз выбрано "
                              f"автоматически: {n_auto} из {len(c['choices'])}")
        self.status.setText("Комп собран. Перетащите его в Cubase.")
        if switch:
            self.tabs.setCurrentIndex(1)
        if self._comp_view():
            self._selection_changed()
        self._refresh_buttons()
        if self._comp_pending is not None:
            choices, sw = self._comp_pending
            self._comp_pending = None
            if choices is not None and self.comp is not None:
                self.comp["choices"] = choices
            self._build_comp(reset=False, switch=sw)

    def _comp_failed(self, msg):
        self._comp_busy = False
        self._comp_pending = None
        self.status.setText("Комп не собран.")
        QMessageBox.warning(self, "VoxMatch", f"Не удалось собрать комп:\n{msg}")
        self._refresh_buttons()

    def _fill_comp_table(self):
        c = self.comp
        t = self.comp_table
        t.setRowCount(len(c["choices"]))
        takes = [d for d in self.dubs if d in c["scores"]]
        for i, ((t0, t1), chosen) in enumerate(zip(c["phrases"], c["choices"])):
            it = QTableWidgetItem(str(i + 1))
            it.setTextAlignment(Qt.AlignCenter)
            t.setItem(i, 0, it)
            t.setItem(i, 1, QTableWidgetItem(f"{fmt_time(t0)} – {fmt_time(t1)}"))
            cmb = QComboBox()
            for d in takes:
                sc = c["scores"][d].phrases[i] if i < len(c["scores"][d].phrases) else float("nan")
                star = " ★" if d == c["auto"][i] else ""
                cmb.addItem(f"{os.path.basename(d)} — {sc:.0f}{star}", d)
            cmb.setCurrentIndex(max(0, takes.index(chosen)) if chosen in takes else 0)
            cmb.currentIndexChanged.connect(lambda _k, row=i, box=cmb: self._comp_choice(row, box))
            t.setCellWidget(i, 2, cmb)
            sc = c["scores"][chosen].phrases[i] if chosen in c["scores"] else float("nan")
            si = QTableWidgetItem(f"{sc:.0f}")
            si.setTextAlignment(Qt.AlignCenter)
            si.setForeground(QColor(score_color(sc)))
            det = c["scores"][chosen].details[i] if chosen in c["scores"] else None
            if det:
                si.setToolTip(f"Ноты: ±{det[0]} центов · тайминг: ±{det[1]} мс · спето: {det[2]}%"
                              + (" · ПЕРЕГРУЗ" if det[3] else ""))
            t.setItem(i, 3, si)

    def _comp_choice(self, row, box):
        if self.comp is None:
            return
        d = box.currentData()
        if d and row < len(self.comp["choices"]) and self.comp["choices"][row] != d:
            self.comp["choices"][row] = d
            self._build_comp(reset=False)

    def _comp_row_clicked(self, row, _col):
        if self.comp and row < len(self.comp["phrases"]):
            self._seek(max(0.0, self.comp["phrases"][row][0] - 0.3))

    # ---------------------------------------------------------------- график
    def _update_plot(self):
        if self._comp_view():
            self.wave.set_labels("Эталон", "Комп", COL_REF)
            c = self.comp
            if c is None:
                self.lbl_stats.setText("Комп ещё не собран.")
                self.wave.clear()
                return
            used = {}
            for d in c["choices"]:
                used[os.path.basename(d)] = used.get(os.path.basename(d), 0) + 1
            parts = [f"Комп из {len(c['choices'])} фраз: " +
                     ", ".join(f"{k} × {v}" for k, v in used.items())]
            if c["notes"]:
                parts.append("; ".join(c["notes"]))
            self.lbl_stats.setText("\n".join(parts))
            ref_pk = self._peaks_for_file(self.ref_path) if self.ref_path else None
            key = ("comp", id(c))
            if key not in self._peak_cache:
                self._peak_cache[key] = (wave_peaks(c["y"]), len(c["y"]) / c["sr"])
            self.wave.set_data(ref_pk, self._peak_cache[key], c["breaths"])
            return
        self.wave.set_labels("До", "После", COL_DUB)
        cur = self._current_dub()
        r = self.results.get(cur) if cur else None
        if r is None:
            self.lbl_stats.setText("Выберите дубль в таблице, чтобы увидеть результат."
                                   if self.results else "")
            self.wave.clear()
            return
        parts = [f"Тайминг: {r.timing_before_ms:.0f} → {r.timing_after_ms:.0f} мс"]
        if not np.isnan(r.pitch_before_c):
            parts.append(f"питч: {r.pitch_before_c:.0f} → {r.pitch_after_c:.0f} центов")
        if not np.isnan(r.dyn_before_db):
            parts.append(f"громкость: {r.dyn_before_db:.1f} → {r.dyn_after_db:.1f} дБ")
        if r.breaths:
            parts.append(f"вдохов не тронуто: {len(r.breaths)}")
        parts.append(f"обработка {r.seconds:.1f} с")
        self.lbl_stats.setText(" · ".join(parts) + ("\n" + " ".join(r.notes) if r.notes else ""))
        self.wave.set_data(self._peaks_for_file(cur), self._peaks_for_result(r), r.breaths)

    def _peaks_for_file(self, path):
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            return None
        if key not in self._peak_cache:
            try:
                x, sr = read_mono(path)
                self._peak_cache[key] = (wave_peaks(x), len(x) / sr)
            except Exception:  # noqa: BLE001
                return None
        return self._peak_cache[key]

    def _peaks_for_result(self, r):
        key = ("result", id(r))
        if key not in self._peak_cache:
            self._peak_cache[key] = (wave_peaks(r.y), len(r.y) / r.sr)
            if len(self._peak_cache) > 40:
                self._peak_cache.pop(next(iter(self._peak_cache)))
        return self._peak_cache[key]

    # ---------------------------------------------------------------- звук
    def _load_player(self):
        """Дорожки для плеера: эталон, выбранный дубль и его результат (или комп)."""
        comp_view = self._comp_view()
        cur = None if comp_view else self._current_dub()
        r = self.results.get(cur) if cur else None
        self.cb_listen["dub"].setEnabled(not comp_view)
        self.cb_listen["out"].setText("Комп" if comp_view else "Результат")
        tracks = {}
        try:
            if self.ref_path:
                tracks["ref"] = self._audio_for(self.ref_path)
            if cur:
                tracks["dub"] = self._audio_for(cur)
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())
        if r is not None:
            tracks["out"] = (r.y, r.sr)
        if comp_view and self.comp is not None:
            tracks["out"] = (self.comp["y"], self.comp["sr"])
            r = self.comp
        sr = (r.sr if hasattr(r, "sr") else r["sr"]) if r is not None else \
            next((v[1] for v in tracks.values() if v), 48000)
        self.player.set_tracks(sr, tracks)
        self._update_time()
        self._refresh_buttons()

    def _audio_for(self, path):
        key = ("audio", path, os.path.getmtime(path))
        if key not in self._peak_cache:
            self._peak_cache[key] = read_mono(path)
            for k in [k for k in self._peak_cache if k[0] == "audio" and k[1] == path and k != key]:
                self._peak_cache.pop(k)
        return self._peak_cache[key]

    def _toggle_play(self):
        if sd is None:
            return
        try:
            if self.player.playing:
                self.player.pause()
                self.play_timer.stop()
            else:
                if not any(self.player.on[k] and k in self.player.tracks for k in Player.NAMES):
                    for k in ("out", "dub", "ref"):        # ничего не выбрано — включить что есть
                        if k in self.player.tracks:
                            self.cb_listen[k].setChecked(True)
                            break
                self.player.play()
                self.play_timer.start()
        except Exception as exc:  # noqa: BLE001
            log_error(traceback.format_exc())
            QMessageBox.warning(self, "VoxMatch", f"Не удалось воспроизвести звук:\n{exc}")
        self._update_time()

    def _play(self, which):
        """Слушать одну дорожку с начала (двойной клик по строке)."""
        for k, cb in self.cb_listen.items():
            cb.setChecked(k == which)
        self.player.seek(0)
        if not self.player.playing:
            self._toggle_play()

    def _listen_toggled(self, key, on):
        self.player.on[key] = on

    def _seek(self, seconds):
        self.player.seek(seconds)
        self._update_time()

    def _seek_frac(self, frac):
        if self.player.length:
            self._seek(frac * self.player.length / self.player.sr)

    def _stop(self):
        if hasattr(self, "player"):
            self.player.pause()
        if hasattr(self, "play_timer"):
            self.play_timer.stop()
        if hasattr(self, "btn_toggle"):
            self._update_time()

    def _update_time(self):
        p = self.player
        total = p.length / p.sr if p.sr else 0.0
        t = min(p.time, total)
        self.lbl_time.setText(f"{fmt_time(t)} / {fmt_time(total)}")
        if not self.pos_slider.isSliderDown():
            self.pos_slider.blockSignals(True)
            self.pos_slider.setValue(int(1000 * t / total) if total else 0)
            self.pos_slider.blockSignals(False)
        self.btn_toggle.setText("⏸" if p.playing else "▶")
        self.wave.set_cursor(t if (p.playing or t > 0) else None)

    def _update_playhead(self):
        self._update_time()
        if not self.player.playing:
            self.play_timer.stop()

    def _open_folder(self):
        paths = self._selected_outputs() or [r.out_path for r in self.results.values()]
        if paths:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(paths[0])))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space:
            self._toggle_play()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        self._sess_save()
        self._stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancelled = True
            self.worker.wait(15000)
        for w in self._retired:
            w.wait(3000)
        self._save_settings()
        try:
            if _engine is not None:
                _engine.shutdown_workers()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(e)


class Player:
    """Простой плеер: несколько дорожек на одной шкале времени, одна позиция.
    Дорожки включаются и выключаются на лету, без остановки."""
    NAMES = ("ref", "dub", "out")

    def __init__(self):
        self.sr = 48000
        self.tracks: dict = {}               # имя → моно float32 при self.sr
        self.on = {"ref": False, "dub": False, "out": True}
        self.pos = 0                          # позиция в сэмплах
        self.playing = False
        self._stream = None

    @property
    def length(self) -> int:
        return max((len(t) for t in self.tracks.values()), default=0)

    def set_tracks(self, sr: int, tracks: dict):
        """tracks: имя → (сигнал, его частота). Всё приводится к sr."""
        out = {}
        for name, item in tracks.items():
            if item is None:
                continue
            x, xsr = item
            x = np.asarray(x, dtype=np.float32)
            if xsr != sr and len(x):
                try:
                    import soxr
                    x = soxr.resample(x, xsr, sr).astype(np.float32)
                except Exception:  # noqa: BLE001
                    n = int(len(x) * sr / xsr)
                    x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
            out[name] = x
        restart = self.playing and sr != self.sr
        if restart:
            self.pause()
        self.sr = sr
        self.tracks = out
        self.pos = min(self.pos, self.length)
        if restart:
            self.play()

    def clear(self):
        self.pause()
        self.tracks = {}
        self.pos = 0

    def _callback(self, outdata, frames, _time, _status):
        pos = self.pos
        buf = np.zeros(frames, dtype=np.float32)
        active = [n for n in self.NAMES if self.on.get(n) and n in self.tracks]
        for n in active:
            t = self.tracks[n]
            seg = t[pos:pos + frames]
            buf[:len(seg)] += seg
        if len(active) > 1:
            buf *= 0.7
        np.clip(buf, -1.0, 1.0, out=buf)
        outdata[:, 0] = buf
        self.pos = pos + frames
        if self.pos >= self.length:
            self.playing = False
            raise sd.CallbackStop

    def play(self):
        if sd is None or not self.tracks:
            return
        if self.pos >= self.length:
            self.pos = 0
        self._close()
        self._stream = sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32",
                                       blocksize=1024, callback=self._callback)
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

    def seek(self, seconds: float):
        self.pos = int(max(0.0, min(seconds * self.sr, self.length)))

    @property
    def time(self) -> float:
        return self.pos / self.sr if self.sr else 0.0


def score_color(sc: float) -> str:
    if sc != sc:            # nan
        return "#8a94a1"
    if sc >= 80:
        return "#3fb6a8"
    if sc >= 60:
        return "#e8a33d"
    return "#e5534b"


class FuncWorker(QThread):
    """Фоновая задача: выполнить функцию и вернуть результат."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as exc:  # noqa: BLE001
            log_error(traceback.format_exc())
            self.failed.emit(str(exc))


def fmt_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{t % 60:04.1f}"


def wave_peaks(x: np.ndarray, cols: int = 1600) -> np.ndarray:
    """Огибающая для рисования: максимум модуля в каждом столбце (считается один раз)."""
    x = np.abs(np.asarray(x, dtype=np.float32))
    if len(x) == 0:
        return np.zeros(cols, dtype=np.float32)
    step = max(1, len(x) // cols)
    n = (len(x) // step) * step
    p = x[:n].reshape(-1, step).max(axis=1)
    return p


class WaveView(QWidget):
    """Две простые звуковые волны: дубль до обработки и после.
    Рисуются один раз в картинку; при воспроизведении двигается только линия.
    Клик — перемотка."""
    ROWS = (("До", COL_DUB), ("После", COL_OUT))
    seekRequested = Signal(float)

    def set_labels(self, top, bottom, top_color=COL_DUB):
        rows = ((top, top_color), (bottom, COL_OUT))
        if rows != self.ROWS:
            self.ROWS = rows
            self._pix = None
            self.update()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._peaks = [None, None]      # огибающие
        self._dur = [0.0, 0.0]          # длительности, с
        self._breaths = []
        self._pix = None
        self._cursor = None

    def set_data(self, before, after, breaths):
        """before/after: (огибающая, длительность в секундах) или None."""
        for i, d in enumerate((before, after)):
            self._peaks[i], self._dur[i] = (d if d is not None else (None, 0.0))
        self._breaths = list(breaths or [])
        self._pix = None
        self.update()

    def clear(self):
        self.set_data(None, None, [])

    def set_cursor(self, t):
        self._cursor = t
        self.update()

    def resizeEvent(self, e):
        self._pix = None
        super().resizeEvent(e)

    def mousePressEvent(self, e):
        total = max(self._dur)
        if total <= 0:
            return
        x = e.position().x()
        frac = (x - 52) / max(1.0, self.width() - 60)
        self.seekRequested.emit(max(0.0, min(1.0, frac)) * total)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.mousePressEvent(e)

    def _render(self):
        w, h = max(1, self.width()), max(1, self.height())
        pix = QPixmap(w, h)
        pix.fill(QColor("#1a1d22"))
        p = QPainter(pix)
        total = max(self._dur) or 1.0
        label_w = 52
        row_h = h / 2
        p.setPen(QColor("#8a94a1"))
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        for i, (name, color) in enumerate(self.ROWS):
            y0 = i * row_h
            mid = y0 + row_h / 2
            p.setPen(QColor("#8a94a1"))
            p.drawText(8, int(mid + 4), name)
            if i == 1:                                   # вдохи, которые не трогали
                for a, b in self._breaths:
                    xa = label_w + (w - label_w - 8) * a / total
                    xb = label_w + (w - label_w - 8) * b / total
                    p.fillRect(int(xa), int(y0 + 3), max(1, int(xb - xa)), int(row_h - 6),
                               QColor(160, 170, 185, 45))
            pk = self._peaks[i]
            if pk is None or len(pk) == 0:
                continue
            width = (w - label_w - 8) * (self._dur[i] / total)
            cols = max(1, int(width))
            idx = np.linspace(0, len(pk) - 1, cols).astype(int)
            col = pk[idx]
            scale = (row_h / 2 - 4) / max(float(pk.max()), 1e-6)
            p.setPen(QPen(QColor(color), 1))
            for c in range(cols):
                a = col[c] * scale
                x = label_w + c
                p.drawLine(x, int(mid - a), x, int(mid + a))
        p.setPen(QColor("#2b3038"))
        p.drawLine(0, int(row_h), w, int(row_h))
        p.end()
        return pix

    def paintEvent(self, e):
        if self._pix is None or self._pix.size() != self.size():
            self._pix = self._render()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._pix)
        total = max(self._dur)
        if self._cursor is not None and total > 0:
            x = 52 + (self.width() - 60) * min(1.0, self._cursor / total)
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawLine(int(x), 0, int(x), self.height())
        p.end()


STYLE = f"""
QWidget {{ background: #1e2126; color: #e6e8eb; font-size: 10pt; }}
QLabel, QCheckBox {{ background: transparent; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: #1e2126; width: 10px; }}
QScrollBar::handle:vertical {{ background: #3a414b; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QLabel#appTitle {{ font-size: 17pt; font-weight: 700; color: #ffffff; }}
QLabel#hint {{ color: #8a94a1; }}
QLabel#dropTitle {{ font-size: 11pt; font-weight: 600; }}
QGroupBox {{ border: 1px solid #343a42; border-radius: 8px; margin-top: 14px; padding: 10px 10px 8px 10px;
            font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #9fb3c8; }}
#dropArea {{ border: 2px dashed #4a5563; border-radius: 10px; background: #23272e; }}
#dropArea:hover {{ border-color: #5d6875; }}
#dropArea[hover="true"] {{ border-color: {ACCENT}; background: #1f3332; }}
#dropArea[filled="true"] {{ border-style: solid; border-color: #2c6e66; }}
QPushButton {{ background: #2b3038; border: 1px solid #3a414b; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ background: #343a44; }}
QPushButton:pressed {{ background: #262a31; }}
QPushButton:disabled {{ color: #5f6772; border-color: #30353d; }}
QPushButton#primary {{ background: #2f9e91; border: none; color: #ffffff; font-weight: 700;
                       padding: 9px 26px; font-size: 11pt; }}
QPushButton#primary:hover {{ background: #36b3a4; }}
QPushButton#primary:disabled {{ background: #2b3038; color: #5f6772; }}
QSlider::groove:horizontal {{ height: 4px; background: #3a414b; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::sub-page:horizontal:disabled {{ background: #4a5563; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; background: #e6e8eb; }}
QSlider::handle:horizontal:disabled {{ background: #5f6772; }}
QProgressBar {{ border: 1px solid #3a414b; border-radius: 5px; background: #23272e; height: 14px; }}
QProgressBar::chunk {{ background: #2f9e91; border-radius: 4px; }}
QTableWidget {{ background: #23272e; alternate-background-color: #262b32; border: 1px solid #343a42;
               border-radius: 6px; selection-background-color: #2c4a47; selection-color: #ffffff; }}
QHeaderView::section {{ background: #2b3038; color: #9fb3c8; border: none; padding: 5px; font-weight: 600; }}
QComboBox, QSpinBox {{ background: #2b3038; border: 1px solid #3a414b; border-radius: 5px; padding: 3px 6px; }}
QComboBox QAbstractItemView {{ background: #2b3038; selection-background-color: #2c4a47; }}
QCheckBox::indicator {{ width: 15px; height: 15px; }}
QLabel#chip {{ border: 1px dashed {ACCENT}; border-radius: 6px; padding: 7px 10px; color: {ACCENT};
               font-weight: 600; background: #1f2b2b; }}
QLabel#chip:disabled {{ border-color: #3a414b; color: #5f6772; background: #23272e; }}
QPushButton#playBtn {{ background: #2f9e91; border: none; color: #ffffff; font-size: 13pt;
                       min-width: 44px; padding: 4px 10px; }}
QPushButton#playBtn:hover {{ background: #36b3a4; }}
QPushButton#playBtn:disabled {{ background: #2b3038; color: #5f6772; }}
QPushButton#advToggle {{ background: transparent; border: 1px dashed #3a414b; color: #9fb3c8;
                         text-align: left; padding: 6px 10px; font-weight: 600; }}
QPushButton#advToggle:hover {{ color: {ACCENT}; border-color: {ACCENT}; }}
QPushButton#advToggle:checked {{ color: {ACCENT}; border-style: solid; }}
QLabel#tplDesc {{ color: #b9c4cf; background: #23272e; border-left: 3px solid {ACCENT};
                  padding: 5px 8px; border-radius: 3px; }}
QLabel#sectionTitle {{ color: #9fb3c8; font-weight: 700; }}
QPushButton#primarySmall {{ background: #2f9e91; border: none; color: #ffffff; font-weight: 700;
                            padding: 6px 14px; }}
QPushButton#primarySmall:hover {{ background: #36b3a4; }}
QPushButton#primarySmall:disabled {{ background: #2b3038; color: #5f6772; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: #23272e; color: #9fb3c8; padding: 6px 16px; border-top-left-radius: 6px;
               border-top-right-radius: 6px; margin-right: 2px; font-weight: 600; }}
QTabBar::tab:selected {{ background: #2c4a47; color: #ffffff; }}
QFrame#inbox {{ background: #1f2b2b; border: 1px solid #2c6e66; border-radius: 8px; }}
QLabel#inboxTitle {{ color: {ACCENT}; font-weight: 700; }}
QPushButton#inboxPath {{ background: transparent; border: none; color: #e6e8eb; text-align: left;
                         padding: 2px 4px; font-weight: 600; }}
QPushButton#inboxPath:hover {{ color: {ACCENT}; }}
QRadioButton {{ background: transparent; }}
QFrame#sep {{ color: #343a42; background: #343a42; max-height: 1px; border: none; }}
QToolTip {{ background: #2b3038; color: #e6e8eb; border: 1px solid #3a414b; padding: 4px; }}
"""


def main():
    multiprocessing.freeze_support()

    def excepthook(t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        log_error(text)
        try:
            QMessageBox.critical(None, "VoxMatch", f"Непредвиденная ошибка:\n{v}\n\n"
                                 f"Подробности записаны в {LOG_PATH}")
        except Exception:  # noqa: BLE001
            print(text, file=sys.stderr)

    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    app.setApplicationName("VoxMatch")
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
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
