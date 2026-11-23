#!/usr/bin/env python3
"""PressVoice Local Test — the window.

Two modes: Whisper (accurate, ~2 s to recognise) and Vosk (fast, more
mistakes, Russian and English only). Hold Right ⌥ — or the big button — and
speak; on release the phrase is recognised, translated and spoken into the
headphones. Every phrase is listed with what took how long.

All work happens on one worker thread, one job at a time; the window only
shows results.
"""
import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (QApplication, QButtonGroup, QComboBox, QFormLayout, QLabel,
                               QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QVBoxLayout, QWidget)

import engine

HERE = Path(__file__).resolve().parent
LOG = HERE / "local_test.log"
SETTINGS = Path.home() / ".config/pressvoice/local_test.json"
WINDOW_SETTINGS = Path.home() / ".config/pressvoice/live_window.json"
METER_FLOOR = -60.0
RIGHT_OPTION = 0x40  # the right ⌥ bit of the HID modifier flags
# The microphone stays on this long after release (the last syllable); a new
# press within it continues the same phrase.
TAIL = 0.3
# (label, seconds of silence that end a phrase)
PAUSES = [("0,5 с — быстро, но может резать фразу", 0.5), ("0,7 с", 0.7), ("1,0 с", 1.0), ("1,5 с — для медленной речи", 1.5)]


def log(line: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {line}\n")


def load_settings() -> dict:
    """Own settings; the first time, languages and devices from the PressVoice window."""
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        pass
    try:
        w = json.loads(WINDOW_SETTINGS.read_text())
    except Exception:
        w = {}
    src, dst = engine.base_lang(w.get("my_lang", "ru")), engine.base_lang(w.get("direction", "en"))
    return {"mode": "whisper", "phrases": True, "pause": 0.7,
            "src": src if src in engine.LANGUAGES else "ru",
            "dst": dst if dst in engine.LANGUAGES else "en",
            "voices": {}, "mic": w.get("mic"), "out": w.get("headphones")}


def save_settings(s: dict) -> None:
    try:
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS.write_text(json.dumps(s, ensure_ascii=False, indent=2))
    except Exception:
        pass


class Bus(QObject):
    """Worker thread → window."""
    status = Signal(str)
    ready = Signal(bool)
    phrase = Signal(dict)


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.s = load_settings()
        self.bus = Bus()
        self.bus.status.connect(self._set_status)
        self.bus.ready.connect(self._set_ready)
        self.bus.phrase.connect(self._show_phrase)
        self.work = ThreadPoolExecutor(max_workers=1)
        self.recognizers = {}
        self.audio = None
        self.ready = False
        self.recording = False
        self.option_held = False
        self.cutter = None  # by-phrase mode: cuts the recording at pauses
        # The end of a press waits TAIL on the main thread (one thread: no locks).
        self.tail = QTimer(self)
        self.tail.setSingleShot(True)
        self.tail.timeout.connect(self._finish)

        self.setWindowTitle("PressVoice Local Test — бесплатный перевод на этом Mac")
        self.setMinimumSize(760, 640)
        big = QFont()
        big.setPointSizeF(big.pointSizeF() * 1.25)

        self.whisper = QRadioButton("Whisper — точнее (~2 с на распознавание)")
        self.vosk = QRadioButton("Vosk — быстрее, но больше ошибок (только русский и английский)")
        modes = QButtonGroup(self)
        modes.addButton(self.whisper)
        modes.addButton(self.vosk)
        (self.whisper if self.s["mode"] == "whisper" else self.vosk).setChecked(True)

        self.by_phrase = QRadioButton("По фразам — каждая фраза переводится сама, как только вы сделали паузу")
        self.whole = QRadioButton("Целиком — всё сказанное переводится после отпускания клавиши")
        sending = QButtonGroup(self)
        sending.addButton(self.by_phrase)
        sending.addButton(self.whole)
        (self.by_phrase if self.s.get("phrases", True) else self.whole).setChecked(True)
        self.pause = QComboBox()
        for label, seconds in PAUSES:
            self.pause.addItem(label, seconds)
        self.pause.setCurrentIndex(max(self.pause.findData(self.s.get("pause", 0.7)), 0))
        self.pause.setEnabled(self.by_phrase.isChecked())

        self.src = QComboBox()
        self.dst = QComboBox()
        for code, (name, _) in engine.LANGUAGES.items():
            self.src.addItem(name, code)
            self.dst.addItem(name, code)
        self.src.setCurrentIndex(self.src.findData(self.s["src"]))
        self.dst.setCurrentIndex(self.dst.findData(self.s["dst"]))
        self.voice = QComboBox()
        self.mic = QComboBox()
        self.out = QComboBox()
        for combo, names, saved in ((self.mic, engine.devices(False), self.s.get("mic")),
                                    (self.out, engine.devices(True), self.s.get("out"))):
            combo.addItem("Системный по умолчанию", None)
            for name in names:
                if name != "PressVoice Microphone":
                    combo.addItem(name, name)
            index = combo.findData(saved)
            combo.setCurrentIndex(max(index, 0))
        self._fill_voices()

        form = QFormLayout()
        mode_box = QVBoxLayout()
        mode_box.addWidget(self.whisper)
        mode_box.addWidget(self.vosk)
        form.addRow("Распознавание:", mode_box)
        send_box = QVBoxLayout()
        send_box.addWidget(self.by_phrase)
        send_box.addWidget(self.whole)
        form.addRow("Отправлять:", send_box)
        form.addRow("Конец фразы — пауза:", self.pause)
        form.addRow("Я говорю на:", self.src)
        form.addRow("Перевод на:", self.dst)
        form.addRow("Голос перевода:", self.voice)
        form.addRow("Микрофон:", self.mic)
        form.addRow("Наушники:", self.out)

        self.talk = QPushButton("Удерживайте, чтобы говорить  (или правый ⌥ Option)")
        self.talk.setMinimumHeight(56)
        self.talk.setFont(big)
        self.talk.pressed.connect(self._press)
        self.talk.released.connect(self._release)
        self.meter = QProgressBar()
        self.meter.setRange(0, int(-METER_FLOOR))
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(10)
        self.status = QLabel("Загрузка…")
        self.status.setFont(big)
        self.status.setAlignment(Qt.AlignCenter)
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        font = self.history.font()
        font.setPointSizeF(font.pointSizeF() * 1.1)
        self.history.setFont(font)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addWidget(self.talk)
        layout.addWidget(self.meter)
        layout.addWidget(self.status)
        layout.addWidget(QLabel("Фразы (с временем каждого шага):"))
        layout.addWidget(self.history, 1)

        self.whisper.toggled.connect(self._reconfigure)
        self.src.currentIndexChanged.connect(self._reconfigure)
        self.dst.currentIndexChanged.connect(self._dst_changed)
        self.voice.currentIndexChanged.connect(self._reconfigure)
        self.by_phrase.toggled.connect(self._send_mode_changed)
        self.pause.currentIndexChanged.connect(self._send_mode_changed)
        self.mic.currentIndexChanged.connect(self._reopen_audio)
        self.out.currentIndexChanged.connect(self._reopen_audio)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(15)
        self._reopen_audio()
        self._reconfigure()

    # ── Settings ─────────────────────────────────────────────────────────────

    def _fill_voices(self):
        self.voice.blockSignals(True)
        self.voice.clear()
        lang = self.dst.currentData()
        for label, path in engine.LANGUAGES[lang][1]:
            self.voice.addItem(label, path)
        saved = self.s.get("voices", {}).get(lang)
        self.voice.setCurrentIndex(max(self.voice.findData(saved), 0))
        self.voice.blockSignals(False)

    def _dst_changed(self):
        self._fill_voices()
        self._reconfigure()

    def _current(self) -> dict:
        self.s.update(mode="whisper" if self.whisper.isChecked() else "vosk",
                      phrases=self.by_phrase.isChecked(), pause=self.pause.currentData(),
                      src=self.src.currentData(), dst=self.dst.currentData(),
                      mic=self.mic.currentData(), out=self.out.currentData())
        self.s.setdefault("voices", {})[self.s["dst"]] = self.voice.currentData()
        save_settings(self.s)
        return dict(self.s, voice=self.voice.currentData())

    def _send_mode_changed(self):
        self.pause.setEnabled(self.by_phrase.isChecked())
        self._current()  # saved; takes effect at the next press

    def _reopen_audio(self):
        cfg = self._current()
        if self.audio:
            self.audio.close()
        try:
            self.audio = engine.Audio(cfg["mic"], cfg["out"])
        except Exception as e:
            self.audio = None
            self._set_status(f"Устройство не открывается: {e}")

    def _reconfigure(self):
        """Loads what the chosen mode, languages and voice need (on the worker)."""
        cfg = self._current()
        self._set_ready(False)
        self.status.setText("Загрузка…")
        self.work.submit(self._configure, cfg)

    def _configure(self, cfg):
        try:
            if cfg["mode"] == "vosk" and (cfg["src"] not in engine.VOSK_MODELS):
                raise RuntimeError("Vosk знает только русский и английский — выберите Whisper")
            if cfg["mode"] not in self.recognizers:
                self.bus.status.emit("Загружаю распознавание…")
                self.recognizers[cfg["mode"]] = engine.Whisper() if cfg["mode"] == "whisper" else engine.Vosk()
            engine.ensure_pair(cfg["src"], cfg["dst"], self.bus.status.emit)
            engine.translate("ok", cfg["src"], cfg["dst"])  # loads the dictionaries now, not on the first phrase
            self.voice_model = engine.voice(cfg["voice"], self.bus.status.emit)
            engine.speak(self.voice_model, "ok")
            self.cfg = cfg
            log(f"=== {cfg['mode']} {cfg['src']}→{cfg['dst']}, голос {cfg['voice'].rsplit('/', 1)[1]}, "
                f"mic={cfg['mic']}, out={cfg['out']}")
            self.bus.ready.emit(True)
        except Exception as e:
            self.bus.status.emit(f"Ошибка: {e}")

    # ── Talking ──────────────────────────────────────────────────────────────

    def _press(self):
        if not self.ready or self.recording or not self.audio:
            return
        if self.tail.isActive():
            self.tail.stop()  # pressed again within the tail: the same recording goes on
        else:
            self.audio.start()
            self.cutter = (engine.PhraseCutter(self.audio.mic_rate, self.s["pause"])
                           if self.s.get("phrases") else None)
        self.recording = True
        self.talk.setText("● Говорите… отпустите, когда закончите")

    def _release(self):
        if not self.recording:
            return
        self.recording = False
        self.talk.setText("Удерживайте, чтобы говорить  (или правый ⌥ Option)")
        self.status.setText("Перевожу…")
        self.tail.start(int(TAIL * 1000))

    def _cut_phrases(self):
        """By-phrase mode, while recording: every finished phrase goes to the worker."""
        if self.cutter is not None and self.audio and (self.recording or self.tail.isActive()):
            for phrase in self.cutter.feed(self.audio.drain()):
                self.work.submit(self._process, phrase, self.s["pause"], True)

    def _finish(self):
        """The tail has passed: what is left goes to the worker."""
        if self.cutter is not None:
            self._cut_phrases()
            rest = self.audio.drain()
            self.audio.take()  # stops recording
            last = self.cutter.flush(rest)
            self.cutter = None
            if last is not None:
                self.work.submit(self._process, last, TAIL, True)
        else:
            self.work.submit(self._process, self.audio.take(), TAIL, False)

    def _process(self, pcm, waited, by_phrase):
        """`waited`: how long after the speech ended the phrase was sent
        (the pause that ended it, or the tail after release)."""
        seconds = len(pcm) / engine.RATE
        if seconds < 0.4:
            self.bus.status.emit("Слишком коротко — держите клавишу, пока говорите")
            return
        cfg = self.cfg
        t0 = time.perf_counter()
        try:
            text = self.recognizers[cfg["mode"]](pcm, cfg["src"])
            t_stt = time.perf_counter() - t0
            if not text:
                self.bus.status.emit(f"Ничего не распознано ({t_stt:.1f} с)")
                return
            t = time.perf_counter()
            translated = engine.translate(text, cfg["src"], cfg["dst"])
            t_mt = time.perf_counter() - t
            t = time.perf_counter()
            samples, rate = engine.speak(self.voice_model, translated)
            t_tts = time.perf_counter() - t
            if self.audio:
                self.audio.play(samples, rate)
            total = time.perf_counter() - t0 + waited
        except Exception as e:
            self.bus.status.emit(f"Ошибка: {e}")
            return
        result = dict(text=text, translated=translated, seconds=seconds, stt=t_stt, mt=t_mt, tts=t_tts,
                      total=total, waited=waited, length=len(samples) / rate, mode=cfg["mode"])
        log(f"{cfg['mode']} {'по фразам' if by_phrase else 'целиком'} {seconds:.1f}s · пауза {waited:.1f} · stt {t_stt:.2f} mt {t_mt:.2f} tts {t_tts:.2f} → {total:.2f}s | "
            f"{text} → {translated}")
        self.bus.phrase.emit(result)

    # ── Window ───────────────────────────────────────────────────────────────

    def _tick(self):
        """Right ⌥ (polled from the system — no key interception needed) and the meter."""
        try:
            import Quartz
            held = bool(Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateHIDSystemState) & RIGHT_OPTION)
        except Exception:
            held = False
        if held != self.option_held:
            self.option_held = held
            self._press() if held else self._release()
        self._cut_phrases()
        if self.audio:
            level = self.audio.mic_level if self.recording else max(self.audio.mic_level, self.audio.out_level)
            self.meter.setValue(int(max(0, min(-METER_FLOOR, level - METER_FLOOR))))

    def _set_status(self, text):
        self.status.setText(text)

    def _set_ready(self, ready):
        self.ready = ready
        self.talk.setEnabled(ready)
        if ready:
            self.status.setText("Готово · держите правый ⌥ или кнопку и говорите")

    def _show_phrase(self, r):
        self.history.moveCursor(QTextCursor.End)
        self.history.insertPlainText(
            f"Вы:       {r['text']}\nПеревод:  {r['translated']}\n"
            f"          фраза {r['seconds']:.1f} с · распознавание ({r['mode']}) {r['stt']:.2f} с · "
            f"перевод {r['mt']:.2f} с · голос {r['tts']:.2f} с → звучит через {r['total']:.1f} с "
            f"после конца речи (из них ожидание паузы {r['waited']:.1f} с), длится {r['length']:.1f} с\n\n")
        self.history.moveCursor(QTextCursor.End)
        self.status.setText(f"Перевод через {r['total']:.1f} с после конца речи")

    def closeEvent(self, event):
        for recognizer in self.recognizers.values():
            recognizer.close()
        if self.audio:
            self.audio.close()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv[:1])
    window = Window()
    # Closing the Terminal window must also stop whisper-server.
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: window.close() or app.quit())
    keepalive = QTimer()  # lets Python see signals while Qt runs
    keepalive.start(200)
    keepalive.timeout.connect(lambda: None)
    window.show()
    window.raise_()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
