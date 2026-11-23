import threading
import time
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import config as cfg_module
import i18n
from audio import AudioRecorder, wav_peak
from i18n import t
from live.session import TEXT_MODES, text_mode
from qtapp import platform_utils
from qtapp.hotkey import DEFAULT_HOTKEY, HOTKEYS, hotkey_id, hotkey_label
from transcribe import (
    friendly_error,
    is_digital_silence,
    list_transcription_models,
    make_client,
    model_choices,
)
from version import __version__

MODES = ["push_to_talk", "toggle"]
_KEYS_URL = "https://platform.openai.com/api-keys"


def _fetch_models(api_key: str) -> list:
    return list_transcription_models(make_client(api_key))


def _test_microphone(index: int, name: str) -> tuple:
    """Record 1.5 s from the chosen microphone → (works, explanation)."""
    rec = AudioRecorder()
    try:
        rec.start(index, name)
    except Exception as e:
        block = platform_utils.windows_microphone_block()
        if block:
            return False, t("mic_test.blocked", reason=block)
        return False, t("mic_test.open_failed", error=str(e)[:120])
    time.sleep(1.5)
    wav = rec.stop()
    if not wav:
        return False, t("mic_test.no_audio")
    if is_digital_silence(wav):
        return False, t("mic_test.silence")
    peak = f"{wav_peak(wav):.0%}"
    if rec.backend not in ("", "default"):
        message = t("mic_test.ok_via", backend=rec.backend, level=peak)
    else:
        message = t("mic_test.ok", level=peak)
    if wav_peak(wav) < 0.02:
        message += " " + t("mic_test.quiet")
    return True, message


def _row(*widgets) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for i, x in enumerate(widgets):
        lay.addWidget(x, 1 if i == 0 else 0)
    return w


def _dim(label: QLabel, smaller: bool = True) -> QLabel:
    pal = label.palette()
    pal.setColor(QPalette.WindowText, pal.color(QPalette.PlaceholderText))
    label.setPalette(pal)
    if smaller:
        font = label.font()
        font.setPointSizeF(font.pointSizeF() * 0.9)
        label.setFont(font)
    return label


def _hint(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    return _dim(label)


class SettingsDialog(QDialog):
    """Created once and reused: hiding instead of destroying keeps a model
    fetch that is still running from signalling into a deleted object.
    Every visible text is set in retranslate(), so the dialog can switch
    language in place."""

    _models_ready = Signal(list, str)
    _mic_tested = Signal(bool, str)

    def __init__(self, list_devices: Callable = AudioRecorder.list_devices,
                 fetch_models: Callable = _fetch_models,
                 test_microphone: Callable = _test_microphone, parent=None):
        super().__init__(parent)
        self._list_devices = list_devices
        self._fetch_models = fetch_models
        self._test_microphone = test_microphone
        self._testing_mic = False
        self._on_save: Optional[Callable] = None
        self._mic_entries: list = []
        self._fetching = False

        self.setMinimumWidth(540)
        self._models_ready.connect(self._on_models_ready)
        self._mic_tested.connect(self._on_mic_tested)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self._labels = {}

        def add_row(key: Optional[str], field) -> None:
            if key is None:
                form.addRow("", field)
            else:
                self._labels[key] = QLabel()
                form.addRow(self._labels[key], field)

        # API key
        self.api_edit = QLineEdit()
        self.api_edit.setEchoMode(QLineEdit.Password)
        self.api_edit.setPlaceholderText("sk-...")
        self.show_key_btn = QToolButton()
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self._toggle_key_visible)
        add_row("settings.api_key", _row(self.api_edit, self.show_key_btn))
        self.key_hint = _hint()
        self.key_hint.setOpenExternalLinks(True)
        add_row(None, self.key_hint)

        # Transcription model
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.refresh_btn = QToolButton()
        self.refresh_btn.clicked.connect(self.refresh_models)
        add_row("settings.model", _row(self.model_combo, self.refresh_btn))
        self.model_status = _hint()
        add_row(None, self.model_status)

        # Microphone
        self.mic_combo = QComboBox()
        self.mic_test_btn = QToolButton()
        self.mic_test_btn.clicked.connect(self.test_microphone)
        add_row("settings.microphone", _row(self.mic_combo, self.mic_test_btn))
        self.mic_status = _hint()
        self.mic_status.setOpenExternalLinks(True)
        self.mic_status.setVisible(False)
        add_row(None, self.mic_status)

        # Hotkey and recording mode (items are filled in retranslate)
        self.hotkey_combo = QComboBox()
        add_row("settings.hotkey", self.hotkey_combo)
        self.mode_combo = QComboBox()
        # Mode descriptions are sentences, long in some languages.
        self.mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        add_row("settings.mode", self.mode_combo)
        self.text_combo = QComboBox()
        self.text_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        add_row("settings.text_mode", self.text_combo)
        self.text_hint = _hint()
        add_row(None, self.text_hint)

        # Interface language
        self.language_combo = QComboBox()
        add_row("settings.language", self.language_combo)

        # Autostart
        self.autostart_check = QCheckBox()
        self.autostart_check.setVisible(platform_utils.autostart_supported())
        if platform_utils.autostart_supported():
            add_row(None, self.autostart_check)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.save)
        self.buttons.rejected.connect(self.hide)
        version = _dim(QLabel(f"PressVoice {__version__}"), smaller=False)
        bottom = QHBoxLayout()
        bottom.addWidget(version)
        bottom.addStretch(1)
        bottom.addWidget(self.buttons)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addSpacing(8)
        root.addLayout(bottom)
        self.retranslate()

    # ── Texts ────────────────────────────────────────────────────────────────

    def retranslate(self) -> None:
        """Apply every text in the current language, keeping selections."""
        self.setWindowTitle(t("settings.title"))
        for key, label in self._labels.items():
            label.setText(t(key))
        self._toggle_key_visible(self.show_key_btn.isChecked())
        self.key_hint.setText(t("settings.key_hint",
                                link=f'<a href="{_KEYS_URL}">platform.openai.com/api-keys</a>'))
        self.refresh_btn.setText(t("settings.refresh"))
        self.refresh_btn.setToolTip(t("settings.refresh_tooltip"))
        self.model_status.setText(t("settings.models_hint"))
        self.mic_test_btn.setText(t("settings.test"))
        self.mic_test_btn.setToolTip(t("settings.test_tooltip"))
        if self.mic_combo.count():
            self.mic_combo.setItemText(0, t("settings.system_default"))

        def refill(combo: QComboBox, items: list) -> None:
            current = combo.currentData()
            combo.clear()
            for data, label in items:
                combo.addItem(label, data)
            idx = combo.findData(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)

        refill(self.hotkey_combo, [(k, hotkey_label(k)) for k in HOTKEYS])
        refill(self.mode_combo, [(m, t(f"mode.{m}")) for m in MODES])
        refill(self.text_combo, [(m, t(f"text_mode.{m}")) for m in TEXT_MODES])
        self.text_hint.setText(t("settings.text_mode_hint"))
        refill(self.language_combo, i18n.language_choices())
        self.autostart_check.setText(platform_utils.autostart_label())
        self.buttons.button(QDialogButtonBox.Save).setText(t("settings.save"))
        self.buttons.button(QDialogButtonBox.Cancel).setText(t("settings.cancel"))

    # ── Show / load ──────────────────────────────────────────────────────────

    def open_settings(self, on_save: Optional[Callable] = None,
                      message: str = "") -> None:
        self._on_save = on_save
        if not self.isVisible():
            self._load(cfg_module.load())
        if message:
            self.model_status.setText(message)
        self.show()
        self.raise_()
        self.activateWindow()
        if not self.api_edit.text():
            self.api_edit.setFocus()

    def _load(self, cfg: dict) -> None:
        self.api_edit.setText(cfg.get("api_key", ""))
        self.show_key_btn.setChecked(False)

        self.model_combo.clear()
        self.model_combo.addItems(model_choices(cfg))
        self.model_combo.setCurrentText(
            cfg.get("transcription_model") or cfg_module.DEFAULT_MODEL)
        self.model_status.setText(t("settings.models_hint"))

        self.mic_combo.clear()
        self._mic_entries = [(-1, "")]
        self.mic_combo.addItem(t("settings.system_default"))
        try:
            devices = self._list_devices()
        except Exception:
            devices = []
        for dev in devices:
            self.mic_combo.addItem(dev["name"])
            self._mic_entries.append((dev["index"], dev["name"]))
        saved_name = cfg.get("microphone_name", "")
        saved_index = cfg.get("microphone_index", -1)
        names = [n for _, n in self._mic_entries]
        indices = [i for i, _ in self._mic_entries]
        if saved_name and saved_name in names:
            self.mic_combo.setCurrentIndex(names.index(saved_name))
        elif saved_index in indices:
            self.mic_combo.setCurrentIndex(indices.index(saved_index))
        else:
            self.mic_combo.setCurrentIndex(0)

        self.mic_status.setVisible(False)

        def select(combo: QComboBox, value) -> None:
            idx = combo.findData(value)
            combo.setCurrentIndex(idx if idx >= 0 else 0)

        select(self.hotkey_combo, hotkey_id(cfg.get("hotkey", DEFAULT_HOTKEY)))
        select(self.mode_combo, cfg.get("recording_mode", "push_to_talk"))
        select(self.text_combo, text_mode(cfg))
        select(self.language_combo, cfg.get("language") or i18n.AUTO)

        if platform_utils.autostart_supported():
            try:
                self.autostart_check.setChecked(platform_utils.is_autostart_enabled())
            except Exception:
                self.autostart_check.setChecked(False)

    def _toggle_key_visible(self, visible: bool) -> None:
        self.api_edit.setEchoMode(QLineEdit.Normal if visible else QLineEdit.Password)
        self.show_key_btn.setText(t("settings.hide") if visible else t("settings.show"))

    # ── Model list refresh ───────────────────────────────────────────────────

    def refresh_models(self) -> None:
        if self._fetching:
            return
        key = self.api_edit.text().strip()
        if not key:
            self.model_status.setText(t("settings.enter_key_first"))
            return
        self._fetching = True
        self.refresh_btn.setEnabled(False)
        self.model_status.setText(t("settings.loading_models"))

        def work():
            try:
                models, error = self._fetch_models(key), ""
            except Exception as e:
                models, error = [], friendly_error(e)
            self._models_ready.emit(models, error)

        threading.Thread(target=work, name="models", daemon=True).start()

    def _on_models_ready(self, models: list, error: str) -> None:
        self._fetching = False
        self.refresh_btn.setEnabled(True)
        if error:
            self.model_status.setText(error)
            return
        if not models:
            self.model_status.setText(t("settings.no_models"))
            return
        # Cache the list right away, so it survives a Cancel.
        cfg = cfg_module.load()
        cfg["transcription_models"] = models
        try:
            cfg_module.save(cfg)
        except Exception:
            pass
        current = self.model_combo.currentText().strip() or models[0]
        self.model_combo.clear()
        self.model_combo.addItems(model_choices({**cfg, "transcription_model": current}))
        self.model_combo.setCurrentText(current)
        self.model_status.setText(
            t("settings.models_found", count=len(models)) + " " + t("settings.key_works"))

    # ── Microphone test ──────────────────────────────────────────────────────

    def test_microphone(self) -> None:
        if self._testing_mic:
            return
        sel = self.mic_combo.currentIndex()
        index, name = (self._mic_entries[sel]
                       if 0 <= sel < len(self._mic_entries) else (-1, ""))
        self._testing_mic = True
        self.mic_test_btn.setEnabled(False)
        self.mic_status.setText(t("mic_test.recording"))
        self.mic_status.setVisible(True)

        def work():
            try:
                ok, message = self._test_microphone(index, name)
            except Exception as e:
                ok, message = False, t("mic_test.failed", error=e)
            self._mic_tested.emit(ok, message)

        threading.Thread(target=work, name="mic-test", daemon=True).start()

    def _on_mic_tested(self, ok: bool, message: str) -> None:
        self._testing_mic = False
        self.mic_test_btn.setEnabled(True)
        if not ok and platform_utils.IS_WINDOWS:
            message += (f' <a href="{platform_utils.MIC_PRIVACY_URL}">'
                        f'{t("mic_test.open_privacy")}</a>')
        self.mic_status.setText(message)
        self.mic_status.setVisible(True)

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self) -> None:
        cfg = cfg_module.load()
        cfg["api_key"] = self.api_edit.text().strip()
        cfg["transcription_model"] = (
            self.model_combo.currentText().strip() or cfg_module.DEFAULT_MODEL)
        sel = self.mic_combo.currentIndex()
        index, name = (self._mic_entries[sel]
                       if 0 <= sel < len(self._mic_entries) else (-1, ""))
        cfg["microphone_index"] = index
        cfg["microphone_name"] = name
        cfg["hotkey"] = self.hotkey_combo.currentData() or DEFAULT_HOTKEY
        cfg["recording_mode"] = self.mode_combo.currentData() or "push_to_talk"
        cfg["text_mode"] = self.text_combo.currentData() or "whole"
        cfg["language"] = self.language_combo.currentData() or i18n.AUTO
        try:
            cfg_module.save(cfg)
        except Exception as e:
            QMessageBox.warning(self, "PressVoice", t("settings.save_failed", error=e))
            return

        if platform_utils.autostart_supported():
            want = self.autostart_check.isChecked()
            try:
                if want != platform_utils.is_autostart_enabled():
                    platform_utils.set_autostart(want)
            except Exception as e:
                QMessageBox.warning(self, "PressVoice",
                                    t("settings.autostart_failed", error=e))

        self.hide()
        if self._on_save:
            self._on_save(cfg)
