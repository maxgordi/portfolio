from typing import Callable, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from i18n import t
from styles import CLEAN_NAME, StylesManager


class StylesDialog(QDialog):
    """Edit post-processing styles. Same behaviour as the macOS window:
    edits are saved when another style is selected or the window closes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mgr: Optional[StylesManager] = None
        self._on_change: Optional[Callable] = None
        self._names: list = []
        self._selected: Optional[str] = None
        self._loading = False

        self.resize(640, 420)

        # Left: list + add/remove
        self.list = QListWidget()
        self.list.setMinimumWidth(180)
        self.list.currentRowChanged.connect(self._row_changed)
        self.add_btn = QPushButton("+")
        self.del_btn = QPushButton("−")
        for b in (self.add_btn, self.del_btn):
            b.setFixedWidth(36)
        self.add_btn.clicked.connect(self.add_style)
        self.del_btn.clicked.connect(self.delete_style)
        left_buttons = QHBoxLayout()
        left_buttons.addWidget(self.add_btn)
        left_buttons.addWidget(self.del_btn)
        left_buttons.addStretch(1)
        left = QVBoxLayout()
        left.addWidget(self.list, 1)
        left.addLayout(left_buttons)

        # Right: editor
        self.name_edit = QLineEdit()
        self.prompt_edit = QPlainTextEdit()
        self.active_check = QCheckBox()
        self.active_check.clicked.connect(self._active_toggled)
        self.hint = QLabel()
        self.hint.setEnabled(False)
        self.close_btn = QPushButton()
        self.close_btn.setDefault(True)
        self.close_btn.clicked.connect(self.close)
        self.name_label = QLabel()
        self.prompt_label = QLabel()

        name_row = QHBoxLayout()
        name_row.addWidget(self.name_label)
        name_row.addWidget(self.name_edit, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(self.hint)
        bottom.addStretch(1)
        bottom.addWidget(self.close_btn)
        right = QVBoxLayout()
        right.addLayout(name_row)
        right.addWidget(self.prompt_label)
        right.addWidget(self.prompt_edit, 1)
        right.addWidget(self.active_check)
        right.addLayout(bottom)

        root = QHBoxLayout(self)
        root.addLayout(left)
        root.addSpacing(8)
        root.addLayout(right, 1)
        self.retranslate()

    def retranslate(self) -> None:
        self.setWindowTitle(t("styles.title"))
        self.name_label.setText(t("styles.name"))
        self.prompt_label.setText(t("styles.prompt"))
        self.prompt_edit.setPlaceholderText(t("styles.prompt_placeholder"))
        self.active_check.setText(t("styles.active"))
        self.hint.setText(t("styles.clean_hint", clean=t("styles.clean")))
        self.close_btn.setText(t("styles.close"))
        if self._mgr is not None and self.isVisible():
            self._reload(self._selected or CLEAN_NAME)

    @staticmethod
    def _shown(name: str) -> str:
        """The built-in style is stored as "Clean" but shown translated."""
        return t("styles.clean") if name == CLEAN_NAME else name

    def open_styles(self, manager: StylesManager,
                    on_change: Optional[Callable] = None) -> None:
        self._mgr = manager
        self._on_change = on_change
        self._reload(self._mgr.active_name)
        self.show()
        self.raise_()
        self.activateWindow()

    # ── State ────────────────────────────────────────────────────────────────

    def _reload(self, select_name: str) -> None:
        self._loading = True
        try:
            self._names = [s["name"] for s in self._mgr.get_all()]
            self.list.clear()
            active = self._mgr.active_name
            for name in self._names:
                shown = self._shown(name)
                self.list.addItem(f"{shown}  ✓" if name == active else shown)
            if select_name not in self._names:
                select_name = self._names[0]
            self.list.setCurrentRow(self._names.index(select_name))
        finally:
            self._loading = False
        self._load_style(select_name)

    def _load_style(self, name: str) -> None:
        style = next((s for s in self._mgr.get_all() if s["name"] == name), None)
        if style is None:
            return
        self._selected = name
        self.name_edit.setText(self._shown(style["name"]))
        self.prompt_edit.setPlainText(style["prompt"])
        is_clean = name == CLEAN_NAME
        self.name_edit.setEnabled(not is_clean)
        self.prompt_edit.setReadOnly(is_clean)
        self.del_btn.setEnabled(not is_clean)
        self.active_check.setChecked(name == self._mgr.active_name)

    def _apply_edits(self) -> None:
        name = self._selected
        if not name or name == CLEAN_NAME or self._mgr is None:
            return
        final = self._mgr.update(name, self.name_edit.text(),
                                 self.prompt_edit.toPlainText())
        self._selected = final
        self._notify()

    def _notify(self) -> None:
        if self._on_change:
            self._on_change()

    # ── Actions ──────────────────────────────────────────────────────────────

    def _row_changed(self, row: int) -> None:
        if self._loading:
            return
        self._apply_edits()
        if 0 <= row < len(self._names):
            # A rename may have changed the list; rebuild before loading.
            target = self._names[row]
            if target not in [s["name"] for s in self._mgr.get_all()]:
                target = self._selected
            self._reload(target)

    def add_style(self) -> None:
        self._apply_edits()
        name = self._mgr.add(t("styles.new"))
        self._reload(name)
        self.name_edit.setFocus()
        self.name_edit.selectAll()
        self._notify()

    def delete_style(self) -> None:
        name = self._selected
        if not name or name == CLEAN_NAME:
            return
        self._selected = None  # gone; nothing to apply
        self._mgr.delete(name)
        self._reload(CLEAN_NAME)
        self._notify()

    def _active_toggled(self, checked: bool) -> None:
        self._apply_edits()
        self._mgr.set_active(self._selected if checked else CLEAN_NAME)
        self._reload(self._selected or CLEAN_NAME)
        self._notify()

    def done(self, result: int) -> None:
        # Close button, window ✕ and Esc all end up here.
        self._apply_edits()
        self._notify()
        super().done(result)
