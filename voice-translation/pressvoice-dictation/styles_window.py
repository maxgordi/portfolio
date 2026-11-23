import objc
from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSBezelBorder,
    NSButton,
    NSColor,
    NSFont,
    NSMakeRect,
    NSMakeSize,
    NSPanel,
    NSScrollView,
    NSSwitchButton,
    NSTableColumn,
    NSTableView,
    NSTextField,
    NSTextView,
    NSViewWidthSizable,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskUtilityWindow,
)
from Foundation import NSIndexSet, NSObject

from i18n import t
from styles import CLEAN_NAME, StylesManager

W = 560
H = 400
MARGIN = 20
LIST_W = 170
RIGHT_X = MARGIN + LIST_W + 20  # 210
RIGHT_W = W - RIGHT_X - MARGIN  # 330
_BIG = 1e7  # effectively "unlimited" for text container sizing


def _label(text: str, x: float, y: float, w: float) -> NSTextField:
    field = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, 20))
    field.setStringValue_(text)
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(False)
    field.setFont_(NSFont.systemFontOfSize_(13))
    return field


class StylesWindowController(NSObject):
    _instance = None

    @classmethod
    def shared(cls):
        if cls._instance is None:
            cls._instance = cls.alloc().init()
        return cls._instance

    def init(self):
        self = objc.super(StylesWindowController, self).init()
        if self is None:
            return None
        self._window = None
        self._mgr = None
        self._on_change = None
        self._names = []
        self._selected_name = None
        self._loading = False
        return self

    @objc.python_method
    def forget_window(self):
        """Language changed: build the window afresh next time it opens."""
        if self._window is not None and not self._window.isVisible():
            self._window = None

    @objc.python_method
    def show(self, manager: StylesManager, on_change=None):
        self._mgr = manager
        self._on_change = on_change
        if self._window is None:
            self._build_window()
        self._reload(select_name=self._mgr.active_name)
        self._window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_window(self):
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskUtilityWindow
        )
        self._window = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H), style, NSBackingStoreBuffered, False
        )
        self._window.setTitle_(t("styles.title"))
        self._window.center()
        self._window.setReleasedWhenClosed_(False)
        self._window.setDelegate_(self)
        content = self._window.contentView()

        # ── Left: styles list ──────────────────────────────────────────────
        list_y = MARGIN + 36
        list_h = H - MARGIN - list_y
        scroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(MARGIN, list_y, LIST_W, list_h)
        )
        scroll.setHasVerticalScroller_(True)
        scroll.setBorderType_(NSBezelBorder)

        self._table = NSTableView.alloc().initWithFrame_(
            NSMakeRect(0, 0, LIST_W, list_h)
        )
        col = NSTableColumn.alloc().initWithIdentifier_("style")
        col.setWidth_(LIST_W - 4)
        self._table.addTableColumn_(col)
        self._table.setHeaderView_(None)
        self._table.setAllowsEmptySelection_(False)
        self._table.setAllowsMultipleSelection_(False)
        self._table.setDataSource_(self)
        self._table.setDelegate_(self)
        scroll.setDocumentView_(self._table)
        content.addSubview_(scroll)

        add_btn = NSButton.alloc().initWithFrame_(NSMakeRect(MARGIN, MARGIN, 36, 26))
        add_btn.setTitle_("+")
        add_btn.setBezelStyle_(1)
        add_btn.setTarget_(self)
        add_btn.setAction_("addClicked:")
        content.addSubview_(add_btn)

        del_btn = NSButton.alloc().initWithFrame_(NSMakeRect(MARGIN + 40, MARGIN, 36, 26))
        del_btn.setTitle_("−")
        del_btn.setBezelStyle_(1)
        del_btn.setTarget_(self)
        del_btn.setAction_("deleteClicked:")
        content.addSubview_(del_btn)
        self._del_btn = del_btn

        # ── Right: name ────────────────────────────────────────────────────
        name_y = H - MARGIN - 24
        content.addSubview_(_label(t("styles.name"), RIGHT_X, name_y + 2, 85))
        self._name_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(RIGHT_X + 90, name_y, RIGHT_W - 90, 24)
        )
        content.addSubview_(self._name_field)

        # ── Right: prompt (scrollable) ─────────────────────────────────────
        prompt_label_y = name_y - 30
        content.addSubview_(_label(t("styles.prompt"), RIGHT_X, prompt_label_y, RIGHT_W))

        prompt_y = MARGIN + 76
        prompt_h = prompt_label_y - 8 - prompt_y
        pscroll = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(RIGHT_X, prompt_y, RIGHT_W, prompt_h)
        )
        pscroll.setHasVerticalScroller_(True)
        pscroll.setBorderType_(NSBezelBorder)

        self._prompt_view = NSTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, pscroll.contentSize().width, prompt_h)
        )
        self._prompt_view.setMinSize_(NSMakeSize(0.0, prompt_h))
        self._prompt_view.setMaxSize_(NSMakeSize(_BIG, _BIG))
        self._prompt_view.setVerticallyResizable_(True)
        self._prompt_view.setHorizontallyResizable_(False)
        self._prompt_view.setAutoresizingMask_(NSViewWidthSizable)
        self._prompt_view.textContainer().setContainerSize_(
            NSMakeSize(pscroll.contentSize().width, _BIG)
        )
        self._prompt_view.textContainer().setWidthTracksTextView_(True)
        self._prompt_view.setRichText_(False)
        self._prompt_view.setFont_(NSFont.systemFontOfSize_(13))
        self._prompt_view.setAllowsUndo_(True)
        pscroll.setDocumentView_(self._prompt_view)
        content.addSubview_(pscroll)

        # ── Right: active checkbox + close ─────────────────────────────────
        self._active_check = NSButton.alloc().initWithFrame_(
            NSMakeRect(RIGHT_X, MARGIN + 40, 200, 22)
        )
        self._active_check.setButtonType_(NSSwitchButton)
        self._active_check.setTitle_(t("styles.active"))
        self._active_check.setTarget_(self)
        self._active_check.setAction_("activeToggled:")
        content.addSubview_(self._active_check)

        hint = _label(t("styles.clean_hint", clean=t("styles.clean")),
                      RIGHT_X, MARGIN + 16, RIGHT_W - 100)
        hint.setFont_(NSFont.systemFontOfSize_(11))
        hint.setTextColor_(NSColor.secondaryLabelColor())
        content.addSubview_(hint)

        close_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(W - MARGIN - 80, MARGIN, 80, 28)
        )
        close_btn.setTitle_(t("styles.close"))
        close_btn.setBezelStyle_(1)
        close_btn.setKeyEquivalent_("\r")
        close_btn.setTarget_(self)
        close_btn.setAction_("closeClicked:")
        content.addSubview_(close_btn)

    # ── Data source / delegate for the styles table ──────────────────────────

    def numberOfRowsInTableView_(self, table):
        return len(self._names)

    def tableView_objectValueForTableColumn_row_(self, table, column, row):
        name = self._names[row]
        shown = t("styles.clean") if name == CLEAN_NAME else name
        return f"{shown}  ✓" if name == self._mgr.active_name else shown

    def tableViewSelectionDidChange_(self, notification):
        if self._loading:
            return
        self._apply_edits()
        row = self._table.selectedRow()
        if 0 <= row < len(self._names):
            self._load_style(self._names[row])

    # ── Editor state ──────────────────────────────────────────────────────────

    @objc.python_method
    def _reload(self, select_name: str):
        """Refresh names from the manager, reload the table, select a row."""
        self._loading = True
        try:
            self._names = [s["name"] for s in self._mgr.get_all()]
            self._table.reloadData()
            if select_name not in self._names:
                select_name = self._names[0]
            row = self._names.index(select_name)
            self._table.selectRowIndexes_byExtendingSelection_(
                NSIndexSet.indexSetWithIndex_(row), False
            )
        finally:
            self._loading = False
        self._load_style(select_name)

    @objc.python_method
    def _load_style(self, name: str):
        style = next((s for s in self._mgr.get_all() if s["name"] == name), None)
        if style is None:
            return
        self._selected_name = name
        self._name_field.setStringValue_(
            t("styles.clean") if name == CLEAN_NAME else style["name"])
        self._prompt_view.setString_(style["prompt"])
        is_clean = name == CLEAN_NAME
        self._name_field.setEnabled_(not is_clean)
        self._prompt_view.setEditable_(not is_clean)
        self._del_btn.setEnabled_(not is_clean)
        self._active_check.setState_(1 if name == self._mgr.active_name else 0)

    def _apply_edits(self):
        """Persist the editor fields into the currently selected style."""
        name = self._selected_name
        if not name or name == CLEAN_NAME:
            return
        new_name = str(self._name_field.stringValue())
        prompt = str(self._prompt_view.string())
        final = self._mgr.update(name, new_name, prompt)
        if final != name:
            self._selected_name = final
        self._notify_change()

    def _notify_change(self):
        if self._on_change:
            self._on_change()

    # ── Actions ───────────────────────────────────────────────────────────────

    @objc.IBAction
    def addClicked_(self, sender):
        self._apply_edits()
        name = self._mgr.add(t("styles.new"))
        self._reload(select_name=name)
        self._window.makeFirstResponder_(self._name_field)
        self._notify_change()

    @objc.IBAction
    def deleteClicked_(self, sender):
        name = self._selected_name
        if not name or name == CLEAN_NAME:
            return
        self._selected_name = None  # style is gone; nothing to apply
        self._mgr.delete(name)
        self._reload(select_name=CLEAN_NAME)
        self._notify_change()

    @objc.IBAction
    def activeToggled_(self, sender):
        self._apply_edits()
        if self._active_check.state():
            self._mgr.set_active(self._selected_name)
        else:
            self._mgr.set_active(CLEAN_NAME)
        self._reload(select_name=self._selected_name or CLEAN_NAME)
        self._notify_change()

    @objc.IBAction
    def closeClicked_(self, sender):
        self._window.performClose_(None)

    def windowWillClose_(self, notification):
        self._apply_edits()
        self._notify_change()
