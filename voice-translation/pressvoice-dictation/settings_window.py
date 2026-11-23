import threading

import objc
from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSButton,
    NSColor,
    NSComboBox,
    NSFont,
    NSMakeRect,
    NSPanel,
    NSPopUpButton,
    NSSecureTextField,
    NSSegmentedControl,
    NSSwitchButton,
    NSTextAlignmentLeft,
    NSTextAlignmentRight,
    NSTextField,
    NSView,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskUtilityWindow,
)
from Foundation import NSObject
from PyObjCTools import AppHelper

import config as cfg_module
import i18n
import mac_support
from audio import AudioRecorder
from hotkey import HOTKEY_IDS, hotkey_label
from i18n import t
from live.session import TEXT_MODES, text_mode
from transcribe import (
    friendly_error,
    list_transcription_models,
    make_client,
    model_choices,
)


W = 500
H = 396
LABEL_X = 20
FIELD_X = 200  # wide label column: translations run longer than English
FIELD_W = 270
ROW_H = 36
LOGIN_ROW_H = 30  # extra height for "Start at login" in the packaged app


def _label(text: str, x: float, y: float, w: float, h: float = 20) -> NSTextField:
    field = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    field.setStringValue_(text)
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(False)
    field.setFont_(NSFont.systemFontOfSize_(13))
    field.setAlignment_(NSTextAlignmentRight)
    return field


def _separator(y: float, width: float) -> NSView:
    line = NSView.alloc().initWithFrame_(NSMakeRect(LABEL_X, y, width - LABEL_X * 2, 1))
    line.setWantsLayer_(True)
    line.layer().setBackgroundColor_(NSColor.separatorColor().CGColor())
    return line


class SettingsWindowController(NSObject):
    _instance = None

    @classmethod
    def shared(cls):
        if cls._instance is None:
            cls._instance = cls.alloc().init()
        return cls._instance

    def init(self):
        self = objc.super(SettingsWindowController, self).init()
        if self is None:
            return None
        self._window = None
        self._on_save = None
        self._mic_indices = []
        self._mic_names = []
        self._fetching = False
        return self

    def show(self, on_save=None):
        self._on_save = on_save
        if self._window and self._window.isVisible():
            self._window.makeKeyAndOrderFront_(None)
            NSApp.activateIgnoringOtherApps_(True)
            return
        self._build_window()
        self._window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    def _build_window(self):
        cfg = cfg_module.load()
        login_row = mac_support.login_item_supported()
        height = H + (LOGIN_ROW_H if login_row else 0)
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskUtilityWindow
        )
        self._window = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, height),
            style,
            NSBackingStoreBuffered,
            False,
        )
        self._window.setTitle_(t("settings.title"))
        self._window.center()
        self._window.setReleasedWhenClosed_(False)

        content = self._window.contentView()

        # ── Row 1: API Key ─────────────────────────────────────────────────
        y1 = height - 50
        content.addSubview_(_label(t("settings.api_key"), LABEL_X, y1, FIELD_X - LABEL_X - 8))
        self._api_field = NSSecureTextField.alloc().initWithFrame_(
            NSMakeRect(FIELD_X, y1 - 2, FIELD_W, 24)
        )
        self._api_field.setStringValue_(cfg.get("api_key", ""))
        self._api_field.setPlaceholderString_("sk-...")
        content.addSubview_(self._api_field)

        # ── Row 2: Transcription model ──────────────────────────────────────
        ym = y1 - ROW_H
        content.addSubview_(_label(t("settings.model"), LABEL_X, ym, FIELD_X - LABEL_X - 8))
        self._model_combo = NSComboBox.alloc().initWithFrame_(
            NSMakeRect(FIELD_X, ym - 3, FIELD_W - 40, 26)
        )
        self._model_combo.setCompletes_(True)
        self._model_combo.setNumberOfVisibleItems_(12)
        self._model_combo.addItemsWithObjectValues_(model_choices(cfg))
        self._model_combo.setStringValue_(
            cfg.get("transcription_model") or cfg_module.DEFAULT_MODEL
        )
        content.addSubview_(self._model_combo)

        self._refresh_btn = NSButton.alloc().initWithFrame_(
            NSMakeRect(FIELD_X + FIELD_W - 36, ym - 3, 36, 28)
        )
        self._refresh_btn.setTitle_("↻")
        self._refresh_btn.setBezelStyle_(1)
        self._refresh_btn.setToolTip_(t("settings.refresh_tooltip"))
        self._refresh_btn.setTarget_(self)
        self._refresh_btn.setAction_("refreshModelsClicked:")
        content.addSubview_(self._refresh_btn)

        self._model_status = _label("", FIELD_X, ym - 24, FIELD_W, 16)
        self._model_status.setAlignment_(NSTextAlignmentLeft)
        self._model_status.setFont_(NSFont.systemFontOfSize_(11))
        self._model_status.setTextColor_(NSColor.secondaryLabelColor())
        self._model_status.setLineBreakMode_(4)  # truncate tail; full text in the tooltip
        self._set_status(t("settings.models_hint_short"))
        content.addSubview_(self._model_status)

        # ── Row 3: Microphone ───────────────────────────────────────────────
        y2 = ym - ROW_H - 18
        content.addSubview_(_label(t("settings.microphone"), LABEL_X, y2, FIELD_X - LABEL_X - 8))
        self._mic_popup = NSPopUpButton.alloc().initWithFrame_(
            NSMakeRect(FIELD_X, y2 - 2, FIELD_W, 26)
        )
        self._mic_popup.removeAllItems()
        self._mic_popup.addItemWithTitle_(t("settings.system_default"))
        self._mic_indices = [-1]
        self._mic_names = [""]
        for dev in AudioRecorder.list_devices():
            self._mic_popup.addItemWithTitle_(dev["name"])
            self._mic_indices.append(dev["index"])
            self._mic_names.append(dev["name"])

        saved_name = cfg.get("microphone_name", "")
        saved_mic = cfg.get("microphone_index", -1)
        if saved_name and saved_name in self._mic_names:
            self._mic_popup.selectItemAtIndex_(self._mic_names.index(saved_name))
        elif saved_mic in self._mic_indices:
            self._mic_popup.selectItemAtIndex_(self._mic_indices.index(saved_mic))
        else:
            self._mic_popup.selectItemAtIndex_(0)
        content.addSubview_(self._mic_popup)

        # ── Row 4: Hotkey ───────────────────────────────────────────────────
        y3 = y2 - ROW_H
        content.addSubview_(_label(t("settings.hotkey"), LABEL_X, y3, FIELD_X - LABEL_X - 8))
        self._hotkey_popup = NSPopUpButton.alloc().initWithFrame_(
            NSMakeRect(FIELD_X, y3 - 2, FIELD_W, 26)
        )
        self._hotkey_popup.removeAllItems()
        for key_id in HOTKEY_IDS:
            self._hotkey_popup.addItemWithTitle_(hotkey_label(key_id))
        saved_hotkey = cfg.get("hotkey", "cmd_r")
        if saved_hotkey in HOTKEY_IDS:
            self._hotkey_popup.selectItemAtIndex_(HOTKEY_IDS.index(saved_hotkey))
        content.addSubview_(self._hotkey_popup)

        # ── Row 5: Recording mode ───────────────────────────────────────────
        y4 = y3 - ROW_H
        content.addSubview_(_label(t("settings.mode"), LABEL_X, y4, FIELD_X - LABEL_X - 8))
        self._mode_ctrl = NSSegmentedControl.alloc().initWithFrame_(
            NSMakeRect(FIELD_X, y4 - 2, FIELD_W, 26)
        )
        self._mode_ctrl.setSegmentCount_(2)
        self._mode_ctrl.setLabel_forSegment_(t("mode.push_to_talk_short"), 0)
        self._mode_ctrl.setLabel_forSegment_(t("mode.toggle_short"), 1)
        self._mode_ctrl.setTrackingMode_(0)  # selectOne
        mode_idx = 0 if cfg.get("recording_mode", "push_to_talk") == "push_to_talk" else 1
        self._mode_ctrl.setSelectedSegment_(mode_idx)
        content.addSubview_(self._mode_ctrl)

        # ── Row 6: When the text appears ────────────────────────────────────
        y_text = y4 - ROW_H
        content.addSubview_(_label(t("settings.text_mode"), LABEL_X, y_text,
                                   FIELD_X - LABEL_X - 8))
        self._text_ctrl = NSSegmentedControl.alloc().initWithFrame_(
            NSMakeRect(FIELD_X, y_text - 2, FIELD_W, 26)
        )
        self._text_ctrl.setSegmentCount_(len(TEXT_MODES))
        for i, mode in enumerate(TEXT_MODES):
            self._text_ctrl.setLabel_forSegment_(t(f"text_mode.{mode}_short"), i)
            self._text_ctrl.setToolTip_forSegment_(t(f"text_mode.{mode}"), i)
        self._text_ctrl.setTrackingMode_(0)  # selectOne
        self._text_ctrl.setSelectedSegment_(TEXT_MODES.index(text_mode(cfg)))
        self._text_ctrl.setToolTip_(t("settings.text_mode_hint"))
        content.addSubview_(self._text_ctrl)

        # ── Row 7: Interface language ───────────────────────────────────────
        y5 = y_text - ROW_H
        content.addSubview_(_label(t("settings.language"), LABEL_X, y5, FIELD_X - LABEL_X - 8))
        self._language_popup = NSPopUpButton.alloc().initWithFrame_(
            NSMakeRect(FIELD_X, y5 - 2, FIELD_W, 26)
        )
        self._language_popup.removeAllItems()
        self._language_codes = []
        for code, label in i18n.language_choices():
            self._language_popup.addItemWithTitle_(label)
            self._language_codes.append(code)
        saved_language = cfg.get("language") or i18n.AUTO
        self._language_popup.selectItemAtIndex_(
            self._language_codes.index(saved_language)
            if saved_language in self._language_codes else 0)
        content.addSubview_(self._language_popup)

        # ── Row 8: Start at login (packaged app only) ───────────────────────
        self._login_check = None
        sep_y = y5 - ROW_H - 6
        if login_row:
            self._login_check = NSButton.alloc().initWithFrame_(
                NSMakeRect(FIELD_X, y5 - 32, FIELD_W, 22)
            )
            self._login_check.setButtonType_(NSSwitchButton)
            self._login_check.setTitle_(t("autostart.login"))
            self._login_check.setState_(1 if mac_support.is_login_item_enabled() else 0)
            content.addSubview_(self._login_check)
            sep_y -= LOGIN_ROW_H

        # ── Separator ───────────────────────────────────────────────────────
        content.addSubview_(_separator(sep_y, W))

        # ── Buttons ─────────────────────────────────────────────────────────
        btn_y = sep_y - 40
        cancel_btn = NSButton.alloc().initWithFrame_(NSMakeRect(W - 250, btn_y, 120, 28))
        cancel_btn.setTitle_(t("settings.cancel"))
        cancel_btn.setBezelStyle_(1)
        cancel_btn.setTarget_(self)
        cancel_btn.setAction_("cancelClicked:")
        content.addSubview_(cancel_btn)

        save_btn = NSButton.alloc().initWithFrame_(NSMakeRect(W - 130, btn_y, 110, 28))
        save_btn.setTitle_(t("settings.save"))
        save_btn.setBezelStyle_(1)
        save_btn.setKeyEquivalent_("\r")
        save_btn.setTarget_(self)
        save_btn.setAction_("saveClicked:")
        content.addSubview_(save_btn)

    @objc.python_method
    def _set_status(self, text: str):
        self._model_status.setStringValue_(text)
        self._model_status.setToolTip_(text)

    # ── Model list refresh ────────────────────────────────────────────────

    @objc.IBAction
    def refreshModelsClicked_(self, sender):
        if self._fetching:
            return
        key = str(self._api_field.stringValue()).strip()
        if not key:
            self._set_status(t("settings.enter_key_first"))
            return
        self._fetching = True
        self._refresh_btn.setEnabled_(False)
        self._set_status(t("settings.loading_models"))
        threading.Thread(
            target=self._fetch_models, args=(key,), name="models", daemon=True
        ).start()

    @objc.python_method
    def _fetch_models(self, key: str):
        try:
            models, error = list_transcription_models(make_client(key)), ""
        except Exception as e:
            models, error = [], friendly_error(e)
        AppHelper.callAfter(self._models_fetched, models, error)

    @objc.python_method
    def _models_fetched(self, models: list, error: str):
        self._fetching = False
        self._refresh_btn.setEnabled_(True)
        if error:
            self._set_status(error)
            return
        if not models:
            self._set_status(t("settings.no_models"))
            return
        # Cache the list right away so it survives a Cancel.
        cfg = cfg_module.load()
        cfg["transcription_models"] = models
        try:
            cfg_module.save(cfg)
        except Exception:
            pass
        current = str(self._model_combo.stringValue()).strip()
        choices = model_choices({**cfg, "transcription_model": current})
        self._model_combo.removeAllItems()
        self._model_combo.addItemsWithObjectValues_(choices)
        self._model_combo.setStringValue_(current or models[0])
        self._set_status(t("settings.models_found", count=len(models)))

    @objc.IBAction
    def saveClicked_(self, sender):
        cfg = cfg_module.load()
        cfg["api_key"] = self._api_field.stringValue()
        model = str(self._model_combo.stringValue()).strip()
        cfg["transcription_model"] = model or cfg_module.DEFAULT_MODEL
        mic_sel = self._mic_popup.indexOfSelectedItem()
        if 0 <= mic_sel < len(self._mic_indices):
            cfg["microphone_index"] = self._mic_indices[mic_sel]
            cfg["microphone_name"] = self._mic_names[mic_sel]
        else:
            cfg["microphone_index"] = -1
            cfg["microphone_name"] = ""
        hk_sel = self._hotkey_popup.indexOfSelectedItem()
        cfg["hotkey"] = HOTKEY_IDS[hk_sel] if 0 <= hk_sel < len(HOTKEY_IDS) else "cmd_r"
        lang_sel = self._language_popup.indexOfSelectedItem()
        cfg["language"] = (self._language_codes[lang_sel]
                           if 0 <= lang_sel < len(self._language_codes) else i18n.AUTO)
        cfg["recording_mode"] = "push_to_talk" if self._mode_ctrl.selectedSegment() == 0 else "toggle"
        text_sel = self._text_ctrl.selectedSegment()
        cfg["text_mode"] = TEXT_MODES[text_sel] if 0 <= text_sel < len(TEXT_MODES) else "whole"
        cfg_module.save(cfg)
        if self._login_check is not None:
            want = bool(self._login_check.state())
            try:
                if want != mac_support.is_login_item_enabled():
                    mac_support.set_login_item(want)
            except Exception:
                pass
        self._window.orderOut_(None)
        if self._on_save:
            self._on_save(cfg)

    @objc.IBAction
    def cancelClicked_(self, sender):
        self._window.orderOut_(None)
