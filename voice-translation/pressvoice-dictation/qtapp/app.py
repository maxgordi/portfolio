"""PressVoice tray app for Windows and Linux. Mirrors the macOS app (app.py):
hold the hotkey, speak, release — the text is pasted where the cursor is."""
import contextlib
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QCursor, QDesktopServices
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

import config as cfg_module
import i18n
from audio import AudioRecorder
from history import HistoryManager
from live.session import LiveDictation, text_mode
from pipeline import NoAudio, TooShort, transcribe_wav
from qtapp import paste as paste_mod
from qtapp import platform_utils
from i18n import t
from qtapp.hotkey import HotkeyListener, hotkey_id, hotkey_label
from qtapp.icons import make_icon
from qtapp.logs import LOG_PATH, log
from qtapp.settings_dialog import SettingsDialog
from qtapp.styles_dialog import StylesDialog
from styles import CLEAN_NAME, StylesManager
from transcribe import friendly_error

AUDIO_STUCK_SECONDS = 25     # an audio call this slow is wedged, not busy
RESTART_COOLDOWN = 600       # never self-restart more than once per 10 min
WAKE_CHECK_MS = 5000
WAKE_GAP_SECONDS = 30        # timers don't fire while the machine sleeps
_RESTART_MARKER = os.path.join(cfg_module.DATA_DIR, ".last_restart")


class PressVoiceQt(QObject):
    _call = Signal(object)

    def __init__(self, recorder=None, transcribe: Callable = transcribe_wav,
                 hotkey_listener=None, paster: Callable = paste_mod.paste_text,
                 settings_dialog: Optional[SettingsDialog] = None,
                 live: Optional[LiveDictation] = None, live_fake_text: str = ""):
        super().__init__()
        self._call.connect(self._run_call)

        self._cfg = cfg_module.load()
        i18n.set_language(self._cfg.get("language"))
        self._history = HistoryManager(self._cfg.get("history_size", 50))
        self._styles = StylesManager()
        self._recorder = recorder or AudioRecorder()
        self._transcribe = transcribe
        self._paste = paster
        self._hotkey = hotkey_listener or HotkeyListener(log=log)
        self._hotkey_started = False
        # Live dictation (text pasted while you speak).
        self._live = live or LiveDictation(log=log)
        self._live_fake_text = live_fake_text
        self._live_paster = paste_mod.LivePaster(paster, before_paste=self._own_keystrokes)
        self._live_turn = False
        self._live_id = 0  # pieces of a cancelled dictation are dropped by it

        self._busy = False
        self._recording = False
        self._start_failed = False
        self._mic_settings_opened = False
        self._click_action: Optional[Callable] = None
        self._rec_lock = threading.Lock()
        # Every recorder call runs here, one at a time and in order, so a
        # quick tap can't run stop() before start() has opened the stream.
        self._audio = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio")

        self._settings = settings_dialog or SettingsDialog()
        self._styles_dialog = StylesDialog()
        self._icons = {s: make_icon(s) for s in ("idle", "recording", "busy", "style")}
        self._state = "idle"
        self._build_tray()

        self._last_tick = time.time()
        self._wake_timer = QTimer(self)
        self._wake_timer.timeout.connect(self._check_wake)
        self._wake_timer.start(WAKE_CHECK_MS)

    # ── Startup / shutdown ───────────────────────────────────────────────────

    def start(self) -> None:
        log(f"startup: PressVoice Qt, language {i18n.current_language()}")
        self._tray.show()
        self._restart_hotkey()
        self._configure_live()
        if os.environ.get("XDG_SESSION_TYPE") == "wayland":
            self._notify(t("notify.wayland.title"), t("notify.wayland.body"))
        if not self._cfg.get("api_key"):
            self._notify(t("notify.welcome.title"), t("notify.welcome.body"))
            self.open_settings()

    def quit(self) -> None:
        log("quit")
        self._hotkey.stop()
        self._tray.hide()
        QApplication.quit()

    # ── Thread helpers ───────────────────────────────────────────────────────

    def _main(self, fn: Callable) -> None:
        """Run fn on the Qt main thread (safe from any thread)."""
        self._call.emit(fn)

    def _run_call(self, fn: Callable) -> None:
        try:
            fn()
        except Exception as e:
            log(f"main-thread callback error: {e}")

    @staticmethod
    def _bg(target: Callable, *args, name: str = "worker") -> threading.Thread:
        def runner():
            try:
                target(*args)
            except Exception as e:
                log(f"thread {name} error: {e}")
        t = threading.Thread(target=runner, name=name, daemon=True)
        t.start()
        return t

    def _audio_call(self, target: Callable, *args) -> None:
        def runner():
            try:
                target(*args)
            except Exception as e:
                log(f"audio task error: {e}")
        self._audio.submit(runner)

    # ── Watchdog ─────────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def _audio_watchdog(self, label: str):
        """If an audio driver call hangs, restart instead of leaving the app
        silently dead (the stuck thread can't be killed from Python)."""
        timer = threading.Timer(AUDIO_STUCK_SECONDS, self._on_audio_stuck, args=(label,))
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    def _on_audio_stuck(self, label: str) -> None:
        log(f"AUDIO STUCK in {label} for {AUDIO_STUCK_SECONDS}s — recovering")
        self._main(lambda: self._notify(
            t("notify.audio_stuck.title"), t("notify.audio_stuck.body")))
        time.sleep(1.0)
        self._self_restart(f"audio stuck in {label}")

    def _self_restart(self, reason: str) -> None:
        try:
            last = 0.0
            if os.path.exists(_RESTART_MARKER):
                with open(_RESTART_MARKER) as f:
                    last = float(f.read().strip() or 0)
            if time.time() - last < RESTART_COOLDOWN:
                log(f"self-restart suppressed ({reason}): too soon after the last one")
                return
            os.makedirs(os.path.dirname(_RESTART_MARKER), exist_ok=True)
            with open(_RESTART_MARKER, "w") as f:
                f.write(str(time.time()))
        except Exception as e:
            log(f"self-restart bookkeeping failed: {e}")
            return
        log(f"self-restart: {reason}")
        try:
            flags = 0
            if platform_utils.IS_WINDOWS:
                flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(platform_utils.launch_command() + ["--restarted"],
                             close_fds=True, creationflags=flags)
        except Exception as e:
            log(f"self-restart failed: {e}")
            return
        os._exit(0)

    # ── Sleep / wake and hook health ─────────────────────────────────────────

    def _check_wake(self) -> None:
        now = time.time()
        gap = now - self._last_tick
        self._last_tick = now
        if gap > WAKE_GAP_SECONDS:
            log(f"system wake (no tick for {gap:.0f}s): resetting audio, state and hotkey")
            with self._rec_lock:
                self._busy = False
                self._recording = False
            self._audio_call(self._reset_audio_bg)
            self._live.warm_up()
            self._set_state("idle")
            self._restart_hotkey()
        elif self._hotkey_started and not self._hotkey.is_alive():
            log("hotkey listener died — restarting it")
            self._restart_hotkey()

    def _reset_audio_bg(self) -> None:
        with self._audio_watchdog("wake reset"):
            try:
                self._recorder.stop()
            except Exception as e:
                log(f"audio reset: stop failed: {e}")
            self._recorder.reset()
        log("audio reset: done")

    # ── Hotkey ───────────────────────────────────────────────────────────────

    def _restart_hotkey(self) -> None:
        try:
            self._hotkey.start(
                hotkey=hotkey_id(self._cfg.get("hotkey", "")),
                mode=self._cfg.get("recording_mode", "push_to_talk"),
                on_start=self._recording_started,
                on_stop=self._recording_stopped,
                on_cancel=self._recording_cancelled,
            )
            self._hotkey_started = True
        except Exception as e:
            self._hotkey_started = False
            log(f"hotkey start failed: {e}")
            self._notify(t("notify.hotkey_unavailable.title"), friendly_error(e))
        self._update_hint()

    def _configure_live(self) -> None:
        try:
            self._live.configure(self._cfg.get("api_key", ""), text_mode(self._cfg),
                                 self._cfg.get("transcription_model", ""),
                                 self._live_fake_text)
        except Exception as e:
            log(f"live setup failed: {e}")

    def _own_keystrokes(self) -> None:
        ignore = getattr(self._hotkey, "ignore_own_input", None)
        if ignore:
            ignore()

    def _hint_text(self) -> str:
        label = hotkey_label(self._cfg.get("hotkey", ""))
        if self._cfg.get("recording_mode") == "toggle":
            return t("hint.toggle", key=label)
        return t("hint.hold", key=label)

    # ── Recording (main thread) ──────────────────────────────────────────────

    def _recording_started(self) -> None:
        log("recording_started")
        if not self._cfg.get("api_key"):
            self._hotkey.reset()
            self._notify(t("notify.no_key.title"), t("notify.no_key.body"))
            self.open_settings()
            return
        with self._rec_lock:
            if self._busy:
                log("recording_started: skipped (busy)")
                return
            if self._recording:
                return
            self._recording = True
            self._live_turn = self._live.active
        self._set_state("recording")
        if self._live_turn:
            self._live_id += 1
            self._live_paster.begin()
        self._audio_call(self._start_recording_bg,
                         self._cfg.get("microphone_index", -1),
                         self._cfg.get("microphone_name", ""), self._live_id)

    def _start_recording_bg(self, mic_index: int, mic_name: str, live_id: int = 0) -> None:
        self._start_failed = False
        try:
            with self._audio_watchdog("recorder start"):
                if self._live_turn:
                    self._live.start(lambda piece: self._main(
                        lambda: self._live_piece(live_id, piece)), mic_index, mic_name)
                    backend = self._live.backend
                else:
                    self._recorder.start(mic_index, mic_name)
                    backend = getattr(self._recorder, "backend", "")
            log(f"recording via {backend or 'default'}"
                + (f" (live, {text_mode(self._cfg)})" if self._live_turn else ""))
        except Exception as e:
            self._start_failed = True
            error = e  # `e` is unbound once the except block ends
            log(f"recorder start error: {e}")
            with self._rec_lock:
                self._recording = False
            self._main(self._hotkey.reset)
            self._main(lambda: self._set_state("idle"))
            self._main(lambda: self._report_mic_failure(error))

    def _report_mic_failure(self, error: Exception) -> None:
        block = platform_utils.windows_microphone_block()
        if block:
            log(f"microphone blocked by Windows privacy settings: {block}")
            self._notify(t("notify.mic_blocked.title"),
                         t("notify.mic_blocked.body", reason=block,
                           path=t("path.windows_mic")),
                         on_click=self._open_mic_privacy)
            if not self._mic_settings_opened:  # take them straight there, once
                self._mic_settings_opened = True
                self._open_mic_privacy()
            return
        short = str(error)[:90]
        if platform_utils.IS_WINDOWS:
            self._notify(t("notify.mic_error.title"),
                         t("notify.mic_error.body_windows", error=short),
                         on_click=self._open_mic_privacy)
        else:
            self._notify(t("notify.mic_error.title"),
                         t("notify.mic_error.body", error=short),
                         on_click=self.open_settings)

    def _open_mic_privacy(self) -> None:
        QDesktopServices.openUrl(QUrl(platform_utils.MIC_PRIVACY_URL))

    def _recording_cancelled(self) -> None:
        log("recording cancelled (key combination)")
        with self._rec_lock:
            was_recording = self._recording
            self._recording = False
            busy = self._busy
        if was_recording:
            self._live_id += 1  # nothing more of it gets pasted
            self._audio_call(self._drop_recording_bg)
        if not busy:
            self._set_state("idle")

    def _drop_recording_bg(self) -> None:
        with self._audio_watchdog("recorder drop"):
            if self._live_turn:
                self._live.cancel()
            else:
                self._recorder.stop()

    def _recording_stopped(self) -> None:
        log("recording_stopped")
        with self._rec_lock:
            was_recording = self._recording
            self._recording = False
            if not was_recording:
                return
            if self._busy:
                log("recording_stopped: skipped transcription (busy)")
                self._audio_call(self._drop_recording_bg)
                return
            self._busy = True
        self._set_state("busy")
        self._audio_call(self._stop_live_bg if self._live_turn else self._stop_bg)

    def _stop_bg(self) -> None:
        wav = b""
        try:
            with self._audio_watchdog("recorder stop"):
                wav = self._recorder.stop()
        finally:
            # The network call runs on its own thread so the audio queue
            # stays free for a wake reset.
            self._bg(self._transcribe_bg, wav, name="transcribe")

    def _transcribe_bg(self, wav: bytes) -> None:
        try:
            if not wav:
                if not self._start_failed:  # that failure was already reported
                    self._main(lambda: self._notify(
                        t("notify.empty.title"), t("notify.empty.body")))
                return
            style = self._styles.get_active()
            try:
                text = self._transcribe(
                    self._cfg.get("api_key", ""), wav,
                    model=self._cfg.get("transcription_model", ""),
                    style_prompt=style.get("prompt", ""),
                    style_name=style.get("name", ""),
                    on_status=self._on_pipeline_status,
                    on_notice=lambda t, m: self._main(lambda: self._notify(t, m)),
                    log=log,
                )
            except TooShort:
                log("transcribe: skipped, recording too short")
                return
            except NoAudio:
                log("transcribe: microphone delivered pure silence")
                hint = (t("notify.no_sound.body_windows", path=t("path.windows_mic"),
                          toggle=t("windows.toggle.desktop"))
                        if platform_utils.IS_WINDOWS else t("notify.no_sound.body"))
                self._main(lambda: self._notify(t("notify.no_sound.title"), hint))
                return
            if text:
                self._main(lambda: self._finish_paste(text))
            else:
                log("transcribe: nothing recognised")
        except Exception as e:
            log(f"transcribe error: {e}")
            msg = friendly_error(e)
            self._main(lambda: self._notify(t("notify.error.title"), msg))
        finally:
            with self._rec_lock:
                self._busy = False
            self._main(self._set_idle_if_free)

    # ── Live dictation ───────────────────────────────────────────────────────

    def _live_piece(self, live_id: int, piece: str) -> None:
        if live_id == self._live_id:
            self._live_paster.write(piece)

    def _stop_live_bg(self) -> None:
        try:
            with self._audio_watchdog("live stop"):
                self._live.stop_mic()
        finally:
            self._bg(self._finish_live_bg, name="live-finish")

    def _finish_live_bg(self) -> None:
        try:
            result = self._live.finish()
            log(f"live: {len(result.text)} chars, error={result.error!r}")
            # The pieces still queued for pasting go first.
            self._main(lambda: self._live_paster.finish(
                lambda ok: self._live_done(result, ok)))
        except Exception as e:
            log(f"live finish error: {e}")
            msg = friendly_error(e)
            self._main(lambda: self._notify(t("notify.error.title"), msg))
            self._live_free()

    def _live_done(self, result, pasted: bool) -> None:
        try:
            for model in result.unavailable:
                self._notify(t("notify.model_unavailable.title"),
                             t("notify.model_unavailable.body",
                               model=model, fallback=result.model))
            if result.silent and not result.text:
                log("live: microphone delivered pure silence")
                hint = (t("notify.no_sound.body_windows", path=t("path.windows_mic"),
                          toggle=t("windows.toggle.desktop"))
                        if platform_utils.IS_WINDOWS else t("notify.no_sound.body"))
                self._notify(t("notify.no_sound.title"), hint)
            elif result.error and not self._start_failed:
                self._notify(t("notify.error.title"), friendly_error(result.error))
            if result.text:
                # The clipboard holds the last piece; the whole text is
                # more useful there.
                paste_mod.copy_to_clipboard(result.text)
                self._history.add(result.text)
                self._rebuild_history_menu()
                if not pasted:
                    self._after_paste(False)
        except Exception as e:
            log(f"live result error: {e}")
        finally:
            self._live_free()

    def _live_free(self) -> None:
        with self._rec_lock:
            self._busy = False
        self._main(self._set_idle_if_free)

    def _on_pipeline_status(self, state: str) -> None:
        if state == "style":
            self._main(lambda: self._set_state("style"))
        elif state.startswith("chunk "):
            self._main(lambda: self._tray.setToolTip(
                "PressVoice — " + t("tray.transcribing_part", progress=state[6:])))

    # ── Paste ────────────────────────────────────────────────────────────────

    def _finish_paste(self, text: str) -> None:
        try:
            self._history.add(text)
            self._rebuild_history_menu()
        except Exception as e:
            log(f"history error: {e}")
        try:
            self._paste(text, self._after_paste)
        except Exception as e:
            log(f"paste error: {e}")

    def _after_paste(self, ok: bool) -> None:
        if not ok:
            log("paste: keystroke was refused")
            self._notify(t("notify.paste_refused.title"), t("notify.paste_refused.body"))

    # ── Tray UI ──────────────────────────────────────────────────────────────

    def _build_tray(self) -> None:
        self._tray = QSystemTrayIcon(self._icons["idle"], self)
        self._tray.setToolTip("PressVoice")
        menu = QMenu()
        self._menu = menu

        self._hint_action = menu.addAction("")
        self._hint_action.setEnabled(False)
        self._rec_action = menu.addAction("")
        self._rec_action.setEnabled(False)
        self._rec_action.setVisible(False)
        menu.addSeparator()
        self._history_menu = menu.addMenu("")
        self._clear_action = self._add_action(menu, "", self._clear_history)
        menu.addSeparator()
        self._style_menu = menu.addMenu("")
        self._style_group: Optional[QActionGroup] = None
        menu.addSeparator()
        self._settings_action = self._add_action(menu, "", self.open_settings)
        self._logs_action = self._add_action(menu, "", self._open_log_folder)
        menu.addSeparator()
        self._quit_action = self._add_action(menu, "", self.quit)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._tray_activated)
        self._tray.messageClicked.connect(self._message_clicked)
        self._retranslate()

    def _retranslate(self) -> None:
        """(Re)apply every menu text in the current language."""
        self._rec_action.setText("●  " + t("menu.recording"))
        self._history_menu.setTitle(t("menu.history"))
        self._clear_action.setText(t("menu.clear_history"))
        self._style_menu.setTitle(t("menu.style"))
        self._settings_action.setText(t("menu.settings"))
        self._logs_action.setText(t("menu.open_logs"))
        self._quit_action.setText(t("menu.quit"))
        self._rebuild_history_menu()
        self._rebuild_style_menu()
        self._update_hint()
        self._set_state(self._state)

    @staticmethod
    def _add_action(menu: QMenu, text: str, slot: Callable) -> QAction:
        action = menu.addAction(text)
        action.triggered.connect(lambda checked=False: slot())
        return action

    def _tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.Trigger:  # left click
            self._menu.popup(QCursor.pos())
            self._menu.activateWindow()

    def _update_hint(self) -> None:
        self._hint_action.setText(self._hint_text())

    def _set_state(self, state: str) -> None:
        self._state = state
        self._tray.setIcon(self._icons.get(state, self._icons["idle"]))
        self._rec_action.setVisible(state == "recording")
        tips = {"idle": self._hint_text(),
                "recording": t("tray.recording"),
                "busy": t("tray.transcribing"),
                "style": t("tray.styling")}
        self._tray.setToolTip("PressVoice — " + tips.get(state, ""))

    def _set_idle_if_free(self) -> None:
        with self._rec_lock:
            recording = self._recording
        if not recording:
            self._set_state("idle")

    def _message_clicked(self) -> None:
        action, self._click_action = self._click_action, None
        if action:
            action()

    def _notify(self, title: str, message: str,
                on_click: Optional[Callable] = None) -> None:
        self._click_action = on_click
        try:
            self._tray.showMessage(title, message, self._icons["idle"], 5000)
        except Exception as e:
            log(f"notification failed: {e}")
        log(f"notify: {title} — {message}")

    def _rebuild_history_menu(self) -> None:
        m = self._history_menu
        m.clear()
        entries = self._history.get_all()
        if not entries:
            m.addAction(t("menu.history_empty")).setEnabled(False)
            return
        for entry in entries:
            text = entry.get("text", "")
            preview = text[:60].replace("\n", " ")
            if len(text) > 60:
                preview += "…"
            ts = entry.get("timestamp", "")[11:16]
            action = m.addAction(f"[{ts}] {preview}".replace("&", "&&"))
            action.triggered.connect(
                lambda checked=False, t=text: paste_mod.copy_to_clipboard(t))

    def _clear_history(self) -> None:
        self._history.clear()
        self._rebuild_history_menu()

    def _rebuild_style_menu(self) -> None:
        m = self._style_menu
        m.clear()
        group = QActionGroup(m)
        group.setExclusive(True)
        active = self._styles.active_name
        for style in self._styles.get_all():
            name = style["name"]
            shown = t("styles.clean") if name == CLEAN_NAME else name
            action = m.addAction(shown.replace("&", "&&"))
            action.setCheckable(True)
            action.setChecked(name == active)
            group.addAction(action)
            action.triggered.connect(lambda checked=False, n=name: self._set_style(n))
        self._style_group = group
        m.addSeparator()
        self._add_action(m, t("menu.edit_styles"), self._open_styles)

    def _set_style(self, name: str) -> None:
        self._styles.set_active(name)
        self._rebuild_style_menu()

    def _open_styles(self) -> None:
        self._styles_dialog.open_styles(self._styles, on_change=self._rebuild_style_menu)

    def _open_log_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(LOG_PATH)))

    # ── Settings ─────────────────────────────────────────────────────────────

    def open_settings(self, message: str = "") -> None:
        self._settings.open_settings(on_save=self._on_settings_saved, message=message)

    def _on_settings_saved(self, cfg: dict) -> None:
        self._cfg = cfg
        self._history.max_size = cfg.get("history_size", 50)
        log(f"settings saved: model={cfg.get('transcription_model')} "
            f"hotkey={cfg.get('hotkey')} mode={cfg.get('recording_mode')} "
            f"text={text_mode(cfg)} "
            f"language={cfg.get('language')}")
        if i18n.resolve(cfg.get("language")) != i18n.current_language():
            i18n.set_language(cfg.get("language"))
            self._settings.retranslate()
            self._styles_dialog.retranslate()
            self._retranslate()
        self._restart_hotkey()
        self._configure_live()
