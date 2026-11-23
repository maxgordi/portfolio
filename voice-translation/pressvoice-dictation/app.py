import contextlib
import datetime
import faulthandler
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import objc
import rumps
import rumps.rumps
from AppKit import NSWorkspace
from Foundation import NSObject

import config as cfg_module
import i18n
import mac_support
from audio import AudioRecorder, FileRecorder
from history import HistoryManager
from hotkey import HotkeyListener, hotkey_label
from i18n import t
from live.mic import FileMic
from live.session import WHOLE, LiveDictation, text_mode
from mac_typer import MacTyper
from pipeline import NoAudio, TooShort, transcribe_wav
from settings_window import SettingsWindowController
from styles import CLEAN_NAME, StylesManager
from styles_window import StylesWindowController
from transcribe import friendly_error

ICON_IDLE = "🎙"
ICON_REC  = "⏺"
ICON_BUSY = "⏳"
ICON_STYLE = "✍️"

LOG_PATH = mac_support.log_path(os.path.dirname(os.path.abspath(__file__)))


LOG_MAX_BYTES = 2_000_000


def _log(msg: str) -> None:
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > LOG_MAX_BYTES:
            os.replace(LOG_PATH, LOG_PATH + ".1")
        with open(LOG_PATH, "a") as f:
            f.write(line)
    except Exception:
        pass


def _install_crash_logging() -> None:
    """Record native crashes and stray thread exceptions in the log, so a
    silent disappearance always leaves a trace."""
    try:
        _crash_file = open(LOG_PATH, "a")
        faulthandler.enable(file=_crash_file, all_threads=True)
        # Keep a reference alive; faulthandler writes to the raw fd.
        globals()["_crash_file"] = _crash_file
    except Exception:
        pass

    def _thread_hook(args):
        _log(f"UNCAUGHT in {args.thread.name if args.thread else '?'}: "
             f"{args.exc_type.__name__}: {args.exc_value}")

    def _hook(exc_type, exc_value, tb):
        _log(f"UNCAUGHT: {exc_type.__name__}: {exc_value}")

    try:
        threading.excepthook = _thread_hook
        sys.excepthook = _hook
    except Exception:
        pass


AUDIO_STUCK_SECONDS = 25       # CoreAudio call this slow is wedged, not busy
RESTART_COOLDOWN = 600         # never self-restart more than once per 10 min
_RESTART_MARKER = os.path.join(cfg_module.CONFIG_DIR, ".last_restart")


def _self_restart(reason: str) -> None:
    """Relaunch the process in place. Used only when a CoreAudio call has hung:
    the stuck thread can't be killed from Python, but exec replaces the whole
    process image with a clean one."""
    try:
        last = 0.0
        if os.path.exists(_RESTART_MARKER):
            with open(_RESTART_MARKER) as f:
                last = float(f.read().strip() or 0)
        if time.time() - last < RESTART_COOLDOWN:
            _log(f"self-restart suppressed ({reason}): too soon after the last one")
            return
        os.makedirs(os.path.dirname(_RESTART_MARKER), exist_ok=True)
        with open(_RESTART_MARKER, "w") as f:
            f.write(str(time.time()))
    except Exception as e:
        _log(f"self-restart bookkeeping failed: {e}")
        return

    _log(f"self-restart: {reason}")
    try:
        if mac_support.is_frozen():
            os.execv(sys.executable, [sys.executable])
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
        os.execv(sys.executable, [sys.executable, script])
    except Exception as e:
        _log(f"self-restart failed: {e}")


def _safe_thread(target, *args, name: str = "worker"):
    """Run target in a daemon thread; log instead of dying on exceptions."""
    def runner():
        try:
            target(*args)
        except Exception as e:
            _log(f"thread {name} error: {e}")
    t = threading.Thread(target=runner, name=name, daemon=True)
    t.start()
    return t


# ── Main-thread dispatcher ────────────────────────────────────────────────────

class _Dispatcher(NSObject):
    def run_(self, func):
        # Anything raised here would escape into the AppKit run loop.
        try:
            func()
        except Exception as e:
            _log(f"main-thread callback error: {e}")

_dispatcher = _Dispatcher.alloc().init()


def _on_main(func):
    """Schedule func() to run on the main thread (safe from any thread)."""
    _dispatcher.performSelectorOnMainThread_withObject_waitUntilDone_(
        "run:", func, False
    )


# Opening PressVoice.app again while it runs (Finder, Launchpad, Spotlight)
# only re-activates the running copy; show Settings so something visible
# happens instead of nothing.
_on_reopen = None


# PyObjC requires a category to carry the name of the class it extends.
class NSApp(objc.Category(rumps.rumps.NSApp)):
    def applicationShouldHandleReopen_hasVisibleWindows_(self, app, has_windows):
        if _on_reopen is not None:
            _on_main(_on_reopen)
        return True


del NSApp  # the method now lives on rumps' own delegate class


# ── App ───────────────────────────────────────────────────────────────────────

class PressVoiceApp(rumps.App):
    def __init__(self):
        super().__init__(ICON_IDLE, quit_button=None)

        self._cfg = cfg_module.load()
        i18n.set_language(self._cfg.get("language"))
        self._history = HistoryManager(self._cfg.get("history_size", 50))
        self._styles = StylesManager()
        # Test hooks for the end-to-end tests (tests/test_e2e_macos.py):
        # a WAV file instead of the microphone, a fixed text instead of the
        # OpenAI call. Never set in normal use.
        fake_mic = os.environ.get("PRESSVOICE_FAKE_MIC")
        fake_text = os.environ.get("PRESSVOICE_FAKE_TRANSCRIPT")
        self._recorder = FileRecorder(fake_mic) if fake_mic else AudioRecorder()
        self._transcribe = (lambda *a, **kw: fake_text) if fake_text else transcribe_wav
        self._test_mode = bool(fake_mic or fake_text)
        self._fake_text = fake_text or ""
        # Live dictation (text typed while you speak). Its microphone calls
        # run on one thread, in order, so a quick tap can't close the
        # microphone before it has been opened.
        self._live = LiveDictation(log=self._log, mic=FileMic(fake_mic) if fake_mic else None)
        self._live_audio = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-audio")
        self._typer = MacTyper(log=self._log)
        self._live_turn = False
        self._live_typing = False
        if self._test_mode:
            _log(f"TEST MODE: fake_mic={bool(fake_mic)} fake_transcript={bool(fake_text)}")
        self._hotkey_listener = HotkeyListener()
        self._busy = False
        self._recording = False
        self._rec_lock = threading.Lock()
        self._startup_done = False
        self._ax_timer = None
        self._ax_settings_opened = False
        _install_crash_logging()
        global _on_reopen
        _on_reopen = lambda: self._open_settings(None)

        # rumps keys menu items by their first title, so they start with
        # fixed ids and get their translated titles in _retranslate().
        self._rec_item = rumps.MenuItem("recording")
        self._rec_item.set_callback(None)
        self._history_menu = rumps.MenuItem("history")
        self._style_menu = rumps.MenuItem("style")
        self._settings_item = rumps.MenuItem("settings", callback=self._open_settings)
        self._clear_item    = rumps.MenuItem("clear", callback=self._clear_history)
        self._quit_item     = rumps.MenuItem("quit", callback=rumps.quit_application)

        self.menu = [
            self._rec_item, None,
            self._history_menu, self._clear_item, None,
            self._style_menu, None,
            self._settings_item,
            self._quit_item,
        ]
        self._rec_item.hidden = True
        self._retranslate()

        # Defer until NSApp run loop is active
        rumps.Timer(self._on_startup, 0.3).start()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _on_startup(self, timer=None):
        if self._startup_done:
            return
        self._startup_done = True
        if timer:
            timer.stop()
        try:
            mac_support.install_edit_menu()
        except Exception as e:
            self._log(f"edit menu setup failed: {e}")
        self._log(f"startup: language {i18n.current_language()}, "
                  f"text mode {text_mode(self._cfg)}, registering hotkey monitor")
        self._restart_hotkey()
        self._configure_live()
        NSWorkspace.sharedWorkspace().notificationCenter().addObserver_selector_name_object_(
            self, "systemDidWake:", "NSWorkspaceDidWakeNotification", None
        )
        trusted = mac_support.is_trusted()
        self._log(f"startup: AXIsProcessTrusted={trusted}")
        if not trusted:
            self._check_accessibility()
        self._setup_login_item()
        if not self._cfg.get("api_key"):
            self._notify(t("notify.welcome.title"), t("notify.welcome.body"))
            self._open_settings(None)

    def _setup_login_item(self):
        """The packaged app starts at login by default (like the Windows
        installer's preselected option); Settings can switch it off."""
        if not mac_support.login_item_supported():
            return
        try:
            # Test runs (fake mic) must never touch the user's real login items.
            if not self._test_mode and mac_support.remove_legacy_login_items():
                mac_support.set_login_item(True)  # keep autostart on after the id change
                self._log("startup: login item migrated to the new bundle id")
            if not self._cfg.get("login_item_initialized"):
                mac_support.set_login_item(True)
                cfg = cfg_module.load()
                cfg["login_item_initialized"] = True
                cfg_module.save(cfg)
                self._cfg = cfg
                self._log("startup: login item enabled (first run)")
            else:
                mac_support.refresh_login_item()
        except Exception as e:
            self._log(f"login item setup failed: {e}")

    def systemDidWake_(self, notification):
        self._log("system wake: resetting audio, state and hotkey monitor")
        with self._rec_lock:
            self._busy = False
            self._recording = False
        _safe_thread(self._reset_audio_bg, name="wake-reset")
        _on_main(self._set_idle_ui)
        _on_main(self._restart_hotkey)

    def _reset_audio_bg(self):
        """Drop any stale stream and force PortAudio to re-scan devices."""
        with self._audio_watchdog("wake reset"):
            try:
                self._recorder.stop()
            except Exception as e:
                self._log(f"audio reset: stop failed: {e}")
            self._recorder.reset()
        self._live.warm_up()
        self._log("audio reset: done")

    # ── Watchdog ──────────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def _audio_watchdog(self, label: str):
        """Guard a CoreAudio call. If it hangs (a known macOS failure mode
        after an audio device dies mid-stream), recover instead of leaving the
        app silently dead."""
        timer = threading.Timer(
            AUDIO_STUCK_SECONDS, self._on_audio_stuck, args=(label,)
        )
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    def _on_audio_stuck(self, label: str):
        self._log(f"AUDIO STUCK in {label} for {AUDIO_STUCK_SECONDS}s — recovering")
        _on_main(lambda: self._notify(
            t("notify.audio_stuck.title"), t("notify.audio_stuck.body")
        ))
        time.sleep(1.0)  # let the notification render before exec
        _self_restart(f"audio stuck in {label}")

    def _log(self, msg: str):
        _log(msg)

    # ── Accessibility ─────────────────────────────────────────────────────────

    def _check_accessibility(self, open_settings: bool = False):
        """Ask macOS to show its own Accessibility prompt (it also adds
        PressVoice to the list in System Settings), optionally open that list
        right away, then wait for the switch to be turned on and carry on
        without needing a restart."""
        if not self._test_mode:  # e2e tests run without anyone to click
            try:
                mac_support.prompt_accessibility()
                if open_settings and not self._ax_settings_opened:
                    self._ax_settings_opened = True
                    mac_support.open_accessibility_settings()
            except Exception as e:
                self._log(f"accessibility prompt failed: {e}")
        if self._ax_timer is None:
            self._ax_timer = rumps.Timer(self._poll_accessibility, 2)
            self._ax_timer.start()

    def _poll_accessibility(self, timer):
        if not mac_support.is_trusted():
            return
        timer.stop()
        self._ax_timer = None
        self._log("accessibility granted: re-registering hotkey monitor")
        self._restart_hotkey()
        key = hotkey_label(self._cfg.get("hotkey", "cmd_r"))
        self._notify(t("notify.ready.title"), t("hint.hold", key=key) + ".")

    # ── Hotkey ────────────────────────────────────────────────────────────────

    def _restart_hotkey(self):
        self._hotkey_listener.start(
            hotkey=self._cfg.get("hotkey", "cmd_r"),
            mode=self._cfg.get("recording_mode", "push_to_talk"),
            on_start=self._recording_started,
            on_stop=self._recording_stopped,
            dispatch_to_main=_on_main,
            poll_release=text_mode(self._cfg) == WHOLE,
        )

    def _configure_live(self):
        try:
            self._live.configure(self._cfg.get("api_key", ""), text_mode(self._cfg),
                                 self._cfg.get("transcription_model", ""), self._fake_text)
        except Exception as e:
            self._log(f"live setup failed: {e}")

    # ── Recording callbacks (called on main thread) ───────────────────────────

    def _recording_started(self):
        """Called on main thread. Audio ops run in background to avoid CoreAudio deadlock."""
        self._log("recording_started called")
        with self._rec_lock:
            if self._busy:
                self._log("recording_started: skipped (busy)")
                return
            if self._recording:
                self._log("recording_started: skipped (already recording)")
                return
            self._recording = True
            # Without an API key the live engine is off: the recording then
            # ends like a normal one, with the "no key" message.
            self._live_turn = self._live.active
        self.title = ICON_REC
        self._rec_item.hidden = False
        mic_index = self._cfg.get("microphone_index", -1)
        mic_name = self._cfg.get("microphone_name", "")
        if self._live_turn:
            # Without Accessibility macOS drops typed keys: the text is
            # then handed over through the clipboard at the end.
            self._live_typing = mac_support.is_trusted()
            self._typer.begin(self._live_typing)
            self._live_audio.submit(self._start_recording_bg, mic_index, mic_name)
        else:
            _safe_thread(self._start_recording_bg, mic_index, mic_name, name="rec-start")

    def _start_recording_bg(self, mic_index: int, mic_name: str):
        try:
            with self._audio_watchdog("recorder start"):
                if self._live_turn:
                    self._live.start(self._typer.write, mic_index, mic_name)
                else:
                    self._recorder.start(mic_index, mic_name)
        except Exception as e:
            self._log(f"recorder start error: {e}")
            with self._rec_lock:
                self._recording = False
            _on_main(self._set_idle_ui)
            body = t("notify.mic_error.body", error=str(e)[:90])  # `e` dies with the block
            _on_main(lambda: self._notify(t("notify.mic_error.title"), body))

    def _recording_stopped(self):
        """Called on main thread (or dispatched there). Stops mic and starts transcription."""
        self._log("recording_stopped called")
        self.title = ICON_BUSY
        self._rec_item.hidden = True
        with self._rec_lock:
            self._recording = False
            if self._busy:
                self._log("recording_stopped: skipped transcription (busy)")
                _safe_thread(self._recorder.stop, name="rec-drop")
                return
            self._busy = True
        if self._live_turn:
            self._live_audio.submit(self._stop_live_bg)
        else:
            _safe_thread(self._stop_and_transcribe, name="transcribe")

    # ── Background: live dictation ends ──────────────────────────────────────

    def _stop_live_bg(self):
        try:
            with self._audio_watchdog("live stop"):
                self._live.stop_mic()
        except Exception as e:
            self._log(f"live stop error: {e}")
        # Waiting for the last words is network time: not on the audio thread.
        _safe_thread(self._finish_live_bg, name="live-finish")

    def _finish_live_bg(self):
        try:
            result = self._live.finish()
            self._typer.finish()
            self._log(f"live: {len(result.text)} chars, error={result.error!r}")
            _on_main(lambda: self._live_done(result))
        except Exception as e:
            self._log(f"live finish error: {e}")
            msg = friendly_error(e)
            _on_main(lambda: self._notify(t("notify.error.title"), msg))
        finally:
            with self._rec_lock:
                self._busy = False
            _on_main(self._set_idle_ui)

    def _live_done(self, result):
        for model in result.unavailable:
            self._notify(t("notify.model_unavailable.title"),
                         t("notify.model_unavailable.body", model=model, fallback=result.model))
        if result.silent and not result.text:
            self._log("live: microphone delivered pure silence")
            self._notify(t("notify.no_sound.title"),
                         t("notify.no_sound.body_mac", path=t("path.mac_mic")))
            return
        if result.error:
            self._notify(t("notify.error.title"), friendly_error(result.error))
        if not result.text:
            return
        try:
            self._history.add(result.text)
            self._rebuild_history_menu()
        except Exception as e:
            self._log(f"history error: {e}")
        if not self._live_typing:
            # Nothing could be typed: same as a normal dictation without
            # the permission — the text waits on the clipboard.
            self._log("live: no Accessibility permission — text left on the clipboard")
            try:
                mac_support.copy_to_clipboard(result.text)
            except Exception as e:
                self._log(f"clipboard error: {e}")
            self._notify(t("notify.copied.title"),
                         t("notify.copied.body", path=t("path.mac_accessibility")))
            self._check_accessibility(open_settings=True)

    # ── Background: stop + transcribe ────────────────────────────────────────

    def _stop_and_transcribe(self):
        try:
            with self._audio_watchdog("recorder stop"):
                wav = self._recorder.stop()
            api_key = self._cfg.get("api_key", "")
            if not api_key:
                _on_main(lambda: self._notify(t("notify.no_key.title"),
                                              t("notify.no_key.body")))
                return
            if not wav:
                _on_main(lambda: self._notify(t("notify.empty.title"),
                                              t("notify.empty.body")))
                return

            style = self._styles.get_active()
            try:
                text = self._transcribe(
                    api_key, wav,
                    model=self._cfg.get("transcription_model", ""),
                    style_prompt=style.get("prompt", ""),
                    style_name=style.get("name", ""),
                    on_status=self._on_pipeline_status,
                    on_notice=lambda t, m: _on_main(lambda: self._notify(t, m)),
                    log=self._log,
                )
            except TooShort:
                self._log("transcribe: skipped, recording too short")
                return
            except NoAudio:
                self._log("transcribe: microphone delivered pure silence")
                _on_main(lambda: self._notify(
                    t("notify.no_sound.title"),
                    t("notify.no_sound.body_mac", path=t("path.mac_mic"))))
                return
            if text:
                _on_main(lambda: self._finish_paste(text))
        except Exception as e:
            self._log(f"transcribe error: {e}")
            msg = friendly_error(e)
            _on_main(lambda: self._notify(t("notify.error.title"), msg))
        finally:
            with self._rec_lock:
                self._busy = False
            _on_main(lambda: self._set_idle_ui())

    def _on_pipeline_status(self, state: str):
        if state == "style":
            _on_main(lambda: setattr(self, "title", ICON_STYLE))
        elif state.startswith("chunk "):
            _on_main(lambda: setattr(self, "title", f"⏳ {state[6:]}"))

    def _set_idle_ui(self):
        self.title = ICON_IDLE
        self._rec_item.hidden = True

    # ── Paste ─────────────────────────────────────────────────────────────────

    def _finish_paste(self, text: str):
        # Runs on the main thread: an exception here would surface inside the
        # AppKit run loop, so every step is guarded.
        try:
            mac_support.copy_to_clipboard(text)
        except Exception as e:
            self._log(f"clipboard error: {e}")
            return
        try:
            self._history.add(text)
            self._rebuild_history_menu()
        except Exception as e:
            self._log(f"history error: {e}")
        if not mac_support.is_trusted():
            # macOS would silently drop the keystroke; say so instead.
            self._log("paste: no Accessibility permission — text left on the clipboard")
            self._notify(
                t("notify.copied.title"),
                t("notify.copied.body", path=t("path.mac_accessibility")))
            self._check_accessibility(open_settings=True)
            return
        try:
            mac_support.send_cmd_v()
        except Exception as e:
            self._log(f"paste error: {e}")

    def _notify(self, title: str, message: str):
        self._log(f"notify: {title} — {message}")
        try:
            rumps.notification("PressVoice", title, message)
        except Exception as e:
            self._log(f"notification failed: {e}")

    # ── History menu ──────────────────────────────────────────────────────────

    def _rebuild_history_menu(self):
        if self._history_menu._menu is not None:
            self._history_menu.clear()
        entries = self._history.get_all()
        if not entries:
            p = rumps.MenuItem(t("menu.history_empty"))
            p.set_callback(None)
            self._history_menu.add(p)
            return
        for entry in entries:
            preview = entry["text"][:60].replace("\n", " ")
            if len(entry["text"]) > 60:
                preview += "…"
            ts = entry["timestamp"][11:16]
            item = rumps.MenuItem(
                f"[{ts}] {preview}",
                callback=self._make_copy_cb(entry["text"])
            )
            self._history_menu.add(item)

    def _make_copy_cb(self, text: str):
        def cb(_):
            mac_support.copy_to_clipboard(text)
        return cb

    def _clear_history(self, _):
        self._history.clear()
        self._rebuild_history_menu()

    # ── Style menu ────────────────────────────────────────────────────────────

    def _rebuild_style_menu(self):
        if self._style_menu._menu is not None:
            self._style_menu.clear()
        active = self._styles.active_name
        for style in self._styles.get_all():
            shown = t("styles.clean") if style["name"] == CLEAN_NAME else style["name"]
            item = rumps.MenuItem(shown, callback=self._make_style_cb(style["name"]))
            item.state = 1 if style["name"] == active else 0
            self._style_menu.add(item)
        self._style_menu.add(None)
        self._style_menu.add(
            rumps.MenuItem(t("menu.edit_styles"), callback=self._open_styles)
        )

    def _make_style_cb(self, name: str):
        def cb(_):
            self._styles.set_active(name)
            self._rebuild_style_menu()
        return cb

    def _open_styles(self, _):
        StylesWindowController.shared().show(
            self._styles, on_change=self._rebuild_style_menu
        )

    # ── Settings ──────────────────────────────────────────────────────────────

    def _open_settings(self, _):
        SettingsWindowController.shared().show(on_save=self._on_settings_saved)

    def _on_settings_saved(self, new_cfg: dict):
        self._cfg = new_cfg
        self._history.max_size = new_cfg.get("history_size", 50)
        if i18n.resolve(new_cfg.get("language")) != i18n.current_language():
            i18n.set_language(new_cfg.get("language"))
            self._log(f"language: {i18n.current_language()}")
            StylesWindowController.shared().forget_window()
            self._retranslate()
        self._restart_hotkey()
        self._configure_live()

    def _retranslate(self):
        """(Re)apply every menu title in the current language."""
        self._rec_item.title = "⏺  " + t("menu.recording")
        self._history_menu.title = t("menu.history")
        self._style_menu.title = t("menu.style")
        self._settings_item.title = t("menu.settings")
        self._clear_item.title = t("menu.clear_history")
        self._quit_item.title = t("menu.quit")
        self._rebuild_history_menu()
        self._rebuild_style_menu()
