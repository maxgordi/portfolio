from __future__ import annotations

import asyncio
import dataclasses
import difflib
import logging
import os
import random
import signal
import queue
import sys
import threading
import time
import urllib.request
import json as _json
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import sounddevice as sd
import customtkinter as ctk

from babellang.audio.devices import resolve_devices
from babellang.config import LANGUAGE_NAMES, load_settings, save_settings, get_api_key
from babellang.logger import DialogLogger
from babellang.pipeline import IncomingPipeline, OutgoingPipeline
from babellang.ptt import PttState, TK_TO_HOTKEY, hotkey_display
from babellang.stt.openai_stt import normalize_text

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".config" / "babellang"
_PID_PATH = _CONFIG_DIR / "babellang.pid"


def _setup_file_logging() -> None:
    """Rotating file log for GUI sessions — without it crashes leave no trace."""
    log_dir = _CONFIG_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("babellang")
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return
    handler = RotatingFileHandler(log_dir / "babellang.log", maxBytes=2_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _install_exception_hooks() -> None:
    """Log uncaught exceptions from every thread instead of losing them."""
    def _hook(exc_type, exc, tb):
        logger.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc, tb))
    sys.excepthook = _hook

    def _thread_hook(args):
        logger.critical("UNCAUGHT THREAD EXCEPTION in %s",
                        args.thread.name if args.thread else "?",
                        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = _thread_hook


def _humanize_error(exc: BaseException) -> str:
    """Translate common API failures into actionable Russian messages."""
    s = str(exc).lower()
    if "401" in s or "authentication" in s or "invalid api key" in s or "unauthorized" in s:
        return "Неверный API-ключ — проверь ключи через babellang --setup"
    if "429" in s or "quota" in s or "insufficient" in s or "rate limit" in s:
        return "Закончилась квота OpenAI/ElevenLabs или превышен лимит — проверь баланс"
    if "timed out" in s or "timeout" in s:
        return "Нет ответа от сервера — проверь интернет-соединение"
    if "getaddrinfo" in s or "nodename" in s or "connection" in s:
        return "Нет соединения с сервером — проверь интернет"
    return str(exc)[:150]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _acquire_single_instance_lock() -> bool:
    """Second running copy would fight over the mic and BlackHole → refuse to start."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if _PID_PATH.exists():
        try:
            old_pid = int(_PID_PATH.read_text().strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid and old_pid != os.getpid() and _pid_alive(old_pid):
            return False
    try:
        _PID_PATH.write_text(str(os.getpid()))
    except OSError:
        pass
    return True


def _release_single_instance_lock() -> None:
    try:
        if _PID_PATH.exists() and _PID_PATH.read_text().strip() == str(os.getpid()):
            _PID_PATH.unlink()
    except OSError:
        pass

# Hotkey string → tkinter KeySym for GUI key bindings
_PTT_KEYSYM: dict[str, str] = {
    "space":   "space",
    "ctrl_l":  "Control_L",
    "ctrl_r":  "Control_R",
    "cmd_l":   "Meta_L",
    "cmd":     "Meta_R",
    "alt_l":   "Alt_L",
    "alt_r":   "Alt_R",
    "shift_l": "Shift_L",
    "shift_r": "Shift_R",
    **{f"f{i}": f"F{i}" for i in range(1, 13)},
}

# ── Colour palettes ────────────────────────────────────────────────────────────
# Tuples: (light_mode, dark_mode) — CTkinter picks automatically via set_appearance_mode()
_C_BG     = ("#F2F2F7", "#1C1C1E")
_C_PANEL  = ("#FFFFFF",  "#2C2C2E")
_C_PANEL2 = ("#E5E5EA",  "#3A3A3C")
_C_TEXT   = ("#1C1C1E",  "#FFFFFF")
_C_SUB    = ("#6C6C70",  "#8E8E93")
_C_GREEN  = ("#34C759",  "#30D158")
_C_RED    = ("#FF3B30",  "#FF453A")
_C_YELLOW = ("#FF9F0A",  "#FFD60A")
_ACCENT   = "#007AFF"

_STATE_COLOR = {
    "idle":        _C_SUB,
    "capturing":   _C_GREEN,
    "speaking":    _C_YELLOW,
    "translating": _ACCENT,
    "playing":     "#BF5AF2",
    "ptt":         _ACCENT,   # key held, waiting for voice
}

_LANG_DISPLAY = {v: k for k, v in LANGUAGE_NAMES.items()}
_LANG_OPTIONS = list(LANGUAGE_NAMES.values())

_LANG_MODE_KEYS = {          # label → settings value
    "🔒 Строгий":   "strict",
    "💡 Подсказка": "hint",
    "🌐 Авто":      "auto",
}
_LANG_MODE_LABELS = {v: k for k, v in _LANG_MODE_KEYS.items()}
_LANG_MODE_HINTS = {
    "strict": "Речь всегда распознаётся как выбранный язык — не «уплывает» в похожие",
    "hint":   "Язык указывается как подсказка — модель может отклониться",
    "auto":   "Без обработки: модель сама определяет язык каждой фразы",
}

_QUALITY_OPTIONS = ["⚡ Быстрый", "⚖ Баланс", "✨ Качество"]
_QUALITY_MODELS  = {
    "⚡ Быстрый":  "eleven_flash_v2_5",
    "⚖ Баланс":   "eleven_turbo_v2_5",
    "✨ Качество": "eleven_multilingual_v2",
}
_MODEL_TO_QUALITY = {v: k for k, v in _QUALITY_MODELS.items()}

def _code_to_display(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def _display_to_code(display: str) -> str:
    return _LANG_DISPLAY.get(display, display)


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    """Interpolate between two hex colour strings."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = max(0, min(255, int(r1 + (r2 - r1) * t)))
    g = max(0, min(255, int(g1 + (g2 - g1) * t)))
    b = max(0, min(255, int(b1 + (b2 - b1) * t)))
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Glass panel ────────────────────────────────────────────────────────────────

class _GlassPanel(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        kwargs.setdefault("fg_color", _C_PANEL)
        kwargs.setdefault("corner_radius", 16)
        super().__init__(master, **kwargs)


# ── Animated voice level indicator ────────────────────────────────────────────

class _VoiceIndicator(ctk.CTkFrame):
    """Animated Unicode waveform: random bars when active, flat line when idle.
    Optional caption below the bars lights up in the accent colour while active."""
    _BARS = "▁▂▃▄▅▆▇█"
    _N    = 5

    def __init__(self, master, accent_color: str, caption: str = "", **kwargs):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self._accent  = accent_color
        self._active  = False
        self._tick_id: Optional[str] = None
        self._lbl = ctk.CTkLabel(
            self, text="─────",
            font=ctk.CTkFont(family="Menlo", size=13),
            text_color=_C_SUB, width=58,
        )
        self._lbl.pack()
        self._caption_lbl: Optional[ctk.CTkLabel] = None
        if caption:
            self._caption_lbl = ctk.CTkLabel(
                self, text=caption,
                font=ctk.CTkFont(size=9),
                text_color=_C_SUB, width=58,
            )
            self._caption_lbl.pack()
        self._tick()

    def set_active(self, active: bool) -> None:
        self._active = active

    def _tick(self) -> None:
        try:
            if self._active:
                txt = "".join(random.choice(self._BARS) for _ in range(self._N))
                self._lbl.configure(text=txt, text_color=self._accent)
                if self._caption_lbl:
                    self._caption_lbl.configure(text_color=self._accent)
            else:
                self._lbl.configure(text="─────", text_color=_C_SUB)
                if self._caption_lbl:
                    self._caption_lbl.configure(text_color=_C_SUB)
            self._tick_id = self.after(110, self._tick)
        except Exception:
            self._tick_id = None  # widget destroyed — stop ticking

    def destroy(self) -> None:
        if self._tick_id:
            try:
                self.after_cancel(self._tick_id)
            except Exception:
                pass
            self._tick_id = None
        super().destroy()


# ── Apple Music-style lyrics panel ────────────────────────────────────────────

class _LyricsPanel(ctk.CTkScrollableFrame):
    """
    Apple Music-style animated caption display.

    Layout:
      Me (out)          → LEFT-aligned, blue accent
      Interlocutor (in) → RIGHT-aligned, green accent

    Each entry: speaker tag (tiny) + original spoken text (large bold, animated)
                + translation (smaller, fixed dim colour).
    Active entries glow bright; after _ACTIVE_MS ms they smoothly dim to a
    readable gray — they never disappear.
    """

    _MAX       = 50      # max entries kept in the widget
    _ACTIVE_MS = 8000    # ms before dimming starts
    _FADE_STEPS = 24
    _FADE_MS   = 60      # ms per step → ~1.4 s total fade

    def __init__(self, master, **kwargs):
        kwargs.setdefault("fg_color", _C_PANEL2)
        kwargs.setdefault("corner_radius", 12)
        super().__init__(master, **kwargs)
        self._entries: list[dict] = []
        self._wrap_labels: list[ctk.CTkLabel] = []
        self._auto_scroll = True  # False when user scrolled up to read history

        self._parent_canvas.bind("<Configure>", self._on_canvas_resize)
        # Detect user scrolling — macOS trackpad/wheel + Linux buttons
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._parent_canvas.bind(seq, self._on_user_scroll, add="+")

    def _on_user_scroll(self, event=None) -> None:
        # Check position after the scroll event is processed
        self._parent_canvas.after(20, self._sync_auto_scroll)

    def _sync_auto_scroll(self) -> None:
        try:
            _, bottom = self._parent_canvas.yview()
            self._auto_scroll = bottom >= 0.98
        except Exception:
            pass

    def _on_canvas_resize(self, event) -> None:
        wrap = max(120, event.width - 90)
        for lbl in self._wrap_labels:
            try:
                lbl.configure(wraplength=wrap)
            except Exception:
                pass

    def _current_wraplength(self) -> int:
        w = self._parent_canvas.winfo_width()
        return max(120, w - 90) if w > 10 else 390

    def add_entry(self, source: str, translation: str, side: str, my: str, inter: str) -> None:
        if len(self._entries) >= self._MAX:
            oldest = self._entries.pop(0)
            # Cancel any pending fade callback before destroying the frame
            oid = oldest.get("after_id")
            if oid:
                try:
                    self.after_cancel(oid)
                except Exception:
                    pass
            try:
                oldest["frame"].destroy()
            except Exception:
                pass
            # Remove the 2 labels (src + trn) for the evicted entry
            if len(self._wrap_labels) >= 2:
                self._wrap_labels = self._wrap_labels[2:]

        is_dark    = ctk.get_appearance_mode() == "Dark"
        # Active colour: full brightness; dim colour: readable grey (never invisible)
        active_col = "#FFFFFF" if is_dark else "#000000"
        dim_col    = "#636366" if is_dark else "#AEAEB2"
        trn_col    = "#8E8E93" if is_dark else "#8E8E93"  # translation fixed dim
        accent     = _ACCENT if side == "out" else ("#30D158" if is_dark else "#34C759")

        # Me → LEFT; interlocutor → RIGHT
        is_me   = side == "out"
        anchor  = "w" if is_me else "e"
        justify = "left" if is_me else "right"
        pk_kw   = dict(anchor="w", padx=(10, 60)) if is_me else dict(anchor="e", padx=(60, 10))

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", pady=(8, 0), **pk_kw)

        wrap = self._current_wraplength()

        # Speaker tag — small, coloured, never fades
        speaker = f"{my}" if is_me else f"{inter}"
        ctk.CTkLabel(
            frame, text=speaker,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=accent, anchor=anchor,
        ).pack(fill="x")

        # Original spoken text — LARGE BOLD, animated from bright → dim
        src_lbl = ctk.CTkLabel(
            frame, text=source,
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=active_col, anchor=anchor,
            wraplength=wrap, justify=justify,
        )
        src_lbl.pack(fill="x")
        self._wrap_labels.append(src_lbl)

        # Translation — smaller, fixed dim colour (not animated)
        trn_lbl = ctk.CTkLabel(
            frame, text=translation,
            font=ctk.CTkFont(size=13),
            text_color=trn_col, anchor=anchor,
            wraplength=wrap, justify=justify,
        )
        trn_lbl.pack(fill="x", pady=(2, 0))
        self._wrap_labels.append(trn_lbl)

        entry: dict = {"frame": frame, "src_lbl": src_lbl, "after_id": None}
        self._entries.append(entry)

        # Schedule smooth dim; store the after-ID so it can be cancelled on eviction
        entry["after_id"] = self.after(
            self._ACTIVE_MS,
            lambda l=src_lbl, c1=active_col, c2=dim_col, e=entry: self._begin_fade(e, l, c1, c2),
        )
        self.after(80, self._scroll_to_bottom)

    def _begin_fade(self, entry: dict, label: ctk.CTkLabel, c1: str, c2: str) -> None:
        entry["after_id"] = None
        self._fade(entry, label, c1, c2, self._FADE_STEPS)

    def _fade(self, entry: dict, label: ctk.CTkLabel, c1: str, c2: str, steps: int) -> None:
        if steps <= 0:
            try:
                label.configure(text_color=c2)
            except Exception:
                pass
            entry["after_id"] = None
            return
        t = 1.0 - steps / self._FADE_STEPS
        try:
            label.configure(text_color=_lerp_hex(c1, c2, t))
        except Exception:
            entry["after_id"] = None
            return
        entry["after_id"] = self.after(
            self._FADE_MS, lambda: self._fade(entry, label, c1, c2, steps - 1)
        )

    def _scroll_to_bottom(self) -> None:
        if not self._auto_scroll:
            return
        try:
            self._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def clear(self) -> None:
        for entry in self._entries:
            oid = entry.get("after_id")
            if oid:
                try:
                    self.after_cancel(oid)
                except Exception:
                    pass
            try:
                entry["frame"].destroy()
            except Exception:
                pass
        self._entries.clear()
        self._wrap_labels.clear()
        self._auto_scroll = True


# ── Main GUI ───────────────────────────────────────────────────────────────────

class BabelLangGUI:
    """
    Main GUI window.
    tkinter runs in the main thread.
    asyncio pipelines run in a background daemon thread.
    GUI→async: loop.call_soon_threadsafe()
    async→GUI: root.after(0, callback)
    """

    def __init__(self) -> None:
        self._settings = load_settings()
        ctk.set_appearance_mode("Dark" if self._settings.theme == "dark" else "Light")
        ctk.set_default_color_theme("blue")

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None
        self._outgoing: Optional[OutgoingPipeline] = None
        self._incoming: Optional[IncomingPipeline] = None
        self._running = False
        self._logger: Optional[DialogLogger] = None
        self._log_dir = Path(__file__).parent.parent / "logs"

        self._ptt_state: Optional[PttState] = None
        self._voices: list[dict] = []
        self._voice_name_to_id: dict[str, str] = {}
        # Subtitles travel asyncio-thread → main thread through this queue and are
        # drained by _poll_status every 250ms. root.after(0, …) from a background
        # thread can silently drop calls — that's why subtitles randomly vanished.
        self._subtitle_q: queue.Queue = queue.Queue()

        self._root = ctk.CTk()
        self._root.title("BabelLang")
        self._root.geometry(self._settings.window_geometry)
        self._root.resizable(True, True)
        self._root.minsize(520, 820)
        self._root.configure(fg_color=_C_BG)
        try:
            self._root.wm_attributes("-alpha", 0.96)
        except Exception:
            pass

        self._build_ui()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Theme ──────────────────────────────────────────────────────────────────

    def _toggle_theme(self) -> None:
        new_theme = "light" if self._settings.theme == "dark" else "dark"
        self._settings.theme = new_theme
        ctk.set_appearance_mode("Dark" if new_theme == "dark" else "Light")
        self._root.configure(fg_color=_C_BG)
        self._theme_btn.configure(text="🌙" if new_theme == "dark" else "☀️")
        # Clear lyrics because per-entry colours are baked in at creation time
        self._lyrics.clear()
        save_settings(self._settings)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        pad = {"padx": 16, "pady": 7}

        self._scroll = ctk.CTkScrollableFrame(
            self._root, fg_color=_C_BG, corner_radius=0,
            scrollbar_button_color=_C_PANEL2,
            scrollbar_button_hover_color=_C_PANEL,
        )
        self._scroll.pack(fill="both", expand=True)

        # ── Title bar ─────────────────────────────────────────────────────────
        title_frame = ctk.CTkFrame(self._scroll, fg_color="transparent")
        title_frame.pack(fill="x", padx=16, pady=(16, 4))

        ctk.CTkLabel(
            title_frame, text="BabelLang",
            font=ctk.CTkFont(size=24, weight="bold"), text_color=_C_TEXT,
        ).pack(side="left")

        # Device settings gear
        ctk.CTkButton(
            title_frame, text="⚙",
            font=ctk.CTkFont(size=17), width=36, height=36,
            fg_color=_C_PANEL2, hover_color=_C_PANEL,
            text_color=_C_SUB, corner_radius=10,
            command=self._open_device_settings,
        ).pack(side="right", padx=(4, 0))

        # Theme toggle — moon = currently dark (click for light), sun = currently light
        theme_icon = "🌙" if self._settings.theme == "dark" else "☀️"
        self._theme_btn = ctk.CTkButton(
            title_frame, text=theme_icon,
            font=ctk.CTkFont(size=17), width=36, height=36,
            fg_color=_C_PANEL2, hover_color=_C_PANEL,
            text_color=_C_SUB, corner_radius=10,
            command=self._toggle_theme,
        )
        self._theme_btn.pack(side="right", padx=(4, 0))

        self._lang_label = ctk.CTkLabel(
            title_frame, text="",
            font=ctk.CTkFont(size=13), text_color=_C_SUB,
        )
        self._lang_label.pack(side="right", padx=4)

        # ── Language picker ───────────────────────────────────────────────────
        lang_panel = _GlassPanel(self._scroll)
        lang_panel.pack(fill="x", **pad)
        ctk.CTkLabel(
            lang_panel, text="Языки",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=_C_SUB,
        ).pack(anchor="w", padx=14, pady=(10, 4))

        lang_row = ctk.CTkFrame(lang_panel, fg_color="transparent")
        lang_row.pack(fill="x", padx=14, pady=(0, 12))

        my_col = ctk.CTkFrame(lang_row, fg_color="transparent")
        my_col.pack(side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkLabel(my_col, text="Мой язык",
                     font=ctk.CTkFont(size=11), text_color=_C_SUB).pack(anchor="w")
        self._my_lang_var = ctk.StringVar(value=_code_to_display(self._settings.my_language))
        ctk.CTkOptionMenu(
            my_col, values=_LANG_OPTIONS, variable=self._my_lang_var,
            command=self._on_my_lang_change,
            fg_color=_C_PANEL2, button_color=_ACCENT, button_hover_color="#0056CC",
            dropdown_fg_color=_C_PANEL, dropdown_hover_color=_C_PANEL2,
            text_color=_C_TEXT, dropdown_text_color=_C_TEXT,
            font=ctk.CTkFont(size=13), corner_radius=10,
        ).pack(fill="x", pady=(2, 0))

        ctk.CTkLabel(lang_row, text="⇄",
                     font=ctk.CTkFont(size=18), text_color=_C_SUB, width=30).pack(side="left", padx=4)

        inter_col = ctk.CTkFrame(lang_row, fg_color="transparent")
        inter_col.pack(side="left", expand=True, fill="x", padx=(6, 0))
        ctk.CTkLabel(inter_col, text="Язык собеседника",
                     font=ctk.CTkFont(size=11), text_color=_C_SUB).pack(anchor="w")
        self._inter_lang_var = ctk.StringVar(value=_code_to_display(self._settings.interlocutor_language))
        ctk.CTkOptionMenu(
            inter_col, values=_LANG_OPTIONS, variable=self._inter_lang_var,
            command=self._on_inter_lang_change,
            fg_color=_C_PANEL2, button_color=_ACCENT, button_hover_color="#0056CC",
            dropdown_fg_color=_C_PANEL, dropdown_hover_color=_C_PANEL2,
            text_color=_C_TEXT, dropdown_text_color=_C_TEXT,
            font=ctk.CTkFont(size=13), corner_radius=10,
        ).pack(fill="x", pady=(2, 0))

        # ── Language-lock mode for STT ────────────────────────────────────────
        self._lang_mode_var = ctk.StringVar(
            value=_LANG_MODE_LABELS.get(self._settings.stt_language_mode, "🔒 Строгий"))
        ctk.CTkSegmentedButton(
            lang_panel,
            values=list(_LANG_MODE_KEYS),
            variable=self._lang_mode_var,
            command=self._on_lang_mode_change,
            fg_color=_C_PANEL2, selected_color=_ACCENT, selected_hover_color="#0056CC",
            unselected_color=_C_PANEL2, unselected_hover_color=_C_PANEL,
            text_color=_C_TEXT, font=ctk.CTkFont(size=12), corner_radius=10,
        ).pack(fill="x", padx=14, pady=(0, 2))
        self._lang_mode_hint = ctk.CTkLabel(
            lang_panel, text=_LANG_MODE_HINTS.get(self._settings.stt_language_mode, ""),
            font=ctk.CTkFont(size=10), text_color=_C_SUB,
        )
        self._lang_mode_hint.pack(anchor="w", padx=14, pady=(0, 10))

        # ── Status panel ──────────────────────────────────────────────────────
        status_panel = _GlassPanel(self._scroll)
        status_panel.pack(fill="x", **pad)
        status_row = ctk.CTkFrame(status_panel, fg_color="transparent")
        status_row.pack(fill="x", padx=12, pady=10)

        self._status_dot = ctk.CTkLabel(
            status_row, text="●", font=ctk.CTkFont(size=18), text_color=_C_SUB, width=24)
        self._status_dot.pack(side="left")
        self._status_label = ctk.CTkLabel(
            status_row, text="IDLE",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_C_SUB)
        self._status_label.pack(side="left", padx=(4, 16))
        self._latency_label = ctk.CTkLabel(
            status_row, text="", font=ctk.CTkFont(size=12), text_color=_C_SUB)
        self._latency_label.pack(side="left")

        self._error_label = ctk.CTkLabel(
            status_panel, text="",
            font=ctk.CTkFont(size=11), text_color=_C_RED, wraplength=510)
        self._error_label.pack(fill="x", padx=12, pady=(0, 2))

        # Routing hint (e.g. "no sound from interlocutor") — yellow, not an error
        self._hint_label = ctk.CTkLabel(
            status_panel, text="",
            font=ctk.CTkFont(size=11), text_color=_C_YELLOW, wraplength=510)
        self._hint_label.pack(fill="x", padx=12, pady=(0, 6))

        # ── Lyrics panel (Apple Music style) ──────────────────────────────────
        dialog_panel = _GlassPanel(self._scroll)
        dialog_panel.pack(fill="x", **pad)

        # Header: three live indicators driven by REAL audio state (not guesses):
        #   🎤 my voice being captured | 🔊 TTS actually going to interlocutor | 🎧 TTS in my ears
        hdr = ctk.CTkFrame(dialog_panel, fg_color="transparent")
        hdr.pack(fill="x", padx=14, pady=(10, 4))
        self._mic_indicator = _VoiceIndicator(hdr, accent_color=_ACCENT, caption="🎤 я говорю")
        self._mic_indicator.pack(side="left", expand=True)
        self._out_tts_indicator = _VoiceIndicator(hdr, accent_color="#BF5AF2", caption="🔊 собеседнику")
        self._out_tts_indicator.pack(side="left", expand=True)
        self._in_tts_indicator = _VoiceIndicator(hdr, accent_color=_C_GREEN[1], caption="🎧 мне")
        self._in_tts_indicator.pack(side="left", expand=True)

        self._lyrics = _LyricsPanel(
            dialog_panel, height=270,
            scrollbar_button_color=_C_PANEL2,
            scrollbar_button_hover_color=_C_PANEL,
        )
        self._lyrics.pack(fill="x", padx=10, pady=(0, 10))

        # ── Voice panel (ElevenLabs) ──────────────────────────────────────────
        voice_panel = _GlassPanel(self._scroll)
        voice_panel.pack(fill="x", **pad)
        ctk.CTkLabel(
            voice_panel, text="Голос ElevenLabs",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=_C_SUB,
        ).pack(anchor="w", padx=14, pady=(10, 6))

        # ── ElevenLabs section ────────────────────────────────────────────────
        self._el_section = ctk.CTkFrame(voice_panel, fg_color="transparent")
        self._el_section.pack(fill="x")

        vrow = ctk.CTkFrame(self._el_section, fg_color="transparent")
        vrow.pack(fill="x", padx=14, pady=(0, 8))
        self._voice_menu_var = ctk.StringVar(value="— нажми 'Загрузить' —")
        self._voice_menu = ctk.CTkOptionMenu(
            vrow,
            values=["— нажми 'Загрузить' —"],
            variable=self._voice_menu_var,
            command=self._on_voice_change,
            state="disabled",
            fg_color=_C_PANEL2, button_color=_ACCENT, button_hover_color="#0056CC",
            dropdown_fg_color=_C_PANEL, dropdown_hover_color=_C_PANEL2,
            text_color=_C_TEXT, dropdown_text_color=_C_TEXT,
            font=ctk.CTkFont(size=13), corner_radius=10,
            width=300,
        )
        self._voice_menu.pack(side="left", expand=True, fill="x", padx=(0, 8))
        self._voice_btn = ctk.CTkButton(
            vrow, text="Загрузить",
            font=ctk.CTkFont(size=13),
            fg_color=_C_PANEL2, hover_color=_C_PANEL,
            text_color=_ACCENT, corner_radius=10, height=36, width=100,
            command=self._start_load_voices,
        )
        self._voice_btn.pack(side="left")

        qrow = ctk.CTkFrame(self._el_section, fg_color="transparent")
        qrow.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkLabel(qrow, text="Качество:",
                     font=ctk.CTkFont(size=13), text_color=_C_TEXT, width=80, anchor="w").pack(side="left")
        current_quality = _MODEL_TO_QUALITY.get(self._settings.pipeline.elevenlabs_model, "⚡ Быстрый")
        self._quality_var = ctk.StringVar(value=current_quality)
        ctk.CTkSegmentedButton(
            qrow,
            values=_QUALITY_OPTIONS,
            variable=self._quality_var,
            command=self._on_quality_change,
            fg_color=_C_PANEL2, selected_color=_ACCENT, selected_hover_color="#0056CC",
            unselected_color=_C_PANEL2, unselected_hover_color=_C_PANEL,
            text_color=_C_TEXT, font=ctk.CTkFont(size=12), corner_radius=10,
        ).pack(side="left", fill="x", expand=True)

        self._quality_hint = ctk.CTkLabel(
            self._el_section, text=self._quality_hint_text(current_quality),
            font=ctk.CTkFont(size=10), text_color=_C_SUB,
        )
        self._quality_hint.pack(anchor="w", padx=14, pady=(0, 10))

        # ── Volume sliders ────────────────────────────────────────────────────
        vol_panel = _GlassPanel(self._scroll)
        vol_panel.pack(fill="x", **pad)
        ctk.CTkLabel(
            vol_panel, text="Громкость живого голоса",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=_C_SUB,
        ).pack(anchor="w", padx=14, pady=(10, 2))
        ctk.CTkLabel(
            vol_panel,
            text="0% = только перевод ElevenLabs  |  100% = живой голос в максимальном качестве",
            font=ctk.CTkFont(size=10), text_color=_C_SUB,
        ).pack(anchor="w", padx=14, pady=(0, 4))

        self._out_ratio_var = ctk.DoubleVar(value=self._settings.outgoing_original_ratio)
        self._in_ratio_var  = ctk.DoubleVar(value=self._settings.incoming_original_ratio)
        self._out_ratio_str = ctk.StringVar(value=f"{int(self._settings.outgoing_original_ratio * 100)}%")
        self._in_ratio_str  = ctk.StringVar(value=f"{int(self._settings.incoming_original_ratio * 100)}%")

        self._build_slider_row(vol_panel, "Мой голос (исходящий):",
                               self._out_ratio_var, self._out_ratio_str, self._on_out_ratio)
        self._build_slider_row(vol_panel, "Голос собеседника (входящий):",
                               self._in_ratio_var, self._in_ratio_str, self._on_in_ratio)
        ctk.CTkFrame(vol_panel, fg_color="transparent", height=6).pack()

        # ── TTS volume sliders ────────────────────────────────────────────────
        tts_vol_panel = _GlassPanel(self._scroll)
        tts_vol_panel.pack(fill="x", **pad)
        ctk.CTkLabel(
            tts_vol_panel, text="Громкость перевода (ElevenLabs)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=_C_SUB,
        ).pack(anchor="w", padx=14, pady=(10, 2))

        self._out_tts_var = ctk.DoubleVar(value=self._settings.outgoing_tts_volume)
        self._in_tts_var  = ctk.DoubleVar(value=self._settings.incoming_tts_volume)
        self._out_tts_str = ctk.StringVar(value=f"{int(self._settings.outgoing_tts_volume * 100)}%")
        self._in_tts_str  = ctk.StringVar(value=f"{int(self._settings.incoming_tts_volume * 100)}%")

        self._build_slider_row(tts_vol_panel, "Перевод собеседнику:",
                               self._out_tts_var, self._out_tts_str, self._on_out_tts_volume)
        self._build_slider_row(tts_vol_panel, "Перевод мне (в уши):",
                               self._in_tts_var, self._in_tts_str, self._on_in_tts_volume)
        ctk.CTkFrame(tts_vol_panel, fg_color="transparent", height=6).pack()

        # ── Ducking mode ──────────────────────────────────────────────────────
        duck_panel = _GlassPanel(self._scroll)
        duck_panel.pack(fill="x", **pad)
        ctk.CTkLabel(
            duck_panel, text="Затухание голоса при озвучке ElevenLabs",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=_C_SUB,
        ).pack(anchor="w", padx=14, pady=(10, 4))
        duck_row = ctk.CTkFrame(duck_panel, fg_color="transparent")
        duck_row.pack(fill="x", padx=14, pady=(0, 12))
        self._smooth_var = ctk.BooleanVar(value=self._settings.smooth_ducking)
        ctk.CTkCheckBox(
            duck_row,
            text="Плавное затухание (1.5 сек)",
            variable=self._smooth_var,
            command=self._on_smooth_toggle,
            font=ctk.CTkFont(size=13), text_color=_C_TEXT,
            fg_color=_ACCENT, hover_color="#0056CC",
            checkmark_color=("#000000", "#FFFFFF"), corner_radius=6,
        ).pack(side="left")
        ctk.CTkLabel(
            duck_row, text="  Снято = мгновенное отключение",
            font=ctk.CTkFont(size=11), text_color=_C_SUB,
        ).pack(side="left")

        # ── Control buttons ───────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self._scroll, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=(4, 6))
        self._start_btn = ctk.CTkButton(
            btn_frame, text="▶  СТАРТ",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=_C_GREEN, hover_color=("#28B34D", "#28B34D"),
            text_color="#000000", corner_radius=14, height=52,
            command=self._on_start,
        )
        self._start_btn.pack(side="left", expand=True, fill="x", padx=(0, 8))
        self._stop_btn = ctk.CTkButton(
            btn_frame, text="■  СТОП",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=_C_RED, hover_color=("#CC3A2F", "#CC3A2F"),
            text_color="#FFFFFF", corner_radius=14, height=52,
            state="disabled", command=self._on_stop,
        )
        self._stop_btn.pack(side="left", expand=True, fill="x")

        self._log_path_label = ctk.CTkLabel(
            self._scroll, text="",
            font=ctk.CTkFont(size=10), text_color=_C_SUB,
        )
        self._log_path_label.pack(pady=(0, 12))

        self._refresh_lang_label()

    def _build_slider_row(self, parent, label, var, str_var, command) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(
            row, text=label, font=ctk.CTkFont(size=13),
            text_color=_C_TEXT, width=230, anchor="w",
        ).pack(side="left")
        ctk.CTkSlider(
            row, from_=0.0, to=1.0, variable=var, command=command,
            fg_color=_C_PANEL2, progress_color=_ACCENT,
            button_color=_C_TEXT, button_hover_color="#CCCCCC", width=180,
        ).pack(side="left", padx=8)
        ctk.CTkLabel(
            row, textvariable=str_var,
            font=ctk.CTkFont(size=12), text_color=_C_SUB, width=36,
        ).pack(side="left")

    # ── Voice loading ──────────────────────────────────────────────────────────

    def _start_load_voices(self) -> None:
        el_key = get_api_key("elevenlabs") or ""
        if not el_key:
            self._error_label.configure(text="Нет API ключа ElevenLabs")
            return
        self._voice_btn.configure(state="disabled", text="Загрузка...")
        self._voice_menu.configure(state="disabled")
        threading.Thread(target=self._fetch_voices, args=(el_key,), daemon=True).start()

    def _fetch_voices(self, api_key: str) -> None:
        try:
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": api_key},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
            voices = data.get("voices", [])
            voices.sort(key=lambda v: 0 if v.get("category") in ("cloned", "generated", "professional") else 1)
            self._root.after(0, lambda: self._on_voices_loaded(voices))
        except Exception as exc:
            self._root.after(0, lambda e=exc: self._on_voices_error(str(e)))

    def _on_voices_loaded(self, voices: list[dict]) -> None:
        self._voices = voices
        self._voice_name_to_id = {}
        display_names = []
        current_id = self._settings.pipeline.elevenlabs_voice_id
        current_display = None

        for v in voices:
            cat = v.get("category", "")
            tag = " ★" if cat in ("cloned", "generated", "professional") else ""
            name = v["name"] + tag
            vid = v["voice_id"]
            self._voice_name_to_id[name] = vid
            display_names.append(name)
            if vid == current_id:
                current_display = name

        self._voice_menu.configure(values=display_names, state="normal")
        if current_display:
            self._voice_menu_var.set(current_display)
        elif display_names:
            self._voice_menu_var.set(display_names[0])
            self._on_voice_change(display_names[0])

        self._voice_btn.configure(state="normal", text="Обновить")

    def _on_voices_error(self, msg: str) -> None:
        self._voice_btn.configure(state="normal", text="Загрузить")
        self._error_label.configure(text=f"Ошибка загрузки голосов: {msg[:80]}")

    # ── Voice / quality callbacks ──────────────────────────────────────────────

    def _on_voice_change(self, name: str) -> None:
        vid = self._voice_name_to_id.get(name)
        if vid:
            self._settings.pipeline = dataclasses.replace(
                self._settings.pipeline, elevenlabs_voice_id=vid
            )
            save_settings(self._settings)

    def _on_quality_change(self, label: str) -> None:
        model = _QUALITY_MODELS.get(label, "eleven_flash_v2_5")
        self._settings.pipeline = dataclasses.replace(
            self._settings.pipeline, elevenlabs_model=model
        )
        save_settings(self._settings)
        self._quality_hint.configure(text=self._quality_hint_text(label))

    @staticmethod
    def _quality_hint_text(label: str) -> str:
        hints = {
            "⚡ Быстрый":  "eleven_flash_v2_5 — задержка ~0.5с, хорошее качество",
            "⚖ Баланс":   "eleven_turbo_v2_5 — задержка ~1с, отличное качество",
            "✨ Качество": "eleven_multilingual_v2 — задержка ~2с, максимальное качество",
        }
        return hints.get(label, "")

    # ── Language callbacks ─────────────────────────────────────────────────────

    def _on_lang_mode_change(self, label: str) -> None:
        self._settings.stt_language_mode = _LANG_MODE_KEYS.get(label, "strict")
        self._lang_mode_hint.configure(
            text=_LANG_MODE_HINTS.get(self._settings.stt_language_mode, ""))
        save_settings(self._settings)
        if self._running:
            self._hint_label.configure(text="Режим языка применится после СТОП → СТАРТ")

    def _on_my_lang_change(self, display: str) -> None:
        self._settings.my_language = _display_to_code(display)
        save_settings(self._settings)
        self._refresh_lang_label()

    def _on_inter_lang_change(self, display: str) -> None:
        self._settings.interlocutor_language = _display_to_code(display)
        save_settings(self._settings)
        self._refresh_lang_label()

    # ── Volume / ducking callbacks ─────────────────────────────────────────────

    def _on_out_ratio(self, value: float) -> None:
        self._out_ratio_str.set(f"{int(value * 100)}%")
        self._settings.outgoing_original_ratio = value
        if self._outgoing and self._loop:
            self._loop.call_soon_threadsafe(self._outgoing.set_passthrough_volume, value)

    def _on_in_ratio(self, value: float) -> None:
        self._in_ratio_str.set(f"{int(value * 100)}%")
        self._settings.incoming_original_ratio = value
        if self._incoming and self._loop:
            self._loop.call_soon_threadsafe(self._incoming._playback.set_original_ratio, value)

    def _on_out_tts_volume(self, value: float) -> None:
        self._out_tts_str.set(f"{int(value * 100)}%")
        self._settings.outgoing_tts_volume = value
        if self._outgoing and self._loop:
            self._loop.call_soon_threadsafe(self._outgoing.set_tts_volume, value)

    def _on_in_tts_volume(self, value: float) -> None:
        self._in_tts_str.set(f"{int(value * 100)}%")
        self._settings.incoming_tts_volume = value
        if self._incoming and self._loop:
            self._loop.call_soon_threadsafe(self._incoming.set_tts_volume, value)

    def _on_smooth_toggle(self) -> None:
        enabled = self._smooth_var.get()
        self._settings.smooth_ducking = enabled
        if self._loop:
            if self._outgoing:
                self._loop.call_soon_threadsafe(self._outgoing._playback.set_smooth_ducking, enabled)
            if self._incoming:
                self._loop.call_soon_threadsafe(self._incoming._playback.set_smooth_ducking, enabled)

    # ── PTT key bindings (tkinter replaces pynput in GUI mode) ───────────────

    def _bind_ptt(self, keysym: str, ptt_state: PttState) -> None:
        self._ptt_keysym      = keysym
        self._ptt_release_id  = None

        def _press(e, p=ptt_state):
            # Cancel debounced release (handles auto-repeat KeyRelease on some systems)
            if self._ptt_release_id:
                self._root.after_cancel(self._ptt_release_id)
                self._ptt_release_id = None
            p.press()
            # Status is driven by _poll_status — no direct update here to avoid flicker

        def _do_release(p):
            self._ptt_release_id = None
            p.release()

        def _release(e, p=ptt_state):
            # Debounce 100ms: ignore fake KeyRelease from auto-repeat
            if self._ptt_release_id:
                self._root.after_cancel(self._ptt_release_id)
            self._ptt_release_id = self._root.after(100, lambda: _do_release(p))

        self._root.bind_all(f"<KeyPress-{keysym}>",   _press,   add="+")
        self._root.bind_all(f"<KeyRelease-{keysym}>", _release, add="+")

    def _unbind_ptt(self) -> None:
        rid = getattr(self, "_ptt_release_id", None)
        if rid:
            try:
                self._root.after_cancel(rid)
            except Exception:
                pass
            self._ptt_release_id = None
        ks = getattr(self, "_ptt_keysym", None)
        if ks:
            try:
                self._root.unbind_all(f"<KeyPress-{ks}>")
                self._root.unbind_all(f"<KeyRelease-{ks}>")
            except Exception:
                pass
            self._ptt_keysym = None
        self._ptt_state = None

    # ── Pipeline start / stop ──────────────────────────────────────────────────

    def _on_start(self) -> None:
        # The previous session may still be closing its audio streams in the
        # background thread. resolve_devices() re-initializes PortAudio, and doing
        # that while old streams are alive corrupts the heap (SIGABRT in
        # Pa_Terminate/CloseStream). Wait for the thread to fully die first.
        if self._async_thread is not None and self._async_thread.is_alive():
            self._start_retries = getattr(self, "_start_retries", 0) + 1
            if self._start_retries == 1:
                self._start_btn.configure(state="disabled")
                self._error_label.configure(text="Жду остановки предыдущей сессии…")
            if self._start_retries > 40:  # ~10s — something is stuck
                self._start_retries = 0
                self._start_btn.configure(state="normal")
                self._error_label.configure(
                    text="Предыдущая сессия не остановилась — перезапустите приложение")
                return
            self._root.after(250, self._on_start)
            return
        self._start_retries = 0
        self._start_btn.configure(state="normal")

        openai_key = get_api_key("openai") or ""
        el_key     = get_api_key("elevenlabs") or ""
        if not openai_key:
            self._error_label.configure(text="API ключ OpenAI не найден. Запустите: babellang --setup")
            return
        if not el_key:
            self._error_label.configure(text="API ключ ElevenLabs не найден. Запустите: babellang --setup")
            return
        if self._settings.input_device is None or self._settings.output_device is None:
            self._error_label.configure(text="Устройства не настроены. Запустите: babellang --setup")
            return

        # Re-resolve devices by NAME: macOS renumbers audio devices whenever
        # Bluetooth headphones / iPhone mics connect, so saved indices go stale.
        res = resolve_devices(self._settings)
        if res.changed:
            save_settings(self._settings)
        if not res.ok:
            self._error_label.configure(
                text="Не найдено: " + "; ".join(res.errors) + " — выбери устройства заново ⚙"
            )
            self._open_device_settings()
            return
        if res.warnings:
            self._error_label.configure(text="\n".join(res.warnings))

        self._api_keys = {"openai": openai_key, "elevenlabs": el_key}
        self._running = True
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        if not res.warnings:
            self._error_label.configure(text="")

        self._logger = DialogLogger(self._log_dir)
        self._log_path_label.configure(text=f"Лог: {self._logger.path.name}")

        self._ptt_state = None
        if self._settings.vad_mode == "manual":
            self._ptt_state = PttState()
            ks = _PTT_KEYSYM.get(self._settings.ptt_hotkey, self._settings.ptt_hotkey)
            self._bind_ptt(ks, self._ptt_state)

        self._loop = asyncio.new_event_loop()
        self._async_thread = threading.Thread(target=self._run_asyncio, daemon=True)
        self._async_thread.start()
        self._root.after(250, self._poll_status)

    def _on_stop(self) -> None:
        self._running = False
        self._unbind_ptt()
        self._lyrics.clear()
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._status_label.configure(text="IDLE", text_color=_C_SUB)
        self._status_dot.configure(text_color=_C_SUB)
        self._latency_label.configure(text="")
        self._hint_label.configure(text="")
        if self._logger:
            self._logger.close()
            self._logger = None
        save_settings(self._settings)

    def _run_asyncio(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.set_exception_handler(
            lambda loop, ctx: logger.error("asyncio: %s", ctx.get("message"),
                                           exc_info=ctx.get("exception"))
        )
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as exc:
            logger.exception("Pipeline thread crashed")
            self._root.after(0, lambda e=exc: self._handle_async_crash(e))
        finally:
            self._loop.close()

    def _handle_async_crash(self, exc: BaseException) -> None:
        """Async thread died — reset the GUI to IDLE so СТАРТ works again."""
        msg = _humanize_error(exc)
        if self._running:
            self._on_stop()
        self._error_label.configure(text=f"Сессия прервана: {msg}")

    async def _async_main(self) -> None:
        my    = self._settings.my_language.upper()
        inter = self._settings.interlocutor_language.upper()

        self._outgoing = OutgoingPipeline(
            settings=self._settings,
            api_keys=self._api_keys,
            ptt_state=self._ptt_state,
        )
        self._outgoing._playback._smooth_ducking = self._settings.smooth_ducking

        # Recent outgoing translations — reference set for the text echo filter
        recent_out: deque = deque(maxlen=5)

        def _out_trn(text: str) -> None:
            try:
                self._outgoing.status.last_translated_text = text
                src = self._outgoing.status.last_source_text
                if text:
                    recent_out.append((time.time(), normalize_text(text)))
                if self._logger and text:
                    self._logger.log_outgoing(src, text, my, inter)
                self._subtitle_q.put((src, text, "out", my, inter))
            except Exception:
                logger.exception("outgoing subtitle callback failed")

        def _is_own_echo(incoming_text: str) -> bool:
            """Far-end echo: interlocutor's mic returned our own TTS. If the incoming
            transcript is nearly identical to a translation we sent in the last 60s,
            it is our voice coming back — not the interlocutor speaking."""
            norm = normalize_text(incoming_text)
            if len(norm) < 6:
                return False  # too short to compare reliably
            now = time.time()
            for ts, sent in recent_out:
                if now - ts > 60.0:
                    continue
                if difflib.SequenceMatcher(None, norm, sent).ratio() > 0.8:
                    return True
            return False

        if self._settings.stt_mode == "audio":
            self._outgoing._audio_translator._on_translation = _out_trn
        else:
            self._outgoing._translator._on_translation = _out_trn

        self._incoming = None
        if self._settings.incoming_device is not None and self._settings.speaker_device is not None:
            self._incoming = IncomingPipeline(
                settings=self._settings, api_keys=self._api_keys,
                transcript_filter=_is_own_echo,
            )
            self._incoming._playback._smooth_ducking = self._settings.smooth_ducking

            def _in_trn(text: str) -> None:
                try:
                    self._incoming.status.last_translated_text = text
                    src = self._incoming.status.last_source_text
                    if self._logger and text:
                        self._logger.log_incoming(src, text, my, inter)
                    self._subtitle_q.put((src, text, "in", my, inter))
                except Exception:
                    logger.exception("incoming subtitle callback failed")

            if self._settings.stt_mode == "audio":
                self._incoming._audio_translator._on_translation = _in_trn
            else:
                self._incoming._translator._on_translation = _in_trn

            # Anti-echo gates: while my TTS goes to the call, the far end may echo it
            # back into BlackHole 16ch — don't treat that as interlocutor speech.
            # +0.75s hangover covers the network round-trip of the echo.
            out_pb = self._outgoing._playback
            in_pb  = self._incoming._playback
            self._incoming.set_capture_gate(lambda: out_pb.tts_active_within(0.75))
            # Symmetric: incoming TTS through speakers must not trigger my VAD (auto mode)
            self._outgoing.set_capture_gate(lambda: in_pb.tts_active_within(0.3))

        await self._outgoing.start()
        if self._incoming:
            await self._incoming.start()

        try:
            while self._running:
                await asyncio.sleep(0.1)
        finally:
            # Close BOTH pipelines even if one stop() raises — a leaked open
            # stream makes the next PortAudio re-init crash the whole process.
            try:
                await self._outgoing.stop()
            except Exception:
                pass
            if self._incoming:
                try:
                    await self._incoming.stop()
                except Exception:
                    pass

    # ── Status polling ─────────────────────────────────────────────────────────

    def _poll_status(self) -> None:
        if not self._running:
            return
        self._drain_subtitles()
        out = self._outgoing
        if out:
            raw = out.status.state
            # PTT: key held but no voice yet → show "PTT ▶" instead of "CAPTURING"
            if (self._ptt_state is not None
                    and self._ptt_state.active
                    and raw not in ("speaking", "translating", "playing")):
                state = "ptt"
            else:
                state = raw
            color = _STATE_COLOR.get(state, _C_SUB)
            self._status_dot.configure(text_color=color)
            label = "PTT ▶" if state == "ptt" else state.upper()
            self._status_label.configure(text=label, text_color=color)
            lat = out.status.last_latency_ms
            if lat > 0:
                self._latency_label.configure(text=f"Задержка: {lat:.0f}мс")
            err = out.status.tts_error
            if not err and self._incoming:
                err = self._incoming.status.tts_error
            self._error_label.configure(text=err or "")

            # Voice indicators — driven by REAL audio state
            if self._ptt_state is not None:
                mic_active = self._ptt_state.active  # PTT: animate while key held
            else:
                mic_active = raw == "speaking"        # Auto: animate when VAD fires
            self._mic_indicator.set_active(mic_active)
            # 🔊 translated voice is being written to BlackHole 2ch right now
            self._out_tts_indicator.set_active(out.tts_playing)
            # 🎧 incoming translation is playing in my speakers right now
            self._in_tts_indicator.set_active(
                bool(self._incoming) and self._incoming.tts_playing
            )

            # Routing hint: BlackHole 16ch silent for 30s+ → call app output is
            # probably not routed into it
            if self._incoming and self._incoming.seconds_since_signal() > 30.0:
                self._hint_label.configure(
                    text="🔇 Нет звука от собеседника — проверь, что динамик "
                         "в Teams/Zoom = BlackHole 16ch")
            else:
                self._hint_label.configure(text="")

        self._root.after(250, self._poll_status)

    # ── Dialog / lyrics ────────────────────────────────────────────────────────

    def _drain_subtitles(self) -> None:
        """Runs in the main (Tk) thread — pulls subtitle entries queued by pipelines."""
        while True:
            try:
                src, trn, side, my, inter = self._subtitle_q.get_nowait()
            except queue.Empty:
                return
            try:
                self._append_dialog(src, trn, side, my, inter)
            except Exception:
                logger.exception("subtitle rendering failed (side=%s)", side)

    def _append_dialog(self, source: str, translated: str, side: str, my: str, inter: str) -> None:
        if not source and not translated:
            return
        self._lyrics.add_entry(source, translated, side, my, inter)

    # ── Settings dialog ────────────────────────────────────────────────────────

    def _open_device_settings(self) -> None:
        _DeviceSettingsDialog(self._root, self._settings, self._on_devices_saved)

    def _on_devices_saved(
        self,
        mic_idx: int, mic_name: str,
        out_idx: int, out_name: str,
        inc_idx: Optional[int], inc_name: str,
        spk_idx: Optional[int], spk_name: str,
        stt_mode: str,
        vad_mode: str,
        ptt_hotkey: str,
    ) -> None:
        self._settings.input_device       = mic_idx
        self._settings.input_device_name  = mic_name
        self._settings.output_device      = out_idx
        self._settings.output_device_name = out_name
        self._settings.incoming_device    = inc_idx
        self._settings.incoming_device_name = inc_name
        self._settings.speaker_device     = spk_idx
        self._settings.speaker_device_name = spk_name
        self._settings.stt_mode           = stt_mode
        self._settings.vad_mode           = vad_mode
        self._settings.ptt_hotkey         = ptt_hotkey
        save_settings(self._settings)

    def _refresh_lang_label(self) -> None:
        my    = self._settings.my_language.upper()
        inter = self._settings.interlocutor_language.upper()
        self._lang_label.configure(text=f"{my} ⇄ {inter}")

    # ── Window lifecycle ───────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._running:
            self._on_stop()
            if self._async_thread and self._async_thread.is_alive():
                self._async_thread.join(timeout=5.0)
        self._settings.window_geometry = self._root.geometry().split("+")[0]  # "WxH" without position
        save_settings(self._settings)
        self._root.destroy()
        self._kill_shell()

    @staticmethod
    def _kill_shell() -> None:
        try:
            pid_file = Path("/tmp/babellang_shell.pid")
            if pid_file.exists():
                shell_pid = int(pid_file.read_text().strip())
                pid_file.unlink(missing_ok=True)
                os.kill(shell_pid, signal.SIGTERM)
        except Exception:
            pass

    def run(self) -> None:
        self._root.after(150, self._root.lift)
        self._root.after(200, self._root.focus_force)
        self._root.after(200, lambda: self._root.attributes("-topmost", True))
        self._root.after(400, lambda: self._root.attributes("-topmost", False))
        self._root.mainloop()


# ── Device settings dialog ─────────────────────────────────────────────────────

class _DeviceSettingsDialog(ctk.CTkToplevel):
    """Modal dialog for selecting audio input/output devices and STT mode."""

    def __init__(self, parent, settings, on_save) -> None:
        super().__init__(parent)
        self.title("Аудиоустройства")
        self.geometry("500x820")
        self.resizable(False, True)
        self.configure(fg_color=_C_BG)
        self.grab_set()
        self.focus()

        devices    = sd.query_devices()
        self._in_opts  = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        self._out_opts = [(i, d["name"]) for i, d in enumerate(devices) if d["max_output_channels"] > 0]

        # Human-friendly labels: BlackHole names mean nothing to a user, so show
        # their role in the translation chain. Real names stay in parentheses —
        # they are what Teams/Zoom display.
        def _fmt(i: int, n: str) -> str:
            ln = n.lower()
            if "blackhole 2ch" in ln:
                return f"{i}: 📤 Исходящее — перевод собеседнику ({n})"
            if "blackhole 16ch" in ln:
                return f"{i}: 📥 Входящее — звук собеседника ({n})"
            return f"{i}: {n}"

        # label → (index, real device name); labels are unique thanks to the index prefix
        self._in_map  = {_fmt(i, n): (i, n) for i, n in self._in_opts}
        self._out_map = {_fmt(i, n): (i, n) for i, n in self._out_opts}
        in_labels  = list(self._in_map)
        out_labels = list(self._out_map)

        def _cur_in(idx):
            for lbl in in_labels:
                if lbl.startswith(f"{idx}:"):
                    return lbl
            return in_labels[0] if in_labels else ""

        def _cur_out(idx):
            for lbl in out_labels:
                if lbl.startswith(f"{idx}:"):
                    return lbl
            return out_labels[0] if out_labels else ""

        self._mic_var = ctk.StringVar(value=_cur_in(settings.input_device))
        self._out_var = ctk.StringVar(value=_cur_out(settings.output_device))
        self._inc_var = ctk.StringVar(value=_cur_in(settings.incoming_device or 0))
        self._spk_var = ctk.StringVar(value=_cur_out(settings.speaker_device or 0))

        pad = {"padx": 20, "pady": 6}

        ctk.CTkLabel(self, text="Аудиоустройства",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=_C_TEXT).pack(pady=(18, 4))

        self._build_row("🎤  Микрофон (твой голос):",            in_labels,  self._mic_var, pad)
        self._build_row("🔊  Выход (в Zoom / BlackHole 2ch):",   out_labels, self._out_var, pad)
        self._build_row("📥  Входящий (BlackHole 16ch):",         in_labels,  self._inc_var, pad)
        self._build_row("🔈  Динамики (ты слышишь):",            out_labels, self._spk_var, pad)

        mode_frame = ctk.CTkFrame(self, fg_color=_C_PANEL, corner_radius=12)
        mode_frame.pack(fill="x", **pad)
        ctk.CTkLabel(mode_frame, text="🧠  Режим распознавания + перевода",
                     font=ctk.CTkFont(size=12), text_color=_C_SUB,
                     anchor="w").pack(fill="x", padx=12, pady=(8, 4))
        self._mode_var = ctk.StringVar(value=settings.stt_mode)
        ctk.CTkSegmentedButton(
            mode_frame, values=["audio", "classic"], variable=self._mode_var,
            fg_color=_C_PANEL2, selected_color=_ACCENT, selected_hover_color="#0056CC",
            unselected_color=_C_PANEL2, unselected_hover_color=_C_PANEL,
            text_color=_C_TEXT, font=ctk.CTkFont(size=12), corner_radius=8,
        ).pack(fill="x", padx=12, pady=(0, 4))
        self._mode_hint = ctk.CTkLabel(
            mode_frame, text=self._mode_hint_text(settings.stt_mode),
            font=ctk.CTkFont(size=10), text_color=_C_SUB,
        )
        self._mode_hint.pack(anchor="w", padx=12, pady=(0, 8))
        self._mode_var.trace_add("write", lambda *_: self._mode_hint.configure(
            text=self._mode_hint_text(self._mode_var.get())
        ))

        # ── VAD / PTT mode ──────────────────────────────────────────────────────
        vad_frame = ctk.CTkFrame(self, fg_color=_C_PANEL, corner_radius=12)
        vad_frame.pack(fill="x", **pad)
        ctk.CTkLabel(vad_frame, text="🎯  Режим записи (только твой голос)",
                     font=ctk.CTkFont(size=12), text_color=_C_SUB,
                     anchor="w").pack(fill="x", padx=12, pady=(8, 4))

        self._vad_mode_var = ctk.StringVar(
            value="Ручной (PTT)" if settings.vad_mode == "manual" else "Авто (VAD)"
        )
        ctk.CTkSegmentedButton(
            vad_frame, values=["Авто (VAD)", "Ручной (PTT)"],
            variable=self._vad_mode_var,
            command=self._on_vad_mode_change,
            fg_color=_C_PANEL2, selected_color=_ACCENT, selected_hover_color="#0056CC",
            unselected_color=_C_PANEL2, unselected_hover_color=_C_PANEL,
            text_color=_C_TEXT, font=ctk.CTkFont(size=12), corner_radius=8,
        ).pack(fill="x", padx=12, pady=(0, 4))

        # Hotkey picker row (visible only in PTT mode)
        self._ptt_hotkey = settings.ptt_hotkey
        self._ptt_frame = ctk.CTkFrame(vad_frame, fg_color="transparent")
        self._ptt_key_label = ctk.CTkLabel(
            self._ptt_frame,
            text=f"Клавиша: {hotkey_display(self._ptt_hotkey)}",
            font=ctk.CTkFont(size=12), text_color=_C_TEXT,
        )
        self._ptt_key_label.pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            self._ptt_frame, text="Изменить",
            font=ctk.CTkFont(size=12), width=100, height=30,
            fg_color=_C_PANEL2, hover_color=_C_PANEL,
            text_color=_ACCENT, corner_radius=8,
            command=self._capture_hotkey,
        ).pack(side="left")
        ctk.CTkFrame(vad_frame, fg_color="transparent", height=4).pack()

        if settings.vad_mode == "manual":
            self._ptt_frame.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(self, text="Изменения вступят в силу при следующем СТАРТ",
                     font=ctk.CTkFont(size=10), text_color=_C_SUB).pack(pady=(8, 4))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(4, 18))
        ctk.CTkButton(
            btn_row, text="Отмена",
            fg_color=_C_PANEL2, hover_color=_C_PANEL,
            text_color=_C_SUB, font=ctk.CTkFont(size=14),
            corner_radius=12, height=42, command=self.destroy,
        ).pack(side="left", expand=True, fill="x", padx=(0, 8))
        ctk.CTkButton(
            btn_row, text="Сохранить",
            fg_color=_ACCENT, hover_color="#0056CC",
            text_color="#FFFFFF", font=ctk.CTkFont(size=14),
            corner_radius=12, height=42,
            command=lambda: self._save(on_save),
        ).pack(side="left", expand=True, fill="x")

    @staticmethod
    def _mode_hint_text(mode: str) -> str:
        if mode == "audio":
            return "audio — gpt-4o-audio-preview: один вызов API, минимальная задержка"
        return "classic — gpt-4o-mini-transcribe → gpt-4o-mini: два вызова, видна транскрипция"

    def _on_vad_mode_change(self, value: str) -> None:
        if value == "Ручной (PTT)":
            self._ptt_frame.pack(fill="x", padx=12, pady=(0, 8))
        else:
            self._ptt_frame.pack_forget()

    def _capture_hotkey(self) -> None:
        self._ptt_key_label.configure(text="Нажми клавишу...")
        # Delay 150ms so the button-click event doesn't get captured as the hotkey
        self.after(150, self._arm_hotkey_capture)

    def _arm_hotkey_capture(self) -> None:
        self.focus_set()
        self.bind("<KeyPress>", self._on_hotkey_press)

    def _on_hotkey_press(self, event) -> None:
        self.unbind("<KeyPress>")
        # Try known special keys first; fall back to raw keysym for letters/digits
        key = TK_TO_HOTKEY.get(event.keysym)
        if not key:
            # Accept single printable chars and function keys directly
            sym = event.keysym
            if len(sym) == 1 and sym.isprintable():
                key = sym.lower()
            elif sym.startswith("F") and sym[1:].isdigit():
                key = sym.lower()
        if key:
            self._ptt_hotkey = key
            # Display: use hotkey_display if available, else the key string itself
            display = hotkey_display(key)
            self._ptt_key_label.configure(text=f"Клавиша: {display}")
        else:
            self._ptt_key_label.configure(text=f"Клавиша: {hotkey_display(self._ptt_hotkey)}")

    def _build_row(self, label: str, values: list, var: ctk.StringVar, pad: dict) -> None:
        frame = ctk.CTkFrame(self, fg_color=_C_PANEL, corner_radius=12)
        frame.pack(fill="x", **pad)
        ctk.CTkLabel(frame, text=label,
                     font=ctk.CTkFont(size=12), text_color=_C_SUB,
                     anchor="w").pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkOptionMenu(
            frame, values=values, variable=var,
            fg_color=_C_PANEL2, button_color=_ACCENT, button_hover_color="#0056CC",
            dropdown_fg_color=_C_PANEL, dropdown_hover_color=_C_PANEL2,
            text_color=_C_TEXT, dropdown_text_color=_C_TEXT,
            font=ctk.CTkFont(size=12), corner_radius=8,
        ).pack(fill="x", padx=12, pady=(0, 8))

    def _save(self, on_save) -> None:
        def _parse_in(var):
            return self._in_map.get(var.get(), (None, ""))

        def _parse_out(var):
            return self._out_map.get(var.get(), (None, ""))

        mic_i, mic_n = _parse_in(self._mic_var)
        out_i, out_n = _parse_out(self._out_var)
        inc_i, inc_n = _parse_in(self._inc_var)
        spk_i, spk_n = _parse_out(self._spk_var)
        vad_mode = "manual" if self._vad_mode_var.get() == "Ручной (PTT)" else "auto"
        on_save(mic_i, mic_n, out_i, out_n, inc_i, inc_n, spk_i, spk_n,
                self._mode_var.get(), vad_mode, self._ptt_hotkey)
        self.destroy()


def launch_gui() -> None:
    try:
        import customtkinter  # noqa: F401
    except ImportError:
        print("customtkinter не установлен. Запустите: pip install customtkinter")
        sys.exit(1)

    _setup_file_logging()
    _install_exception_hooks()

    if not _acquire_single_instance_lock():
        # A second copy would fight the first over the mic and BlackHole devices
        from tkinter import messagebox
        hidden = ctk.CTk()
        hidden.withdraw()
        messagebox.showwarning(
            "BabelLang", "BabelLang уже запущен.\nЗакрой предыдущее окно перед новым запуском.")
        hidden.destroy()
        return

    logger.info("=== BabelLang GUI started (pid=%d) ===", os.getpid())
    try:
        BabelLangGUI().run()
    finally:
        _release_single_instance_lock()
        logger.info("=== BabelLang GUI exited ===")
