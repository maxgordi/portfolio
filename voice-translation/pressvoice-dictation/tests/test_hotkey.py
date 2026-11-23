import sys

import pytest

from qtapp.hotkey import HOTKEYS, HotkeyLogic, hotkey_id


def make(mode="push_to_talk", is_modifier=True):
    events = []
    logic = HotkeyLogic(mode, is_modifier,
                        on_start=lambda: events.append("start"),
                        on_stop=lambda: events.append("stop"),
                        on_cancel=lambda: events.append("cancel"))
    return logic, events


def test_push_to_talk_hold_and_release():
    logic, ev = make()
    logic.hotkey_down()
    logic.hotkey_down()  # auto-repeat
    logic.hotkey_down()
    logic.hotkey_up()
    logic.hotkey_up()    # duplicate release (e.g. from the poller)
    assert ev == ["start", "stop"]


def test_push_to_talk_shortcut_cancels():
    logic, ev = make()
    logic.hotkey_down()
    logic.other_key_down()  # Right Ctrl + C
    logic.other_key_down()
    logic.hotkey_up()
    assert ev == ["start", "cancel"]


def test_own_paste_keystrokes_do_not_cancel():
    """Live dictation pastes with Ctrl+V while Right Ctrl is still held."""
    import time
    logic, ev = make()
    logic.hotkey_down()
    logic.own_input_until = time.monotonic() + 0.5
    logic.other_key_down()  # our own Ctrl+V
    logic.own_input_until = 0.0
    logic.other_key_down()  # the user's Right Ctrl + C
    logic.hotkey_up()
    assert ev == ["start", "cancel"]


def test_non_modifier_ignores_other_keys():
    logic, ev = make(is_modifier=False)
    logic.hotkey_down()
    logic.other_key_down()
    logic.hotkey_up()
    assert ev == ["start", "stop"]


def test_toggle_taps():
    logic, ev = make("toggle")
    logic.hotkey_down(); logic.hotkey_up()
    assert ev == ["start"]
    logic.other_key_down()  # typing while recording is fine in toggle mode
    logic.hotkey_down(); logic.hotkey_down(); logic.hotkey_up()
    assert ev == ["start", "stop"]


def test_toggle_shortcut_is_not_a_tap():
    logic, ev = make("toggle")
    logic.hotkey_down(); logic.other_key_down(); logic.hotkey_up()
    assert ev == []


def test_release_without_press_and_reset():
    logic, ev = make()
    logic.hotkey_up()
    logic.hotkey_down()
    logic.reset()
    logic.hotkey_up()
    assert ev == ["start"]
    assert not logic.recording and not logic.held


def test_hotkey_id_fallback():
    assert hotkey_id("cmd_r") == "ctrl_r"  # a macOS config value
    assert hotkey_id("f13") == "f13"


@pytest.mark.skipif(sys.platform == "darwin", reason="Windows/X11 key codes")
def test_key_table_matches_pynput():
    """Our virtual-key / keysym table must agree with pynput's own codes."""
    from pynput.keyboard import Key
    names = {"ctrl_r": "ctrl_r", "alt_r": "alt_r", "shift_r": "shift_r",
             "menu": "menu", "pause": "pause", "scroll_lock": "scroll_lock",
             "insert": "insert", **{f"f{n}": f"f{n}" for n in range(13, 20)}}
    for hk, key_name in names.items():
        _label, win_vks, x_syms, _mod = HOTKEYS[hk]
        codes = win_vks if sys.platform == "win32" else x_syms
        assert getattr(Key, key_name).value.vk in codes, hk
