import io
import os
import sys
import tempfile
import time
import wave

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Never touch the real user config, even before a fixture kicks in.
os.environ["PRESSVOICE_CONFIG_DIR"] = tempfile.mkdtemp(prefix="pressvoice-tests-")
if sys.platform == "darwin":
    # Windows and Linux CI run with a real desktop (Linux under Xvfb).
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def make_wav(seconds: float, rate: int = 16000, silent: bool = False) -> bytes:
    """A mono 16-bit WAV of faint noise (or exact zeros with silent=True)."""
    buf = io.BytesIO()
    frame = b"\x00\x00\x00\x00" if silent else b"\x01\x00\xff\xff"  # 1, -1
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frame * (int(seconds * rate) // 2))
    return buf.getvalue()


def qwait(app, cond, timeout: float = 3.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(cond())


@pytest.fixture(autouse=True)
def english_interface():
    """Tests assert English texts whatever the machine's language."""
    import i18n
    i18n.set_language("en")
    yield
    i18n.set_language("en")


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    import config
    import history
    import styles
    monkeypatch.setattr(config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setattr(history, "HISTORY_PATH", str(tmp_path / "history.json"))
    monkeypatch.setattr(styles, "STYLES_PATH", str(tmp_path / "styles.json"))
    return tmp_path


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


@pytest.fixture
def no_autostart(monkeypatch):
    """Keep tests away from the real registry / autostart folder."""
    from qtapp import platform_utils
    state = {"enabled": False, "calls": []}
    monkeypatch.setattr(platform_utils, "is_autostart_enabled", lambda: state["enabled"])

    def fake_set(enabled):
        state["calls"].append(enabled)
        state["enabled"] = enabled
    monkeypatch.setattr(platform_utils, "set_autostart", fake_set)
    return state
