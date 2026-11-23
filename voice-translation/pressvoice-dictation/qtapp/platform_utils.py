import os
import sys

from i18n import t
from version import APP_NAME

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

AUMID = "PressVoice.PressVoice"  # must match AppUserModelID in installer.iss
_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_DESKTOP_FILE = os.path.join(
    os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
    "autostart", "pressvoice.desktop",
)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_path(*parts: str) -> str:
    """A bundled data file, both from source and inside the PyInstaller build."""
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


def launch_command() -> list:
    """How to start this app again (autostart, self-restart)."""
    appimage = os.environ.get("APPIMAGE")
    if appimage:
        return [appimage]
    if is_frozen():
        return [sys.executable]
    exe = sys.executable
    if IS_WINDOWS and exe.lower().endswith("python.exe"):
        pythonw = exe[:-len("python.exe")] + "pythonw.exe"
        if os.path.exists(pythonw):
            exe = pythonw  # no console window
    main = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "main_qt.py")
    return [exe, main]


def _quote(cmd: list) -> str:
    return " ".join(f'"{c}"' for c in cmd)


MIC_PRIVACY_URL = "ms-settings:privacy-microphone"
_MIC_CONSENT = (r"Software\Microsoft\Windows\CurrentVersion"
                r"\CapabilityAccessManager\ConsentStore\microphone")


def windows_microphone_block() -> str:
    """Which Windows privacy switch keeps desktop apps away from the
    microphone, as a sentence in the interface language ("" if none). A
    blocked microphone fails to open with a meaningless "Unanticipated host
    error [MME error 1]"."""
    if not IS_WINDOWS:
        return ""
    import winreg

    def value(root, path, name="Value") -> str:
        try:
            with winreg.OpenKey(root, path) as key:
                return str(winreg.QueryValueEx(key, name)[0])
        except OSError:
            return ""

    hklm, hkcu = winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER
    if value(hklm, r"SOFTWARE\Policies\Microsoft\Windows\AppPrivacy",
             "LetAppsAccessMicrophone") == "2":
        return t("windows.policy")
    if value(hklm, _MIC_CONSENT) == "Deny":
        return t("windows.toggle_off", toggle=t("windows.toggle.device"))
    if value(hkcu, _MIC_CONSENT) == "Deny":
        return t("windows.toggle_off", toggle=t("windows.toggle.apps"))
    if value(hkcu, _MIC_CONSENT + r"\NonPackaged") == "Deny":
        return t("windows.toggle_off", toggle=t("windows.toggle.desktop"))
    return ""


def autostart_supported() -> bool:
    return IS_WINDOWS or IS_LINUX


def autostart_label() -> str:
    return t("autostart.windows") if IS_WINDOWS else t("autostart.login")


def is_autostart_enabled() -> bool:
    if IS_WINDOWS:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
                winreg.QueryValueEx(key, APP_NAME)
                return True
        except OSError:
            return False
    if IS_LINUX:
        return os.path.exists(_DESKTOP_FILE)
    return False


def set_autostart(enabled: bool) -> None:
    if IS_WINDOWS:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0,
                            winreg.KEY_SET_VALUE) as key:
            if enabled:
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ,
                                  _quote(launch_command()))
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
        return
    if IS_LINUX:
        if enabled:
            os.makedirs(os.path.dirname(_DESKTOP_FILE), exist_ok=True)
            with open(_DESKTOP_FILE, "w", encoding="utf-8") as f:
                f.write(
                    "[Desktop Entry]\n"
                    "Type=Application\n"
                    f"Name={APP_NAME}\n"
                    f"Exec={_quote(launch_command())}\n"
                    "X-GNOME-Autostart-enabled=true\n"
                    "Terminal=false\n"
                )
        elif os.path.exists(_DESKTOP_FILE):
            os.remove(_DESKTOP_FILE)
