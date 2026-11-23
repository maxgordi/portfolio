"""Thin ctypes wrappers over the few Win32 calls the app needs. Import only
on Windows."""
import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
MAPVK_VK_TO_VSC = 0

VK_CONTROL = 0x11
VK_MENU = 0x12
VK_V = 0x56
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_RMENU = 0xA5
VK_RSHIFT = 0xA1
# Unassigned virtual key. Tapping it while Alt is held stops Windows from
# opening the focused window's menu bar when Alt is released alone.
VK_MENU_MASK = 0xE8

# Keys whose scan codes carry the E0 prefix; without the extended flag an
# injected Right Ctrl arrives as Left Ctrl, Insert as Numpad 0, and so on.
_EXTENDED = {
    0xA3, 0xA5, 0x5B, 0x5C, 0x5D, 0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22,
    0x25, 0x26, 0x27, 0x28, 0x6F, 0x90,
}

ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
user32.MapVirtualKeyW.restype = wintypes.UINT
user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
user32.BringWindowToTop.argtypes = (wintypes.HWND,)
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
user32.LoadKeyboardLayoutW.argtypes = (wintypes.LPCWSTR, wintypes.UINT)
user32.LoadKeyboardLayoutW.restype = wintypes.HANDLE
user32.ActivateKeyboardLayout.argtypes = (wintypes.HANDLE, wintypes.UINT)
user32.ActivateKeyboardLayout.restype = wintypes.HANDLE
user32.GetKeyboardLayout.argtypes = (wintypes.DWORD,)
user32.GetKeyboardLayout.restype = wintypes.HANDLE
kernel32.GetCurrentThreadId.restype = wintypes.DWORD


def _key_input(vk: int, up: bool) -> INPUT:
    flags = KEYEVENTF_KEYUP if up else 0
    if vk in _EXTENDED:
        flags |= KEYEVENTF_EXTENDEDKEY
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.ki = KEYBDINPUT(
        wVk=vk,
        wScan=user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) & 0xFF,
        dwFlags=flags,
        time=0,
        dwExtraInfo=0,
    )
    return inp


def send_keys(events: list) -> int:
    """events: [(vk, is_up), ...] sent as one atomic batch. Returns how many
    the system accepted (0 means input is blocked, e.g. by UIPI)."""
    arr = (INPUT * len(events))(*[_key_input(vk, up) for vk, up in events])
    return user32.SendInput(len(events), arr, ctypes.sizeof(INPUT))


def send_ctrl_v() -> int:
    # Virtual-key codes, not characters: "v" doesn't exist on a Russian
    # layout, but VK_V + Ctrl is Paste on every layout.
    return send_keys([(VK_CONTROL, False), (VK_V, False),
                      (VK_V, True), (VK_CONTROL, True)])


def tap_menu_mask() -> None:
    send_keys([(VK_MENU_MASK, False), (VK_MENU_MASK, True)])


def is_key_down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def force_foreground(hwnd: int) -> bool:
    """SetForegroundWindow is refused to background processes; joining the
    foreground thread's input queue for a moment lifts that restriction."""
    hwnd = wintypes.HWND(hwnd)
    fg = user32.GetForegroundWindow()
    fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    me = kernel32.GetCurrentThreadId()
    attached = False
    if fg_thread and fg_thread != me:
        attached = bool(user32.AttachThreadInput(me, fg_thread, True))
    try:
        user32.ShowWindow(hwnd, 5)  # SW_SHOW
        user32.BringWindowToTop(hwnd)
        return bool(user32.SetForegroundWindow(hwnd))
    finally:
        if attached:
            user32.AttachThreadInput(me, fg_thread, False)


def activate_keyboard_layout(klid: str) -> int:
    """klid like "00000419" (Russian) or "00000409" (US). Returns the
    previous layout handle."""
    prev = user32.GetKeyboardLayout(0)
    hkl = user32.LoadKeyboardLayoutW(klid, 1)  # KLF_ACTIVATE
    user32.ActivateKeyboardLayout(hkl, 0)
    return prev or 0


def current_keyboard_layout() -> int:
    return user32.GetKeyboardLayout(0) or 0


def set_app_user_model_id(app_id: str) -> None:
    """Groups the taskbar entry and attributes toast notifications to the
    Start-menu shortcut (name and icon) instead of python.exe."""
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass
