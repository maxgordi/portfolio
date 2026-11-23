# PyInstaller build for Windows and Linux:
#     pyinstaller --noconfirm --clean packaging/pressvoice.spec
# Produces dist/PressVoice/ (one folder: starts fast and trips antivirus
# heuristics far less often than a one-file build).
import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, ROOT)
from version import __version__  # noqa: E402

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    hidden = ["live.phrases", "pynput.keyboard._win32", "pynput.mouse._win32",
              "pynput._util.win32", "pynput._util.win32_vks"]
    runtime_hooks = []
else:
    hidden = ["live.phrases", "pynput.keyboard._xorg", "pynput.mouse._xorg",
              "pynput._util.xorg", "pynput._util.xorg_keysyms"]
    runtime_hooks = [os.path.join(SPECPATH, "rthook_portaudio.py")]

a = Analysis(
    [os.path.join(ROOT, "main_qt.py")],
    pathex=[ROOT],
    datas=[(os.path.join(ROOT, "assets", "pressvoice.png"), "assets"),
           (os.path.join(ROOT, "i18n", "locales"), os.path.join("i18n", "locales"))],
    hiddenimports=hidden,
    runtime_hooks=runtime_hooks,
    excludes=["tkinter", "PIL", "rumps", "objc", "AppKit", "Foundation", "Quartz",
              "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtPdf",
              "PySide6.QtOpenGL", "PySide6.QtSvg"],
    noarchive=False,
)

# Trim what a tray app never touches: Qt's translations and the 20 MB
# software OpenGL fallback.
a.datas = [d for d in a.datas if "translations" not in d[0].replace("\\", "/")]
a.binaries = [b for b in a.binaries if "opengl32sw" not in b[0].lower()]

version_file = None
if IS_WINDOWS:
    parts = tuple(int(x) for x in __version__.split(".")) + (0,)
    version_file = os.path.join(workpath, "version_info.txt")
    with open(version_file, "w", encoding="utf-8") as f:
        f.write(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={parts}, prodvers={parts}, mask=0x3f, flags=0x0,
                    OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'PressVoice'),
      StringStruct('FileDescription', 'PressVoice — voice to text'),
      StringStruct('FileVersion', '{__version__}'),
      StringStruct('InternalName', 'PressVoice'),
      StringStruct('OriginalFilename', 'PressVoice.exe'),
      StringStruct('ProductName', 'PressVoice'),
      StringStruct('ProductVersion', '{__version__}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""")

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PressVoice",
    console=False,
    icon=os.path.join(ROOT, "assets", "pressvoice.ico") if IS_WINDOWS else None,
    version=version_file,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PressVoice", upx=False)
