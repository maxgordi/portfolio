# PyInstaller build of the macOS menu bar app:
#     pyinstaller --noconfirm --clean packaging/macos/pressvoice_mac.spec
# Produces dist/PressVoice.app (ad-hoc signed by PyInstaller).
import json
import os
import plistlib
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
sys.path.insert(0, ROOT)
from i18n import LANGUAGES  # noqa: E402
from version import __version__  # noqa: E402

# macOS shows the microphone permission prompt from Info.plist; a
# <lang>.lproj/InfoPlist.strings per language translates it.
_LPROJ = {"zh-CN": "zh-Hans", "zh-TW": "zh-Hant"}
lproj_root = os.path.join(workpath, "lproj")
lproj_datas = []
for code in LANGUAGES:
    with open(os.path.join(ROOT, "i18n", "locales", f"{code}.json"), encoding="utf-8") as f:
        usage = json.load(f)["macos.mic_usage"]
    folder = _LPROJ.get(code, code) + ".lproj"
    os.makedirs(os.path.join(lproj_root, folder), exist_ok=True)
    # A .strings file may be a property list, which sidesteps any doubt
    # about its text encoding.
    with open(os.path.join(lproj_root, folder, "InfoPlist.strings"), "wb") as f:
        plistlib.dump({"NSMicrophoneUsageDescription": usage}, f)
    lproj_datas.append((os.path.join(lproj_root, folder, "InfoPlist.strings"), folder))

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    datas=[(os.path.join(ROOT, "i18n", "locales"), os.path.join("i18n", "locales"))]
    + lproj_datas,
    hiddenimports=["app", "mac_selftest", "mac_support", "live.phrases"],
    excludes=["tkinter", "PIL", "PySide6", "shiboken6", "pynput", "qtapp",
              "pytest", "main_qt"],
    noarchive=False,
)


def _min_macos(binaries) -> str:
    """The newest LC_BUILD_VERSION minos among bundled binaries: the app
    can't run on anything older, so Finder should say so up front."""
    best = (11, 0)
    for _dest, src, _kind in binaries:
        try:
            out = subprocess.run(["otool", "-l", src], capture_output=True,
                                 text=True, timeout=30).stdout
        except Exception:
            continue
        for m in re.finditer(r"minos (\d+)\.(\d+)", out):
            best = max(best, (int(m.group(1)), int(m.group(2))))
    return f"{best[0]}.{best[1]}"


MIN_MACOS = _min_macos(a.binaries)
print(f"PressVoice.app minimum macOS: {MIN_MACOS}")

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PressVoice",
    console=False,
    argv_emulation=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PressVoice", upx=False)
app = BUNDLE(
    coll,
    name="PressVoice.app",
    icon=os.path.join(ROOT, "assets", "pressvoice.icns"),
    bundle_identifier="com.pressvoice.mac",
    version=__version__,
    info_plist={
        "CFBundleName": "PressVoice",
        "CFBundleDisplayName": "PressVoice",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "LSUIElement": True,  # menu bar only, no Dock icon
        "CFBundleDevelopmentRegion": "en",
        "CFBundleLocalizations": [_LPROJ.get(c, c) for c in LANGUAGES],
        "LSMinimumSystemVersion": MIN_MACOS,
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription":
            "PressVoice records your voice while you hold the hotkey and "
            "sends it to OpenAI to turn it into text.",
    },
)
