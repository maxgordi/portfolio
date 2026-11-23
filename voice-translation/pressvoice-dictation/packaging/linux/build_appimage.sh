#!/bin/bash
# Wraps the PyInstaller build (dist/PressVoice) into
# dist/PressVoice-<version>-x86_64.AppImage. Run from the repository root
# after: pyinstaller --noconfirm --clean packaging/pressvoice.spec
set -euo pipefail

VERSION=$(python -c "from version import __version__; print(__version__)")
APPDIR=build/AppDir
TOOL=build/appimagetool-x86_64.AppImage

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp -a dist/PressVoice/. "$APPDIR/usr/bin/"
cp packaging/linux/AppRun "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun"
cp packaging/linux/pressvoice.desktop "$APPDIR/pressvoice.desktop"
# Translated descriptions for app menus (Comment[uk]=…), from i18n/locales.
python - "$APPDIR/pressvoice.desktop" <<'PY'
import json, sys
sys.path.insert(0, ".")
from i18n import LANGUAGES
with open(sys.argv[1], "a", encoding="utf-8") as desktop:
    for code in LANGUAGES:
        if code == "en":
            continue
        with open(f"i18n/locales/{code}.json", encoding="utf-8") as f:
            comment = json.load(f)["app.comment"]
        desktop.write(f"Comment[{code.replace('-', '_')}]={comment}\n")
PY
cp assets/pressvoice.png "$APPDIR/pressvoice.png"
cp assets/pressvoice.png "$APPDIR/.DirIcon"

if [ ! -x "$TOOL" ]; then
    curl -fsSL -o "$TOOL" \
        https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$TOOL"
fi

ARCH=x86_64 "$TOOL" --appimage-extract-and-run --no-appstream \
    "$APPDIR" "dist/PressVoice-${VERSION}-x86_64.AppImage"
echo "Built dist/PressVoice-${VERSION}-x86_64.AppImage"
