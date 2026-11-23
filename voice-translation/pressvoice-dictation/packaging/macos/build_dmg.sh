#!/bin/bash
# Wraps dist/PressVoice.app into dist/PressVoice-<version>-macOS-<arch>.dmg:
# the usual "drag PressVoice into Applications" disk image.
set -euo pipefail

VERSION=$(python3 -c "import sys; sys.path.insert(0, '.'); from version import __version__; print(__version__)")
APP=dist/PressVoice.app
ARCH=$(lipo -archs "$APP/Contents/MacOS/PressVoice" | tr ' ' '+')
STAGE=build/dmg-stage
DMG="dist/PressVoice-${VERSION}-macOS-${ARCH}.dmg"

codesign --verify --deep --strict "$APP"

rm -rf "$STAGE"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/PressVoice.app"   # ditto keeps signatures and metadata intact
ln -s /Applications "$STAGE/Applications"

rm -f "$DMG"
hdiutil create -volname "PressVoice" -srcfolder "$STAGE" -fs HFS+ \
    -format UDZO -imagekey zlib-level=9 -ov "$DMG" >/dev/null
hdiutil verify "$DMG" >/dev/null
echo "Built $DMG"
