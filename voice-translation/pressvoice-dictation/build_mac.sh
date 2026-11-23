#!/bin/bash
# Build PressVoice.app and the installer disk image on this Mac:
#     ./build_mac.sh
# Needs Python 3.11+ (python.org or Homebrew). Output: dist/PressVoice-*.dmg
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if [ ! -d .venv-mac-build ]; then
    "$PY" -m venv .venv-mac-build
fi
source .venv-mac-build/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -r requirements-mac.txt pyinstaller==6.22.3 pyinstaller-hooks-contrib==2026.8

echo "=== PyInstaller ==="
pyinstaller --noconfirm --clean packaging/macos/pressvoice_mac.spec

echo "=== Signing ==="
SIGN_DIR="$HOME/.config/pressvoice-signing"
if [ -f "$SIGN_DIR/pressvoice-signing.p12" ]; then
    PRESSVOICE_SIGN_P12="$SIGN_DIR/pressvoice-signing.p12" \
    PRESSVOICE_SIGN_PASSWORD="$(cat "$SIGN_DIR/password.txt")" \
        packaging/macos/sign_app.sh
else
    packaging/macos/sign_app.sh   # warns and keeps the ad-hoc signature
fi

echo "=== Self-test of the built app ==="
dist/PressVoice.app/Contents/MacOS/PressVoice --selftest "$PWD/selftest-mac.json" || {
    cat selftest-mac.json; exit 1; }

echo "=== Disk image ==="
packaging/macos/build_dmg.sh
