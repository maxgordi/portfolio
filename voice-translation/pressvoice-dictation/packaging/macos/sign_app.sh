#!/bin/bash
# Re-sign dist/PressVoice.app with PressVoice's own certificate instead of
# PyInstaller's ad-hoc signature.
#
# Why: macOS ties the Accessibility permission (needed to paste) to the code
# signature. An ad-hoc signature changes with every build, so each update
# would silently lose the permission while System Settings still shows the
# switch on. A fixed certificate keeps the permission across updates.
#
#   PRESSVOICE_SIGN_P12       path to the .p12 (certificate + private key)
#   PRESSVOICE_SIGN_PASSWORD  its password
# Without them the app stays ad-hoc signed (a warning is printed).
set -euo pipefail

APP=${1:-dist/PressVoice.app}
IDENTITY="PressVoice Code Signing"

if [ -z "${PRESSVOICE_SIGN_P12:-}" ] || [ ! -f "${PRESSVOICE_SIGN_P12:-}" ]; then
    echo "WARNING: no signing certificate — $APP stays ad-hoc signed;" \
         "Accessibility will have to be re-granted after every update." >&2
    exit 0
fi

KC_DIR=$(mktemp -d)
KC="$KC_DIR/pressvoice-signing.keychain-db"
KC_PASS=$(uuidgen)

# codesign only finds identities in keychains on the search list, so the
# temporary keychain joins it for the duration and the list is restored.
ORIG_KEYCHAINS=()
while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"; line="${line#\"}"; line="${line%\"}"
    [ -n "$line" ] && ORIG_KEYCHAINS+=("$line")
done < <(security list-keychains -d user)

cleanup() {
    security list-keychains -d user -s "${ORIG_KEYCHAINS[@]}" 2>/dev/null || true
    security delete-keychain "$KC" 2>/dev/null || true
    rm -rf "$KC_DIR"
}
trap cleanup EXIT

security create-keychain -p "$KC_PASS" "$KC"
security set-keychain-settings -lut 3600 "$KC"
security unlock-keychain -p "$KC_PASS" "$KC"
security import "$PRESSVOICE_SIGN_P12" -k "$KC" -P "$PRESSVOICE_SIGN_PASSWORD" \
    -T /usr/bin/codesign >/dev/null
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$KC_PASS" "$KC" >/dev/null
security list-keychains -d user -s "$KC" "${ORIG_KEYCHAINS[@]}"

# Inside-out: nested libraries and frameworks first, then the bundle.
codesign --force --deep --keychain "$KC" --sign "$IDENTITY" --timestamp=none "$APP"
codesign --verify --deep --strict "$APP"
codesign -dr - "$APP" 2>&1 | grep designated
