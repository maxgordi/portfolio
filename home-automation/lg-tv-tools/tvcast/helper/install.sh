#!/bin/bash
# Установка помощника «На телевизор» для Chrome (Native Messaging) на macOS.
# Пароль не нужен: всё ставится в папку пользователя.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
HOST_NAME="com.tvcast.helper"
EXT_ID="ohlnmnflkccnigplbpgidflfdmfhbeah"   # постоянный: задан ключом "key" в chrome-extension/manifest.json
HOST="$DIR/tvcast_host.py"

chmod +x "$HOST"
if ! /usr/bin/python3 -c "import sys; assert sys.version_info >= (3, 8)" 2>/dev/null; then
  echo "✖ Нет /usr/bin/python3. Установи Command Line Tools: xcode-select --install"
  exit 1
fi

MANIFEST=$(cat <<JSON
{
  "name": "$HOST_NAME",
  "description": "На телевизор — помощник для телевизора LG",
  "path": "$HOST",
  "type": "stdio",
  "allowed_origins": ["chrome-extension://$EXT_ID/"]
}
JSON
)

installed=0
for BROWSER in "Google/Chrome" "Google/Chrome Beta" "Google/Chrome Canary" "Chromium"; do
  BASE="$HOME/Library/Application Support/$BROWSER"
  if [ "$BROWSER" = "Google/Chrome" ] || [ -d "$BASE" ]; then
    mkdir -p "$BASE/NativeMessagingHosts"
    echo "$MANIFEST" > "$BASE/NativeMessagingHosts/$HOST_NAME.json"
    echo "✓ $BROWSER: $BASE/NativeMessagingHosts/$HOST_NAME.json"
    installed=1
  fi
done

echo
echo "Проверяю связь с телевизором…"
/usr/bin/python3 - "$HOST" <<'PY'
import json, struct, subprocess, sys
msg = json.dumps({"cmd": "status"}).encode()
p = subprocess.run([sys.argv[1], "chrome-extension://test/"], input=struct.pack("=I", len(msg)) + msg,
                   capture_output=True, timeout=30)
out = p.stdout
res = {}
while len(out) >= 4:
    n = struct.unpack("=I", out[:4])[0]
    res = json.loads(out[4:4 + n]); out = out[4 + n:]
if res.get("power") == "on":
    print("✓ Телевизор в сети. Сейчас на экране:", res.get("appName") or "—")
elif res.get("power") == "off":
    print("• Телевизор сейчас выключен — это нормально, помощник установлен.")
else:
    print("⚠", res.get("error") or res)
PY
echo
echo "Готово. Расширение: chrome://extensions → «Режим разработчика» → «Загрузить распакованное» →"
echo "  $(dirname "$DIR")/chrome-extension"
echo "ID расширения должен быть: $EXT_ID"
