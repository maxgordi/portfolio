#!/bin/bash
# Автозапуск агента при вході в macOS + авто-перезапуск, якщо впаде.
#   bash autostart_mac.sh          — увімкнути
#   bash autostart_mac.sh off      — вимкнути
set -e
cd "$(dirname "$0")"
DIR="$(pwd)"
LABEL="com.example.jobagent"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "$1" = "off" ]; then
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Автозапуск вимкнено."; exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$DIR/.venv/bin/python</string><string>-m</string><string>app.main</string>
  </array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>StandardOutPath</key><string>$DIR/data/agent.log</string>
  <key>StandardErrorPath</key><string>$DIR/data/agent.log</string>
</dict></plist>
PL
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "✅ Агент працює у фоні і стартуватиме сам після входу в систему."
echo "   Лог: tail -f \"$DIR/data/agent.log\"     Вимкнути: bash autostart_mac.sh off"
