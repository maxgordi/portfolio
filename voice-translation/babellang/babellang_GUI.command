#!/bin/zsh
cd "$(dirname "$0")"
nohup .venv/bin/babellang --gui > /tmp/babellang_console.log 2>&1 &
disown
sleep 0.3
osascript -e 'tell application "Terminal" to close front window' 2>/dev/null || true
