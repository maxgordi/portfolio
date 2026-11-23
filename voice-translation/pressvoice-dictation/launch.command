#!/bin/bash
cd "$(dirname "$0")"
pkill -f "supervise.sh" 2>/dev/null
pkill -f "main.py" 2>/dev/null
sleep 0.5
nohup /bin/bash ./supervise.sh &>/dev/null &
disown
sleep 0.3
osascript -e 'tell application "Terminal" to close front window' 2>/dev/null || true
