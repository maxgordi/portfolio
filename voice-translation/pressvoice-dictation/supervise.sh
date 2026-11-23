#!/bin/bash
# Keeps PressVoice running: if the process dies unexpectedly (native crash,
# killed audio thread), it is restarted. A clean quit from the menu exits.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PRESSVOICE_PYTHON:-/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3}"
LOG="$SCRIPT_DIR/pressvoice.log"

cd "$SCRIPT_DIR" || exit 1

fails=0
window_start=$(date +%s)

while true; do
    "$PYTHON" main.py >> "$LOG" 2>&1
    code=$?

    if [ $code -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] supervisor: clean exit, stopping" >> "$LOG"
        break
    fi

    now=$(date +%s)
    # Crashes more than 5 minutes apart don't count toward the give-up limit.
    if [ $((now - window_start)) -gt 300 ]; then
        fails=0
        window_start=$now
    fi
    fails=$((fails + 1))

    if [ $fails -ge 5 ]; then
        echo "[$(date +%H:%M:%S)] supervisor: exit $code, 5 crashes in a row — giving up" >> "$LOG"
        osascript -e 'display notification "PressVoice keeps crashing — check pressvoice.log" with title "PressVoice"' 2>/dev/null
        break
    fi

    echo "[$(date +%H:%M:%S)] supervisor: exit $code, restart #$fails in 2s" >> "$LOG"
    sleep 2
done
