#!/bin/bash
# Удаляет «PressVoice Microphone» из системы и перезапускает звуковую службу.
set -euo pipefail

osascript -e "do shell script \"rm -rf /Library/Audio/Plug-Ins/HAL/PressVoice.driver && killall -9 coreaudiod\" with prompt \"PressVoice удаляет виртуальный микрофон «PressVoice Microphone».\" with administrator privileges"
echo "Удалено."
