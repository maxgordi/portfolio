#!/bin/bash
# Устанавливает собранный «PressVoice Microphone» в систему (нужен пароль администратора:
# macOS покажет обычное окно) и перезапускает звуковую службу — звук пропадёт на секунду.
set -euo pipefail
cd "$(dirname "$0")"

DRIVER="$PWD/.build/PressVoice.driver"
[ -d "$DRIVER" ] || ./build.sh

osascript -e "do shell script \"mkdir -p /Library/Audio/Plug-Ins/HAL && rm -rf /Library/Audio/Plug-Ins/HAL/PressVoice.driver && cp -R '$DRIVER' /Library/Audio/Plug-Ins/HAL/ && chown -R root:wheel /Library/Audio/Plug-Ins/HAL/PressVoice.driver && killall -9 coreaudiod\" with prompt \"PressVoice устанавливает виртуальный микрофон «PressVoice Microphone».\" with administrator privileges"
echo "Установлено: /Library/Audio/Plug-Ins/HAL/PressVoice.driver"
