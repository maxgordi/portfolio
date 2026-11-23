#!/bin/bash
# Бесплатный перевод на этом Mac, в окне: Whisper или Vosk → Argos → Piper.
# Сначала закройте окно PressVoice Rust — оно слушает ту же клавишу (правый ⌥).
cd "$(dirname "$0")"
exec .venv/bin/python gui.py "$@"
