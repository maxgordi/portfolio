#!/bin/bash
# VocalSync для macOS. Первый запуск ставит библиотеки (нужен интернет, 3–10 минут).
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$PATH"
LOG="install.log"
VENV=".venv"
if [ ! -f "$VENV/installed.ok" ]; then
  exec > >(tee -a "$LOG") 2>&1
  echo "VocalSync: первый запуск, ставлю библиотеки…"
  PY=""
  for v in python3.11 python3.10 python3.12 python3; do
    if command -v "$v" >/dev/null 2>&1 && "$v" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then
      PY="$(command -v $v)"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Не найден Python 3.10–3.12. Установите: brew install python@3.11"
    read -n 1 -s -r -p "Нажмите любую клавишу…"; exit 1
  fi
  if ! "$PY" -m venv "$VENV" || ! "$VENV/bin/python" -m pip install --upgrade pip \
     || ! "$VENV/bin/python" -m pip install -r requirements.txt; then
    echo "Ошибка установки. Подробности в install.log"
    read -n 1 -s -r -p "Нажмите любую клавишу…"; exit 1
  fi
  touch "$VENV/installed.ok"
fi
if ! command -v rubberband >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Ставлю Rubber Band CLI (лучшее качество растяжения): brew install rubberband"
    brew install rubberband || echo "Не вышло — будет использован pylibrb (тоже Rubber Band R3)."
  fi
fi
echo "Запускаю VocalSync…"
"$VENV/bin/python" app.py
