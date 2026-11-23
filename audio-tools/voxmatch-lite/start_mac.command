#!/bin/bash
# VoxMatch Простой для macOS. Первый запуск доустанавливает движок Rubber Band (нужен интернет).
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$PATH"
LOG="install.log"
SHARED="../VoxMatch/.venv"
if [ -x "$SHARED/bin/python" ] && [ -f "$SHARED/installed.ok" ]; then
  VENV="$SHARED"                     # библиотеки уже стоят для большого VoxMatch — берём их
else
  VENV=".venv"
  if [ ! -f "$VENV/installed.ok" ]; then
    exec > >(tee -a "$LOG") 2>&1
    echo "VoxMatch Простой: первый запуск, ставлю библиотеки (3-10 минут)…"
    PY=""
    for v in python3.12 python3.11 python3.13 python3.10 python3; do
      if command -v "$v" >/dev/null 2>&1 && "$v" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
        PY="$(command -v $v)"; break
      fi
    done
    if [ -z "$PY" ]; then
      echo "Не найден Python 3.10-3.13. Установите Python 3.12 с https://www.python.org/downloads/"
      read -n 1 -s -r -p "Нажмите любую клавишу…"; exit 1
    fi
    if ! "$PY" -m venv "$VENV" || ! "$VENV/bin/python" -m pip install --upgrade pip \
       || ! "$VENV/bin/python" -m pip install -r requirements.txt; then
      echo "Ошибка установки. Подробности в install.log"
      read -n 1 -s -r -p "Нажмите любую клавишу…"; exit 1
    fi
    touch "$VENV/installed.ok"
  fi
fi
if ! "$VENV/bin/python" -c "import pylibrb" >/dev/null 2>&1; then
  echo "Ставлю движок Rubber Band (pylibrb)…"
  if ! "$VENV/bin/python" -m pip install "pylibrb>=0.1.2" 2>&1 | tee -a "$LOG"; then
    echo "Не удалось установить pylibrb. Подробности в install.log"
    read -n 1 -s -r -p "Нажмите любую клавишу…"; exit 1
  fi
fi
echo "Запускаю VoxMatch Простой…"
"$VENV/bin/python" vms_app.py
