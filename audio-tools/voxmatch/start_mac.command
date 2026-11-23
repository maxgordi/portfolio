#!/bin/bash
# VoxMatch для macOS: двойной клик запускает программу (первый раз — с установкой библиотек).
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$PATH"
VENV=".venv"
LOG="install.log"
if [ ! -f "$VENV/installed.ok" ]; then
  exec > >(tee -a "$LOG") 2>&1
  echo "=================================================="
  echo " VoxMatch: первый запуск, устанавливаю библиотеки."
  echo " Это займёт 3-10 минут, нужен интернет."
  echo "=================================================="
  PY=""
  for v in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$v" >/dev/null 2>&1 && "$v" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
      PY="$(command -v $v)"; break
    fi
  done
  echo "Python: ${PY:-не найден}"
  if [ -z "$PY" ]; then
    echo "STATUS: NO_PYTHON"
    echo "Не найден Python 3.10-3.13. Установите Python 3.12 с https://www.python.org/downloads/"
    read -n 1 -s -r -p "Нажмите любую клавишу…"; exit 1
  fi
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "STATUS: NO_XCODE_TOOLS"
    echo "Нужны инструменты разработчика Apple. Сейчас откроется окно установки — нажмите «Установить»,"
    echo "дождитесь окончания и запустите этот файл ещё раз."
    xcode-select --install
    read -n 1 -s -r -p "Нажмите любую клавишу…"; exit 1
  fi
  if ! "$PY" -m venv "$VENV" || ! "$VENV/bin/python" -m pip install --upgrade pip \
     || ! "$VENV/bin/python" -m pip install -r requirements.txt; then
    echo "STATUS: PIP_FAILED"
    echo "Ошибка установки. Подробности в install.log"
    read -n 1 -s -r -p "Нажмите любую клавишу…"; exit 1
  fi
  touch "$VENV/installed.ok"
  echo "STATUS: INSTALLED"
fi
echo "Запускаю VoxMatch…"
"$VENV/bin/python" voxmatch.py
