#!/bin/bash
# Букви для macOS: двойной клик открывает окно ручной правки букв (первый раз — с установкой библиотек).
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [ ! -x ".venv/bin/python" ]; then
  echo "Первый запуск: устанавливаю библиотеки (5-10 минут, нужен интернет)…"
  PY="$(command -v python3.12 || command -v python3.11 || command -v python3)"
  "$PY" -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt \
    || { echo "Ошибка установки"; read -n 1 -s -r -p "Нажмите любую клавишу…"; exit 1; }
fi
.venv/bin/python bukvi_app.py "$@"
