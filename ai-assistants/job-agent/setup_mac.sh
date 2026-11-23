#!/bin/bash
# Одноразове встановлення на Mac. Запуск: bash setup_mac.sh
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3)
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Потрібен Python 3.10+. Встанови: brew install python@3.12  — і запусти скрипт ще раз."; exit 1
fi
echo "Python: $($PY --version)"
"$PY" -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m playwright install chromium
[ -f .env ] || cp .env.example .env
echo
echo "✅ Встановлено. Далі:"
echo "  1) open -e .env                         # впиши токен бота і LLM-ключ"
echo "  2) .venv/bin/python tools/login.py djinni   # і dou, robota, workua"
echo "  3) .venv/bin/python -m app.main          # запуск (Ctrl+C — зупинити)"
echo "  4) bash autostart_mac.sh                 # щоб стартував сам після входу в систему"
