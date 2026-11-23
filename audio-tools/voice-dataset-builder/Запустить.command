#!/bin/bash
# macOS: двойной клик. Кладёшь WAV/FLAC в папку input/ и запускаешь этот файл.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Первый запуск: устанавливаю программу (5–10 минут, нужен интернет)…"
  PY=$(command -v python3.11 || command -v python3.12 || command -v python3.10 || command -v python3)
  "$PY" -m venv .venv || { echo "Не нашёл Python 3. Установите с python.org/downloads"; read -n1; exit 1; }
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -e . || { echo "Установка не удалась"; read -n1; exit 1; }
fi
mkdir -p input
if ! ls input/*.wav input/*.flac >/dev/null 2>&1; then
  echo "Положите свои записи (WAV или FLAC) в папку input/ и запустите снова."
  open input; read -n1; exit 0
fi
.venv/bin/voiceset all input -o out
echo
echo "Готово. Файл для Suno: out/suno/voice_set_2min.wav, отчёт: out/report/report.html"
open out/report/report.html 2>/dev/null; open out/suno 2>/dev/null
read -n1 -p "Нажмите любую клавишу…"
