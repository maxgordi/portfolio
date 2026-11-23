@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  echo Первый запуск: устанавливаю программу (5-10 минут, нужен интернет)...
  py -3.11 -m venv .venv || py -3 -m venv .venv || (echo Не нашёл Python 3. Установите с python.org/downloads & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip >nul
  .venv\Scripts\pip install -e . || (echo Установка не удалась & pause & exit /b 1)
)
if not exist input mkdir input
dir /b input\*.wav input\*.flac >nul 2>&1 || (echo Положите свои записи WAV или FLAC в папку input и запустите снова. & start "" input & pause & exit /b 0)
.venv\Scripts\voiceset all input -o out
echo.
echo Готово. Файл для Suno: out\suno\voice_set_2min.wav, отчёт: out\report\report.html
start "" out\report\report.html
pause
