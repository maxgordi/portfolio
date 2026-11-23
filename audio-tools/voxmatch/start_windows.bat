@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "VENV=%~dp0.venv"
if exist "%VENV%\installed.ok" goto run

echo ==================================================
echo  VoxMatch: первый запуск, устанавливаю библиотеки.
echo  Это займёт 3-10 минут, нужен интернет.
echo ==================================================
set "PY="
for %%V in (3.12 3.13 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo Не найден подходящий Python ^(нужна версия 3.10 - 3.13^).
  echo Установите Python 3.12 с https://www.python.org/downloads/
  echo и при установке отметьте "Add python.exe to PATH".
  pause
  exit /b 1
)
echo Использую %PY%
%PY% -m venv "%VENV%" || goto fail
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip || goto fail
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
echo ok> "%VENV%\installed.ok"
echo Установка завершена.

:run
start "" "%VENV%\Scripts\pythonw.exe" "%~dp0voxmatch.py"
exit /b 0

:fail
echo.
echo Ошибка установки. Проверьте интернет и запустите этот файл ещё раз.
pause
exit /b 1
