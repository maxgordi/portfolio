@echo off
rem Build PressVoice-Setup-<version>.exe on any Windows 10/11 PC.
rem Needs Python 3.12 (python.org, "Add to PATH") and Inno Setup 6 (jrsoftware.org).
setlocal
cd /d "%~dp0"

if not exist .venv-build ( python -m venv .venv-build || goto :error )
call .venv-build\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip || goto :error
pip install -r requirements-qt.txt -r requirements-dev.txt || goto :error

echo === Tests ===
python -m pytest -q || goto :error

echo === PyInstaller ===
pyinstaller --noconfirm --clean packaging\pressvoice.spec || goto :error

echo === Self-test of the built app ===
start "" /wait dist\PressVoice\PressVoice.exe --selftest "%CD%\selftest.json"
if errorlevel 1 ( type selftest.json & goto :error )

echo === Installer ===
python tools\gen_installer_i18n.py || goto :error
for /f %%v in ('python -c "from version import __version__; print(__version__)"') do set VERSION=%%v
set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
%ISCC% /DMyAppVersion=%VERSION% packaging\installer.iss || goto :error

echo.
echo Done: dist\PressVoice-Setup-%VERSION%.exe
exit /b 0

:error
echo Build failed.
exit /b 1
