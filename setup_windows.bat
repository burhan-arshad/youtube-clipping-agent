@echo off
REM One-click setup for Windows. Double-click this file.
cd /d "%~dp0"
echo === 1/5 Python ===
python --version || (echo Python not found. Install Python 3.11/3.12 from python.org and tick "Add to PATH". & pause & exit /b 1)
echo === 2/5 ffmpeg ===
where ffmpeg >nul 2>nul || (echo Installing ffmpeg ... & winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements & echo.& echo ffmpeg installed. CLOSE this window and double-click setup_windows.bat AGAIN. & pause & exit /b 0)
echo === 3/5 virtual environment + packages (a few minutes) ===
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo pip install failed, send me this screen & pause & exit /b 1)
echo === 4/5 config files ===
python -m featureclip init
echo === 5/5 doctor ===
python -m featureclip doctor
echo.
echo Done. Next: open .env in Notepad and paste GEMINI_API_KEY=... then run: python -m featureclip selftest
pause
