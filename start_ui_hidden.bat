@echo off
REM Login par khud chalne wala version: browser nahi kholta, crash par 10 second baad dobara chalata hai.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
:loop
python -m featureclip ui --no-browser
timeout /t 10 /nobreak >nul
goto loop
