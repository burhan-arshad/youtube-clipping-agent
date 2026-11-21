@echo off
REM Login par khud chalne wala version: browser nahi kholta, crash ya hang par watchdog restart karta hai.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m featureclip.supervisor --root .
