@echo off
REM Watchdog: UI + Assistant chalata hai aur crash ya hang (atakne) par khud restart karta hai.
cd /d "%~dp0"
call D:\venv\featureclip\Scripts\activate.bat
echo Watchdog shuru. Band karne ke liye Ctrl+C dabao.
python -m featureclip.supervisor --root .
pause
