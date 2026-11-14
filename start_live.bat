@echo off
REM Usage: start_live.bat SOURCE_ID [LIVE_URL]     (example: start_live.bat ducky https://www.youtube.com/watch?v=XXXX)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if "%~2"=="" (python -m featureclip watch %1) else (python -m featureclip watch %1 --url %2)
pause
