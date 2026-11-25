@echo off
REM featureclip ko Windows login par chupke (bina window ke) khud shuru karne ke liye. Ek baar chalao.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_autostart.ps1"
if errorlevel 1 (echo Kuch masla aaya. Is file par Right click - Run as administrator try karo.) else (echo Theek hai! Ab laptop on karke login karte hi featureclip khud chalega.)
pause
