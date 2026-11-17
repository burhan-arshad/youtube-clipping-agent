@echo off
REM featureclip ko Windows login par khud shuru karne ke liye (ek baar chalao).
cd /d "%~dp0"
schtasks /Create /TN "featureclip" /TR "\"%~dp0start_ui_hidden.bat\"" /SC ONLOGON /DELAY 0000:30 /F
if errorlevel 1 (echo Kuch masla aaya. Is file par Right click - Run as administrator try karo.) else (echo Theek hai! Ab PC restart ke baad featureclip khud chalega aur Assistant wapis shuru ho jayega.)
pause
