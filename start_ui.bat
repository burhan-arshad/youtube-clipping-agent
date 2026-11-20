@echo off
cd /d "%~dp0"
call D:\venv\featureclip\Scripts\activate.bat
echo featureclip UI shuru ho raha hai... browser khud khulega.
echo Band karne ke liye is window me Ctrl+C dabao.
python -m featureclip ui
pause
