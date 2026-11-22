@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -c "import yaml, fastapi" >nul 2>nul || (echo Packages install ho rahe hain, thora intezar karo... & python -m pip install -r requirements.txt)
if "%~2"=="" (python -m featureclip watch %1) else (python -m featureclip watch %1 --url %2)
pause
