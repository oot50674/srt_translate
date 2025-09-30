@echo off
chcp 65001 >nul
setlocal

rem Determine the directory where this script resides.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

rem Configuration
set "VENV_DIR=.venv"


rem Check if virtual environment exists.
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment not found. Please run install_app.bat first.
    pause
    goto :END
)


rem Launch the Flask application in a new window.
echo Starting Flask server...
start "Flask Server" "%VENV_DIR%\Scripts\python.exe" app.py


rem Give the server a moment to start then open the browser.
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5000"


echo Flask server started.

:END
popd >nul
endlocal
