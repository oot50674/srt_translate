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


rem Wait for the server to be ready by checking if it responds
echo Waiting for Flask server to start...
powershell -Command "& { $url = 'http://127.0.0.1:5000'; $timeout = 30; $startTime = Get-Date; while (((Get-Date) - $startTime).TotalSeconds -lt $timeout) { try { $response = Invoke-WebRequest -Uri $url -Method GET -TimeoutSec 1 -ErrorAction Stop; if ($response.StatusCode -eq 200) { Write-Host 'Server is ready!'; break; } } catch { Start-Sleep -Seconds 1; } } }" >nul 2>&1

rem Open the browser
start "" "http://127.0.0.1:5000"


echo Flask server started.

:END
popd >nul
endlocal
