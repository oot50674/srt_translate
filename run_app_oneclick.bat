
@echo off
chcp 65001 >nul
setlocal

rem Determine the directory where this script resides.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

rem Configuration
set "VENV_DIR=.venv"
set "REQUIREMENTS_FILE=requirements.txt"
set "REQUIREMENTS_URL=https://raw.githubusercontent.com/oot50674/srt_translate/main/requirements.txt"


rem Ensure Python is available.
where python >nul 2>nul
if errorlevel 1 (
    echo Python interpreter를 찾을 수 없습니다. PATH 설정을 확인하세요.
    goto :END
)


rem Create the virtual environment if it does not exist.
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo 가상환경을 생성합니다...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo 가상환경 생성에 실패했습니다.
        goto :END
    )
)

rem Upgrade pip in the virtual environment.
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip >nul


rem Download requirements.txt when it is missing.
if not exist "%REQUIREMENTS_FILE%" (
    echo requirements.txt를 다운로드합니다...
    powershell -NoLogo -NoProfile -Command "try { Invoke-WebRequest -Uri '%REQUIREMENTS_URL%' -OutFile '%REQUIREMENTS_FILE%' -UseBasicParsing } catch { Write-Error $_; exit 1 }"
    if errorlevel 1 (
        echo requirements.txt 다운로드에 실패했습니다.
        goto :END
    )
) else (
    echo 로컬 requirements.txt 파일을 사용합니다.
)


rem Install dependencies inside the virtual environment.
echo 패키지 설치를 진행합니다...
"%VENV_DIR%\Scripts\python.exe" -m pip install --no-warn-script-location -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo 패키지 설치에 실패했습니다.
    goto :END
)


rem Launch the Flask application in a new window.
echo Flask 서버를 시작합니다...
start "Flask Server" "%VENV_DIR%\Scripts\python.exe" app.py


rem Give the server a moment to start then open the browser.
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5000"


echo 모든 준비가 완료되었습니다.

:END
popd >nul
endlocal
