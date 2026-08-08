@echo off
REM ===========================================================================
REM  IPL Analytics - start the REST API
REM
REM  Double-click this to serve the prediction API and open its interactive
REM  documentation at http://localhost:8000/docs
REM
REM  Run this alongside run.bat to demonstrate the dashboard and the API
REM  together - they are separate windows and do not conflict.
REM ===========================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo.
echo  ============================================================
echo    IPL Match Prediction ^& Analytics - starting REST API
echo  ============================================================
echo.

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo  [X] No virtual environment found. Run setup.bat first.
    echo.
    pause
    exit /b 1
)
echo  [OK] Virtual environment

"%PY%" -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo  [X] FastAPI is not installed. Run setup.bat.
    echo.
    pause
    exit /b 1
)
echo  [OK] Dependencies

if not exist "models\artifacts\chase_model.joblib" (
    echo  [!] Models are not trained - prediction endpoints will return 503.
    echo      Train them with:  .venv\Scripts\python.exe scripts\train_models.py
    echo.
)

set "PORT=8000"
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo  [!] Port 8000 is already in use - using 8010 instead.
    set "PORT=8010"
)

echo.
echo  ------------------------------------------------------------
echo    API docs:  http://localhost:!PORT!/docs
echo    Press Ctrl+C in this window to stop the API.
echo  ------------------------------------------------------------
echo.

start "" /b cmd /c "timeout /t 5 /nobreak >nul && start http://localhost:!PORT!/docs"

"%PY%" -m uvicorn --app-dir src ipl.api.main:app --host 127.0.0.1 --port !PORT!

echo.
echo  API stopped.
pause
