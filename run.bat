@echo off
REM ===========================================================================
REM  IPL Analytics - start the dashboard
REM
REM  Double-click this file. It checks the environment, then opens the app in
REM  your browser at http://localhost:8501
REM
REM  If anything is missing it tells you exactly which command to run.
REM ===========================================================================
setlocal EnableDelayedExpansion

REM Work from the folder this script lives in, so it can be run from anywhere
REM (including a desktop shortcut).
cd /d "%~dp0"

echo.
echo  ============================================================
echo    IPL Match Prediction ^& Analytics - starting dashboard
echo  ============================================================
echo.

REM --- 1. Virtual environment ------------------------------------------------
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo  [X] No virtual environment found.
    echo.
    echo      Run setup.bat first - it creates .venv and installs everything.
    echo.
    pause
    exit /b 1
)
echo  [OK] Virtual environment

REM --- 2. Streamlit installed? -----------------------------------------------
"%PY%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo  [X] Streamlit is not installed in .venv
    echo.
    echo      Run setup.bat, or:  .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
    echo.
    pause
    exit /b 1
)
echo  [OK] Dependencies

REM --- 3. Is there any data? --------------------------------------------------
if not exist "data\ipl.db" (
    if not exist "data\ipl_deploy.db" (
        echo  [X] No database found.
        echo.
        echo      Collect the data first:  .venv\Scripts\python.exe scripts\ingest.py
        echo      That takes about 20 minutes the first time.
        echo.
        pause
        exit /b 1
    )
)
echo  [OK] Database

REM --- 4. Are the models trained? ---------------------------------------------
if not exist "models\artifacts\winner_model.joblib" (
    echo  [!] Models are not trained - predictions will be unavailable.
    echo      Train them with:  .venv\Scripts\python.exe scripts\train_models.py
    echo.
) else (
    echo  [OK] Trained models
)

REM --- 5. Port already in use? ------------------------------------------------
REM Streamlit would otherwise silently pick a different port and the URL below
REM would be wrong.
set "PORT=8501"
netstat -ano | findstr /R /C:":8501 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo  [!] Port 8501 is already in use - using 8502 instead.
    set "PORT=8502"
)

echo.
echo  ------------------------------------------------------------
echo    Opening http://localhost:!PORT!
echo    Press Ctrl+C in this window to stop the app.
echo  ------------------------------------------------------------
echo.

REM Give the server a moment to bind before the browser opens, otherwise the
REM first page load can race it and show a connection error.
start "" /b cmd /c "timeout /t 6 /nobreak >nul && start http://localhost:!PORT!"

"%PY%" -m streamlit run streamlit_app.py --server.port !PORT! --server.headless true

echo.
echo  Dashboard stopped.
pause
