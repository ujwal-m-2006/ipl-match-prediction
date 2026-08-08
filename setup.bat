@echo off
REM ===========================================================================
REM  IPL Analytics - first-time setup
REM
REM  Double-click this once on a new machine. It creates the virtual
REM  environment, installs everything, and tells you what to do next.
REM
REM  Safe to re-run: it reuses an existing .venv rather than rebuilding it.
REM ===========================================================================
setlocal

cd /d "%~dp0"

echo.
echo  ============================================================
echo    IPL Match Prediction ^& Analytics - setup
echo  ============================================================
echo.

REM --- 1. Find a usable Python ------------------------------------------------
REM Prefer 3.12: it has the widest wheel coverage for scikit-learn, XGBoost,
REM LightGBM and CatBoost. Fall back to whatever "py" or "python" resolves to.
set "BASEPY="
py -3.12 --version >nul 2>&1 && set "BASEPY=py -3.12"
if not defined BASEPY (py -3.13 --version >nul 2>&1 && set "BASEPY=py -3.13")
if not defined BASEPY (py -3.11 --version >nul 2>&1 && set "BASEPY=py -3.11")
if not defined BASEPY (python --version >nul 2>&1 && set "BASEPY=python")

if not defined BASEPY (
    echo  [X] Python was not found on this machine.
    echo.
    echo      Install Python 3.12 from https://www.python.org/downloads/
    echo      Tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('%BASEPY% --version 2^>^&1') do set "PYVER=%%v"
echo  [OK] Found %PYVER%

REM --- 2. Virtual environment -------------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo  [OK] Virtual environment already exists - reusing it
) else (
    echo  [..] Creating virtual environment...
    %BASEPY% -m venv .venv
    if errorlevel 1 (
        echo  [X] Could not create the virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created
)

set "PY=.venv\Scripts\python.exe"

REM --- 3. Dependencies --------------------------------------------------------
echo  [..] Installing dependencies - this takes a few minutes...
echo.
"%PY%" -m pip install --upgrade pip --quiet
"%PY%" -m pip install -r requirements-dev.txt
if errorlevel 1 (
    echo.
    echo  [X] Some packages failed to install.
    echo      Scroll up to see which one, then try again.
    echo.
    pause
    exit /b 1
)
echo.
echo  [OK] Dependencies installed

REM --- 4. Configuration -------------------------------------------------------
if not exist ".env" (
    copy /y ".env.example" ".env" >nul
    echo  [OK] Created .env from the template
) else (
    echo  [OK] .env already exists
)

REM --- 5. Database schema -----------------------------------------------------
"%PY%" scripts\init_db.py
echo  [OK] Database schema ready

echo.
echo  ============================================================
echo    Setup complete.
echo  ============================================================
echo.

if exist "data\ipl.db" (
    echo    You already have data. Just run:   run.bat
) else (
    echo    Next, collect the data ^(about 20 minutes^):
    echo.
    echo        .venv\Scripts\python.exe scripts\ingest.py
    echo.
    echo    Then train the models ^(about 3 minutes^):
    echo.
    echo        .venv\Scripts\python.exe scripts\train_models.py
    echo.
    echo    Then start the app:   run.bat
)
echo.
pause
