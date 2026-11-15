@echo off
setlocal EnableExtensions
rem ===========================================================================
rem  One-click launcher: Weather-Station Behaviour Grouping dashboard
rem  Double-click this file. It sets everything up and opens the dashboard.
rem ===========================================================================
title Weather Station Dashboard
set "PORT=8501"

rem 1. Go to the project folder (the folder this file is in)
cd /d "%~dp0"
echo Project folder: %CD%
echo.

rem 2. Find Python (only needed to create the virtual environment)
set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)

rem 3. Create the virtual environment if it does not exist
if not exist ".venv\Scripts\python.exe" (
    if not defined PY (
        echo [ERROR] Python 3 was not found. Install it from https://www.python.org/downloads/
        echo         and tick "Add python.exe to PATH" during installation. Then run this file again.
        goto :fail
    )
    echo Creating virtual environment in .venv ... this happens only once.
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        goto :fail
    )
)

rem 4. Activate it
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Could not activate .venv. Delete the .venv folder and run this file again.
    goto :fail
)

rem 5. Install missing dependencies (only if an import fails)
python -c "import streamlit, plotly, pandas, numpy, sklearn, scipy, matplotlib, seaborn, requests" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages ... this can take a few minutes the first time.
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Package installation failed. Check your internet connection and run this file again.
        goto :fail
    )
)

rem 6. Open the browser when the server is ready, then start the dashboard
echo.
echo Starting the dashboard at http://localhost:%PORT%
echo Keep this window open while you use the dashboard. Close it (or press Ctrl+C) to stop.
echo.
start "" /b python dashboard\open_browser.py %PORT%
python -m streamlit run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false
if errorlevel 1 (
    echo.
    echo [ERROR] The dashboard stopped with an error. If the port is busy, close other dashboards and retry.
    goto :fail
)
goto :end

:fail
echo.
echo Setup or launch failed - read the message above.
pause
exit /b 1

:end
endlocal
