@echo off
setlocal
title Healthcare Assistant

cd /d "%~dp0"

echo ==========================================
echo        HEALTHCARE ASSISTANT
echo ==========================================
echo.

REM Create virtual environment if it does not exist
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Python was not found or the virtual environment could not be created.
        echo Please install Python 3.10+ and try again.
        pause
        exit /b 1
    )
)

REM Check whether Streamlit is actually installed in this venv
".venv\Scripts\python.exe" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Streamlit is not installed in this virtual environment.
    echo Installing required packages...
    echo This may take several minutes on the first setup.
    echo.

    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 (
        echo.
        echo ERROR: Could not upgrade pip.
        pause
        exit /b 1
    )

    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Package installation failed.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment and Streamlit found.
    echo Skipping package installation.
)

echo.
echo Starting Healthcare Assistant...
echo.

".venv\Scripts\python.exe" -m streamlit run app.py

echo.
echo Healthcare Assistant has stopped.
pause
endlocal
