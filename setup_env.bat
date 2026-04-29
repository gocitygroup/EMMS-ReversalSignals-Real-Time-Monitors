@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo   EMMS + Reversal Signals — Environment Setup
echo ============================================================
echo.

:: — check python —
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found on PATH.  Install Python 3.10+ first.
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do echo [OK] Found Python %%v

:: — create venv —
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [INFO] Creating virtual environment .venv ...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] venv creation failed.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment .venv already exists.
)

:: — activate & upgrade pip —
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

:: — install core dependencies —
echo.
echo [INFO] Installing core dependencies ...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARN] Some core packages may not have installed.
    echo        MetaTrader5 requires Windows and a running MT5 terminal.
)

:: — offer socketio —
echo.
set /p INSTALL_SIO="Install optional Socket.IO packages? (y/n): "
if /i "!INSTALL_SIO!"=="y" (
    echo [INFO] Installing Socket.IO packages ...
    pip install -r requirements-socketio.txt
)

echo.
echo ============================================================
echo   Setup complete.
echo.
echo   Activate the environment before running scripts:
echo     .venv\Scripts\activate
echo.
echo   Run the monitors:
echo     python EstimatedManipulationMovementSignal.py --realtime
echo     python ReversalSignals.py --realtime
echo ============================================================
pause
