@echo off
setlocal enabledelayedexpansion
title VisualSearch — Setup

echo ============================================================
echo  VisualSearch Setup
echo ============================================================
echo.

REM Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo.
    echo Please install Python 3.10 or newer from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found.
echo.

REM Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found. Please reinstall Python and ensure pip is included.
    pause
    exit /b 1
)

REM Upgrade pip silently
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install PyTorch CPU-only first (smaller download, sufficient for most users)
echo.
echo Installing PyTorch (CPU-only build — smaller download)...
echo This may take several minutes depending on your connection.
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

if errorlevel 1 (
    echo.
    echo [WARNING] PyTorch install failed. Trying default PyTorch...
    pip install torch torchvision
)

REM Install remaining dependencies
echo.
echo Installing remaining dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Some dependencies failed to install.
    echo Check the output above for details.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!
echo  Run VisualSearch by double-clicking run.bat
echo ============================================================
echo.
pause
