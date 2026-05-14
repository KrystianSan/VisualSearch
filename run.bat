@echo off
title VisualSearch

REM Change to the directory where this .bat file lives
cd /d "%~dp0"

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM Launch the app (pythonw suppresses the console window)
pythonw main.py

REM If pythonw fails fall back to python (shows console — useful for debugging)
if errorlevel 1 (
    python main.py
)
