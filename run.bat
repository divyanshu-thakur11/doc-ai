@echo off
title DocMind AI

echo.
echo  ============================================
echo   DocMind AI - RAG PDF Q and A System
echo   Powered by Google Gemini (FREE)
echo  ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found.
    echo  Download from https://python.org and tick "Add to PATH"
    pause
    exit /b 1
)
echo  [1/5] Python found.

if "%GEMINI_API_KEY%"=="" (
    echo.
    echo  Get your FREE Gemini API key - no credit card needed:
    echo  1. Go to: https://aistudio.google.com/app/apikey
    echo  2. Sign in with your Google account
    echo  3. Click "Create API Key" and copy it
    echo.
    set /p GEMINI_API_KEY="  Paste your key here and press Enter: "
)
echo  [2/5] API key set.

if not exist "venv\Scripts\python.exe" (
    echo  [3/5] Creating virtual environment...
    python -m venv venv
) else (
    echo  [3/5] Virtual environment ready.
)

echo  [4/5] Installing dependencies (first run takes 1-2 min)...
venv\Scripts\pip.exe install -r requirements.txt --quiet
if errorlevel 1 (
    echo  ERROR: pip install failed. Check internet connection.
    pause
    exit /b 1
)
echo  [4/5] Done.

echo  [5/5] Starting server...
echo.
echo  ============================================
echo   App:      http://localhost:8000
echo   API docs: http://localhost:8000/docs
echo   Press Ctrl+C to stop
echo  ============================================
echo.

start "" "http://localhost:8000"
venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

pause
