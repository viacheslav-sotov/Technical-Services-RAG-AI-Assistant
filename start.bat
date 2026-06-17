@echo off
REM ===========================================================================
REM  Technical Services RAG Assistant - one-click launcher (Windows)
REM  Double-click this file, or run it from a terminal.
REM  It opens the backend API and the Streamlit frontend in two windows.
REM ===========================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found at .venv
  echo Run setup first:  py -3.12 -m venv .venv  ^&^&  .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)

echo Starting backend API on http://localhost:8000 ...
start "TSRAG API" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

REM Give the API a moment to boot before the UI starts polling it.
timeout /t 4 /nobreak >nul

echo Starting frontend on http://localhost:8501 ...
start "TSRAG Frontend" cmd /k ".venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501"

echo.
echo Both services are starting in their own windows.
echo   Frontend (open this): http://localhost:8501
echo   API docs:             http://localhost:8000/docs
echo Close those two windows to stop the app.
