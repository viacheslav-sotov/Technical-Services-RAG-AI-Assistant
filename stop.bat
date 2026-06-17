@echo off
REM Stop the Technical Services RAG Assistant servers (frees ports 8000 and 8501).
echo Stopping backend (port 8000) and frontend (port 8501) ...
for %%P in (8000 8501) do (
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%P " ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
  )
)
echo Done.
