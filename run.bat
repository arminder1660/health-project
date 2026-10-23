@echo off
title CardiaNova Platform
cd /d "%~dp0"

echo ====================================================
echo   CardiaNova - Launching Platform
echo ====================================================
echo.
echo Server starting at http://127.0.0.1:8000 ...
echo API Documentation  http://127.0.0.1:8000/docs
echo.
echo Press Ctrl+C in this window to stop the server.
echo ====================================================
echo.

:: Train the risk model once if it has not been built yet
if not exist "%~dp0models\cardio_model.joblib" (
    echo No trained model found. Building it now ^(one-time step^) ...
    "%~dp0venv\Scripts\python.exe" -m src.train
)

:: Open browser after 2 seconds in the background
start "" cmd /c "ping 127.0.0.1 -n 3 >nul & start http://127.0.0.1:8000"

:: Run FastAPI server using venv Python
"%~dp0venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload

echo.
echo Server stopped.
pause
