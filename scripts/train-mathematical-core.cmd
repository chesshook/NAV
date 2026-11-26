@echo off
setlocal
cd /d "%~dp0\.."
py -3 tools\train_mathematical_core.py --download --epochs 5
if errorlevel 1 (
  echo MCR-L mathematical syntax training failed.
  exit /b 1
)
echo.
echo Mathematical syntax model trained successfully.
