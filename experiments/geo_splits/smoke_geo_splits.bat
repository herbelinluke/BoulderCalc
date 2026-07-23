@echo off
REM Smoke-test all geo-split setups (RGB+DSM, offline+jitter, --no-rich-aug).
REM Run from B:\ (or your short project root) after activating the conda/venv env.
REM See README.md in this folder.

setlocal
cd /d "%~dp0..\..\.."
if not exist "BoulderCalculator\experiments\geo_splits\smoke_geo_splits.py" (
  echo ERROR: run this .bat from the cloned repo layout, or cd to project root first.
  exit /b 1
)

python BoulderCalculator\experiments\geo_splits\smoke_geo_splits.py --mode smoke --device cuda --num-workers 2 %*
if errorlevel 1 exit /b 1
echo.
echo Smoke OK. For full weekend runs: run_geo_weekend.bat
endlocal
