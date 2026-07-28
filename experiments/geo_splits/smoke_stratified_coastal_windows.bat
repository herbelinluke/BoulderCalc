@echo off
REM Smoke: full stratified_coastal build (RGB + local-relief) + 3-iter trains.
REM Run this BEFORE the overnight bat. From project root.
setlocal
cd /d "%~dp0..\..\.."
if not exist "BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_overnight.py" (
  echo ERROR: project root not found. cwd=%CD%
  exit /b 1
)

echo Smoke test from %CD%
python BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_overnight.py --mode smoke --device cuda --skip-leakage-check %*
set ERR=%ERRORLEVEL%
if %ERR% neq 0 (
  echo SMOKE FAILED with exit %ERR% — do not start overnight until fixed.
  exit /b %ERR%
)
echo.
echo Smoke OK. Start overnight with:
echo   BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_overnight.bat
endlocal
