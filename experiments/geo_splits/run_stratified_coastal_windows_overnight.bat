@echo off
REM Windows overnight: stratified_coastal RGB A/B + local-relief A @ 2000.
REM Run from project root (folder with BoulderCalculator\ and segmentation\).
REM Requires activated conda/venv with Detectron2 + CUDA.

setlocal
cd /d "%~dp0..\..\.."
if not exist "BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_overnight.py" (
  echo ERROR: run from / find project root failed. Expected BoulderCalculator\ under cwd.
  echo cwd=%CD%
  exit /b 1
)

echo Project root: %CD%
python BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_overnight.py --mode weekend --device cuda %*
set ERR=%ERRORLEVEL%
if %ERR% neq 0 (
  echo FAILED with exit %ERR%
  exit /b %ERR%
)
echo.
echo Finished. See segmentation\training_run_geo_stratified_coastal_*
endlocal
