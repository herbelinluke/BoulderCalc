@echo off
REM Smoke: stratified_coastal RGB @ 512+1024 (build + 3-iter balanced trains).
REM Windows guest. Run from project root before the weekend bat.
setlocal
cd /d "%~dp0..\..\.."
if not exist "BoulderCalculator\experiments\geo_splits\run_stratified_coastal_rgb_chips_windows.py" (
  echo ERROR: project root not found. cwd=%CD%
  exit /b 1
)
echo Smoke RGB chips from %CD%
python BoulderCalculator\experiments\geo_splits\run_stratified_coastal_rgb_chips_windows.py --mode smoke --device cuda %*
set ERR=%ERRORLEVEL%
if %ERR% neq 0 (
  echo SMOKE FAILED exit %ERR%
  exit /b %ERR%
)
echo.
echo Smoke OK. Weekend:
echo   BoulderCalculator\experiments\geo_splits\run_stratified_coastal_rgb_chips_windows.bat
endlocal
