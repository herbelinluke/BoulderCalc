@echo off
REM Weekend: stratified_coastal RGB balanced @ 512 then 1024.
REM Windows guest. Smoke first with smoke_stratified_coastal_rgb_chips_windows.bat
setlocal
cd /d "%~dp0..\..\.."
if not exist "BoulderCalculator\experiments\geo_splits\run_stratified_coastal_rgb_chips_windows.py" (
  echo ERROR: project root not found. cwd=%CD%
  exit /b 1
)
echo Weekend RGB chips from %CD%
python BoulderCalculator\experiments\geo_splits\run_stratified_coastal_rgb_chips_windows.py --mode weekend --device cuda --skip-leakage-check %*
set ERR=%ERRORLEVEL%
if %ERR% neq 0 (
  echo FAILED exit %ERR%
  exit /b %ERR%
)
echo.
echo Finished. See segmentation\training_run_geo512_stratified_coastal_rgb_balanced\
echo           and segmentation\training_run_geo1024_stratified_coastal_rgb_balanced\
endlocal
