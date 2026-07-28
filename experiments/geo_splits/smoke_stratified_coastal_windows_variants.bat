@echo off
REM Smoke: stratified_coastal 4 variants (build + 3-iter trains).
REM Run BEFORE the overnight variants bat. From project root.
setlocal
cd /d "%~dp0..\..\.."
if not exist "BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_variants.py" (
  echo ERROR: project root not found. cwd=%CD%
  exit /b 1
)

echo Smoke variants from %CD%
python BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_variants.py --mode smoke --device cuda %*
set ERR=%ERRORLEVEL%
if %ERR% neq 0 (
  echo SMOKE FAILED with exit %ERR% — do not start overnight until fixed.
  exit /b %ERR%
)
echo.
echo Smoke OK. Start overnight with:
echo   BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_variants.bat
endlocal
