@echo off
REM Windows overnight: stratified_coastal variants A–D @ 2000 (balanced).
REM A RGB+DSM elevation | B RGB min1.0 | C drop deposits | D drop deposits+small
REM Skip leakage check is ON by default. Pass %%* for --force / --only …
setlocal
cd /d "%~dp0..\..\.."
if not exist "BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_variants.py" (
  echo ERROR: run from / find project root failed. Expected BoulderCalculator\ under cwd.
  echo cwd=%CD%
  exit /b 1
)

echo Project root: %CD%
python BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_variants.py --mode weekend --device cuda %*
set ERR=%ERRORLEVEL%
if %ERR% neq 0 (
  echo FAILED with exit %ERR%
  exit /b %ERR%
)
echo.
echo Finished. See segmentation\training_run_geo_stratified_coastal_*_balanced\
echo QGIS audits: segmentation\resample_audits_stratified_coastal\
endlocal
