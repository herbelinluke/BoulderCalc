@echo off
REM Smoke 512-chip geo splits (RGB + RGB+DSM). Prefer one setup first if unsure.
setlocal
cd /d "%~dp0..\..\.."
python BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py --mode smoke --device cuda --num-workers 2 %*
if errorlevel 1 exit /b 1
echo Smoke OK. Weekend: run_geo512_weekend.bat
endlocal
