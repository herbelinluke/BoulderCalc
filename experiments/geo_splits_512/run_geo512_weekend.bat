@echo off
REM Full weekend: 512 chips, 5 geo setups x RGB + RGB+DSM (10 trains).
REM Defaults: hard links, skip-existing, batch 4, no rich augs, min-area 1.5.
REM Prefer smoke_geo_splits_512.bat successfully first.
setlocal
cd /d "%~dp0..\..\.."
if not exist "BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py" (
  echo ERROR: expected project root with BoulderCalculator\ and segmentation\
  exit /b 1
)

echo === Smoke (short) before long runs ===
python BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py --mode smoke --device cuda --num-workers 2
if errorlevel 1 (
  echo Smoke failed — aborting weekend runs.
  exit /b 1
)

echo.
echo === Weekend full training ===
python BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py --mode weekend --device cuda --num-workers 2 --batch-size 4 %*
if errorlevel 1 exit /b 1

echo.
echo Weekend finished. See segmentation\training_run_geo512_*\
endlocal
