@echo off
REM Local-relief geo pass only: 5 setups x rgb_local_relief (5 trains).
REM Encoding: fixed 0.5 m positive-only clip, 10 m Gaussian, 60 m parent pad.
REM Prefer a short smoke with --modalities rgb_local_relief first.
setlocal
cd /d "%~dp0..\..\.."
if not exist "BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py" (
  echo ERROR: expected project root with BoulderCalculator\ and segmentation\
  exit /b 1
)

echo === Smoke local-relief (baseline) ===
python BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py --mode smoke --device cuda --num-workers 2 --setups baseline --modalities rgb_local_relief
if errorlevel 1 (
  echo Smoke failed — aborting weekend runs.
  exit /b 1
)

echo.
echo === Weekend local-relief geo training ===
python BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py --mode weekend --device cuda --num-workers 2 --batch-size 4 --modalities rgb_local_relief %*
if errorlevel 1 exit /b 1

echo.
echo Local-relief weekend finished. See segmentation\training_run_geo512_*_rgb_local_relief\
endlocal
