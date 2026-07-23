@echo off
REM Full weekend geo-split training loop (RGB+DSM, offline+jitter, no online augs).
REM Prefer running smoke_geo_splits.bat successfully first.
REM
REM Outputs: segmentation\training_run_geo_<setup>\
REM   model_final.pth, sparse checkpoints (~every 2000 iters), metrics.json (AP every ~500)

setlocal
cd /d "%~dp0..\..\.."
if not exist "BoulderCalculator\experiments\geo_splits\smoke_geo_splits.py" (
  echo ERROR: expected project root with BoulderCalculator\ and segmentation\
  exit /b 1
)

echo === Building shared RGB+DSM tiles if missing ===
python BoulderCalculator\experiments\geo_splits\smoke_geo_splits.py --mode weekend --device cuda --num-workers 2 --build-rgb-dsm-tiles --skip-train --skip-aug
if errorlevel 1 exit /b 1

echo.
echo === Smoke all setups (shared pool + short train) before long runs ===
python BoulderCalculator\experiments\geo_splits\smoke_geo_splits.py --mode smoke --device cuda --num-workers 2
if errorlevel 1 (
  echo Smoke failed — aborting weekend runs.
  exit /b 1
)

echo.
echo === Weekend full training (5000 iters each; reuse shared aug pool) ===
python BoulderCalculator\experiments\geo_splits\smoke_geo_splits.py --mode weekend --device cuda --num-workers 2 %*
if errorlevel 1 exit /b 1

echo.
echo Weekend loop finished. Check segmentation\training_run_geo_*\metrics.json
echo Shared aug pool: segmentation\coco_geo_all_rgb_dsm_aug\
endlocal
