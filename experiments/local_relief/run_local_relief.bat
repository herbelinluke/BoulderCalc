@echo off
REM Dual full train: RGB+elevation DSM and RGB+local-relief (baseline tiles).
REM Defaults: no rich augs, min-area 1.5, jitter 0.15, batch 1, workers 2,
REM 3000 iters, early-stop patience 500. Prefer smoke_local_relief.bat first.
setlocal
cd /d "%~dp0..\..\.."
python BoulderCalculator\experiments\local_relief\run_local_relief.py --mode full --device cuda --num-workers 2 --batch-size 1 --models both %*
if errorlevel 1 exit /b 1
echo Done. See segmentation\training_run_rgb_dsm\ and segmentation\training_run_local_relief\
endlocal
