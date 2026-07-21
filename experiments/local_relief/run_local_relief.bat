@echo off
REM Full local-relief RGB+DSM training. Prefer smoke_local_relief.bat first.
setlocal
cd /d "%~dp0..\..\.."
python BoulderCalculator\experiments\local_relief\run_local_relief.py --mode full --device cuda --num-workers 2 --batch-size 1 %*
if errorlevel 1 exit /b 1
echo Done. See segmentation\training_run_local_relief\
endlocal
