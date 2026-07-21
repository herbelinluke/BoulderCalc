@echo off
REM Smoke: local-relief RGB+DSM short train. Run from project root after activating env.
setlocal
cd /d "%~dp0..\..\.."
python BoulderCalculator\experiments\local_relief\run_local_relief.py --mode smoke --device cuda --num-workers 2 --batch-size 1 %*
if errorlevel 1 exit /b 1
echo Smoke OK. Full run: run_local_relief.bat
endlocal
