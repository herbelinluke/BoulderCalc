@echo off
REM Create a user-local Python venv for boulder training (no admin required).
REM Usage: setup_venv.bat D:\boulder_project

setlocal
set ROOT=%~1
if "%ROOT%"=="" set ROOT=%CD%

echo Using project root: %ROOT%
cd /d "%ROOT%"

where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install Python 3.10 or 3.11 for current user only from https://www.python.org/downloads/windows/
  echo Enable "Add python.exe to PATH" during install.
  exit /b 1
)

python -m venv .venv_boulder
call .venv_boulder\Scripts\activate.bat
python -m pip install --upgrade pip

REM --- Choose ONE PyTorch line based on the target machine GPU/CUDA ---
REM GPU (CUDA 12.4 example):
REM pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
REM CPU only:
pip install torch torchvision torchaudio

REM Detectron2 wheel must match your torch+CUDA version.
REM See https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md
REM Example for CUDA 12.4 + torch 2.4:
REM pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu124/torch2.4/index.html
REM If no wheel exists, install Visual Studio Build Tools (user-level) and:
REM pip install git+https://github.com/facebookresearch/detectron2.git

pip install -r segmentation\requirements-training.txt

echo.
echo Venv ready. Activate with:
echo   call .venv_boulder\Scripts\activate.bat
endlocal
