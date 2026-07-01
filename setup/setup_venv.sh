#!/usr/bin/env bash
# Create a user-local Python venv for boulder training (no admin required).
# Usage: bash segmentation/setup_venv.sh /path/to/boulder_project

set -euo pipefail
ROOT="${1:-$(pwd)}"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ for your user account."
  exit 1
fi

python3 -m venv .venv_boulder
# shellcheck disable=SC1091
source .venv_boulder/bin/activate
python -m pip install --upgrade pip

# Choose ONE:
# GPU (CUDA 12.4 example):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# CPU:
pip install torch torchvision torchaudio

# Detectron2 (match torch/CUDA): see README_PORTABLE.md
# pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu124/torch2.4/index.html
# or:
# pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

pip install -r segmentation/requirements-training.txt

echo "Venv ready. Activate with: source .venv_boulder/bin/activate"
