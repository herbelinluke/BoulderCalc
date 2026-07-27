#!/usr/bin/env bash
# Overnight stratified_coastal RGB @ 2000 (balanced then control).
# Run from project root (folder with BoulderCalculator/ and segmentation/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/.venv_boulder/bin/python" ]]; then
  PY="$ROOT/.venv_boulder/bin/python"
elif [[ -x "$ROOT/BoulderCalculator/.venv_boulder/bin/python" ]]; then
  PY="$ROOT/BoulderCalculator/.venv_boulder/bin/python"
else
  PY="${PYTHON:-python}"
fi
echo "Using $PY"
exec "$PY" BoulderCalculator/experiments/geo_splits/run_stratified_coastal_rgb_overnight.py \
  --device auto \
  --python "$PY" \
  "$@"
