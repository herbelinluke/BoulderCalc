#!/usr/bin/env bash
# Build a matcher eval dataset from july14 manual GPKGs (optional bbox).
#
# Usage:
#   ./run_build_gt_dataset.sh
#   ./run_build_gt_dataset.sh --bbox 458800 5879880 458900 5880000
#   ./run_build_gt_dataset.sh --outdir /path/to/out --search-radius 20

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv_boulder/bin/python"
MATCH_DIR="${ROOT}/BoulderCalculator/Matching"
OUT="${ROOT}/segmentation/match_datasets/july14_manual"

cd "$MATCH_DIR"
export PYTHONUNBUFFERED=1

exec "$PY" -m matching.build_gt_dataset \
  --outdir "$OUT" \
  --search-radius 15 \
  --candidate-radius 25 \
  "$@"
