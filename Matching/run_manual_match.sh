#!/usr/bin/env bash
# Match manual july14_24 ↔ july14_25 annotations (shape-first, ≥0.5 m³).
#
# Usage:
#   ./run_manual_match.sh
#   ./run_manual_match.sh --search-radius 100
#   ./run_manual_match.sh --no-compute-volume   # area-only filter / no DSM

set -euo pipefail

MATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "${MATCH_DIR}/.." && pwd)"
ROOT="$(cd "${CODE_DIR}/.." && pwd)"

if [[ -x "${ROOT}/.venv_boulder/bin/python" ]]; then
  PY="${ROOT}/.venv_boulder/bin/python"
elif [[ -x "${CODE_DIR}/.venv_boulder/bin/python" ]]; then
  PY="${CODE_DIR}/.venv_boulder/bin/python"
else
  PY="${PYTHON:-python3}"
fi

OUT="${ROOT}/segmentation/manual_match_2024_2025"
cd "$MATCH_DIR"
export PYTHONUNBUFFERED=1
export MPLBACKEND="${MPLBACKEND:-Agg}"

exec "$PY" -m matching.cli \
  --use-path-config \
  --boulder-class-only \
  --min-volume 0.5 \
  --search-radius 200 \
  --min-score 0.55 \
  --compute-volume \
  --no-dedupe \
  --no-dod-qc \
  --outdir "$OUT" \
  "$@"
