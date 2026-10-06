#!/usr/bin/env bash
# Label matches for matcher evaluation (confirm / not-match / unsure).
#
# Usage:
#   ./run_match_eval.sh
#   ./run_match_eval.sh --outdir /path/to/matching
#   ./run_match_eval.sh --labels-json /path/to/match_labels.json

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

# Default to manual GT↔GT match outdir if present, else detection match outdir.
if [[ -d "${ROOT}/segmentation/manual_match_2024_2025/results" ]]; then
  OUT="${ROOT}/segmentation/manual_match_2024_2025"
elif [[ -d "${ROOT}/segmentation/training_run_rgb_dsm_14/matching/results" ]]; then
  OUT="${ROOT}/segmentation/training_run_rgb_dsm_14/matching"
else
  OUT="${ROOT}/segmentation/training_run_rgb_dsm_4000/matching"
fi

cd "$MATCH_DIR"
unset MPLBACKEND || true
export PYTHONUNBUFFERED=1

exec "$PY" -m matching.evaluate_matches --outdir "$OUT" "$@"
