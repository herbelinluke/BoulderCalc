#!/usr/bin/env bash
# Label inferred matches for matcher evaluation (confirm / not-match / unsure).
#
# Usage:
#   ./run_match_eval.sh
#   ./run_match_eval.sh --outdir /path/to/matching
#   ./run_match_eval.sh --labels-json /path/to/match_labels.json

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv_boulder/bin/python"
MATCH_DIR="${ROOT}/BoulderCalculator/Matching"
OUT="${ROOT}/segmentation/training_run_rgb_dsm_4000/matching"

cd "$MATCH_DIR"
unset MPLBACKEND || true
export PYTHONUNBUFFERED=1

exec "$PY" -m matching.evaluate_matches --outdir "$OUT" "$@"
