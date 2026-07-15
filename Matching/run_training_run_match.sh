#!/usr/bin/env bash
# Inference → match → side-by-side screenshots for training_run_rgb_dsm_4000.
# Default test set = gpkg_to_coco TEST_24 (27) + TEST_25 (15) = 42 tiles.
#
# Usage:
#   ./run_training_run_match.sh                 # full pipeline
#   ./run_training_run_match.sh --gui           # full pipeline, then open browser
#   ./run_training_run_match.sh --gui-only      # browse existing results (no inference)
#   ./run_training_run_match.sh --screenshots-only

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv_boulder/bin/python"
MATCH_DIR="${ROOT}/BoulderCalculator/Matching"
OUT="${ROOT}/segmentation/training_run_rgb_dsm_4000/matching"
MODEL="${ROOT}/segmentation/training_run_rgb_dsm_4000/model_final.pth"

GUI_ONLY=0
SHOTS_ONLY=0
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --gui-only) GUI_ONLY=1 ;;
    --screenshots-only) SHOTS_ONLY=1 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

cd "$MATCH_DIR"

export PYTHONUNBUFFERED=1

if [[ "$GUI_ONLY" -eq 1 || "$SHOTS_ONLY" -eq 1 ]]; then
  VIEW_ARGS=( -m matching.view_results --outdir "$OUT" )
  [[ "$GUI_ONLY" -eq 1 ]] && VIEW_ARGS+=( --gui )
  [[ "$SHOTS_ONLY" -eq 1 ]] && VIEW_ARGS+=( --screenshots )
  # GUI needs a real display backend
  if [[ "$GUI_ONLY" -eq 1 ]]; then
    unset MPLBACKEND || true
  else
    export MPLBACKEND="${MPLBACKEND:-Agg}"
  fi
  exec "$PY" "${VIEW_ARGS[@]}"
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"
"$PY" -m matching.run_inference_match \
  --model "$MODEL" \
  --outdir "$OUT" \
  --project-root "$ROOT" \
  --score-thresh 0.4 \
  --search-radius 5.0 \
  --min-score 0.55 \
  --device cpu \
  "${EXTRA[@]}"

echo "Done."
echo "  Predictions: $OUT/predictions"
echo "  Results:     $OUT/results"
echo "  Screenshots: $OUT/screenshots"
echo "  Summary:     $OUT/match_summary.json"
echo
echo "Browse without re-inferring:"
echo "  ./run_training_run_match.sh --gui-only"
