#!/usr/bin/env bash
# Run boulder matching + screenshot export for training_run_rgb_dsm_4000.
# Usage:
#   ./run_training_run_match.sh              # match + screenshots
#   ./run_training_run_match.sh --gui        # also open interactive browser

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/.venv_boulder/bin/python"
MATCH_DIR="${ROOT}/BoulderCalculator/Matching"
OUT="${ROOT}/segmentation/training_run_rgb_dsm_4000/matching"
ANN="${ROOT}/segmentation/annotations"

GUI=0
for arg in "$@"; do
  [[ "$arg" == "--gui" ]] && GUI=1
done

mkdir -p "$OUT/inputs" "$OUT/results" "$OUT/screenshots"

echo "=== Preparing boulder-only GeoJSONs (EPSG:25829) ==="
"$PY" <<PY
import geopandas as gpd
from pathlib import Path

ann = Path("${ANN}")
out = Path("${OUT}/inputs")

def load_boulders(path):
    gdf = gpd.read_file(path)
    cls = gdf["Class"].astype(str).str.strip()
    gdf = gdf[cls.isin(["0", "0.0"]) & gdf.geometry.notnull()].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.geometry.is_empty].to_crs("EPSG:25829")
    return gdf[gdf.geometry.area >= 0.1].reset_index(drop=True)

before = load_boulders(ann / "july14_24.gpkg")
after = load_boulders(ann / "july14_25.gpkg")
before.to_file(out / "before_boulders_25829.geojson", driver="GeoJSON")
after.to_file(out / "after_boulders_25829.geojson", driver="GeoJSON")
print(f"before={len(before)} after={len(after)}")
PY

echo "=== Matching with DSM volumes ==="
cd "$MATCH_DIR"
"$PY" -m matching.cli \
  --before "$OUT/inputs/before_boulders_25829.geojson" \
  --after "$OUT/inputs/after_boulders_25829.geojson" \
  --before-dsm "${ROOT}/2024/Sites1and2_2024_DSM_30mm.tif" \
  --after-dsm "${ROOT}/2025/25IniSouthDSM.tif" \
  --compute-volume \
  --outdir "$OUT/results" \
  --search-radius 5.0 \
  --min-score 0.55

echo "=== Screenshots ==="
export MPLBACKEND=Agg
VIZ_ARGS=(
  -m matching.visualize
  --results-dir "$OUT/results"
  --outdir "$OUT/screenshots"
  --before "$OUT/inputs/before_boulders_25829.geojson"
  --after "$OUT/inputs/after_boulders_25829.geojson"
  --before-ortho "${ROOT}/2024/Sites1and2_2024_Orthomosaic.tif"
  --after-ortho "${ROOT}/2025/25IniSouthOrt.tif"
  --max-matches 30
)
if [[ "$GUI" -eq 1 ]]; then
  unset MPLBACKEND
  VIZ_ARGS+=(--gui)
fi
"$PY" "${VIZ_ARGS[@]}"

echo "Done. Results: $OUT/results"
echo "Screenshots: $OUT/screenshots"
