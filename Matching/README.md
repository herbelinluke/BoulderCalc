# Boulder Matching Module

## Overview

The **Boulder Matching Module** matches segmented boulder polygons
between two surveys (e.g., 2024 and 2025) to identify:

-   Matched boulders
-   Newly appeared boulders
-   Disappeared boulders
-   Movement vectors between matched boulders

The matcher is intended to be used after instance segmentation (e.g.,
Detectron2) and can optionally incorporate DSM-derived volume estimates
to improve matching.

## Features

-   Polygon matching using the Hungarian assignment algorithm
-   Distance- and area-based similarity scoring
-   Optional DSM-based volume estimation
-   Overlap dedupe for multi-tile / sliding-window duplicate masks
-   DSM-of-Difference (DoD) QC layer for match / appeared / disappeared review
-   GeoJSON outputs for visualization in QGIS
-   Synthetic test data generation
-   Unit tests for the matching pipeline

## Repository Structure

``` text
Matching/
    ├── matching/
    │   ├── __init__.py
    │   ├── cli.py
    │   ├── matcher.py
    │   ├── survey.py
    │   ├── attributes.py
    │   ├── dedupe.py
    │   ├── qc.py
    │   └── visualize.py
    ├── matching_tests/
    │   ├── generate_test_data.py
    │   ├── check_results.py
    │   ├── test_matcher.py
    │   └── test_dedupe_qc.py
    └── requirements.txt
```

## Installation

Create a Python environment and install dependencies:

``` bash
pip install -r requirements.txt
```

## Inputs

Required:

-   Before-survey polygon layer (`.gpkg` or `.geojson`)
-   After-survey polygon layer (`.gpkg` or `.geojson`)

Optional:

-   Before DSM (`.tif`)
-   After DSM (`.tif`)

For best results, all datasets should use the same projected CRS.

## Running the Matcher

Without DSM volumes:

``` bash
python -m matching.cli --before data/before.gpkg --after data/after.gpkg --outdir data/results
```

With DSM-derived volumes:

``` bash
python -m matching.cli --before data/before.gpkg --after data/after.gpkg --before-dsm data/before_dsm.tif --after-dsm data/after_dsm.tif --compute-volume --outdir data/results
```

Dedupe (default on) collapses overlapping detections from sliding tiles
before matching (`--no-dedupe` to disable). When both DSMs are provided,
a DoD QC folder is written under the outdir (`--no-dod-qc` to skip):

-   `dod_qc/match_dod_qc.geojson` — per-match source/sink volumes + `qc_label`
-   `dod_qc/disappeared_dod_qc.geojson` — flags `likely_missed_mover_source`
-   `dod_qc/appeared_dod_qc.geojson` — flags `likely_missed_mover_sink`
-   `dod_qc/dod_qc_summary.json`

## Outputs

The matcher generates:

-   `matched_boulders.geojson`
-   `appeared_boulders.geojson`
-   `disappeared_boulders.geojson`
-   `movement_vectors.geojson`

These outputs can be loaded directly into QGIS for visualization and
quality control.

Quick look without QGIS (overview + ortho crops, optional GUI):

``` bash
python -m matching.visualize \
  --results-dir data/results \
  --outdir data/screenshots \
  --before data/before.geojson \
  --after data/after.geojson \
  --after-ortho /path/to/after_ortho.tif

# Interactive browser (n/p to flip matches; o toggles overview zoom;
# left panel starts zoomed on the current pair; screenshots draw displacement arrows):
python -m matching.visualize --results-dir data/results --gui \
  --before data/before.geojson --after data/after.geojson \
  --after-ortho /path/to/after_ortho.tif --no-screenshots
```

For the `training_run_rgb_dsm_4000` model against the full **42-tile**
hold-out set from `gpkg_to_coco.py` (`TEST_24` 27 + `TEST_25` 15):
for each test tile, build a same-extent opposite-year RGB+DSM window,
run the 4-band model on both years, match, and write side-by-side shots.

``` bash
./run_training_run_match.sh                 # inference + match + screenshots
./run_training_run_match.sh --gui           # same, then open browser
./run_training_run_match.sh --gui-only      # browse existing results (no inference)
./run_training_run_match.sh --screenshots-only
```

## Matching Method

Candidate matches are evaluated using:

-   Centroid distance
-   Polygon area
-   DSM-derived volume (optional)
-   Orientation (if available)

A global optimal assignment is computed using the Hungarian algorithm to
maximize overall match quality.

Before matching, overlapping instance masks from multi-tile inference are
collapsed with IoU / centroid NMS (highest score kept). After matching, the
optional DoD QC layer compares elevation change under before/after footprints
to label consistent movers vs likely missed movers among appeared/disappeared.

## Testing

Generate a synthetic dataset:

``` bash
python matching_tests/generate_test_data.py --outdir test_data
```

Run the matcher on the generated data, then evaluate the results:

``` bash
python matching_tests/check_results.py --outdir test_data
```

Run unit tests:

``` bash
pytest matching_tests/test_matcher.py
```

## Future Improvements

-   Adaptive search radius (including DoD-guided expansion for long movers)
-   Shape descriptors
-   Confidence-weighted matching
-   Integration with BoulderCalc volume utilities
-   Improved handling of dense boulder deposits
-   Feed DoD source–sink pairs back into the matcher score
