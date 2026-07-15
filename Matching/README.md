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
│   └── attributes.py
├── matching_tests/
│   ├── generate_test_data.py
│   ├── check_results.py
│   └── test_matcher.py
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

# Interactive browser (n/p to flip matches):
python -m matching.visualize --results-dir data/results --gui \
  --before data/before.geojson --after data/after.geojson \
  --after-ortho /path/to/after_ortho.tif --no-screenshots
```

For the `training_run_rgb_dsm_4000` july14 annotations + DSMs:

``` bash
./run_training_run_match.sh          # rematch + screenshots
./run_training_run_match.sh --gui    # also open the browser
```

## Matching Method

Candidate matches are evaluated using:

-   Centroid distance
-   Polygon area
-   DSM-derived volume (optional)
-   Orientation (if available)

A global optimal assignment is computed using the Hungarian algorithm to
maximize overall match quality.

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

-   Adaptive search radius
-   Shape descriptors
-   Confidence-weighted matching
-   Integration with BoulderCalc volume utilities
-   Improved handling of dense boulder deposits
