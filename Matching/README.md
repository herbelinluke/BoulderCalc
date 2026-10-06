# Boulder Matching Module

> For the end-to-end model pipeline (dataset build, training, inference,
> evaluation) that produces the polygons matched here, see the canonical guide
> [`../MODEL_TRAINING.md`](../MODEL_TRAINING.md). Run the commands below from
> this `Matching/` directory.

## Overview

The **Boulder Matching Module** matches segmented boulder polygons
between two surveys (e.g., 2024 and 2025) to identify:

-   Matched boulders
-   Newly appeared boulders
-   Disappeared boulders
-   Movement vectors between matched boulders

Matching is **shape-first**: PCA-aligned contour Chamfer similarity dominates
the score. Volume is a secondary cue; centroid distance is only a weak prior
inside a large candidate gate (default **200 m**) for storm transport.

## Features

-   Shape-first Hungarian assignment (contour Chamfer + weak distance/volume)
-   Optional DSM-based volume estimation and `--min-volume` filter (default recipe: 0.5 m³)
-   Class filter for manual GPKGs (`Class == 0` boulders)
-   Portable `paths.local.yaml` for machine-specific orthos/DSMs/annotations
-   Overlap dedupe for multi-tile / sliding-window duplicate masks (off by default)
-   DSM-of-Difference (DoD) QC layer for match / appeared / disappeared review
-   GeoJSON outputs + eye-label GUI + verified-dataset export
-   Synthetic test data generation and unit tests

## Path configuration

Copy [`../paths.example.yaml`](../paths.example.yaml) to `../paths.local.yaml`
and point orthos/DSMs/annotations at your machine (USB, NAS, etc.).
`paths.local.yaml` is gitignored.

``` bash
# From Matching/
python -c "from matching.paths import load_paths; print(load_paths())"
```

Helper for manual 2024↔2025 GT matching:

``` bash
./run_manual_match.sh
```

## Installation

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

All datasets should use the same projected CRS (**EPSG:25829** at this site).

## Running the Matcher

### Manual GT ↔ GT (primary recipe)

Matches `july14_24.gpkg` ↔ `july14_25.gpkg`, keeps Class=0, volumes ≥ 0.5 m³,
search radius 200 m, no dedupe:

``` bash
./run_manual_match.sh
# equivalent:
python -m matching.cli \
  --use-path-config \
  --boulder-class-only \
  --min-volume 0.5 \
  --search-radius 200 \
  --compute-volume \
  --no-dedupe \
  --outdir ../../segmentation/manual_match_2024_2025
```

Outputs land in `segmentation/manual_match_2024_2025/` (`results/`, `predictions/`,
`match_summary.json`).

### Generic polygon layers

``` bash
python -m matching.cli \
  --before data/before.gpkg \
  --after data/after.gpkg \
  --outdir data/results \
  --search-radius 200
```

With DSM volumes:

``` bash
python -m matching.cli \
  --before data/before.gpkg --after data/after.gpkg \
  --before-dsm data/before_dsm.tif --after-dsm data/after_dsm.tif \
  --compute-volume --min-volume 0.5 \
  --outdir data/results
```

Use `--dedupe` only for overlapping Mask R-CNN detections. When both DSMs are
provided, a DoD QC folder is written under the outdir (`--no-dod-qc` to skip).

## Outputs

-   `results/matched_boulders.geojson` (includes `before_fid` / `after_fid` when present)
-   `results/appeared_boulders.geojson`
-   `results/disappeared_boulders.geojson`
-   `results/movement_vectors.geojson`
-   `predictions/before_inferred_boulders.geojson` / `after_…` (filtered inputs)
-   `match_summary.json`

## Eye verification → verified dataset

Label proposed matches:

``` bash
./run_match_eval.sh --gt-gt
# or:
python -m matching.evaluate_matches \
  --outdir ../../segmentation/manual_match_2024_2025 \
  --gt-gt
```

Keys: `y` confirm, `x` not a match, `?` unsure, `j` next unlabeled,
`c` print QGIS extent/centroids, `s` save, `q` save+quit, `n`/`p` navigate.

Writes:

-   `<outdir>/eval/match_labels.json`
-   `<outdir>/eval/match_labels.geojson`

Export only confirmed pairs:

``` bash
python -m matching.export_verified \
  --outdir ../../segmentation/manual_match_2024_2025
```

Creates `<outdir>/verified/verified_matches.geojson`, `verified_pairs.csv`,
and `match_labels_confirmed.json`.

## Matching Method

Score weights (defaults):

| Term | Weight | Notes |
|------|--------|-------|
| Shape | 0.70 | PCA-aligned contour Chamfer (+ Hausdorff), with a small area log-ratio fold-in |
| Volume | 0.20 | DSM log-ratio; 0.5 if missing |
| Distance | 0.10 | Soft prior inside `search_radius` (default 200 m) |

Candidates are pairs with centroid distance ≤ `search_radius`. Global optimal
assignment uses the Hungarian algorithm on cost = `1 − score`.

## Detection inference matching (secondary)

``` bash
./run_training_run_match.sh                 # inference + match + screenshots
./run_training_run_match.sh --gui-only      # browse existing results
```

## Testing

``` bash
pytest matching_tests/ -v
```

## Future Improvements

-   Staged search radii (50 → 100 → 200 m) for speed
-   Flip-aware matching using volume / 3D cues
-   Confidence-weighted matching
-   Integration with BoulderCalc MATLAB volume utilities
-   Feed DoD source–sink pairs back into the matcher score
