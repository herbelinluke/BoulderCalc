# Experiment evaluation tools

Utilities for comparing training runs and mapping per-tile AP / AR.
**No Jupyter required** — use the Python CLI scripts below.

| Path | Role |
|------|------|
| [`run_compare_report.py`](run_compare_report.py) | **Start here** — compare geo runs → CSV + curve PNGs + `report.md` |
| [`../../scripts/eval_utils.py`](../../scripts/eval_utils.py) | Library |
| [`../../scripts/eval_compare_runs.py`](../../scripts/eval_compare_runs.py) | Same comparison as a lower-level CLI |
| [`../../scripts/eval_per_tile.py`](../../scripts/eval_per_tile.py) | Per-tile scores, heatmaps, optional merge, geo difficulty |
| [`compare_training_runs.ipynb`](compare_training_runs.ipynb) | Optional notebook (same logic; needs a working kernel) |

## Quick: compare geo-split weekend runs (no Jupyter)

From the project root (`tamucc/`, the folder that contains `BoulderCalculator/` and `segmentation/`):

```bash
source .venv_boulder/bin/activate   # or: .venv_boulder/bin/python ...
python BoulderCalculator/experiments/eval/run_compare_report.py
```

Writes under `segmentation/eval_compare_geo/`:
- `metrics_valid_comparison.csv` / `metrics_valid_ranked.csv`
- `curve_bbox_AP50.png` (and AR / segm variants)
- `report.md`

## Per-tile heatmaps

Weekend models were **4-band (RGB+DSM)**. On many laptops
`segmentation/coco_geo_baseline_rgb_dsm/` is only an empty stub (no
`testing_annotations.json`). Use the RGB annotations that *do* exist and point
`--image-dir` at the 4-band tile folders (partial coverage is OK — missing tiles
are skipped by default):

```bash
python BoulderCalculator/scripts/eval_per_tile.py \
  --gt-json segmentation/coco_geo_baseline/testing_annotations.json \
  --image-dir segmentation/tiling_rgb_dsm_24 \
  --image-dir segmentation/tiling_rgb_dsm_25 \
  --model segmentation/training_run_geo_baseline/model_final.pth \
  --four-band --device cuda \
  --output-dir segmentation/eval_per_tile_baseline_test \
  --split-config BoulderCalculator/experiments/geo_splits/baseline.yaml \
  --split-config BoulderCalculator/experiments/geo_splits/blocks_alt_a.yaml \
  --split-config BoulderCalculator/experiments/geo_splits/blocks_alt_b.yaml \
  --split-config BoulderCalculator/experiments/geo_splits/north_south.yaml
```

Or, if you have a complete 4-band COCO dir:

```bash
python BoulderCalculator/scripts/eval_per_tile.py \
  --dataset-dir segmentation/coco_geo_baseline_rgb_dsm \
  --split test \
  --model segmentation/training_run_geo_baseline/model_final.pth \
  --four-band --device cuda \
  --output-dir segmentation/eval_per_tile_baseline_test
```

### With / without merging overlapping detections

```bash
# raw
python BoulderCalculator/scripts/eval_per_tile.py ... --output-dir segmentation/eval_per_tile_raw

# NMS merge
python BoulderCalculator/scripts/eval_per_tile.py ... --merge-iou 0.4 \
  --output-dir segmentation/eval_per_tile_merged
```

### Rebuild a full 4-band COCO (when tiling dirs are complete)

```bash
python BoulderCalculator/scripts/build_coco_rgb_dsm.py \
  --source-coco segmentation/coco_geo_baseline \
  --tile-dirs segmentation/tiling_rgb_dsm_24 segmentation/tiling_rgb_dsm_25 \
  --output-dir segmentation/coco_geo_baseline_rgb_dsm
```

## Optional notebook

If you want the notebook, pick the **`.venv_boulder`** kernel and open it from
the project root (or `experiments/eval/`). Path discovery walks upward until it
finds `BoulderCalculator/scripts/eval_utils.py`. Prefer `run_compare_report.py`
if Jupyter install is painful.
