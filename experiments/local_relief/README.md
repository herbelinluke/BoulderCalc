# Local-relief DSM training experiment

Branch: **`exp/local-relief-dsm`**

Same RGB+DSM Mask R-CNN recipe as the standard 4-band path, except band 4 is
**local relief** (DSM − Gaussian-smoothed DSM, then 2–98% stretch) instead of
absolute elevation. Implemented by existing
`build_rgb_dsm_tiles.py --dsm-mode local_relief`.

Tiles go to **separate** dirs so they do not overwrite elevation DSM tiles:

| Path | Role |
|------|------|
| `segmentation/tiling_rgb_dsm_local_relief_24` | 2024 4-band tiles |
| `segmentation/tiling_rgb_dsm_local_relief_25` | 2025 4-band tiles |
| `segmentation/coco_dataset_local_relief` | COCO pointing at those tiles |
| `segmentation/coco_dataset_local_relief_aug` | Offline dihedral + jitter (train) |
| `segmentation/training_run_local_relief` | Weights + `metrics.json` |

Default relief radius: **10 m** (`--relief-radius-m`).

## Prerequisites

Same as RGB+DSM in [`MODEL_TRAINING.md`](../../MODEL_TRAINING.md) /
[`setup/README_WINDOWS_GUEST.md`](../../setup/README_WINDOWS_GUEST.md):

- Project root with `BoulderCalculator/`, `segmentation/`, `2024/`, `2025/`
- CUDA env with Detectron2
- Ortho tiles + GPKGs already available

## Smoke (short train)

From project root:

```bat
python BoulderCalculator\experiments\local_relief\run_local_relief.py --mode smoke --device cuda --num-workers 2 --batch-size 1
```

Or:

```bat
BoulderCalculator\experiments\local_relief\smoke_local_relief.bat
```

## Full training run

```bat
python BoulderCalculator\experiments\local_relief\run_local_relief.py --mode full --device cuda --num-workers 2 --batch-size 1 --min-area-m2 1.5
```

Or:

```bat
BoulderCalculator\experiments\local_relief\run_local_relief.bat --min-area-m2 1.5 --batch-size 1
```

Defaults for `full`: `--max-iter 5000`, `--checkpoint-period 2000`,
`--eval-period 500`, `--four-band`, online rich augs **on** (same as standard
RGB+DSM recipe). Pass `--no-rich-aug` to match the geo-split weekend offline-only
recipe.

## Manual commands (equivalent)

```bat
python BoulderCalculator\scripts\gpkg_to_coco.py --segmentation-dir segmentation --years 24,25 --output-dir segmentation\coco_dataset_both --min-area-m2 1.5

python BoulderCalculator\scripts\build_rgb_dsm_tiles.py --year 24 --dsm-mode local_relief --relief-radius-m 10
python BoulderCalculator\scripts\build_rgb_dsm_tiles.py --year 25 --dsm-mode local_relief --relief-radius-m 10

python BoulderCalculator\scripts\build_coco_rgb_dsm.py --source-coco segmentation\coco_dataset_both --tile-dirs segmentation\tiling_rgb_dsm_local_relief_24 segmentation\tiling_rgb_dsm_local_relief_25 --output-dir segmentation\coco_dataset_local_relief

python BoulderCalculator\scripts\augment_coco_dataset.py --input-dir segmentation\coco_dataset_local_relief --output-dir segmentation\coco_dataset_local_relief_aug --jitter 0.15

python BoulderCalculator\scripts\train_boulder_local.py --dataset-dir segmentation\coco_dataset_local_relief_aug --output-dir segmentation\training_run_local_relief --four-band --max-iter 5000 --batch-size 1 --num-workers 2 --device cuda --checkpoint-period 2000 --eval-period 500
```

## Compare to elevation DSM

Keep elevation tiles under `tiling_rgb_dsm_{24,25}` and local-relief under
`tiling_rgb_dsm_local_relief_{24,25}`. Compare
`training_run_rgb_dsm/metrics*.json` vs `training_run_local_relief/metrics*.json`
(or use `experiments/eval/compare_training_runs.ipynb` if present on another
branch).
