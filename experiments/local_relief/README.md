# Local-relief vs elevation DSM (baseline)

Branch: **`exp/local-relief-dsm`**

Trains **two** Mask R-CNN models on the same baseline tile/annotation set:

| Model | Band 4 | Tile dirs | Train output |
|-------|--------|-----------|--------------|
| elevation | absolute DSM | `tiling_rgb_dsm_{24,25}` | `training_run_rgb_dsm` |
| local_relief | DSM − Gaussian smooth (2–98% stretch) | `tiling_rgb_dsm_local_relief_{24,25}` | `training_run_local_relief` |

Shared upstream: `coco_dataset_both` from July GPKGs + `tiles_used.txt` (default
geographic baseline). Deposits and boulders below the area cutoff are
**`iscrowd=1`** (Detectron2 ignore). Default cutoff: **1.5 m²**.

## Recipe (Windows defaults)

| Setting | Value |
|---------|-------|
| Online rich augs | **off** (`--no-rich-aug`) |
| Offline aug | full dihedral **8×** + `--jitter 0.15` |
| `--min-area-m2` | `1.5` (small → iscrowd) |
| Deposits | iscrowd (`--boulder-only` default) |
| `--batch-size` / `--num-workers` | `1` / `2` |
| `--max-iter` | `3000` |
| Early stop | stop if `segm/AP` flat for **500** iters since best (saves `model_best.pth`) |
| Relief radius | **10 m** |

## Prerequisites

Same as RGB+DSM in [`MODEL_TRAINING.md`](../../MODEL_TRAINING.md) /
[`setup/README_WINDOWS_GUEST.md`](../../setup/README_WINDOWS_GUEST.md):

- Project root with `BoulderCalculator/`, `segmentation/`, `2024/`, `2025/`
- CUDA env with Detectron2
- Ortho tiles + GPKGs already available

## Smoke (both models, short train)

```bat
python BoulderCalculator\experiments\local_relief\run_local_relief.py --mode smoke --device cuda --num-workers 2 --batch-size 1
```

Or:

```bat
BoulderCalculator\experiments\local_relief\smoke_local_relief.bat
```

## Full dual training

```bat
python BoulderCalculator\experiments\local_relief\run_local_relief.py --mode full --device cuda --num-workers 2 --batch-size 1
```

Or:

```bat
BoulderCalculator\experiments\local_relief\run_local_relief.bat
```

One model only: `--models elevation` or `--models local_relief`.

Reuse tiles / COCO / aug by default (core scripts skip existing outputs). Rebuild
with ``--force``. Optional hard skips: `--skip-build-tiles`, `--skip-coco`,
`--skip-train`.

## Compare

```bat
python BoulderCalculator\scripts\run_provenance.py segmentation\training_run_rgb_dsm
python BoulderCalculator\scripts\run_provenance.py segmentation\training_run_local_relief
```

Or `experiments/eval/compare_training_runs.ipynb` when available.
