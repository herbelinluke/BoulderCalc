# Geo-split weekend experiment (RGB+DSM)

Branch: **`exp/geo-split-weekend`**

Five geographic train / valid / test setups, trained with **offline dihedral + jitter**
on a **shared all-tiles pool** and **no online rich augs** (`--no-rich-aug`).
Modality is **RGB+DSM only** (`--four-band`). Default long runs use **5000**
iterations, checkpoints every **2000**, validation AP every **500**.

2024 tile `3_36` is removed from `tiles_used` / hold-outs (2025 `3_36` unchanged).

## Setups

| ID | Idea | Leakage check |
|----|------|----------------|
| `baseline` | Current coastal blocks (valid≈PCA {2,8}, test≈{5,11}) + buffer | geographic |
| `blocks_alt_a` | PCA bins valid={1,7}, test={4,10} + buffer | geographic |
| `blocks_alt_b` | PCA bins valid={0,6}, test={3,9} + buffer | geographic |
| `north_south` | Contiguous coastal ends + buffer | geographic |
| `sporadic_aligned` | Random year-aligned locations, **no buffer** | location_consistency |

QGIS review: open `tile_extents_<id>.geojson` (EPSG:25829) and style by the
`split` property (`train` / `valid` / `test` / `excluded`).

Regenerate alternate YAMLs + GeoJSONs (does **not** overwrite `baseline.yaml`):

```bat
python BoulderCalculator\scripts\generate_coastal_splits.py --segmentation-dir segmentation --output-dir BoulderCalculator\experiments\geo_splits --also-baseline-geojson
```

## Shared aug pool (one folder, not five)

There is **one** offline-augmented dataset for all tiles:

| Path | Role |
|------|------|
| `segmentation\coco_geo_all` | All tiles as train (`all_tiles.yaml`) |
| `segmentation\coco_geo_all_rgb_dsm` | Same, 4-band images |
| `segmentation\coco_geo_all_rgb_dsm_aug` | Offline 8× + jitter on that pool |
| `segmentation\coco_geo_<setup>_from_pool` | Thin per-setup COCO: JSONs + **symlinks** into the shared aug images |

Each training run only materializes which pool tiles belong to train/valid/test
for that geographic setup (`materialize_geo_split_coco.py`).

## Windows machine setup

Use the existing short-path guest / admin guides — do not reinvent the env here:

- [README_WINDOWS_GUEST.md](../../setup/README_WINDOWS_GUEST.md) (no admin / long paths)
- [README_WINDOWS.md](../../setup/README_WINDOWS.md)
- Canonical flags: [MODEL_TRAINING.md](../../MODEL_TRAINING.md)

From your short root (`B:\` or similar), you need:

- `BoulderCalculator\` (this branch)
- `segmentation\` (tiling, annotations, GPKGs)
- `2024\` / `2025\` DSM GeoTIFFs for 4-band tiles
- Activated conda/venv with Detectron2 + `requirements-training.txt` (includes PyYAML)

## One-time: shared RGB+DSM tiles

```bat
python BoulderCalculator\scripts\build_rgb_dsm_tiles.py --year 24
python BoulderCalculator\scripts\build_rgb_dsm_tiles.py --year 25
```

Or let the smoke / weekend runner fill gaps via `--from-coco` (requires DSM
GeoTIFFs). Optional full rebuild: `--build-rgb-dsm-tiles` / `--force-tiles`.

Rebuild the shared COCO+aug pool after changing annotations or small-boulder
policy: `--force-pool`.

## Smoke all setups (do this before leaving for the weekend)

From project root, with CUDA env active:

```bat
BoulderCalculator\experiments\geo_splits\smoke_geo_splits.bat
```

Or:

```bat
python BoulderCalculator\experiments\geo_splits\smoke_geo_splits.py --mode smoke --device cuda --num-workers 2
```

Optional flags:

- `--setups baseline,sporadic_aligned`
- `--skip-train` (pool + materialize only)
- `--drop-below-min-area` — omit boulders below `--min-area-m2` instead of `iscrowd=1`
- `--force-pool` — rebuild shared aug pool
- `--build-rgb-dsm-tiles` · `--device cpu`

Smoke flow:

1. Build shared pool once (`all_tiles.yaml` → RGB+DSM → offline aug)
2. For each setup: `materialize_geo_split_coco.py` → short `--four-band --no-rich-aug` train

Stops on the first failing setup so you can fix before the long loop.

## Weekend full runs

```bat
BoulderCalculator\experiments\geo_splits\run_geo_weekend.bat
```

That script: ensures RGB+DSM tiles → smoke all setups → trains each setup at
`--max-iter 5000 --batch-size 2 --checkpoint-period 2000 --eval-period 500`
(reuses the shared aug pool).

Outputs under `segmentation\training_run_geo_<id>\`:

| File | Role |
|------|------|
| `metrics.json` | Train loss + periodic val AP (every ~500 iters) — use to see saturation |
| `metrics_valid.json` | Final validation metrics |
| `model_final.pth` | Final weights |
| `model_XXXX.pth` | Sparse checkpoints (~2000 / 4000), not every 50 |

Manual single setup:

```bat
python BoulderCalculator\experiments\geo_splits\smoke_geo_splits.py --mode weekend --device cuda --setups baseline
```

Drop small boulders entirely (rebuilds pool when combined with `--force-pool`):

```bat
python BoulderCalculator\experiments\geo_splits\smoke_geo_splits.py --mode smoke --device cuda --drop-below-min-area --force-pool
```

## Disk note

Only **one** 8× 4-band aug of all tiles is stored (`coco_geo_all_rgb_dsm_aug`).
Per-setup dirs are mostly symlinks + JSON. Prefer a second drive / USB for the
pool and `training_run_geo_*`. Delete `training_run_geo_*_smoke` after a
successful smoke if space is tight.

## Matching / inference

The Matching tools still default to the baked-in baseline `TEST_*` lists unless
you point them at a setup’s test tiles. For this experiment, compare setups via
each run’s `metrics_valid.json` / `metrics.json` AP curves first.
