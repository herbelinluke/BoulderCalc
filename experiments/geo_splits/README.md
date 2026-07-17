# Geo-split weekend experiment (RGB+DSM)

Branch: **`exp/geo-split-weekend`**

Five geographic train / valid / test setups, trained with **offline dihedral + jitter
on all splits** and **no online rich augs** (`--no-rich-aug`). Modality is
**RGB+DSM only** (`--four-band`). Default long runs use **5000** iterations,
checkpoints every **2000**, validation AP every **500**.

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

Or let the weekend / smoke runner fill gaps: it calls
`build_rgb_dsm_tiles.py --from-coco` for any tiles a setup still needs (requires
`2024\` / `2025\` DSM GeoTIFFs). Optional full rebuild: `--build-rgb-dsm-tiles`
/ `--force-tiles`.

These dirs are **shared** across setups (`segmentation\tiling_rgb_dsm_24` /
`tiling_rgb_dsm_25`). Per-setup COCO dirs are separate and large once all
splits are 8× augmented — keep them on the USB / short drive.

## Smoke all setups (do this before leaving for the weekend)

From project root, with CUDA env active:

```bat
BoulderCalculator\experiments\geo_splits\smoke_geo_splits.bat
```

Or:

```bat
python BoulderCalculator\experiments\geo_splits\smoke_geo_splits.py --mode smoke --device cuda --num-workers 2
```

Optional: `--setups baseline,sporadic_aligned` · `--skip-train` (dataset only) ·
`--build-rgb-dsm-tiles` · `--device cpu` (slow).

Smoke recipe per setup:

1. `gpkg_to_coco.py --split-config …yaml`
2. `build_coco_rgb_dsm.py` → `segmentation\coco_geo_<id>_rgb_dsm`
3. `augment_coco_dataset.py --splits train,valid,test --jitter 0.15`
4. `train_boulder_local.py --four-band --no-rich-aug --max-iter 3 --image-size 800 …`

Stops on the first failing setup so you can fix before the long loop.

## Weekend full runs

```bat
BoulderCalculator\experiments\geo_splits\run_geo_weekend.bat
```

That script: ensures RGB+DSM tiles → smoke all setups → trains each setup at
`--max-iter 5000 --batch-size 2 --checkpoint-period 2000 --eval-period 500`.

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

## Disk note

Augmenting **train + valid + test** at 8× on 4-band GeoTIFFs for five setups is
heavy. Prefer a second drive / USB for `segmentation\coco_geo_*` and
`training_run_geo_*`. Delete smoke output dirs
(`training_run_geo_*_smoke`) after a successful smoke if space is tight.

## Matching / inference

The Matching tools still default to the baked-in baseline `TEST_*` lists unless
you point them at a setup’s test tiles. For this experiment, compare setups via
each run’s `metrics_valid.json` / `metrics.json` AP curves first.
