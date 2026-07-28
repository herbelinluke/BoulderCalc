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
| `segmentation\coco_geo_<setup>_from_pool` | Thin per-setup COCO: JSONs + **hard links** into the shared aug images (no extra GB) |

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

## Stratified coastal overnight (RGB @ 2000)

Density-stratified split [`stratified_coastal.yaml`](stratified_coastal.yaml)
(~74/12/14 boulder counts). Offline **8×+jitter on train only**; valid/test
stay unaugmented. Year-24 east-edge tiles `4_46` / `5_46` (messy mosaic cutoff)
are dropped from the tile list and always excluded. Two sequential trains:

1. **balanced** — oversample positives ~1.5×; thin deposit-heavy + green-hilly
   empties; keep a fraction of coastal empties
2. **control** — same split/aug, no resample

COCO build rejects mild valid↔test cross-year footprint overlap unless you
pass `--skip-leakage-check` (Windows overnight `.bat` includes this for now).

```bash
# From project root (laptop / Linux)
bash BoulderCalculator/experiments/geo_splits/run_stratified_coastal_rgb_overnight.sh
# or with the temporary leakage escape hatch:
python BoulderCalculator/experiments/geo_splits/run_stratified_coastal_rgb_overnight.py \
  --mode weekend --skip-leakage-check
```

### Windows (admin + CUDA): RGB A/B **and** local-relief A

Builds local-relief 2000×2000 parents, then trains Run C with the **same**
resample policy as RGB Run A (`--four-band`).

**1. Smoke first** (full build + 3-iter trains into `*_smoke` dirs):

```bat
BoulderCalculator\experiments\geo_splits\smoke_stratified_coastal_windows.bat
```

**2. Overnight** (only after smoke exits 0):

```bat
BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_overnight.bat
```

Or:

```bat
python BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_overnight.py --mode smoke --device cuda --skip-leakage-check
python BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_overnight.py --mode weekend --device cuda --skip-leakage-check
```

Outputs:

| Run | Dir |
|-----|-----|
| A RGB balanced | `segmentation\training_run_geo_stratified_coastal_rgb_balanced\` |
| B RGB control | `segmentation\training_run_geo_stratified_coastal_rgb\` |
| C local-relief balanced | `segmentation\training_run_geo_stratified_coastal_rgb_local_relief_balanced\` |

Skip pieces: `--skip-rgb-control`, `--skip-local-relief`, `--skip-rgb-balanced`.

Defaults: `--min-area-m2 1.5`, `--no-rich-aug`, jitter 0.15, eval every 500,
early-stop patience 1000, batch auto (2 if CUDA ≥10 GiB else 1), `--link-mode hard`.

### Windows guest: four balanced variants (RGB+DSM + annotation ablations)

Same split / resample / no-rich-aug / 8×+jitter / skip-leakage defaults, then
four sequential **balanced** trains:

| Key | Run | Annotation policy |
|-----|-----|-------------------|
| `A_rgb_dsm` | RGB+DSM elevation @ 2000 | 1.5 m²; iscrowd deposits + small |
| `B_rgb_min1p0` | RGB @ 2000 | **1.0 m²**; iscrowd deposits + small |
| `C_rgb_drop_deposits` | RGB @ 2000 | drop deposits; iscrowd small @ 1.5 |
| `D_rgb_drop_dep_small` | RGB @ 2000 | drop deposits **and** small |

```bat
BoulderCalculator\experiments\geo_splits\smoke_stratified_coastal_windows_variants.bat
BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_variants.bat
```

Subset / rebuild:

```bat
python BoulderCalculator\experiments\geo_splits\run_stratified_coastal_windows_variants.py --mode weekend --device cuda --only A_rgb_dsm,B_rgb_min1p0 --force
```

**QGIS before/after (resample):** each variant writes

`segmentation\resample_audits_stratified_coastal\<key>\`

- `resample_train_audit_parents.geojson` — style by `status`
  (`removed` / `thinned` / `kept` / `oversampled`) or `oversample_factor`
- `resample_train_audit_parents.csv` / `…_images.csv`
- `resample_train_summary.json` — keep rates + `positive_oversample`

Geometries join from
[`tile_extents_stratified_coastal.geojson`](tile_extents_stratified_coastal.geojson).

### Windows guest: RGB chips 512 + 1024 (balanced only)

Same split / aug / iscrowd / resample policy as Run A, but **RGB-only chips**
(no DSM). Faster than 2000² for guest VRAM.

```bat
BoulderCalculator\experiments\geo_splits\smoke_stratified_coastal_rgb_chips_windows.bat
BoulderCalculator\experiments\geo_splits\run_stratified_coastal_rgb_chips_windows.bat
```

Outputs:

| Chip | Run dir |
|------|---------|
| 512 | `segmentation\training_run_geo512_stratified_coastal_rgb_balanced\` |
| 1024 | `segmentation\training_run_geo1024_stratified_coastal_rgb_balanced\` |

Defaults: `--min-area-m2 1.5`, `--no-rich-aug`, jitter 0.15, train-only 8× aug,
eval every 500, early-stop 500, max_iter 3000, batch auto (512→4 / 1024→2 on
≥8 GiB). `--chips 512` or `--chips 1024` to run one size.

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

Only **one** 8× 4-band aug of all tiles is stored (`coco_geo_all_rgb_dsm_aug`,
~20GB). Per-setup dirs use **NTFS hard links** into that pool (no second copy).
Symlinks need admin / Developer Mode on Windows guest — the runners pass
`--link-mode hard` instead. Avoid `--link-mode copy` (would multiply the 20GB).

Leave headroom for five `training_run_geo_*` dirs (checkpoints are sparse:
every 2000 iters + final). Prefer a second drive / USB. Delete
`training_run_geo_*_smoke` after a successful smoke if space is tight.

## Matching / inference

The Matching tools still default to the baked-in baseline `TEST_*` lists unless
you point them at a setup’s test tiles. For this experiment, compare setups via
each run’s `metrics_valid.json` / `metrics.json` AP curves first.

## Evaluating results (compare runs + per-tile maps)

See [`../eval/README.md`](../eval/README.md) and the notebook
[`../eval/compare_training_runs.ipynb`](../eval/compare_training_runs.ipynb).

```bash
# Whole-split table + learning curves
python BoulderCalculator/scripts/eval_compare_runs.py \
  --segmentation-dir segmentation --geo-prefix training_run_geo_ \
  --output-dir segmentation/eval_compare_geo

# Per-tile AP/AR heatmaps (+ optional --merge-iou / --split-config difficulty)
python BoulderCalculator/scripts/eval_per_tile.py \
  --dataset-dir segmentation/coco_geo_baseline_rgb_dsm --split test \
  --model segmentation/training_run_geo_baseline/model_final.pth \
  --four-band --device cuda \
  --output-dir segmentation/eval_per_tile_baseline_test
```
