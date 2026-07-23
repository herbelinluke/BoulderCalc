# Boulder model training & inference — canonical guide

One place for how to build datasets, train, run inference, evaluate (incl.
recall), and match boulders across surveys — on any device.

- **Environment / platform setup** lives in the platform guides (linked below);
  this guide is the single source of truth for **workflows, flags, and
  copy‑paste commands**.
- Everything here runs from the **project root** (the folder that contains
  `BoulderCalculator/` and `segmentation/`).

> **Path/notation note.** Commands are written with forward slashes and one
> command per line so they work as‑is on Linux/macOS and, unchanged, in Windows
> PowerShell. In classic `cmd.exe` you may use `\` instead of `/`. Where older
> guides split a command over lines they use `^` (cmd) — here each command is a
> single line so you can paste it directly.

## Contents

1. [Platform & environment setup](#1-platform--environment-setup)
2. [Device selection (CPU vs CUDA)](#2-device-selection-cpu-vs-cuda)
3. [Data layout](#3-data-layout)
4. [Pipeline at a glance](#4-pipeline-at-a-glance)
5. [Step 1 — Build a COCO dataset (crowd‑ignore behavior)](#5-step-1--build-a-coco-dataset)
6. [Step 2 — Offline augmentation](#6-step-2--offline-augmentation)
7. [Step 3 — Train (RGB)](#7-step-3--train-rgb)
8. [Step 3b — Train (RGB+DSM 4‑band)](#8-step-3b--train-rgbdsm-4-band)
9. [Training controls (resume, weights, no‑eval, …)](#9-training-controls)
10. [Step 4 — Inference (tile & full ortho)](#10-step-4--inference)
11. [Step 5 — Evaluation & recall](#11-step-5--evaluation--recall)
12. [Boulder matching (survey‑to‑survey)](#12-boulder-matching)
13. [Per‑script CLI reference](#13-per-script-cli-reference)
14. [End‑to‑end copy‑paste recipes](#14-end-to-end-copy-paste-recipes)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Platform & environment setup

Pick the guide matching your machine, install the environment there, then come
back here for the workflow:

- **Linux / portable (USB):** [`setup/README_PORTABLE.md`](setup/README_PORTABLE.md)
- **Windows (normal/admin):** [`setup/README_WINDOWS.md`](setup/README_WINDOWS.md)
- **Windows (guest / no admin / long‑path issues):** [`setup/README_WINDOWS_GUEST.md`](setup/README_WINDOWS_GUEST.md)
- **Geo-split weekend experiment (RGB+DSM, five region setups):** [`experiments/geo_splits/README.md`](experiments/geo_splits/README.md)
- **Elevation vs local-relief DSM (baseline dual train):** [`experiments/local_relief/README.md`](experiments/local_relief/README.md)

All three install the same stack: a Python 3.10/3.11 environment,
`setup/requirements-training.txt`, GPU PyTorch, and Detectron2 (wheel on Linux,
source build on Windows). Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## 2. Device selection (CPU vs CUDA)

- Training and tile inference **default to CPU** (`--device cpu`). CPU is only
  practical for smoke tests — pass `--device cuda` for any real run.
- `run_boulder_detection.py` (full‑ortho) and the matching inference wrapper
  have their own device handling; see their sections.
- Out‑of‑memory on GPU: lower `--batch-size` (to 1), lower `--image-size`, or
  use `--no-eval` so periodic COCO eval does not spike VRAM.

## 3. Data layout

Relative to the project root:

```text
project_root/
├── BoulderCalculator/            # this repo (scripts/, setup/, Matching/, MODEL_TRAINING.md)
├── 2024/ , 2025/                 # optional full-year DSM GeoTIFFs (RGB+DSM only)
└── segmentation/
    ├── annotations/              # july14_24/25.gpkg (preferred) + tiles_used.txt
    ├── tiling/{24,25}/           # 2000x2000 RGB ortho tiles (.tif)
    ├── tiling_rgb_dsm_{24,25}/   # optional 4-band RGB+DSM tiles
    ├── coco_dataset*/            # generated COCO datasets
    └── training_run*/            # training outputs (model_final.pth, metrics*.json)
```

Annotation defaults resolve newest‑first: **July 14 → July 13 → merged
`july9_input.gpkg`**. Tile row/col ranges come from
`segmentation/annotations/tiles_used.txt`.

## 4. Pipeline at a glance

```text
GPKG + tiles ──▶ gpkg_to_coco.py ──▶ (augment_coco_dataset.py) ──▶ train_boulder_local.py ──▶ model_final.pth
                                                                         │
   RGB+DSM only: build_rgb_dsm_tiles.py ▶ build_coco_rgb_dsm.py ─────────┘
                                                                         ▼
                                    run_tile_inference.py / run_boulder_detection.py
                                                                         ▼
                          metrics_valid.json (AP + recall)  ·  Matching/ (survey-to-survey)
```

## 5. Step 1 — Build a COCO dataset

Converts year‑tagged GPKG polygons + ortho tiles into Detectron2 COCO JSON
(`train/valid/test_annotations.json` + copied tiles). Splits are leakage‑safe
geographic blocks (~111 train / 27 valid / 42 test, plus a buffer).

```bash
python BoulderCalculator/scripts/gpkg_to_coco.py --segmentation-dir segmentation --years 24,25 --output-dir segmentation/coco_dataset_both --min-area-m2 1.0
```

**Deposit / small‑boulder policies** (orthogonal):

| Deposits | Flag |
|----------|------|
| `iscrowd=1` ignore | `--boulder-only` (default) |
| trainable class 2 | `--no-boulder-only` |
| omitted entirely | `--drop-deposits` |

| Small boulders (`--min-area-m2` > 0) | Flag |
|--------------------------------------|------|
| `iscrowd=1` ignore | default |
| omitted entirely | `--drop-below-min-area` |
| all trainable | `--min-area-m2 0` |

Examples: no iscrowd → `--drop-deposits --drop-below-min-area --min-area-m2 1.0`;
iscrowd small only → `--drop-deposits --min-area-m2 1.0`; iscrowd deposits only →
`--boulder-only --drop-below-min-area --min-area-m2 1.0`; iscrowd both →
`--boulder-only --min-area-m2 1.0` (default deposit handling).

Useful flags: `--gpkg a.gpkg:24,b.gpkg:25` (override annotations),
`--roi path` (re‑enable ROI clipping; off by default), `--tiles-used path`,
`--years 24` or `--years 25` (single year),
`--split-config path.yaml` (alternate geographic hold-outs; see
[`experiments/geo_splits/`](experiments/geo_splits/)). Full list: `--help`.

Sanity‑check the polygons before training:

```bash
python BoulderCalculator/scripts/visualize_coco_annotations.py --dataset-dir segmentation/coco_dataset_both --output-dir segmentation/visualizations/coco_gt_both
```

## 6. Step 2 — Offline augmentation

Multiplies the **train** split with exact geometric variants (polygons
transformed to match); valid/test are copied unchanged by default. The default
variant set is the full dihedral group (8×), e.g. ~111 → ~888 train images.
Scale `--max-iter` up accordingly. Pass `--splits train,valid,test` to
offline-augment every split (geo-split weekend experiment).

```bash
python BoulderCalculator/scripts/augment_coco_dataset.py --input-dir segmentation/coco_dataset_both --output-dir segmentation/coco_dataset_both_aug --jitter 0.15
```

- `--jitter 0.15` adds brightness/contrast jitter (RGB only; DSM band 4 is
  preserved). `--variants ...` customizes the transform list; `--seed`
  (default 42) makes it reproducible.

## 7. Step 3 — Train (RGB)

```bash
python BoulderCalculator/scripts/train_boulder_local.py --dataset-dir segmentation/coco_dataset_both_aug --output-dir segmentation/training_run_both --max-iter 10000 --batch-size 2 --num-workers 4 --device cuda
```

Outputs in `--output-dir`: `model_final.pth`, periodic checkpoints,
`metrics.json` (training + periodic eval), and `metrics_valid.json` (final
validation AP **and recall** — see [§11](#11-step-5--evaluation--recall)).
Class names are read from the dataset JSON automatically.

Smoke test: `--max-iter 40 --batch-size 1 --device cpu`.

## 8. Step 3b — Train (RGB+DSM 4‑band)

Stacks a DSM band onto each ortho tile (band order **R, G, B, DSM**) and trains
a 4‑channel Mask R‑CNN. Requires the year DSM GeoTIFFs under `2024/` / `2025/`.

```bash
python BoulderCalculator/scripts/build_rgb_dsm_tiles.py --year 24
python BoulderCalculator/scripts/build_rgb_dsm_tiles.py --year 25
python BoulderCalculator/scripts/build_coco_rgb_dsm.py --source-coco segmentation/coco_dataset_both --tile-dirs segmentation/tiling_rgb_dsm_24 segmentation/tiling_rgb_dsm_25 --output-dir segmentation/coco_dataset_rgb_dsm
python BoulderCalculator/scripts/augment_coco_dataset.py --input-dir segmentation/coco_dataset_rgb_dsm --output-dir segmentation/coco_dataset_rgb_dsm_aug --jitter 0.15
python BoulderCalculator/scripts/train_boulder_local.py --dataset-dir segmentation/coco_dataset_rgb_dsm_aug --output-dir segmentation/training_run_rgb_dsm --four-band --max-iter 10000 --batch-size 2 --num-workers 4 --device cuda
```

- Pass `--four-band` for **both** training and inference. Never mix a 4‑band
  checkpoint with 3‑band images (or vice versa).
- DSM band: `build_rgb_dsm_tiles.py` defaults to `2024/Sites1and2_2024_DSM_30mm.tif`
  / `2025/25IniSouthDSM.tif` (override with `--dsm`). Default `--dsm-mode
  elevation`; use `--dsm-mode local_relief` (with `--relief-radius-m`) for a
  local‑relief band 4. Local-relief tiles default to
  `segmentation/tiling_rgb_dsm_local_relief_{year}` (does not overwrite
  elevation tiles). Full recipe: [`experiments/local_relief/README.md`](experiments/local_relief/README.md).
- Smoke test: `--four-band --max-iter 3 --image-size 800 --device cpu`.

## 9. Training controls

All of these are flags on `train_boulder_local.py`:

| Flag | Default | What it does |
|------|---------|--------------|
| `--dataset-dir` | `segmentation/coco_dataset` | COCO dataset directory |
| `--output-dir` | `segmentation/training_run` | Where weights/metrics are written |
| `--max-iter` | `60` | Training iterations (scale with dataset size) |
| `--batch-size` | `1` | Images per batch (`IMS_PER_BATCH`) |
| `--num-workers` | `2` | Dataloader workers |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--resume` | off | Resume from the latest checkpoint in `--output-dir` |
| `--weights PATH` | model zoo | Init from a local `.pkl/.pth` (skips zoo download / offline) |
| `--four-band` | off | 4‑band RGB+DSM training |
| `--image-size` | `2000` | Square train/test resize (lower for smoke tests) |
| `--no-eval` | off | Skip periodic + final COCO eval (saves VRAM/time) |
| `--no-rich-aug` | off | Disable the coastal aug stack (rotation/flips/scale/photometric); use resize‑only |
| `--checkpoint-period` | 2000 if `max-iter>=1000`, else short-run formula | Write `model_XXXX.pth` every N iters |
| `--eval-period` | 500 if `max-iter>=1000`, else short-run formula | Validation COCO eval every N iters (AP in `metrics.json`) |
| `--early-stop-patience-iters` | `0` (off) | Stop when `--early-stop-metric` has not improved for N iters since the best eval; also writes `model_best.pth`. Typical: `500`–`1000` with `--eval-period 500`. Requires periodic eval. |
| `--early-stop-metric` | `segm/AP` | Event-storage key watched by early stopping |

Notes:
- `--resume` and `--weights` are different: resume continues a run from its
  checkpoints; `--weights` sets the *initial* backbone weights.
- `--no-eval` means no `metrics_valid.json` is written — evaluate later with the
  steps in [§11](#11-step-5--evaluation--recall).
- Early stop cuts a long `--max-iter` once validation AP plateaus (saturation).
  Prefer `model_best.pth` over `model_final.pth` when it fired.

## 10. Step 4 — Inference

### Single tile (with optional GT comparison)

```bash
python BoulderCalculator/scripts/run_tile_inference.py --image segmentation/coco_dataset_both/test/24_Sites1and2_2024_Orthomosaic_14_15.tif --model segmentation/training_run_both/model_final.pth --gt-json segmentation/coco_dataset_both/testing_annotations.json --output-dir segmentation/visualizations/test_inference_both --score-thresh 0.4 --device cuda --class-names "Boulder"
```

- Add `--four-band` for RGB+DSM models/images.
- `--score-thresh` 0.3–0.5 trades precision vs recall.
- `--exclude-classes "BoulderDeposit"` drops GT classes from the comparison.
- Outputs: `*_gt_vs_pred.jpg`, `*_predictions.jpg`, `*_inference_summary.json`.

### Full orthomosaic (sliding window)

```bash
python BoulderCalculator/scripts/run_boulder_detection.py --ortho segmentation/tiling/24/Sites1and2_2024_Orthomosaic.tif --model segmentation/training_run_both/model_final.pth --output-dir segmentation/visualizations/full_ortho --score-thresh 0.7 --window-size 2000 --step-rate 0.25
```

- `--step-rate` is the window overlap stride fraction; `--max-tiles` limits work
  for a quick test; `--epsg` (default 25829) sets the output CRS.

## 11. Step 5 — Evaluation & recall

Training now saves **Average Recall** alongside AP. After a run,
`metrics_valid.json` (and periodic entries in `metrics.json`) contain, for both
`bbox` and `segm`:

- **AP** metrics: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`
- **AR / recall** metrics: `AR1`, `AR10`, `AR100`, `ARs`, `ARm`, `ARl`

All are COCO‑style, scaled 0–100. `AR100` is the headline recall (max 100
detections/image); `ARs/ARm/ARl` are recall for small/medium/large objects.
This comes from `scripts/coco_eval_with_recall.py` (a `COCOEvaluator` subclass);
there is no separate CLI — it is wired into training.

**Per‑detection error analysis** (true/false positives & negatives, precision,
recall, F1‑style totals) with visual overlays:

```bash
python BoulderCalculator/scripts/visualize_detection_errors.py --gt-json segmentation/coco_dataset_both/testing_annotations.json --predictions-dir segmentation/visualizations/test_inference_both --image-dir segmentation/tiling --output-dir segmentation/visualizations/error_analysis --iou-threshold 0.5
```

Writes per‑tile TP/FP/FN images plus `error_analysis_summary.json` whose
`totals` include aggregate `precision` and `recall`.

**Compare multiple training runs** (tables + learning curves from
`metrics_valid.json` / `metrics.json`):

```bash
python BoulderCalculator/scripts/eval_compare_runs.py --segmentation-dir segmentation --geo-prefix training_run_geo_ --output-dir segmentation/eval_compare_geo
```

**Per‑tile AP / AR heatmaps** (and optional overlap merge for sliding‑window
detections). Interactive walkthrough:
[`experiments/eval/compare_training_runs.ipynb`](experiments/eval/compare_training_runs.ipynb);
CLI details: [`experiments/eval/README.md`](experiments/eval/README.md).

```bash
python BoulderCalculator/scripts/eval_per_tile.py --gt-json segmentation/coco_geo_baseline/testing_annotations.json --image-dir segmentation/tiling_rgb_dsm_24 --image-dir segmentation/tiling_rgb_dsm_25 --model segmentation/training_run_geo_baseline/model_final.pth --four-band --device cuda --output-dir segmentation/eval_per_tile_baseline_test
```

(If `coco_geo_*_rgb_dsm` is a full dataset on disk you can pass `--dataset-dir` instead.
Missing tiles are skipped by default — this laptop often has only a subset of 4‑band tiles.)

Add `--merge-iou 0.4` to NMS‑merge duplicates within each tile before scoring.
Pass several `--split-config` YAMLs to average the same per‑tile scores over each
geo‑setup’s test membership (check whether baseline hold‑outs are intrinsically
easier).

**Provenance sidecars.** Dataset and training scripts write JSON next to their
outputs (`dataset_provenance.json`, `training_run_provenance.json`, …) capturing
flags such as jitter, drop/iscrowd modes, `--four-band`, and `--no-rich-aug`.
See [§13](#13-per-script-cli-reference) (`run_provenance.py`).

## 12. Boulder matching

The matcher (in `Matching/`) pairs boulder polygons between two surveys
(e.g. 2024↔2025). It is an **algorithmic** matcher (Hungarian assignment), not a
learned model — there is no matching "training". Full details:
[`Matching/README.md`](Matching/README.md). Run these from `BoulderCalculator/Matching/`.

### Match two existing polygon layers

```bash
python -m matching.cli --before data/before.gpkg --after data/after.gpkg --outdir data/results
```

With DSM volumes and DoD QC:

```bash
python -m matching.cli --before data/before.gpkg --after data/after.gpkg --before-dsm data/before_dsm.tif --after-dsm data/after_dsm.tif --compute-volume --outdir data/results
```

Outputs: `matched_boulders.geojson`, `appeared_boulders.geojson`,
`disappeared_boulders.geojson`, `movement_vectors.geojson`, and (with both DSMs)
a `dod_qc/` folder. Tuning: `--search-radius`, `--min-score`, `--dedupe-iou`,
`--dedupe-centroid-m`, `--dod-lod-m`, `--dod-min-change-m3` (`--no-dedupe` /
`--no-dod-qc` to disable those stages).

### Run model inference on both years, then match

`matching.run_inference_match` runs a 4‑band model on paired opposite‑year
windows and matches the results. It is configurable (defaults target the 42‑tile
hold‑out and **CPU**):

```bash
python -m matching.run_inference_match --model ../../segmentation/training_run_rgb_dsm/model_final.pth --outdir data/run_match --device cuda
```

Key flags: `--project-root`, `--ortho-24/25`, `--dsm-24/25`, `--score-thresh`,
`--image-size`, `--no-volume`, `--rebuild-tiles`, `--max-matches`, `--gui`,
`--no-screenshots`. The `run_training_run_match.sh` wrapper hard‑codes paths and
**CPU** for the `training_run_rgb_dsm_4000` model; edit it or call the module
directly to change device/model/data.

### Visualize matches

```bash
python -m matching.visualize --results-dir data/results --outdir data/screenshots --before data/before.geojson --after data/after.geojson --after-ortho /path/to/after_ortho.tif
```

Add `--gui` for the interactive browser (`n`/`p` to flip matches, `o` toggles
overview zoom).

## 13. Per‑script CLI reference

Defaults in parentheses. Run any script with `--help` for the authoritative list.

**`scripts/gpkg_to_coco.py`** — GPKG + tiles → COCO.
`--segmentation-dir`, `--years` (24,25), `--gpkg`, `--roi` (off),
`--tiles-used`, `--output-dir`, `--min-area-m2` (0.0),
`--drop-below-min-area` (omit small boulders instead of iscrowd),
`--drop-deposits` (omit deposits instead of iscrowd/trainable),
`--boulder-only`/`--no-boulder-only` (on), `--layer`, `--class-field` (Class),
`--train-tiles`/`--valid-tiles`/`--test-tiles`,
`--split-config` (YAML/JSON geographic hold-outs; default = baked-in baseline).

**`scripts/augment_coco_dataset.py`** — offline split aug.
`--input-dir`*, `--output-dir`*, `--variants` (full dihedral 8×), `--jitter`
(0.0), `--seed` (42), `--splits` (default `train`; use `train,valid,test` to
expand every split).

**`scripts/materialize_geo_split_coco.py`** — filter a shared offline-aug pool
into train/valid/test for one `--split-config`. Default `--link-mode auto`
(hard link → symlink → copy). Smoke/weekend use `--link-mode hard` (Windows
guest friendly, no extra disk).

**`scripts/build_rgb_dsm_tiles.py`** — warp DSM onto ortho tiles.
`--year`* (24|25), `--dsm` (year default), `--ortho-dir` (segmentation/tiling),
`--output-dir` (auto), `--dsm-mode` (elevation|local_relief),
`--relief-radius-m` (10.0), `--tile-keys`, `--from-coco`.

**`scripts/build_coco_rgb_dsm.py`** — COCO from 4‑band tiles.
`--source-coco` (segmentation/coco_dataset), `--tile-dirs`* (nargs+),
`--output-dir` (segmentation/coco_dataset_rgb_dsm).

**`scripts/train_boulder_local.py`** — see [§9](#9-training-controls).

**`scripts/run_tile_inference.py`** — single‑tile inference.
`--image`*, `--model`*, `--output-dir`*, `--score-thresh` (0.5), `--device`
(cpu), `--gt-json`, `--class-names` (Boulder), `--exclude-classes`,
`--four-band`, `--image-size` (2000).

**`scripts/run_boulder_detection.py`** — full‑ortho sliding window.
`--ortho`*, `--model`*, `--output-dir`*, `--score-thresh` (0.7),
`--window-size` (2000), `--step-rate` (0.25), `--epsg` (25829),
`--max-tiles`, `--class-names` (Boulder).

**`scripts/visualize_detection_errors.py`** — TP/FP/FN + precision/recall.
`--gt-json`, `--predictions-dir`, `--image-dir`, `--output-dir`*,
`--iou-threshold` (0.5), `--tiles`, `--exclude-classes`.

**`scripts/visualize_coco_annotations.py`** — GT polygon QA.
`--dataset-dir`, `--output-dir`.

**`scripts/coco_eval_with_recall.py`** — internal evaluator (no CLI); adds
`AR1/AR10/AR100/ARs/ARm/ARl` to saved metrics.

**`scripts/run_provenance.py`** — writes / shows sidecars that record dataset and
training flags (`dataset_provenance.json`, `tiling_provenance.json`,
`training_run_provenance.json`). Auto-written by `gpkg_to_coco`,
`augment_coco_dataset`, `build_coco_rgb_dsm`, `build_rgb_dsm_tiles`,
`materialize_geo_split_coco`, and `train_boulder_local`. Inspect with:

```bash
python BoulderCalculator/scripts/run_provenance.py segmentation/coco_dataset_both
python BoulderCalculator/scripts/run_provenance.py segmentation/training_run_geo_baseline
```

**`scripts/eval_compare_runs.py`** — compare many `metrics_valid.json` + plot
eval curves from `metrics.json`. `--runs name=path` (repeatable), or
`--segmentation-dir` + `--geo-prefix` (default `training_run_geo_`),
`--output-dir`, `--metrics` (defaults include AP50/AR100).

**`scripts/eval_per_tile.py`** — per‑tile COCO AP/AR + precision/recall heatmaps.
`--gt-json` + `--predictions-dir`, or `--dataset-dir` + `--split` + `--model`;
`--merge-iou` (optional NMS), `--extents` (GeoJSON for QGIS), `--split-config`
(repeatable; difficulty summary), `--four-band`, `--device`, `--output-dir`*.

**Matching** — see [§12](#12-boulder-matching) and `Matching/README.md`.

`*` = required.

## 14. End‑to‑end copy‑paste recipes

Run from the project root with the environment activated. Windows `cmd.exe`: swap
`/` for `\`.

**RGB, both years, GPU:**

```bash
python BoulderCalculator/scripts/gpkg_to_coco.py --segmentation-dir segmentation --years 24,25 --output-dir segmentation/coco_dataset_both --min-area-m2 1.0
python BoulderCalculator/scripts/augment_coco_dataset.py --input-dir segmentation/coco_dataset_both --output-dir segmentation/coco_dataset_both_aug --jitter 0.15
python BoulderCalculator/scripts/visualize_coco_annotations.py --dataset-dir segmentation/coco_dataset_both --output-dir segmentation/visualizations/coco_gt_both
python BoulderCalculator/scripts/train_boulder_local.py --dataset-dir segmentation/coco_dataset_both_aug --output-dir segmentation/training_run_both --max-iter 10000 --batch-size 2 --num-workers 4 --device cuda
python BoulderCalculator/scripts/run_tile_inference.py --image segmentation/coco_dataset_both/test/24_Sites1and2_2024_Orthomosaic_14_15.tif --model segmentation/training_run_both/model_final.pth --gt-json segmentation/coco_dataset_both/testing_annotations.json --output-dir segmentation/visualizations/test_inference_both --score-thresh 0.4 --device cuda --class-names "Boulder"
```

**RGB+DSM 4‑band, both years, GPU:**

```bash
python BoulderCalculator/scripts/gpkg_to_coco.py --segmentation-dir segmentation --years 24,25 --output-dir segmentation/coco_dataset_both --min-area-m2 1.0
python BoulderCalculator/scripts/build_rgb_dsm_tiles.py --year 24
python BoulderCalculator/scripts/build_rgb_dsm_tiles.py --year 25
python BoulderCalculator/scripts/build_coco_rgb_dsm.py --source-coco segmentation/coco_dataset_both --tile-dirs segmentation/tiling_rgb_dsm_24 segmentation/tiling_rgb_dsm_25 --output-dir segmentation/coco_dataset_rgb_dsm
python BoulderCalculator/scripts/augment_coco_dataset.py --input-dir segmentation/coco_dataset_rgb_dsm --output-dir segmentation/coco_dataset_rgb_dsm_aug --jitter 0.15
python BoulderCalculator/scripts/train_boulder_local.py --dataset-dir segmentation/coco_dataset_rgb_dsm_aug --output-dir segmentation/training_run_rgb_dsm --four-band --max-iter 10000 --batch-size 2 --num-workers 4 --device cuda
python BoulderCalculator/scripts/run_tile_inference.py --image segmentation/coco_dataset_rgb_dsm/test/24_Sites1and2_2024_Orthomosaic_14_15.tif --model segmentation/training_run_rgb_dsm/model_final.pth --gt-json segmentation/coco_dataset_rgb_dsm/testing_annotations.json --output-dir segmentation/visualizations/test_inference_rgb_dsm --score-thresh 0.4 --device cuda --four-band --class-names "Boulder"
```

**CPU smoke test (verify the pipeline end‑to‑end fast):**

```bash
python BoulderCalculator/scripts/gpkg_to_coco.py --segmentation-dir segmentation --years 25 --output-dir segmentation/coco_dataset_smoke --min-area-m2 1.0
python BoulderCalculator/scripts/train_boulder_local.py --dataset-dir segmentation/coco_dataset_smoke --output-dir segmentation/training_run_smoke --max-iter 40 --batch-size 1 --image-size 800 --device cpu
```

## 15. Troubleshooting

| Problem | Fix |
|---------|-----|
| `detectron2` install fails | Match the torch/CUDA wheel (Linux) or source‑build with VS Build Tools (Windows). Use Python 3.10/3.11. See platform guides. |
| CUDA not available | Install the GPU torch build; check `nvidia-smi`; pass `--device cuda`. |
| Out of memory | `--batch-size 1`, lower `--image-size`, or `--no-eval`. |
| 4‑band shape/channel errors | Confirm tiles have `count=4` and pass `--four-band` on both train and infer. |
| Inference ignores DSM | Use a `--four-band` checkpoint and pass `--four-band` at inference. |
| Polygons shifted in QA overlays | GPKG CRS issue; the converter reprojects to EPSG:25829 automatically. |
| No `metrics_valid.json` | You trained with `--no-eval`; evaluate via [§11](#11-step-5--evaluation--recall). |
| Windows "filename too long" | Enable long paths or use `README_WINDOWS_GUEST.md` short‑path tricks. |

See the platform guides for install‑specific issues and the matching README for
matching‑specific troubleshooting.
