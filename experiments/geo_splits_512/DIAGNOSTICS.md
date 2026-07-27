# Geo-split tiling confound tooling

Branch: **`exp/geo-split-weekend`**

Answers whether the earlier **512×512 vs 2000×2000** comparison is about tile
size, or confounded by effective network GSD, empty chips, boundary truncation,
anchor/object-size mismatch, or unequal training exposure.

## Machine roles

| Mark | Where | Needs |
|------|-------|-------|
| `# RUNS ON: Windows` | Guest with data | COCO JSONs, tiles, checkpoints |
| `# RUNS ON: laptop` | This repo machine | Transferred JSON/CSV only |

## Naming (glob — do not hardcode)

| Artifact | Pattern |
|----------|---------|
| COCO | `segmentation/coco_geo512_{setup}_{modality}_from_pool/` |
| Run | `segmentation/training_run_geo512_{setup}_{modality}/` |
| Modalities | `rgb`, `rgb_dsm`, `rgb_local_relief` |
| Setups | `baseline`, `blocks_alt_a`, `blocks_alt_b`, `north_south`, `sporadic_aligned` |
| Preferred ckpt | `model_best.pth` → `model_final.pth` |

## Windows sequence

From project root (folder with `BoulderCalculator/` + `segmentation/`):

```bat
REM 1) Dataset/config diagnostics (no model)
python BoulderCalculator\scripts\geo_split_dataset_diagnostics.py --include-parent-2000

REM 2) Batch eval_per_tile across discovered runs (resumable)
python BoulderCalculator\scripts\eval_geo_split_runs.py --device cuda --split test --include-parent-2000

REM 3) Test metrics: new trains write metrics_test.json automatically.
REM    Already-finished runs: step 2 covers held-out test via eval.

REM 4) Package for email / USB (no checkpoints / tiles)
python BoulderCalculator\scripts\package_geo_split_transfer.py --include-parent-2000
```

## Laptop sequence

```bash
python BoulderCalculator/scripts/analyze_geo_split_transfer.py \
  --package segmentation/transfer_packages/geo_split_transfer_YYYYMMDD_HHMMSS.zip \
  --output-dir segmentation/geo_split_analysis
```

Outputs: `master_table.csv`, `analysis_report.md`.

## Scripts

| Script | Role |
|--------|------|
| `geo_split_run_index.py` | Shared discovery / GSD / size-bucket helpers |
| `geo_split_dataset_diagnostics.py` | Deliverable 1 |
| `eval_per_tile.py` (+ `--overall-metrics`) | Core eval (refactored importable) |
| `eval_geo_split_runs.py` | Deliverable 2 batch driver |
| `train_boulder_local.py` | Deliverable 3: writes `metrics_test.json` |
| `package_geo_split_transfer.py` | Deliverable 4 |
| `analyze_geo_split_transfer.py` | Deliverable 5 |

## Val vs test (important)

- Periodic eval during training uses **`boulder_valid`** only (`metrics.json` curves).
- Post-train still writes **`metrics_valid.json`** (final in-memory weights) — unchanged.
- New: **`metrics_test.json`** from `model_best.pth` on **`boulder_test`**, plus optional
  **`metrics_valid_at_best.json`** for a fair val−test gap. Never averaged into val curves.

Geo-split YAMLs define distinct `valid` and `test` tile lists; materialize keeps
separate annotation files. First Windows run should still confirm
`testing_annotations.json` ≠ `validation_annotations.json` (hash warning is printed).
