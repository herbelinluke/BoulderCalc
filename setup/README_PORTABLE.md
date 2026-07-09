# Portable Boulder Training Guide

Run Detectron2 boulder training on another machine using files copied to a USB stick.

> **Windows users:** this file is the general pipeline reference. For
> step-by-step Windows setup use:
>
> - `README_WINDOWS.md` — normal or admin account
> - `README_WINDOWS_GUEST.md` — guest / restricted account (miniconda, no
>   admin, 260-char path limit workarounds)

---

## What to copy to the USB stick

Copy this folder structure (paths relative to project root, e.g. `D:\boulder_project\`):

```text
boulder_project/
├── BoulderCalculator/
│   └── scripts/
│       ├── gpkg_to_coco.py
│       ├── geojson_tiles_to_coco.py
│       ├── visualize_coco_annotations.py
│       ├── run_tile_inference.py
│       ├── train_boulder_local.py
│       ├── run_boulder_detection.py      # optional: full-ortho inference
│       └── run_volume_extraction.py      # optional: DSM volume step
├── segmentation/
│   ├── README_PORTABLE.md                # this file
│   ├── requirements-training.txt
│   ├── setup_venv.bat                    # Windows
│   ├── setup_venv.sh                     # Linux
│   ├── tile_geojsons/                    # per-tile boulder polygons (required)
│   ├── tiling/                           # 2000×2000 ortho tiles .tif (required)
│   ├── coco_dataset/                     # optional: pre-built COCO JSON
│   └── training_run/                     # optional: existing model_final.pth
```

**Do not copy** `.venv_boulder`, `.gpkg-wal`, or `.gpkg-shm` files.

### Minimum files for training

If someone already has the full ortho tile set (`tiling/*.tif`), they only need:

| Required | Notes |
|----------|-------|
| `tile_geojsons/*.geojson` | One GeoJSON per annotated tile |
| `tiling/<matching>.tif` | Ortho tile for each GeoJSON |
| `BoulderCalculator/scripts/` | Pipeline scripts |
| `segmentation/requirements-training.txt` + setup script | Environment |

They do **not** need `.gpkg` files, `tile_extents/`, or QGIS project files. Those are only for drawing annotations in QGIS. The converter reads GeoJSON + GeoTIFF directly.

| Optional | What it is |
|----------|------------|
| `*.gpkg` | QGIS annotation source (only if re-exporting GeoJSON) |
| `tile_extents/*.gpkg` | QGIS reference footprints — not used by scripts |
| `*.gpkg-wal` / `*.gpkg-shm` | SQLite journal temp files — ignore |

---

## One-time setup on the new machine (no admin)

### Windows

1. Install **Python 3.10 or 3.11** from [python.org](https://www.python.org/downloads/windows/)
   - Check **“Add python.exe to PATH”**
   - Choose **“Install for me only”** (does not require admin)
2. Plug in USB, open **Command Prompt** or **PowerShell**
3. Run:

```bat
cd D:\boulder_project
segmentation\setup_venv.bat D:\boulder_project
call .venv_boulder\Scripts\activate.bat
pip install -r segmentation\requirements-training.txt
```

4. Install **GPU PyTorch + Detectron2** (edit CUDA version to match the machine):

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu124/torch2.4/index.html
```

If no matching Detectron2 wheel exists, see [Detectron2 INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) for your torch/CUDA combo.

5. Verify GPU:

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Linux

```bash
cd /path/to/boulder_project
bash segmentation/setup_venv.sh /path/to/boulder_project
source .venv_boulder/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu124/torch2.4/index.html
```

**Note:** Python 3.14 may lack pre-built Detectron2 wheels. Use **3.10 or 3.11** on the GPU machine for fewer install issues.

---

## Full workflow (commands)

Activate the venv first (`call .venv_boulder\Scripts\activate.bat` on Windows).

### Step 1 (current) — Build both-years boulder-only COCO from GPKG + tiles

Tiles live under `segmentation/tiling/24/` and `segmentation/tiling/25/`.
Use the merged annotations GPKG; deposit polygons (`Class=1`) are dropped by
default. Both ROIs are unioned automatically.

```bat
python BoulderCalculator\scripts\gpkg_to_coco.py ^
  --segmentation-dir segmentation ^
  --years 24,25 ^
  --output-dir segmentation\coco_dataset_both ^
  --min-area-m2 1.0
```

Defaults for `--years 24,25`: `--gpkg july9_input.gpkg`, ROIs =
`roi_24_0709.gpkg` + `roi.shp`. Single-year: `--years 24` or `--years 25`.

Notes:

- Requires `fiona` (in `requirements-training.txt`).
- Annotations are reprojected to EPSG:25829, clipped to the ROI union and each
  tile extent, then converted to pixel coordinates.
- Categories: `1 = Boulder` only when `--boulder-only` (default).
- Copied tile filenames are year-prefixed (`24_...`, `25_...`) so years never
  collide in one dataset folder.
- Splits (~99 tiles): valid/test include hold-outs from both years
  (24: 13_9, 15_15 / 12_8, 14_15, 16_14; 25: 05_33, 08_24 / 04_35, 05_34, 06_29).

`train_boulder_local.py` reads class names from the dataset JSON automatically.
For inference pass `--class-names "Boulder"`.

### Step 1b (optional) — Offline dataset augmentation

The original BoulderCalculator paper pre-augmented the dataset before training
(its notebook loads data with the note "datasets have already been augmented"
and uses a NonAugmentationsTrainer). `augment_coco_dataset.py` reproduces
that: it multiplies the **train** split with exact geometric variants
(flips + 90/180/270 rotations, polygons transformed accordingly) and optional
brightness/contrast jitter. Valid/test are copied unchanged.

```bat
python BoulderCalculator\scripts\augment_coco_dataset.py ^
  --input-dir segmentation\coco_dataset_both ^
  --output-dir segmentation\coco_dataset_both_aug ^
  --jitter 0.15
```

Then train with `--dataset-dir segmentation\coco_dataset_both_aug`. The default
variant set is the full dihedral group (hflip, vflip, rot90/180/270,
transpose, antitranspose) -- every exact orientation of a square tile -- so
the train split grows 8x (e.g. 89 -> 712 images). With more images per epoch,
scale `--max-iter` accordingly (the paper used images x 15 / batch 2; for 712
images that is ~5300 iterations).

### Step 1 (legacy single-year / two-class)

```bat
python BoulderCalculator\scripts\gpkg_to_coco.py ^
  --segmentation-dir segmentation ^
  --years 25 ^
  --no-boulder-only ^
  --gpkg segmentation\annotations\july9_25input.gpkg ^
  --roi segmentation\tile_extents\roi.shp ^
  --output-dir segmentation\coco_dataset_25
```

### Step 1 (legacy) — Build COCO dataset from GeoJSON + tiles

```bat
python BoulderCalculator\scripts\geojson_tiles_to_coco.py ^
  --segmentation-dir segmentation ^
  --output-dir segmentation\coco_dataset
```

(Linux/mac: use `/` paths and `\` line continuations instead of `^`.)

Creates:

```text
segmentation/coco_dataset/
  train/   + train_annotations.json
  valid/   + validation_annotations.json
  test/    + testing_annotations.json
```

**Current dataset:** 20 annotated tiles, ~676 boulder polygons.

| Split | Tiles | Role |
|-------|-------|------|
| **Train** (15) | 04_34, 04_36, 04_37, 05_31, 05_32, 05_35, 06_27, 06_28, 06_30, 07_24, 07_25, 07_26, 07_28, 08_23, 08_25 | Fine-tuning |
| **Valid** (2) | 05_33, 08_24 | Tune threshold / early stopping |
| **Test** (3) | 04_35, 05_34, 06_29 | Held-out evaluation |

To change tiles or splits, edit `TILE_MAP`, `GEOJSON_MAP`, and `TRAIN_TILES` / `VALID_TILES` / `TEST_TILES` in `geojson_tiles_to_coco.py`.

### Step 2 — Visual QA: verify COCO polygons align with imagery

```bat
python BoulderCalculator\scripts\visualize_coco_annotations.py ^
  --dataset-dir segmentation\coco_dataset ^
  --output-dir segmentation\visualizations\coco_gt
```

**Check these files:**

| Output | Purpose |
|--------|---------|
| `visualizations/coco_gt/train/*.jpg` | Train tile overlays |
| `visualizations/coco_gt/valid/*.jpg` | Valid tile overlays |
| `visualizations/coco_gt/test/*.jpg` | Test tile overlays |
| `visualizations/coco_gt/all_tiles_montage.jpg` | All tiles stacked |

Green polygons = COCO segmentation. Yellow boxes = COCO bounding boxes.

### Step 3 — Train

Smoke test (fast):

```bat
python BoulderCalculator\scripts\train_boulder_local.py ^
  --dataset-dir segmentation\coco_dataset ^
  --output-dir segmentation\training_run ^
  --max-iter 40 ^
  --batch-size 1 ^
  --device cuda
```

Recommended GPU run (20 tiles):

```bat
python BoulderCalculator\scripts\train_boulder_local.py ^
  --dataset-dir segmentation\coco_dataset ^
  --output-dir segmentation\training_run ^
  --max-iter 4000 ^
  --batch-size 2 ^
  --num-workers 4 ^
  --device cuda
```

With 15 train images, use more iterations than the old 4-tile set. Start with 4000; increase if loss is still dropping.

Outputs: `segmentation/training_run/model_final.pth`, checkpoints, `metrics.json`, `metrics_valid.json`.

Use `--device cpu` only for quick smoke tests (very slow at scale).

### Step 4 — Inference + visualization on a test tile

```bat
python BoulderCalculator\scripts\run_tile_inference.py ^
  --image segmentation\coco_dataset\test\25IniSouthOrt_06_29.tif ^
  --model segmentation\training_run\model_final.pth ^
  --gt-json segmentation\coco_dataset\testing_annotations.json ^
  --output-dir segmentation\visualizations\test_inference ^
  --score-thresh 0.4 ^
  --device cuda
```

Try `--score-thresh` between 0.3 and 0.5 depending on precision/recall trade-off.

**Check:**

| Output | Purpose |
|--------|---------|
| `*_gt_vs_pred.jpg` | Left = ground truth, right = model predictions |
| `*_predictions.jpg` | Model-only overlay |
| `*_inference_summary.json` | Detection count and scores |

---

## Tile / GeoJSON naming convention

GeoJSON stem `25_XX_YY` maps to ortho tile `25IniSouthOrt_XX_YY.tif`.

| GeoJSON | Ortho tile |
|---------|------------|
| `tile_geojsons/25_05_31_1st.geojson` | `tiling/25IniSouthOrt_05_31.tif` |
| `tile_geojsons/25_04_35_2nd.geojson` | `tiling/25IniSouthOrt_04_35.tif` |
| `tile_geojsons/25_06_29_2nd.geojson` | `tiling/25IniSouthOrt_06_29.tif` |
| … | (see `TILE_MAP` / `GEOJSON_MAP` in script) |

GeoJSON is in **EPSG:4326 (lon/lat)**. The converter reprojects to each tile’s CRS automatically. No `category` field is required — all polygons are treated as class `Boulder`.

The `_1st` / `_2nd` suffix on GeoJSON files is only a version label; the script maps each tile key to the correct filename in `GEOJSON_MAP`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `detectron2` install fails | Match torch/CUDA wheel URL from Detectron2 docs; use Python 3.10/3.11 |
| CUDA not available | Install correct GPU torch build; check `nvidia-smi` |
| Out of memory | `--batch-size 1`, reduce `ROI_HEADS.BATCH_SIZE_PER_IMAGE` in train script |
| Polygons look shifted in QA | Re-export GeoJSON from QGIS; confirm tile TIF matches annotation extent |
| `FileNotFoundError` on geojson | Check `GEOJSON_MAP` matches actual filenames in `tile_geojsons/` |
| `.gpkg-wal` files appear | Normal when QGIS has DB open; ignore for training |

---

## Quick copy-paste checklist (GPU Windows, v2 two-class dataset)

Files needed on the Windows machine (which already has `segmentation/tiling/`
and the BoulderCalculator repo):

- `segmentation/annotations/july9_input.gpkg`
- `segmentation/tile_extents/roi_24_0709.gpkg`
- `segmentation/tile_extents/roi.shp` + sidecars (`roi.shx`, `roi.dbf`, `roi.prj`, `roi.cpg`)
- `segmentation/tiling/24/` and `segmentation/tiling/25/` if not already present
- Updated repo code (`git pull`)
- `pip install fiona` into the existing venv (or re-run the requirements install)

```bat
cd D:\boulder_project
call .venv_boulder\Scripts\activate.bat
pip install fiona

python BoulderCalculator\scripts\gpkg_to_coco.py --segmentation-dir segmentation --years 24,25 --output-dir segmentation\coco_dataset_both --min-area-m2 1.0
python BoulderCalculator\scripts\augment_coco_dataset.py --input-dir segmentation\coco_dataset_both --output-dir segmentation\coco_dataset_both_aug --jitter 0.15
python BoulderCalculator\scripts\visualize_coco_annotations.py --dataset-dir segmentation\coco_dataset_both --output-dir segmentation\visualizations\coco_gt_both
python BoulderCalculator\scripts\train_boulder_local.py --dataset-dir segmentation\coco_dataset_both_aug --output-dir segmentation\training_run_both --max-iter 5000 --batch-size 2 --num-workers 4 --device cuda
python BoulderCalculator\scripts\run_tile_inference.py --image segmentation\coco_dataset_both\test\24_Sites1and2_2024_Orthomosaic_14_15.tif --model segmentation\training_run_both\model_final.pth --gt-json segmentation\coco_dataset_both\testing_annotations.json --output-dir segmentation\visualizations\test_inference_both --score-thresh 0.4 --device cuda --class-names "Boulder"
```

## Quick copy-paste checklist (GPU Windows, legacy 1-class)

```bat
cd D:\boulder_project
call .venv_boulder\Scripts\activate.bat

python BoulderCalculator\scripts\geojson_tiles_to_coco.py --segmentation-dir segmentation --output-dir segmentation\coco_dataset
python BoulderCalculator\scripts\visualize_coco_annotations.py --dataset-dir segmentation\coco_dataset --output-dir segmentation\visualizations\coco_gt
python BoulderCalculator\scripts\train_boulder_local.py --dataset-dir segmentation\coco_dataset --output-dir segmentation\training_run --max-iter 4000 --batch-size 2 --num-workers 4 --device cuda
python BoulderCalculator\scripts\run_tile_inference.py --image segmentation\coco_dataset\test\25IniSouthOrt_06_29.tif --model segmentation\training_run\model_final.pth --gt-json segmentation\coco_dataset\testing_annotations.json --output-dir segmentation\visualizations\test_inference --score-thresh 0.4 --device cuda
```

---

## Scripts reference

| Script | Purpose |
|--------|---------|
| `gpkg_to_coco.py` | GPKG + ROI (.shp/.gpkg) + year tiles → COCO (boulder-only or two-class) |
| `augment_coco_dataset.py` | Offline train-split augmentation (flips/rotations/jitter, paper-style) |
| `geojson_tiles_to_coco.py` | (legacy) GeoJSON + tiles → 1-class COCO train/valid/test |
| `visualize_coco_annotations.py` | Ground-truth polygon QA images |
| `train_boulder_local.py` | Fine-tune Mask R-CNN |
| `run_tile_inference.py` | Single-tile inference + GT comparison |
| `run_boulder_detection.py` | Full ortho sliding-window detection |
| `run_volume_extraction.py` | DSM volume from detections (Python port of MATLAB) |
