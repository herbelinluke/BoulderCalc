# Windows training guide — normal / admin account

For a Windows machine where you have a normal user (or admin) account and can
install software the usual way. If you are stuck on a **guest account** or
cannot enable long paths, use `README_WINDOWS_GUEST.md` instead.

Assumes the machine already has:

- The BoulderCalculator repo (`git clone` / `git pull`)
- `segmentation/tiling/` (ortho tiles)

Files to copy over for the current (2024 boulder-only) training input:

| File | Purpose |
|------|---------|
| `segmentation/annotations/july8_24annot.gpkg` | 2024 annotations (missing Class / 0 = Boulder; Class 1 = deposit, dropped) |
| `segmentation/tile_extents/roi_24_0709.gpkg` | 2024 ROI mask (GeoPackage — no shapefile sidecars) |
| `segmentation/tiling/24/*.tif` | 2024 ortho tiles (`Sites1and2_2024_Orthomosaic_RR_CC.tif`) |
| `2024/Sites1and2_2024_DSM_30mm.tif` (optional) | Only for hillshade / local-relief experiments |

---

## 1. One-time setup

### Enable long paths (recommended, needs admin)

Deep pip/conda paths can exceed Windows' 260-character limit. In an **admin**
PowerShell:

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1
git config --global core.longpaths true
```

(Reboot not required for new processes.)

### Python environment

Either python.org Python **3.10/3.11** with a venv, or miniconda:

```bat
conda create -n boulder python=3.11 -y
conda activate boulder
```

### Install packages

```bat
cd C:\path\to\boulder_project
pip install -r BoulderCalculator\setup\requirements-training.txt

:: GPU PyTorch. The nvidia-smi "CUDA Version" is the driver's MAX supported
:: version (backward compatible) -- you do not need to match it exactly.
:: cu130 works on any driver reporting CUDA >= 13.0 and is the combo proven
:: on the Linux box (torch 2.12.1 + cu130 + detectron2 source build).
pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu130
```

Detectron2 has **no official Windows wheels** — build from source (needs VS
Build Tools; run from the "x64 Native Tools Command Prompt" so `cl.exe` is on
PATH):

```bat
git clone https://github.com/facebookresearch/detectron2.git
pip install -e detectron2
```

If the build skips CUDA extensions (no `CUDA_HOME`), that is fine: standard
Mask R-CNN gets its GPU ops from torchvision's precompiled wheel, so GPU
training still works without the CUDA toolkit installed.

Verify GPU:

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Full workflow (2024 boulder-only)

Run from the project root (folder containing `BoulderCalculator\` and
`segmentation\`). Tiles live under `segmentation\tiling\24\`.

```bat
:: Step 1 - GPKG + ROI + 2024 tiles -> 1-class COCO (50 tiles).
::   --boulder-only (default) drops Class=1 deposit polygons.
::   --min-area-m2 1.0 drops Boulder polygons under 1 m2 (whole-geometry area).
python BoulderCalculator\scripts\gpkg_to_coco.py --segmentation-dir segmentation --year 24 --output-dir segmentation\coco_dataset_24 --min-area-m2 1.0

:: Step 2 - offline augmentation (8x train split: flips, rotations, transposes + jitter)
python BoulderCalculator\scripts\augment_coco_dataset.py --input-dir segmentation\coco_dataset_24 --output-dir segmentation\coco_dataset_24_aug --jitter 0.15

:: Step 3 - visual QA of ground truth
python BoulderCalculator\scripts\visualize_coco_annotations.py --dataset-dir segmentation\coco_dataset_24 --output-dir segmentation\visualizations\coco_gt_24

:: Step 4 - train (~2600-3000 iters for an ~360-image augmented split at batch 2)
python BoulderCalculator\scripts\train_boulder_local.py --dataset-dir segmentation\coco_dataset_24_aug --output-dir segmentation\training_run_24 --max-iter 3000 --batch-size 2 --num-workers 4 --device cuda

:: Step 5 - inference + GT comparison on a test tile
python BoulderCalculator\scripts\run_tile_inference.py --image segmentation\coco_dataset_24\test\Sites1and2_2024_Orthomosaic_14_15.tif --model segmentation\training_run_24\model_final.pth --gt-json segmentation\coco_dataset_24\testing_annotations.json --output-dir segmentation\visualizations\test_inference_24 --score-thresh 0.4 --device cuda --class-names "Boulder"
```

Notes:

- Defaults for `--year 24`: `--gpkg july8_24annot.gpkg`, `--roi roi_24_0709.gpkg`,
  tiles under `tiling\24\`. ROI may be `.gpkg` or `.shp`.
- `--boulder-only` is on by default (1-class COCO). Pass `--no-boulder-only`
  only if you want deposits kept as a second class.
- `train_boulder_local.py` reads class names from the dataset JSON automatically.
- Splits (2024): valid = 13_9, 15_15; test = 12_8, 14_15, 16_14; train = the
  other 45 tiles. Override with `--train-tiles/--valid-tiles/--test-tiles`.
- For the older 2025 two-class run: `--year 25 --no-boulder-only` and the
  july7 GPKG / `roi.shp` / `tiling\25\` paths.

## 3. Troubleshooting

| Problem | Fix |
|---------|-----|
| pip error "filename too long" | Enable long paths (above) or use `README_WINDOWS_GUEST.md` short-path tricks |
| Detectron2 install fails | Match the wheel URL to your torch/CUDA; or source-build with VS Build Tools |
| CUDA not available | Install the correct GPU torch build; check `nvidia-smi` |
| Out of memory | `--batch-size 1`, or lower `ROI_HEADS.BATCH_SIZE_PER_IMAGE` in the train script |
| `FileNotFoundError` on tile | Confirm tiles are under `segmentation\tiling\24\` (or `25\`) with the expected filename pattern |
| Polygons shifted in QA overlays | Confirm the GPKG CRS is declared correctly (the converter reprojects automatically) |
