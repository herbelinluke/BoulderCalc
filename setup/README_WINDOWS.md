# Windows training guide — normal / admin account

For a Windows machine where you have a normal user (or admin) account and can
install software the usual way. If you are stuck on a **guest account** or
cannot enable long paths, use `README_WINDOWS_GUEST.md` instead.

Assumes the machine already has:

- The BoulderCalculator repo (`git clone` / `git pull`)
- `segmentation/tiling/` (ortho tiles)

Files to copy over for the current (both-years boulder-only) training input:

| File | Purpose |
|------|---------|
| `segmentation/annotations/july9_24input.gpkg` | 2024 annotations (preferred) |
| `segmentation/annotations/july9_25input.gpkg` | 2025 annotations (preferred) |
| `segmentation/annotations/july9_input.gpkg` | Optional merged fallback if per-year files are absent |
| `segmentation/tile_extents/roi_24_0709.gpkg` | 2024 ROI (omit / use `--no-roi` to skip) |
| `segmentation/tile_extents/roi.shp` + `roi.shx`, `roi.dbf`, `roi.prj`, `roi.cpg` | 2025 ROI (all sidecars required) |
| `segmentation/tiling/24/*.tif` | 2024 ortho tiles |
| `segmentation/tiling/25/*.tif` | 2025 ortho tiles |
| `2024/Sites1and2_2024_DSM_30mm.tif` | Optional: for RGB+DSM 4-band training |
| `2025/25IniSouthDSM.tif` | Optional: for RGB+DSM 4-band training |

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

## 2. Full workflow (both years, boulder-only)

Run from the project root. Tiles live under `segmentation\tiling\24\` and
`segmentation\tiling\25\`. One conversion pass loads year-tagged GPKGs (24
polys only label 24 tiles) and optionally unions both ROIs.

```bat
:: Step 1 - per-year GPKGs + both ROIs + 24/25 tiles -> 1-class COCO (~99 tiles).
::   --years 24,25 is the default. --boulder-only (default) drops deposits.
::   Defaults: july9_24input.gpkg:24 + july9_25input.gpkg:25 when present.
::   Skip ROI: add --no-roi   Explicit GPKGs: --gpkg a.gpkg:24,b.gpkg:25
python BoulderCalculator\scripts\gpkg_to_coco.py --segmentation-dir segmentation --years 24,25 --output-dir segmentation\coco_dataset_both --min-area-m2 1.0

:: Step 2 - offline augmentation (8x train split)
python BoulderCalculator\scripts\augment_coco_dataset.py --input-dir segmentation\coco_dataset_both --output-dir segmentation\coco_dataset_both_aug --jitter 0.15

:: Step 3 - visual QA
python BoulderCalculator\scripts\visualize_coco_annotations.py --dataset-dir segmentation\coco_dataset_both --output-dir segmentation\visualizations\coco_gt_both

:: Step 4 - train (scale iters with image count; ~89 train tiles x 8 aug ~= 712 images -> ~5000 iters at batch 2)
python BoulderCalculator\scripts\train_boulder_local.py --dataset-dir segmentation\coco_dataset_both_aug --output-dir segmentation\training_run_both --max-iter 5000 --batch-size 2 --num-workers 4 --device cuda

:: Step 5 - inference (filenames are year-prefixed in the COCO dataset)
python BoulderCalculator\scripts\run_tile_inference.py --image segmentation\coco_dataset_both\test\24_Sites1and2_2024_Orthomosaic_14_15.tif --model segmentation\training_run_both\model_final.pth --gt-json segmentation\coco_dataset_both\testing_annotations.json --output-dir segmentation\visualizations\test_inference_both --score-thresh 0.4 --device cuda --class-names "Boulder"
```

Notes:

- Defaults for `--years 24,25`: per-year GPKGs `july9_24input.gpkg` +
  `july9_25input.gpkg` (year-tagged; fall back to merged `july9_input.gpkg`),
  ROIs = `roi_24_0709.gpkg` + `roi.shp` (unioned). Use `--no-roi` / `--roi none`
  to disable ROI. Single-year: `--years 24` or `--years 25`.
- Copied tile filenames are year-prefixed (`24_...tif`, `25_...tif`) so the
  two years never collide in one dataset folder.
- Hold-outs include both years: valid = 24_13_9, 24_15_15, 25_05_33, 25_08_24;
  test = 24_12_8, 24_14_15, 24_16_14, 25_04_35, 25_05_34, 25_06_29.
- Inference uses `--class-names "Boulder"` (single class).

## 2b. Optional — RGB+DSM 4-band training

Stack the year DSM onto each ortho tile (bands R,G,B,DSM) and train with
`--four-band`. Requires the DSM GeoTIFFs listed above.

```bat
python BoulderCalculator\scripts\build_rgb_dsm_tiles.py --year 24
python BoulderCalculator\scripts\build_rgb_dsm_tiles.py --year 25
python BoulderCalculator\scripts\build_coco_rgb_dsm.py --source-coco segmentation\coco_dataset_both --tile-dirs segmentation\tiling_rgb_dsm_24 segmentation\tiling_rgb_dsm_25 --output-dir segmentation\coco_dataset_rgb_dsm
python BoulderCalculator\scripts\augment_coco_dataset.py --input-dir segmentation\coco_dataset_rgb_dsm --output-dir segmentation\coco_dataset_rgb_dsm_aug --jitter 0.15
python BoulderCalculator\scripts\train_boulder_local.py --dataset-dir segmentation\coco_dataset_rgb_dsm_aug --output-dir segmentation\training_run_rgb_dsm --four-band --max-iter 5000 --batch-size 2 --num-workers 4 --device cuda
python BoulderCalculator\scripts\run_tile_inference.py --image segmentation\coco_dataset_rgb_dsm\test\24_Sites1and2_2024_Orthomosaic_14_15.tif --model segmentation\training_run_rgb_dsm\model_final.pth --gt-json segmentation\coco_dataset_rgb_dsm\testing_annotations.json --output-dir segmentation\visualizations\test_inference_rgb_dsm --score-thresh 0.4 --device cuda --four-band --class-names "Boulder"
```

Pass `--four-band` for both train and inference. Do not mix 3-band and 4-band
checkpoints/images. See `README_PORTABLE.md` Step 3b for DSM modes and smoke-test flags.

## 3. Troubleshooting

| Problem | Fix |
|---------|-----|
| pip error "filename too long" | Enable long paths (above) or use `README_WINDOWS_GUEST.md` short-path tricks |
| Detectron2 install fails | Match the wheel URL to your torch/CUDA; or source-build with VS Build Tools |
| CUDA not available | Install the correct GPU torch build; check `nvidia-smi` |
| Out of memory | `--batch-size 1`, or lower `ROI_HEADS.BATCH_SIZE_PER_IMAGE` in the train script |
| `FileNotFoundError` on tile | Confirm tiles are under `segmentation\tiling\24\` and `25\` with the expected filename patterns |
| Polygons shifted in QA overlays | Confirm the GPKG CRS is declared correctly (the converter reprojects automatically) |
| 4-band shape / channel errors | Confirm GeoTIFFs have `count=4` and you passed `--four-band` on train and infer |
