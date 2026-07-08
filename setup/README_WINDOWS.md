# Windows training guide — normal / admin account

For a Windows machine where you have a normal user (or admin) account and can
install software the usual way. If you are stuck on a **guest account** or
cannot enable long paths, use `README_WINDOWS_GUEST.md` instead.

Assumes the machine already has:

- The BoulderCalculator repo (`git clone` / `git pull`)
- `segmentation/tiling/` (ortho tiles)

Files to copy over for the current (v3) training input:

| File | Purpose |
|------|---------|
| `segmentation/annotations/july7_training_input.gpkg` | Annotations (Class: 0 = Boulder, 1 = BoulderDeposit) |
| `segmentation/tile_extents/roi.shp` + `roi.shx`, `roi.dbf`, `roi.prj`, `roi.cpg` | ROI mask (all sidecars required) |
| `2025/25IniSouthDSM.tif` (optional) | Only for hillshade / local-relief input experiments |

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

:: GPU PyTorch + Detectron2 (adjust CUDA version to the machine; check nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu124/torch2.4/index.html
```

If no matching Detectron2 wheel exists for your torch/CUDA combo, build from
source (needs VS Build Tools):

```bat
git clone https://github.com/facebookresearch/detectron2.git
pip install -e detectron2
```

Verify GPU:

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Full workflow

Run from the project root (folder containing `BoulderCalculator\` and
`segmentation\`).

```bat
:: Step 1 - GPKG + ROI + tiles -> two-class COCO dataset (49 tiles).
::   --min-area-m2 1.0 drops Boulder polygons under 1 m2 (whole-geometry area);
::   deposits are never filtered. Omit the flag to keep everything.
python BoulderCalculator\scripts\gpkg_to_coco.py --segmentation-dir segmentation --output-dir segmentation\coco_dataset_v3 --min-area-m2 1.0

:: Step 2 - offline augmentation (8x train split: flips, rotations, transposes + jitter)
python BoulderCalculator\scripts\augment_coco_dataset.py --input-dir segmentation\coco_dataset_v3 --output-dir segmentation\coco_dataset_v3_aug --jitter 0.15

:: Step 3 - visual QA of ground truth
python BoulderCalculator\scripts\visualize_coco_annotations.py --dataset-dir segmentation\coco_dataset_v3 --output-dir segmentation\visualizations\coco_gt_v3

:: Step 4 - train (~2600-3000 iters matches the paper's images x 15 / batch 2 for a 352-image split)
python BoulderCalculator\scripts\train_boulder_local.py --dataset-dir segmentation\coco_dataset_v3_aug --output-dir segmentation\training_run_v3_aug --max-iter 3000 --batch-size 2 --num-workers 4 --device cuda

:: Step 5 - inference + GT comparison on a test tile
python BoulderCalculator\scripts\run_tile_inference.py --image segmentation\coco_dataset_v3\test\25IniSouthOrt_06_29.tif --model segmentation\training_run_v3_aug\model_final.pth --gt-json segmentation\coco_dataset_v3\testing_annotations.json --output-dir segmentation\visualizations\test_inference_v3 --score-thresh 0.4 --device cuda --class-names "Boulder,BoulderDeposit"
```

Notes:

- `gpkg_to_coco.py` defaults: `--gpkg segmentation\annotations\july7_training_input.gpkg`,
  `--roi segmentation\tile_extents\roi.shp`. Tile extents come from the
  GeoTIFFs; no `tile_extents/*.gpkg` needed.
- `train_boulder_local.py` reads class names from the dataset JSON, so it
  works for 1-class and 2-class datasets without flags.
- First training run downloads the COCO base weights (~170 MB) — internet
  needed once.
- Splits: valid = 05_33, 08_24; test = 04_35, 05_34, 06_29; train = the other
  44 tiles. Override with `--train-tiles/--valid-tiles/--test-tiles`.

## 3. Troubleshooting

| Problem | Fix |
|---------|-----|
| pip error "filename too long" | Enable long paths (above) or use `README_WINDOWS_GUEST.md` short-path tricks |
| Detectron2 install fails | Match the wheel URL to your torch/CUDA; or source-build with VS Build Tools |
| CUDA not available | Install the correct GPU torch build; check `nvidia-smi` |
| Out of memory | `--batch-size 1`, or lower `ROI_HEADS.BATCH_SIZE_PER_IMAGE` in the train script |
| Polygons shifted in QA overlays | Confirm the GPKG CRS is declared correctly (the converter reprojects automatically) |
