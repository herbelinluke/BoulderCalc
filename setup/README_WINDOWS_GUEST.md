# Windows training guide — guest account / no admin

For a Windows machine where you only have a **guest or restricted account**:
no admin rights, so the registry `LongPathsEnabled` fix is unavailable and the
260-character path limit is in force. Miniconda + developer tools are assumed
to be installed already. The strategy is simple: **keep every path short**.

> **Guest-profile warning:** true Windows Guest accounts may wipe the user
> profile at logout. Keep the project, environment, and especially
> `training_run*/model_final.pth` outputs on a USB / second drive (or
> `C:\Users\Public\`), and copy results off before logging out.

---

## 1. Pick a short base path

Everything (project, conda env, caches) lives under one short root. Good
options, in order of preference:

1. A second drive / USB root: `D:\bp`
2. `C:\Users\Public\bp` (writable by all users on most machines)
3. Your profile root only if the username is short

If you are forced into a deep folder (e.g. a long OneDrive path), map it to a
virtual drive letter — `subst` **does not need admin** and instantly shortens
every path:

```bat
subst B: "C:\Users\longguestusername\Some\Deep\Folder"
B:
```

(`subst` mappings vanish at logout; re-run the command each session.)

Below, `B:` stands for your chosen root. Layout:

```text
B:\
├── BoulderCalculator\        (git clone)
├── segmentation\             (tiling, annotations, tile_extents)
├── env\                      (conda env, created below)
└── cache\                    (pip/conda/torch caches)
```

## 2. Redirect caches to short paths (each session)

pip, conda, and torch cache under `C:\Users\<name>\AppData\...`, which is where
long-path failures usually happen. Redirect them:

```bat
set TMP=B:\cache\tmp
set TEMP=B:\cache\tmp
set PIP_CACHE_DIR=B:\cache\pip
set CONDA_PKGS_DIRS=B:\cache\conda
set TORCH_HOME=B:\cache\torch
set FVCORE_CACHE=B:\cache\fvcore
mkdir B:\cache\tmp B:\cache\pip B:\cache\conda B:\cache\torch B:\cache\fvcore 2>nul
```

(`FVCORE_CACHE` is where Detectron2 downloads the COCO base weights;
`TORCH_HOME` covers torchvision weights.)

Also tell git to tolerate long paths (per-user, no admin needed):

```bat
git config --global core.longpaths true
```

## 3. Conda environment at a short prefix

Create the env **by path** (`-p`), not by name, so it lives under `B:\` instead
of deep inside the guest profile:

```bat
conda create -p B:\env python=3.11 -y
conda activate B:\env
```

Then install:

```bat
cd B:\
pip install --no-cache-dir -r BoulderCalculator\setup\requirements-training.txt

:: nvidia-smi "CUDA Version" is the driver's MAX supported version (backward
:: compatible) -- no need to match exactly. cu130 mirrors the proven Linux
:: setup (torch 2.12.1 + cu130).
pip install --no-cache-dir torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu130
```

Detectron2 has **no official Windows wheels** — build from source. Clone to a
*short* directory and run from the "x64 Native Tools Command Prompt" (so
`cl.exe` is on PATH); re-run the cache `set` commands there first:

```bat
:: Move to source destination
git clone https://github.com/facebookresearch/detectron2.git B:\d2

:: Initialize clean x64 developer environment
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Force CPU extension compilation (torchvision wheel provides the actual GPU ops)
set FORCE_CUDA=0
set DISTUTILS_USE_SDK=1

:: Build and install
pip install --no-cache-dir -e B:\d2 --no-build-isolation
```

If the build skips CUDA extensions (no `CUDA_HOME`), that is fine: standard
Mask R-CNN gets its GPU ops from torchvision's precompiled wheel, so GPU
training works without installing the CUDA toolkit.

Verify GPU:

```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Data files to copy in

| File | Purpose |
|------|---------|
| `segmentation\annotations\july7_training_input.gpkg` | Annotations (Class: 0 = Boulder, 1 = BoulderDeposit) |
| `segmentation\tile_extents\roi.shp` + `roi.shx`, `roi.dbf`, `roi.prj`, `roi.cpg` | ROI mask (all sidecars required) |
| `segmentation\tiling\*.tif` | Ortho tiles (if not already present) |

## 5. Workflow (from `B:\`)

```bat
:: 1. GPKG -> COCO. --min-area-m2 1.0 drops Boulder polygons under 1 m2
::    (whole-geometry area); deposits are never filtered. Omit to keep all.
python BoulderCalculator\scripts\gpkg_to_coco.py --segmentation-dir segmentation --output-dir segmentation\coco_dataset_v3 --min-area-m2 1.0

:: 2. Offline augmentation (8x train split)
python BoulderCalculator\scripts\augment_coco_dataset.py --input-dir segmentation\coco_dataset_v3 --output-dir segmentation\coco_dataset_v3_aug --jitter 0.15

:: 3. GT QA overlays
python BoulderCalculator\scripts\visualize_coco_annotations.py --dataset-dir segmentation\coco_dataset_v3 --output-dir segmentation\visualizations\coco_gt_v3

:: 4. Train (needs internet once for base weights, ~170 MB -> B:\cache\fvcore)
python BoulderCalculator\scripts\train_boulder_local.py --dataset-dir segmentation\coco_dataset_v3_aug --output-dir segmentation\training_run_v3_aug --max-iter 3000 --batch-size 2 --num-workers 2 --device cuda

:: 5. Inference on a test tile
python BoulderCalculator\scripts\run_tile_inference.py --image segmentation\coco_dataset_v3\test\25IniSouthOrt_06_29.tif --model segmentation\training_run_v3_aug\model_final.pth --gt-json segmentation\coco_dataset_v3\testing_annotations.json --output-dir segmentation\visualizations\test_inference_v3 --score-thresh 0.4 --device cuda --class-names "Boulder,BoulderDeposit"
```

Before logging out: copy `segmentation\training_run_v3_aug\` (at minimum
`model_final.pth`, `metrics.json`, `metrics_valid.json`) somewhere permanent.

## 6. Guest-account troubleshooting

| Problem | Fix |
|---------|-----|
| pip: `OSError ... filename too long` / `[WinError 206]` | Paths still too deep — move root closer to a drive letter, use `subst`, re-check the `set` cache redirects ran in *this* terminal |
| conda env activates but wrong python | `conda activate B:\env` (full path, not a name); check `where python` |
| Downloads fail mid-train (weights) | Pre-download once on another machine and copy into `B:\cache\fvcore\detectron2\...`, or re-run — the download resumes |
| `subst` drive disappeared | Re-run `subst B: <path>` — mappings reset every logout |
| Permission denied writing outputs | Guest may not write outside its profile/Public — keep everything under the chosen root |
| Profile wiped after logout | Expected on true Guest accounts — work from USB/second drive and copy results off first |
ProblemFixpip: OSError ... filename too long / [WinError 206]Paths still too deep — move root closer to a drive letter, use subst, re-check the set cache redirects ran in this terminal  conda env activates but wrong pythonconda activate B:\env (full path, not a name); check where python  vcvarsall.bat throws Windows SDK errorDo not specify a fallback version parameter (like 8.1). Run vcvarsall.bat x64 to cleanly inherit your machine's primary SDK.Building detectron2 crashes on DISTUTILS_USE_SDKMake sure you explicitly execute set DISTUTILS_USE_SDK=1 and set FORCE_CUDA=0 right before calling pip install.Downloads fail mid-train (weights)Pre-download once on another machine and copy into B:\cache\fvcore\detectron2\..., or re-run — the download resumes  subst drive disappearedRe-run subst B: <path> — mappings reset every logout  Permission denied writing outputsGuest may not write outside its profile/Public — keep everything under the chosen root  Profile wiped after logoutExpected on true Guest accounts — work from USB/second drive and copy results off first  


Session checklist (every login): `subst` (if used) → `set` cache variables →
`conda activate B:\env` → work from `B:\`.
