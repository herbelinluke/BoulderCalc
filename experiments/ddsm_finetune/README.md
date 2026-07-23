# dDSM fine-tune experiment (moved-boulder aware segmentation)

Branch: **`exp/ddsm-finetune`**

**Goal (option A):** fine-tune an existing **RGB+DSM (4-band)** Mask R-CNN by
adding the site-wide **difference DSM** as band 5, so the detector can prefer
boulders that moved (change cues) before Matching.

Matcher DoD QC (option B) stays on the matcher branch — this experiment feeds
**better proposals** into that step.

## Data

| File | Role |
|------|------|
| `25_minus_24_dsm.tif` (project root) | Site-wide ΔDSM (25 − 24), already exported (~8.1G) |
| `segmentation/tiling_rgb_dsm_{24,25}/` | Existing 4-band RGB+DSM tiles (prerequisite) |
| `segmentation/tiling_rgb_dsm_ddsm_{24,25}/` | New 5-band tiles (RGB+DSM+dDSM) |
| 4-band `model_final.pth` | Init weights (`--weights`) |

Band 5 is a **symmetric uint8** stretch of signed Δz (≈127 = no change).

## Labels (v1)

Reuse the same Boulder COCO polygons as the 4-band run. The dDSM channel
supplies change context; the class is still “Boulder”.

**Later (optional):** filter train positives to known movers from Matching GT so
the head is explicitly mover-centric.

## Pipeline

```bat
:: 1. Stack dDSM onto existing 4-band tiles
python BoulderCalculator\scripts\build_rgb_dsm_ddsm_tiles.py --year 24 --from-coco segmentation\coco_dataset_both
python BoulderCalculator\scripts\build_rgb_dsm_ddsm_tiles.py --year 25 --from-coco segmentation\coco_dataset_both

:: 2. COCO pointing at 5-band tiles
python BoulderCalculator\scripts\build_coco_rgb_dsm.py --source-coco segmentation\coco_dataset_both --tile-dirs segmentation\tiling_rgb_dsm_ddsm_24 segmentation\tiling_rgb_dsm_ddsm_25 --output-dir segmentation\coco_dataset_rgb_dsm_ddsm --min-bands 5

:: 3. Offline aug (RGB jitter only; DSM/dDSM untouched)
python BoulderCalculator\scripts\augment_coco_dataset.py --input-dir segmentation\coco_dataset_rgb_dsm_ddsm --output-dir segmentation\coco_dataset_rgb_dsm_ddsm_aug --jitter 0.15

:: 4. Fine-tune from a 4-band checkpoint
python BoulderCalculator\scripts\train_boulder_local.py --dataset-dir segmentation\coco_dataset_rgb_dsm_ddsm_aug --output-dir segmentation\training_run_ddsm_ft --five-band --weights segmentation\training_run_rgb_dsm\model_final.pth --max-iter 3000 --batch-size 1 --num-workers 2 --device cuda --checkpoint-period 1000 --eval-period 500
```

Or the helper:

```bat
python BoulderCalculator\experiments\ddsm_finetune\run_ddsm_finetune.py --mode smoke --weights PATH\to\4band\model_final.pth --device cuda
python BoulderCalculator\experiments\ddsm_finetune\run_ddsm_finetune.py --mode full --weights PATH\to\4band\model_final.pth --device cuda --batch-size 1
```

## Inference

Use `--five-band` once wired on the inference script, or load with the same
5-channel PIXEL_MEAN as training. Prefer running Matching on **mover-biased**
detections from this model, then compare to the plain 4-band + DoD QC path.

## Notes

- Keep elevation 4-band dirs untouched; dDSM tiles are a separate tree.
- Stem init: channels 0–3 copied from the 4-band checkpoint; channel 4 starts
  as a copy of the DSM stem filter (learns Δz during fine-tune).
- Matcher WIP was stashed on `feature/matcher-work` before this branch
  (`git stash list` / `git checkout feature/matcher-work && git stash pop`).
