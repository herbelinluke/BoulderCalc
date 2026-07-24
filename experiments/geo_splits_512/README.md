# Geo-split weekend @ 512×512 chips (RGB + RGB+DSM)

Branch: **`exp/geo-split-weekend`**

Retiles annotated **2000×2000** parents into **512×512** chips (no resampling —
same GSD; last window anchored so edges are covered). Then runs the usual five
geo setups × **two modalities** = **10 training runs**.

| Knob | Value |
|------|-------|
| Chip size / train resize | 512 (no upscale) |
| `--min-area-m2` | 1.5 (iscrowd below) |
| Deposits / small | iscrowd (defaults) |
| Online rich augs | **off** |
| Offline aug | dihedral **8×** + `--jitter 0.15` |
| Batch size | **4** (weekend) / 2 (smoke) |
| Max iter | 3000 weekend (early-stop patience 500) |
| Image links | **`--link-mode hard`** (Windows guest) |

Reuses parent-level YAMLs under [`../geo_splits/`](../geo_splits/) — chips inherit
their parent tile’s train/valid/test membership.

## Layout

| Path | Role |
|------|------|
| `segmentation/tiling_512/{24,25}/` | RGB chips |
| `segmentation/tiling_512_rgb_dsm_{24,25}/` | 4-band chips |
| `segmentation/coco_geo512_all[_aug]/` | shared RGB pool |
| `segmentation/coco_geo512_all_rgb_dsm[_aug]/` | shared 4-band pool |
| `segmentation/coco_geo512_{setup}_{modality}_from_pool/` | hard-linked setups |
| `segmentation/training_run_geo512_{setup}_{modality}/` | weights + metrics |

## Smoke

```bat
python BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py --mode smoke --device cuda --num-workers 2
```

Or `smoke_geo_splits_512.bat`. Prefer one setup first:

```bat
python BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py --mode smoke --device cuda --setups baseline --modalities rgb
```

## Weekend (10 runs)

```bat
BoulderCalculator\experiments\geo_splits_512\run_geo512_weekend.bat
```

After a successful smoke, re-runs **skip** existing chips/COCO/aug unless you
pass `--force`.

## Storage (Windows guest)

Hard links keep setup dirs and COCO image trees from duplicating chip bytes.
Approximate **new** free space to leave before a full weekend (both modalities,
8× aug, 10 runs):

| Asset | ~GB |
|-------|-----|
| RGB chips (~198×16) | 2–3 |
| RGB+DSM chips | 3–4 |
| Offline 8× aug RGB | ~20 |
| Offline 8× aug RGB+DSM | ~24 |
| 10 training runs (checkpoints) | ~10 |
| JSON / misc / slack | ~5 |
| **Total free to have** | **~65–75 GB** |

Parent `tiling/` and `tiling_rgb_dsm_*` are still needed as retile sources (not
deleted). If those are already on disk, only the table above is incremental.

Keep project root short (`B:\` / `subst`) so hard links stay on one NTFS volume.
