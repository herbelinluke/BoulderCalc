# Geo-split weekend @ 512×512 chips

Branch: **`exp/geo-split-weekend`**

Retiles annotated **2000×2000** parents into **512×512** chips (no resampling —
same GSD; last window anchored so edges are covered). Then runs the five geo
setups × selected modalities.

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

**Modalities:** `rgb`, `rgb_dsm` (absolute elevation stretch), `rgb_local_relief`.

### Local-relief encoding

Computed on **padded parents** (not on 512 chips alone), then window-cropped:

| Setting | Default |
|---------|---------|
| Stretch | **fixed** meter clip (not per-tile 2–98%) |
| Clip | **0.5 m**, positive-only (boulder-up residual) |
| Gaussian radius | **10 m** |
| DEM context pad | **60 m** beyond parent (~3σ Gaussian + cliff/flat buffer) |

Percentile stretch is still available via `build_rgb_dsm_tiles.py --relief-stretch percentile` (stats from the padded window). Rebuild parents with `--force` if you change these knobs.

Reuses parent-level YAMLs under [`../geo_splits/`](../geo_splits/) — chips inherit
their parent tile’s train/valid/test membership.

## Layout

| Path | Role |
|------|------|
| `segmentation/tiling_512/{24,25}/` | RGB chips |
| `segmentation/tiling_512_rgb_dsm_{24,25}/` | RGB+elevation chips |
| `segmentation/tiling_512_rgb_dsm_local_relief_{24,25}/` | RGB+local-relief chips |
| `segmentation/coco_geo512_all[_aug]/` | shared RGB pool |
| `segmentation/coco_geo512_all_rgb_dsm[_aug]/` | shared elevation pool |
| `segmentation/coco_geo512_all_rgb_local_relief[_aug]/` | shared local-relief pool |
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

Local-relief smoke:

```bat
python BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py --mode smoke --device cuda --setups baseline --modalities rgb_local_relief
```

## Weekend

Default (RGB + elevation = **10 runs**):

```bat
BoulderCalculator\experiments\geo_splits_512\run_geo512_weekend.bat
```

Local-relief geo pass only (**5 runs**):

```bat
python BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py --mode weekend --device cuda --modalities rgb_local_relief --num-workers 2 --batch-size 4
```

All three modalities (**15 runs**):

```bat
python BoulderCalculator\experiments\geo_splits_512\smoke_geo_splits_512.py --mode weekend --device cuda --modalities rgb,rgb_dsm,rgb_local_relief --num-workers 2 --batch-size 4
```

After a successful smoke, re-runs **skip** existing chips/COCO/aug unless you
pass `--force`.

## Storage (Windows guest)

Hard links keep setup dirs and COCO image trees from duplicating chip bytes.
Approximate **new** free space before a full weekend:

| Asset | ~GB |
|-------|-----|
| RGB chips (~198×16) | 2–3 |
| RGB+DSM chips | 3–4 |
| RGB+local-relief chips | 3–4 |
| Offline 8× aug RGB | ~20 |
| Offline 8× aug RGB+DSM | ~24 |
| Offline 8× aug local-relief | ~24 |
| Training runs (checkpoints) | ~5 per 5-setup modality |
| JSON / misc / slack | ~5 |
| **RGB+DSM weekend (10 runs)** | **~65–75 GB** |
| **+ local-relief geo pass** | **~+30 GB** → **~100–105 GB** |

Parent `tiling/`, `tiling_rgb_dsm_*`, and `tiling_rgb_dsm_local_relief_*` stay on
disk as retile sources. If those are already present, only the incremental
rows above apply.

Keep project root short (`B:\` / `subst`) so hard links stay on one NTFS volume.
