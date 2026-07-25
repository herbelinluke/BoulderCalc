#!/usr/bin/env python3
"""512×512 geo-split weekend: RGB / RGB+DSM / RGB+local-relief × five setups.

Retiles annotated 2000×2000 parents into 512 chips (no resampling), builds
shared hard-linked COCO pools with offline 8×+jitter, materializes each geo
setup, and trains with ``--image-size 512 --batch-size 4 --no-rich-aug``.

Local-relief band 4 defaults (see build_rgb_dsm_tiles.py):
  fixed 0.5 m positive-only clip, morphological opening SE 5 m,
  optional 60 m context pad (not a per-tile 2–98% stretch).

Windows guest defaults: ``--link-mode hard`` everywhere, skip-existing unless
``--force``. Reuses existing geo_splits/*.yaml (parent-level membership; chips
inherit parent split).

Run from project root:

  python BoulderCalculator/experiments/geo_splits_512/smoke_geo_splits_512.py --mode smoke --device cuda
  python BoulderCalculator/experiments/geo_splits_512/smoke_geo_splits_512.py --mode weekend --device cuda
  # local-relief only (5 trains):
  python BoulderCalculator/experiments/geo_splits_512/smoke_geo_splits_512.py \\
      --mode weekend --device cuda --modalities rgb_local_relief
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SETUPS = (
    "baseline",
    "blocks_alt_a",
    "blocks_alt_b",
    "north_south",
    "sporadic_aligned",
)

MODALITIES = ("rgb", "rgb_dsm", "rgb_local_relief")

EXP_DIR = Path(__file__).resolve().parent
GEO_SPLITS_DIR = EXP_DIR.parent / "geo_splits"
REPO_ROOT = EXP_DIR.parents[1]
SCRIPTS = REPO_ROOT / "scripts"

CHIP = 512
# Shared pools (hard-linked into from chip stores / materialized setups).
POOL_RGB = "coco_geo512_all"
POOL_RGB_AUG = "coco_geo512_all_aug"
POOL_4B = "coco_geo512_all_rgb_dsm"
POOL_4B_AUG = "coco_geo512_all_rgb_dsm_aug"
POOL_LR = "coco_geo512_all_rgb_local_relief"
POOL_LR_AUG = "coco_geo512_all_rgb_local_relief_aug"
TILE_RGB = "tiling_512"
TILE_4B_24 = "tiling_512_rgb_dsm_24"
TILE_4B_25 = "tiling_512_rgb_dsm_25"
TILE_LR_24 = "tiling_512_rgb_dsm_local_relief_24"
TILE_LR_25 = "tiling_512_rgb_dsm_local_relief_25"

# Local-relief encoding (computed on padded parents, then chipped).
# Background default = morphological opening (see relief_opening_study/).
RELIEF_BACKGROUND = "opening"
RELIEF_OPENING_SE_M = 5.0
RELIEF_RADIUS_M = 10.0  # gaussian sigma if RELIEF_BACKGROUND=gaussian
RELIEF_CLIP_M = 0.5
RELIEF_STRETCH = "fixed"
RELIEF_CONTEXT_BUFFER_M = 60.0


def is_four_band(modality: str) -> bool:
    return modality in ("rgb_dsm", "rgb_local_relief")


def project_root_from_cwd() -> Path:
    cwd = Path.cwd()
    if (cwd / "BoulderCalculator").is_dir() and (cwd / "segmentation").is_dir():
        return cwd
    return REPO_ROOT.parent


def run(cmd: list[str], *, label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"[FAIL] {label} exited {proc.returncode} after {elapsed:.1f}s")
    print(f"[OK] {label} ({elapsed:.1f}s)", flush=True)


def ensure_parent_rgb_dsm(
    root: Path,
    py: str,
    force: bool,
    *,
    dsm_mode: str = "elevation",
) -> None:
    """Build parent 2000×2000 RGB+band4 tiles needed before 4-band retile."""
    for year in (24, 25):
        cmd = [
            py,
            str(SCRIPTS / "build_rgb_dsm_tiles.py"),
            "--year",
            str(year),
            "--dsm-mode",
            dsm_mode,
        ]
        if dsm_mode == "local_relief":
            cmd.extend(
                [
                    "--relief-background",
                    RELIEF_BACKGROUND,
                    "--relief-opening-se-m",
                    str(RELIEF_OPENING_SE_M),
                    "--relief-radius-m",
                    str(RELIEF_RADIUS_M),
                    "--relief-stretch",
                    RELIEF_STRETCH,
                    "--relief-clip-m",
                    str(RELIEF_CLIP_M),
                    "--relief-positive-only",
                    "--relief-context-buffer-m",
                    str(RELIEF_CONTEXT_BUFFER_M),
                ]
            )
        if force:
            cmd.append("--force")
        run(cmd, label=f"parent build_rgb_dsm_tiles year={year} mode={dsm_mode}")


def retile_all(root: Path, py: str, force: bool, modalities: list[str]) -> None:
    seg = root / "segmentation"
    rgb_cmd = [
        py,
        str(SCRIPTS / "retile_to_chips.py"),
        "--source-dir",
        str(seg / "tiling"),
        "--output-dir",
        str(seg / TILE_RGB),
        "--chip-size",
        str(CHIP),
        "--years",
        "24,25",
        "--segmentation-dir",
        str(seg),
    ]
    if force:
        rgb_cmd.append("--force")
    run(rgb_cmd, label="retile RGB → tiling_512")

    four_band_jobs: list[tuple[str, str, int]] = []
    if "rgb_dsm" in modalities:
        four_band_jobs.extend(
            [
                (f"tiling_rgb_dsm_{24}", TILE_4B_24, 24),
                (f"tiling_rgb_dsm_{25}", TILE_4B_25, 25),
            ]
        )
    if "rgb_local_relief" in modalities:
        four_band_jobs.extend(
            [
                (f"tiling_rgb_dsm_local_relief_{24}", TILE_LR_24, 24),
                (f"tiling_rgb_dsm_local_relief_{25}", TILE_LR_25, 25),
            ]
        )
    for src_name, out_name, year in four_band_jobs:
        cmd = [
            py,
            str(SCRIPTS / "retile_to_chips.py"),
            "--source-dir",
            str(seg / src_name),
            "--output-dir",
            str(seg / out_name),
            "--chip-size",
            str(CHIP),
            "--flat",
            "--year",
            str(year),
            "--segmentation-dir",
            str(seg),
        ]
        if force:
            cmd.append("--force")
        run(cmd, label=f"retile → {out_name}")


def build_modality_pool(
    *,
    root: Path,
    py: str,
    modality: str,
    min_area_m2: float,
    force: bool,
    link_mode: str,
) -> Path:
    """Build shared all-tiles chip COCO + offline aug for one modality."""
    seg = root / "segmentation"
    all_yaml = GEO_SPLITS_DIR / "all_tiles.yaml"
    if modality == "rgb":
        coco = seg / POOL_RGB
        coco_aug = seg / POOL_RGB_AUG
        tile_dirs: list[Path] | None = None
    elif modality == "rgb_dsm":
        coco = seg / POOL_4B
        coco_aug = seg / POOL_4B_AUG
        tile_dirs = [seg / TILE_4B_24, seg / TILE_4B_25]
    elif modality == "rgb_local_relief":
        coco = seg / POOL_LR
        coco_aug = seg / POOL_LR_AUG
        tile_dirs = [seg / TILE_LR_24, seg / TILE_LR_25]
    else:
        raise SystemExit(f"Unknown modality: {modality}")

    four_band = is_four_band(modality)

    if (coco_aug / "train_annotations.json").is_file() and not force:
        print(f"[skip] shared aug pool present: {coco_aug}")
        return coco_aug

    # RGB chip COCO (always; 4-band pool clones annotations then swaps images).
    rgb_coco = seg / POOL_RGB
    gpkg_cmd = [
        py,
        str(SCRIPTS / "gpkg_to_coco.py"),
        "--segmentation-dir",
        str(seg),
        "--tile-dir",
        str(seg / TILE_RGB),
        "--years",
        "24,25",
        "--split-config",
        str(all_yaml),
        "--output-dir",
        str(rgb_coco),
        "--min-area-m2",
        str(min_area_m2),
        "--expand-chips",
        "--link-mode",
        link_mode,
    ]
    if force:
        gpkg_cmd.append("--force")
    run(gpkg_cmd, label="gpkg_to_coco chips → coco_geo512_all")

    if four_band:
        assert tile_dirs is not None
        coco4_cmd = [
            py,
            str(SCRIPTS / "build_coco_rgb_dsm.py"),
            "--source-coco",
            str(rgb_coco),
            "--tile-dirs",
            str(tile_dirs[0]),
            str(tile_dirs[1]),
            "--output-dir",
            str(coco),
            "--link-mode",
            link_mode,
        ]
        if force:
            coco4_cmd.append("--force")
        run(coco4_cmd, label=f"build_coco_rgb_dsm chips → {coco.name}")
        aug_in = coco
    else:
        aug_in = rgb_coco

    aug_cmd = [
        py,
        str(SCRIPTS / "augment_coco_dataset.py"),
        "--input-dir",
        str(aug_in),
        "--output-dir",
        str(coco_aug),
        "--splits",
        "train",
        "--jitter",
        "0.15",
    ]
    if force:
        aug_cmd.append("--force")
    run(aug_cmd, label=f"offline 8x+jitter ({modality})")
    return coco_aug


def train_one(
    *,
    root: Path,
    py: str,
    setup: str,
    modality: str,
    pool_dir: Path,
    max_iter: int,
    batch_size: int,
    image_size: int,
    checkpoint_period: int,
    eval_period: int,
    device: str,
    num_workers: int,
    skip_train: bool,
    no_eval: bool,
    early_stop_patience_iters: int,
    link_mode: str,
) -> None:
    seg = root / "segmentation"
    split_yaml = GEO_SPLITS_DIR / f"{setup}.yaml"
    if not split_yaml.is_file():
        raise SystemExit(f"Missing {split_yaml}")

    out_coco = seg / f"coco_geo512_{setup}_{modality}_from_pool"
    mat_cmd = [
        py,
        str(SCRIPTS / "materialize_geo_split_coco.py"),
        "--pool-dir",
        str(pool_dir),
        "--split-config",
        str(split_yaml),
        "--output-dir",
        str(out_coco),
        "--link-mode",
        link_mode,
    ]
    run(mat_cmd, label=f"materialize {setup}/{modality}")

    if skip_train:
        print(f"[skip] train {setup}/{modality}")
        return

    out_run = seg / f"training_run_geo512_{setup}_{modality}"
    train_cmd = [
        py,
        str(SCRIPTS / "train_boulder_local.py"),
        "--dataset-dir",
        str(out_coco),
        "--output-dir",
        str(out_run),
        "--max-iter",
        str(max_iter),
        "--batch-size",
        str(batch_size),
        "--image-size",
        str(image_size),
        "--checkpoint-period",
        str(checkpoint_period),
        "--eval-period",
        str(eval_period),
        "--num-workers",
        str(num_workers),
        "--device",
        device,
        "--no-rich-aug",
    ]
    if is_four_band(modality):
        train_cmd.append("--four-band")
    if no_eval:
        train_cmd.append("--no-eval")
    elif early_stop_patience_iters > 0:
        train_cmd.extend(
            [
                "--early-stop-patience-iters",
                str(early_stop_patience_iters),
                "--early-stop-metric",
                "segm/AP",
            ]
        )
    run(train_cmd, label=f"train {setup}/{modality}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "weekend"), default="smoke")
    parser.add_argument(
        "--setups",
        default=",".join(SETUPS),
        help="Comma-separated geo setups (default: all five).",
    )
    parser.add_argument(
        "--modalities",
        default="rgb,rgb_dsm",
        help="Comma-separated: rgb, rgb_dsm, rgb_local_relief (default rgb,rgb_dsm).",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=CHIP)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--min-area-m2", type=float, default=1.5)
    parser.add_argument("--checkpoint-period", type=int, default=None)
    parser.add_argument("--eval-period", type=int, default=None)
    parser.add_argument(
        "--early-stop-patience-iters",
        type=int,
        default=None,
        help="Default: 0 smoke / 500 weekend.",
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-retile", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild tiles/COCO/aug even when present.",
    )
    parser.add_argument(
        "--link-mode",
        default="hard",
        choices=("hard", "symlink", "copy", "auto"),
        help="Image placement mode (default hard — Windows guest friendly).",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    setups = [s.strip() for s in args.setups.split(",") if s.strip()]
    unknown = [s for s in setups if s not in SETUPS]
    if unknown:
        raise SystemExit(f"Unknown setups {unknown}; expected {list(SETUPS)}")

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    bad = [m for m in modalities if m not in MODALITIES]
    if bad:
        raise SystemExit(f"Unknown modalities {bad}; expected {list(MODALITIES)}")

    root = project_root_from_cwd()
    py = args.python

    if args.mode == "smoke":
        max_iter = args.max_iter if args.max_iter is not None else 3
        batch_size = args.batch_size if args.batch_size is not None else 2
        checkpoint_period = args.checkpoint_period or 2
        eval_period = args.eval_period or 2
        early_stop = (
            args.early_stop_patience_iters
            if args.early_stop_patience_iters is not None
            else 0
        )
    else:
        max_iter = args.max_iter if args.max_iter is not None else 3000
        batch_size = args.batch_size if args.batch_size is not None else 4
        checkpoint_period = args.checkpoint_period or 1000
        eval_period = args.eval_period or 500
        early_stop = (
            args.early_stop_patience_iters
            if args.early_stop_patience_iters is not None
            else 500
        )

    image_size = args.image_size

    print(f"Project root: {root}")
    print(
        f"mode={args.mode} setups={setups} modalities={modalities} "
        f"chip={CHIP} image_size={image_size} batch_size={batch_size} "
        f"max_iter={max_iter} min_area_m2={args.min_area_m2} "
        f"link_mode={args.link_mode} force={args.force}"
    )

    if not args.skip_retile:
        if "rgb_dsm" in modalities:
            ensure_parent_rgb_dsm(root, py, args.force, dsm_mode="elevation")
        if "rgb_local_relief" in modalities:
            ensure_parent_rgb_dsm(root, py, args.force, dsm_mode="local_relief")
        retile_all(root, py, args.force, modalities)
    else:
        print("[skip] retile")

    pools: dict[str, Path] = {}
    for modality in modalities:
        pools[modality] = build_modality_pool(
            root=root,
            py=py,
            modality=modality,
            min_area_m2=args.min_area_m2,
            force=args.force,
            link_mode=args.link_mode,
        )

    failed: list[str] = []
    for setup in setups:
        for modality in modalities:
            try:
                train_one(
                    root=root,
                    py=py,
                    setup=setup,
                    modality=modality,
                    pool_dir=pools[modality],
                    max_iter=max_iter,
                    batch_size=batch_size,
                    image_size=image_size,
                    checkpoint_period=checkpoint_period,
                    eval_period=eval_period,
                    device=args.device,
                    num_workers=args.num_workers,
                    skip_train=args.skip_train,
                    no_eval=args.no_eval,
                    early_stop_patience_iters=early_stop,
                    link_mode=args.link_mode,
                )
            except SystemExit as exc:
                print(exc, flush=True)
                failed.append(f"{setup}/{modality}")

    if failed:
        raise SystemExit(f"Failed: {', '.join(failed)}")
    print("\nDone. Outputs under segmentation/training_run_geo512_*", flush=True)


if __name__ == "__main__":
    main()
