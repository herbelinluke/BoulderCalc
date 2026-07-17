#!/usr/bin/env python3
"""Smoke-test every geo-split setup (RGB+DSM, offline+jitter, no online augs).

Builds COCO → 4-band remap → offline aug (all splits) → short Detectron2 train
for each YAML under experiments/geo_splits/. Fails fast with the setup id on error.

Run from the project root (parent of BoulderCalculator/ and segmentation/):

  python BoulderCalculator/experiments/geo_splits/smoke_geo_splits.py
  python BoulderCalculator/experiments/geo_splits/smoke_geo_splits.py --setups baseline,sporadic_aligned
  python BoulderCalculator/experiments/geo_splits/smoke_geo_splits.py --skip-train
  python BoulderCalculator/experiments/geo_splits/smoke_geo_splits.py --build-rgb-dsm-tiles

Full weekend training uses run_geo_weekend.bat / --mode weekend.
"""

from __future__ import annotations

import argparse
import json
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

# Repo layout: .../BoulderCalculator/experiments/geo_splits/this_file.py
GEO_SPLITS_DIR = Path(__file__).resolve().parent
REPO_ROOT = GEO_SPLITS_DIR.parents[1]  # BoulderCalculator/
SCRIPTS = REPO_ROOT / "scripts"


def project_root_from_cwd() -> Path:
    """Prefer cwd if it contains BoulderCalculator/ + segmentation/."""
    cwd = Path.cwd()
    if (cwd / "BoulderCalculator").is_dir() and (cwd / "segmentation").is_dir():
        return cwd
    # Fall back: parent of BoulderCalculator (tamucc/)
    return REPO_ROOT.parent


def run(cmd: list[str], *, label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(
            f"[FAIL] {label} exited {proc.returncode} after {elapsed:.1f}s"
        )
    print(f"[OK] {label} ({elapsed:.1f}s)", flush=True)


def ensure_rgb_dsm_tiles(root: Path, py: str, force: bool) -> None:
    """Build full-year RGB+DSM tile sets (all tiles_used) when missing/forced."""
    for year in (24, 25):
        tile_dir = root / "segmentation" / f"tiling_rgb_dsm_{year}"
        has_tif = tile_dir.is_dir() and any(tile_dir.glob("*.tif"))
        if has_tif and not force:
            print(f"[skip] RGB+DSM tiles present: {tile_dir}")
            continue
        run(
            [
                py,
                str(SCRIPTS / "build_rgb_dsm_tiles.py"),
                "--year",
                str(year),
            ],
            label=f"build_rgb_dsm_tiles year={year}",
        )


def missing_four_band_names(coco_dir: Path, tile_dirs: list[Path]) -> list[str]:
    """Return COCO image file_names that are not yet in the 4-band tile dirs."""
    missing: list[str] = []
    for ann_name in (
        "train_annotations.json",
        "validation_annotations.json",
        "testing_annotations.json",
    ):
        data = json.loads((coco_dir / ann_name).read_text())
        for image in data["images"]:
            name = image["file_name"]
            candidates = [name]
            for prefix in ("24_", "25_"):
                if name.startswith(prefix):
                    candidates.append(name[len(prefix) :])
            found = False
            for tile_dir in tile_dirs:
                for cand in candidates:
                    if (tile_dir / cand).is_file():
                        found = True
                        break
                if found:
                    break
            if not found:
                missing.append(name)
    return missing


def ensure_tiles_for_coco(root: Path, py: str, coco_dir: Path, setup: str) -> None:
    """Build any RGB+DSM tiles referenced by coco_dir that are not on disk yet."""
    tile_24 = root / "segmentation" / "tiling_rgb_dsm_24"
    tile_25 = root / "segmentation" / "tiling_rgb_dsm_25"
    tile_24.mkdir(parents=True, exist_ok=True)
    tile_25.mkdir(parents=True, exist_ok=True)
    missing = missing_four_band_names(coco_dir, [tile_24, tile_25])
    if not missing:
        print(f"[skip] all 4-band tiles present for {setup}")
        return
    print(
        f"[{setup}] building {len(missing)} missing RGB+DSM tile(s) via --from-coco",
        flush=True,
    )
    for year in (24, 25):
        run(
            [
                py,
                str(SCRIPTS / "build_rgb_dsm_tiles.py"),
                "--year",
                str(year),
                "--from-coco",
                str(coco_dir),
            ],
            label=f"{setup}: build_rgb_dsm_tiles year={year} --from-coco",
        )
    still = missing_four_band_names(coco_dir, [tile_24, tile_25])
    if still:
        sample = ", ".join(still[:5])
        more = f" (+{len(still) - 5} more)" if len(still) > 5 else ""
        raise SystemExit(
            f"[{setup}] still missing {len(still)} 4-band tile(s) after build: "
            f"{sample}{more}. Check DSM files under 2024/ and 2025/."
        )


def pipeline_one(
    *,
    root: Path,
    py: str,
    setup: str,
    max_iter: int,
    batch_size: int,
    image_size: int,
    checkpoint_period: int,
    eval_period: int,
    device: str,
    num_workers: int,
    skip_train: bool,
    skip_aug: bool,
    min_area_m2: float,
) -> None:
    split_yaml = GEO_SPLITS_DIR / f"{setup}.yaml"
    if not split_yaml.is_file():
        raise SystemExit(f"Missing split config: {split_yaml}")

    coco_rgb = root / "segmentation" / f"coco_geo_{setup}"
    coco_4b = root / "segmentation" / f"coco_geo_{setup}_rgb_dsm"
    coco_aug = root / "segmentation" / f"coco_geo_{setup}_rgb_dsm_aug"
    out_dir = root / "segmentation" / f"training_run_geo_{setup}"
    if max_iter <= 20:
        out_dir = root / "segmentation" / f"training_run_geo_{setup}_smoke"

    run(
        [
            py,
            str(SCRIPTS / "gpkg_to_coco.py"),
            "--segmentation-dir",
            str(root / "segmentation"),
            "--years",
            "24,25",
            "--split-config",
            str(split_yaml),
            "--output-dir",
            str(coco_rgb),
            "--min-area-m2",
            str(min_area_m2),
        ],
        label=f"{setup}: gpkg_to_coco",
    )

    ensure_tiles_for_coco(root, py, coco_rgb, setup)

    tile_24 = root / "segmentation" / "tiling_rgb_dsm_24"
    tile_25 = root / "segmentation" / "tiling_rgb_dsm_25"

    run(
        [
            py,
            str(SCRIPTS / "build_coco_rgb_dsm.py"),
            "--source-coco",
            str(coco_rgb),
            "--tile-dirs",
            str(tile_24),
            str(tile_25),
            "--output-dir",
            str(coco_4b),
        ],
        label=f"{setup}: build_coco_rgb_dsm",
    )

    if not skip_aug:
        run(
            [
                py,
                str(SCRIPTS / "augment_coco_dataset.py"),
                "--input-dir",
                str(coco_4b),
                "--output-dir",
                str(coco_aug),
                "--splits",
                "train,valid,test",
                "--jitter",
                "0.15",
            ],
            label=f"{setup}: augment all splits",
        )
        train_dir = coco_aug
    else:
        train_dir = coco_4b

    if skip_train:
        print(f"[skip] train for {setup}")
        return

    run(
        [
            py,
            str(SCRIPTS / "train_boulder_local.py"),
            "--dataset-dir",
            str(train_dir),
            "--output-dir",
            str(out_dir),
            "--four-band",
            "--no-rich-aug",
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
        ],
        label=f"{setup}: train max_iter={max_iter}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setups",
        type=str,
        default=",".join(SETUPS),
        help=f"Comma-separated setup ids (default: all). Known: {','.join(SETUPS)}",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "weekend"),
        default="smoke",
        help="smoke: short train; weekend: full 5000-iter runs",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-area-m2", type=float, default=1.0)
    parser.add_argument(
        "--build-rgb-dsm-tiles",
        action="store_true",
        help="Build shared tiling_rgb_dsm_24/25 if missing (or rebuild with --force-tiles)",
    )
    parser.add_argument("--force-tiles", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--skip-aug",
        action="store_true",
        help="Skip offline aug (faster path check; not the weekend recipe)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter (default: current)",
    )
    args = parser.parse_args()

    setups = [s.strip() for s in args.setups.split(",") if s.strip()]
    unknown = [s for s in setups if s not in SETUPS]
    if unknown:
        raise SystemExit(f"Unknown setup(s) {unknown}; expected one of {list(SETUPS)}")

    root = project_root_from_cwd()
    py = args.python
    print(f"Project root: {root}")
    print(f"Python: {py}")
    print(f"Setups: {setups}")
    print(f"Mode: {args.mode}")

    if args.build_rgb_dsm_tiles or args.force_tiles:
        ensure_rgb_dsm_tiles(root, py, force=args.force_tiles)

    if args.mode == "smoke":
        max_iter, batch_size, image_size = 3, 1, 800
        checkpoint_period, eval_period = 2, 2
    else:
        max_iter, batch_size, image_size = 5000, 2, 2000
        checkpoint_period, eval_period = 2000, 500

    failed = []
    for setup in setups:
        try:
            pipeline_one(
                root=root,
                py=py,
                setup=setup,
                max_iter=max_iter,
                batch_size=batch_size,
                image_size=image_size,
                checkpoint_period=checkpoint_period,
                eval_period=eval_period,
                device=args.device,
                num_workers=args.num_workers,
                skip_train=args.skip_train,
                skip_aug=args.skip_aug,
                min_area_m2=args.min_area_m2,
            )
        except SystemExit as exc:
            print(exc, flush=True)
            failed.append(setup)
            # Fail fast for smoke; weekend also stops so a bad setup doesn't burn the night
            break

    if failed:
        raise SystemExit(f"Stopped after failure(s): {failed}")
    print("\nAll requested setups completed OK.", flush=True)


if __name__ == "__main__":
    main()
