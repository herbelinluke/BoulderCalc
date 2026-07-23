#!/usr/bin/env python3
"""Build + train RGB+DSM with --dsm-mode local_relief (separate tile dirs).

Run from the project root (parent of BoulderCalculator/ and segmentation/):

  python BoulderCalculator/experiments/local_relief/run_local_relief.py --mode smoke --device cuda
  python BoulderCalculator/experiments/local_relief/run_local_relief.py --mode full --device cuda --batch-size 1 --min-area-m2 1.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]  # BoulderCalculator/
SCRIPTS = REPO_ROOT / "scripts"

TILE_24 = "tiling_rgb_dsm_local_relief_24"
TILE_25 = "tiling_rgb_dsm_local_relief_25"
COCO_RGB = "coco_dataset_both"
COCO_4B = "coco_dataset_local_relief"
COCO_AUG = "coco_dataset_local_relief_aug"
OUT_FULL = "training_run_local_relief"
OUT_SMOKE = "training_run_local_relief_smoke"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--min-area-m2", type=float, default=1.5)
    parser.add_argument("--relief-radius-m", type=float, default=10.0)
    parser.add_argument("--checkpoint-period", type=int, default=None)
    parser.add_argument("--eval-period", type=int, default=None)
    parser.add_argument("--no-rich-aug", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--skip-build-tiles",
        action="store_true",
        help="Reuse existing tiling_rgb_dsm_local_relief_{24,25}",
    )
    parser.add_argument(
        "--skip-coco",
        action="store_true",
        help="Skip gpkg_to_coco if coco_dataset_both already exists",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    root = project_root_from_cwd()
    py = args.python
    seg = root / "segmentation"

    if args.mode == "smoke":
        max_iter = args.max_iter if args.max_iter is not None else 3
        batch_size = args.batch_size if args.batch_size is not None else 1
        image_size = args.image_size if args.image_size is not None else 800
        checkpoint_period = args.checkpoint_period if args.checkpoint_period is not None else 2
        eval_period = args.eval_period if args.eval_period is not None else 2
        out_name = OUT_SMOKE
    else:
        max_iter = args.max_iter if args.max_iter is not None else 5000
        batch_size = args.batch_size if args.batch_size is not None else 1
        image_size = args.image_size if args.image_size is not None else 2000
        checkpoint_period = (
            args.checkpoint_period if args.checkpoint_period is not None else 2000
        )
        eval_period = args.eval_period if args.eval_period is not None else 500
        out_name = OUT_FULL

    print(f"Project root: {root}")
    print(
        f"mode={args.mode} dsm_mode=local_relief relief_radius_m={args.relief_radius_m} "
        f"max_iter={max_iter} batch_size={batch_size} image_size={image_size}"
    )

    coco_rgb = seg / COCO_RGB
    if not args.skip_coco or not (coco_rgb / "train_annotations.json").is_file():
        run(
            [
                py,
                str(SCRIPTS / "gpkg_to_coco.py"),
                "--segmentation-dir",
                str(seg),
                "--years",
                "24,25",
                "--output-dir",
                str(coco_rgb),
                "--min-area-m2",
                str(args.min_area_m2),
            ],
            label="gpkg_to_coco",
        )
    else:
        print(f"[skip] existing {coco_rgb}")

    if not args.skip_build_tiles:
        for year in (24, 25):
            run(
                [
                    py,
                    str(SCRIPTS / "build_rgb_dsm_tiles.py"),
                    "--year",
                    str(year),
                    "--dsm-mode",
                    "local_relief",
                    "--relief-radius-m",
                    str(args.relief_radius_m),
                    "--from-coco",
                    str(coco_rgb),
                ],
                label=f"build_rgb_dsm_tiles year={year} local_relief",
            )
    else:
        print(f"[skip] tile build ({TILE_24}, {TILE_25})")

    coco_4b = seg / COCO_4B
    run(
        [
            py,
            str(SCRIPTS / "build_coco_rgb_dsm.py"),
            "--source-coco",
            str(coco_rgb),
            "--tile-dirs",
            str(seg / TILE_24),
            str(seg / TILE_25),
            "--output-dir",
            str(coco_4b),
        ],
        label="build_coco_rgb_dsm (local_relief)",
    )

    coco_aug = seg / COCO_AUG
    run(
        [
            py,
            str(SCRIPTS / "augment_coco_dataset.py"),
            "--input-dir",
            str(coco_4b),
            "--output-dir",
            str(coco_aug),
            "--jitter",
            "0.15",
        ],
        label="offline aug (train)",
    )

    if args.skip_train:
        print("[skip] train")
        return

    train_cmd = [
        py,
        str(SCRIPTS / "train_boulder_local.py"),
        "--dataset-dir",
        str(coco_aug),
        "--output-dir",
        str(seg / out_name),
        "--four-band",
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
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if args.no_rich_aug:
        train_cmd.append("--no-rich-aug")
    if args.no_eval:
        train_cmd.append("--no-eval")
    run(train_cmd, label=f"train ({args.mode})")
    print(f"\nDone. Outputs: {seg / out_name}", flush=True)


if __name__ == "__main__":
    main()
