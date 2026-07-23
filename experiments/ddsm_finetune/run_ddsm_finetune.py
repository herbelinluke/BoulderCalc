#!/usr/bin/env python3
"""Build 5-band RGB+DSM+dDSM data and fine-tune from a 4-band checkpoint.

Requires project-root ``25_minus_24_dsm.tif`` and existing
``segmentation/tiling_rgb_dsm_{24,25}`` tiles plus a 4-band ``model_final.pth``.

  python BoulderCalculator/experiments/ddsm_finetune/run_ddsm_finetune.py \\
    --mode smoke --weights segmentation/training_run_rgb_dsm/model_final.pth --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
SCRIPTS = REPO_ROOT / "scripts"


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
    if proc.returncode != 0:
        raise SystemExit(f"[FAIL] {label} exited {proc.returncode}")
    print(f"[OK] {label} ({time.time() - t0:.1f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="4-band model_final.pth to fine-tune from",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-area-m2", type=float, default=1.0)
    parser.add_argument("--ddsm", type=Path, default=Path("25_minus_24_dsm.tif"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    root = project_root_from_cwd()
    py = args.python
    seg = root / "segmentation"
    ddsm = args.ddsm if args.ddsm.is_absolute() else root / args.ddsm
    if not ddsm.is_file():
        raise SystemExit(f"dDSM not found: {ddsm}")

    if args.mode == "smoke":
        max_iter, image_size, ckpt, ev = 3, 800, 2, 2
        out = seg / "training_run_ddsm_ft_smoke"
    else:
        max_iter, image_size, ckpt, ev = 3000, 2000, 1000, 500
        out = seg / "training_run_ddsm_ft"

    coco_rgb = seg / "coco_dataset_both"
    if not (coco_rgb / "train_annotations.json").is_file():
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

    if not args.skip_build:
        for year in (24, 25):
            run(
                [
                    py,
                    str(SCRIPTS / "build_rgb_dsm_ddsm_tiles.py"),
                    "--year",
                    str(year),
                    "--ddsm",
                    str(ddsm),
                    "--from-coco",
                    str(coco_rgb),
                ],
                label=f"build 5-band tiles year={year}",
            )

        coco_5 = seg / "coco_dataset_rgb_dsm_ddsm"
        run(
            [
                py,
                str(SCRIPTS / "build_coco_rgb_dsm.py"),
                "--source-coco",
                str(coco_rgb),
                "--tile-dirs",
                str(seg / "tiling_rgb_dsm_ddsm_24"),
                str(seg / "tiling_rgb_dsm_ddsm_25"),
                "--output-dir",
                str(coco_5),
                "--min-bands",
                "5",
            ],
            label="build_coco 5-band",
        )
        coco_aug = seg / "coco_dataset_rgb_dsm_ddsm_aug"
        run(
            [
                py,
                str(SCRIPTS / "augment_coco_dataset.py"),
                "--input-dir",
                str(coco_5),
                "--output-dir",
                str(coco_aug),
                "--jitter",
                "0.15",
            ],
            label="offline aug",
        )
    else:
        coco_aug = seg / "coco_dataset_rgb_dsm_ddsm_aug"
        if not (coco_aug / "train_annotations.json").is_file():
            raise SystemExit(f"--skip-build but missing {coco_aug}")

    run(
        [
            py,
            str(SCRIPTS / "train_boulder_local.py"),
            "--dataset-dir",
            str(coco_aug),
            "--output-dir",
            str(out),
            "--five-band",
            "--weights",
            str(args.weights),
            "--max-iter",
            str(max_iter),
            "--batch-size",
            str(args.batch_size),
            "--image-size",
            str(image_size),
            "--checkpoint-period",
            str(ckpt),
            "--eval-period",
            str(ev),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
        ],
        label=f"fine-tune five-band ({args.mode})",
    )
    print(f"\nDone → {out}", flush=True)


if __name__ == "__main__":
    main()
