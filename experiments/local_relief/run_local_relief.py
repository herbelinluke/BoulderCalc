#!/usr/bin/env python3
"""Train elevation DSM vs local-relief DSM (baseline tiles, Windows-safe).

Builds shared RGB COCO (iscrowd deposits + iscrowd below --min-area-m2), then
two 4-band pipelines:

  elevation     tiling_rgb_dsm_{24,25}           → training_run_rgb_dsm
  local_relief  tiling_rgb_dsm_local_relief_{*}  → training_run_local_relief

Defaults match the dual Windows recipe: --no-rich-aug, --min-area-m2 1.5,
jitter 0.15 / 8× offline aug, --batch-size 1, --num-workers 2, --max-iter 3000,
early stop after 500 iters without val ``segm/AP`` improvement.

Run from the project root (parent of BoulderCalculator/ and segmentation/):

  python BoulderCalculator/experiments/local_relief/run_local_relief.py --mode smoke --device cuda
  python BoulderCalculator/experiments/local_relief/run_local_relief.py --mode full --device cuda
  python BoulderCalculator/experiments/local_relief/run_local_relief.py --mode full --models elevation
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

COCO_RGB = "coco_dataset_both"

# (dsm_mode, tile_24, tile_25, coco_4b, coco_aug, out_full, out_smoke)
MODEL_SPECS: dict[str, tuple[str, str, str, str, str, str, str]] = {
    "elevation": (
        "elevation",
        "tiling_rgb_dsm_24",
        "tiling_rgb_dsm_25",
        "coco_dataset_rgb_dsm",
        "coco_dataset_rgb_dsm_aug",
        "training_run_rgb_dsm",
        "training_run_rgb_dsm_smoke",
    ),
    "local_relief": (
        "local_relief",
        "tiling_rgb_dsm_local_relief_24",
        "tiling_rgb_dsm_local_relief_25",
        "coco_dataset_local_relief",
        "coco_dataset_local_relief_aug",
        "training_run_local_relief",
        "training_run_local_relief_smoke",
    ),
}


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


def parse_models(value: str) -> list[str]:
    if value.strip().lower() == "both":
        return ["elevation", "local_relief"]
    names = [p.strip() for p in value.split(",") if p.strip()]
    unknown = [n for n in names if n not in MODEL_SPECS]
    if unknown:
        raise SystemExit(
            f"Unknown --models {unknown!r}; choose from "
            f"{sorted(MODEL_SPECS)} or 'both'"
        )
    if not names:
        raise SystemExit("--models must list at least one model or 'both'")
    return names


def build_one_model(
    *,
    py: str,
    seg: Path,
    coco_rgb: Path,
    model: str,
    args: argparse.Namespace,
    max_iter: int,
    batch_size: int,
    image_size: int,
    checkpoint_period: int,
    eval_period: int,
    out_name: str,
) -> None:
    (
        dsm_mode,
        tile_24,
        tile_25,
        coco_4b_name,
        coco_aug_name,
        _out_full,
        _out_smoke,
    ) = MODEL_SPECS[model]

    if not args.skip_build_tiles:
        for year in (24, 25):
            cmd = [
                py,
                str(SCRIPTS / "build_rgb_dsm_tiles.py"),
                "--year",
                str(year),
                "--dsm-mode",
                dsm_mode,
                "--from-coco",
                str(coco_rgb),
            ]
            if dsm_mode == "local_relief":
                cmd.extend(["--relief-radius-m", str(args.relief_radius_m)])
            if args.force:
                cmd.append("--force")
            run(cmd, label=f"build_rgb_dsm_tiles year={year} {dsm_mode}")
    else:
        print(f"[skip] tile build ({tile_24}, {tile_25})")

    coco_4b = seg / coco_4b_name
    coco4_cmd = [
        py,
        str(SCRIPTS / "build_coco_rgb_dsm.py"),
        "--source-coco",
        str(coco_rgb),
        "--tile-dirs",
        str(seg / tile_24),
        str(seg / tile_25),
        "--output-dir",
        str(coco_4b),
    ]
    if args.force:
        coco4_cmd.append("--force")
    run(coco4_cmd, label=f"build_coco_rgb_dsm ({model})")

    coco_aug = seg / coco_aug_name
    aug_cmd = [
        py,
        str(SCRIPTS / "augment_coco_dataset.py"),
        "--input-dir",
        str(coco_4b),
        "--output-dir",
        str(coco_aug),
        "--jitter",
        str(args.jitter),
    ]
    if args.force:
        aug_cmd.append("--force")
    run(aug_cmd, label=f"offline aug 8x+jitter ({model})")

    if args.skip_train:
        print(f"[skip] train ({model})")
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
    elif args.early_stop_patience_iters > 0:
        train_cmd.extend(
            [
                "--early-stop-patience-iters",
                str(args.early_stop_patience_iters),
                "--early-stop-metric",
                args.early_stop_metric,
            ]
        )
    run(train_cmd, label=f"train {model} ({args.mode})")
    print(f"Outputs: {seg / out_name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--models",
        default="both",
        help="Comma-separated: elevation, local_relief, or both (default).",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=1.5,
        help="Small-boulder cutoff (m²); below → iscrowd (default 1.5).",
    )
    parser.add_argument("--relief-radius-m", type=float, default=10.0)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument("--checkpoint-period", type=int, default=None)
    parser.add_argument("--eval-period", type=int, default=None)
    parser.add_argument(
        "--early-stop-patience-iters",
        type=int,
        default=None,
        help=(
            "Stop when segm/AP has not improved for N iters since best eval. "
            "Default: 0 (smoke) or 500 (full). Pass 0 to disable."
        ),
    )
    parser.add_argument("--early-stop-metric", default="segm/AP")
    parser.add_argument(
        "--rich-aug",
        action="store_true",
        help="Enable online coastal rich augs (default for this experiment: off).",
    )
    parser.add_argument("--no-rich-aug", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--skip-build-tiles",
        action="store_true",
        help="Do not invoke build_rgb_dsm_tiles at all (default: run it; existing tiles are skipped unless --force).",
    )
    parser.add_argument(
        "--skip-coco",
        action="store_true",
        help="Skip gpkg_to_coco entirely (default: run it; existing coco_dataset_both is reused unless --force).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild tiles/COCO/aug even when outputs already exist.",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    # This experiment defaults to offline-only (no rich online augs).
    args.no_rich_aug = not args.rich_aug

    models = parse_models(args.models)
    root = project_root_from_cwd()
    py = args.python
    seg = root / "segmentation"

    if args.mode == "smoke":
        max_iter = args.max_iter if args.max_iter is not None else 3
        batch_size = args.batch_size if args.batch_size is not None else 1
        image_size = args.image_size if args.image_size is not None else 800
        checkpoint_period = args.checkpoint_period if args.checkpoint_period is not None else 2
        eval_period = args.eval_period if args.eval_period is not None else 2
        early_stop = (
            args.early_stop_patience_iters
            if args.early_stop_patience_iters is not None
            else 0
        )
        out_idx = 6  # out_smoke
    else:
        max_iter = args.max_iter if args.max_iter is not None else 3000
        batch_size = args.batch_size if args.batch_size is not None else 1
        image_size = args.image_size if args.image_size is not None else 2000
        checkpoint_period = (
            args.checkpoint_period if args.checkpoint_period is not None else 1000
        )
        eval_period = args.eval_period if args.eval_period is not None else 500
        early_stop = (
            args.early_stop_patience_iters
            if args.early_stop_patience_iters is not None
            else 500
        )
        out_idx = 5  # out_full

    args.early_stop_patience_iters = early_stop

    print(f"Project root: {root}")
    print(
        f"mode={args.mode} models={models} min_area_m2={args.min_area_m2} "
        f"jitter={args.jitter} no_rich_aug={args.no_rich_aug} "
        f"max_iter={max_iter} batch_size={batch_size} num_workers={args.num_workers} "
        f"early_stop_patience_iters={early_stop} force={args.force}"
    )

    coco_rgb = seg / COCO_RGB
    if args.skip_coco and (coco_rgb / "train_annotations.json").is_file() and not args.force:
        print(f"[skip] existing {coco_rgb} (--skip-coco)")
    else:
        # Defaults: --boulder-only (deposits iscrowd) + small boulders iscrowd.
        # gpkg_to_coco itself skips when complete unless --force.
        gpkg_cmd = [
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
        ]
        if args.force:
            gpkg_cmd.append("--force")
        run(
            gpkg_cmd,
            label="gpkg_to_coco (baseline tiles, iscrowd deposits+small)",
        )

    for model in models:
        out_name = MODEL_SPECS[model][out_idx]
        build_one_model(
            py=py,
            seg=seg,
            coco_rgb=coco_rgb,
            model=model,
            args=args,
            max_iter=max_iter,
            batch_size=batch_size,
            image_size=image_size,
            checkpoint_period=checkpoint_period,
            eval_period=eval_period,
            out_name=out_name,
        )

    outs = [str(seg / MODEL_SPECS[m][out_idx]) for m in models]
    print("\nDone. Training outputs:", flush=True)
    for o in outs:
        print(f"  {o}", flush=True)


if __name__ == "__main__":
    main()
