#!/usr/bin/env python3
"""Overnight stratified_coastal RGB @ 2000×2000 (two trains).

1. Build COCO with ``stratified_coastal.yaml``, ``--min-area-m2 1.5`` (iscrowd
   for deposits + sub-threshold boulders; default boulder-only).
2. Offline 8× dihedral + ``--jitter 0.15`` on **train only**; valid/test copied
   unaugmented.
3. **Run A (balanced):** resample train (oversample positives ~1.5×; thin
   deposit-heavy / green-hilly empties; keep a coastal-empty fraction) → train.
4. **Run B (control):** same aug dataset, **no** resample → train.

Both use ``--no-rich-aug``, eval every 500 iters, early-stop patience 1000 on
``segm/AP``.

Batch size: auto — 2 if CUDA reports ≥10 GB, else 1 (safe for 2000² + 8 GB-class
cards / CPU). Override with ``--batch-size``.

Example (project root)::

  python BoulderCalculator/experiments/geo_splits/run_stratified_coastal_rgb_overnight.py \\
      --device cuda

  # Skip run B if time is short:
  python .../run_stratified_coastal_rgb_overnight.py --skip-control
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SPLIT_YAML = EXP_DIR / "stratified_coastal.yaml"

SETUP = "stratified_coastal"
COCO_RAW = "coco_geo_stratified_coastal"
COCO_AUG = "coco_geo_stratified_coastal_aug"
COCO_BALANCED = "coco_geo_stratified_coastal_aug_balanced"
RUN_BALANCED = "training_run_geo_stratified_coastal_rgb_balanced"
RUN_CONTROL = "training_run_geo_stratified_coastal_rgb"


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
    print(f"[OK] {label} ({elapsed / 3600:.2f} h)", flush=True)


def pick_batch_size(requested: int | None) -> tuple[int, str]:
    if requested is not None:
        return requested, "cli"
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gb = props.total_memory / (1024**3)
            name = torch.cuda.get_device_name(0)
            if gb >= 10:
                return 2, f"cuda {name} ({gb:.1f} GiB) → batch 2"
            return 1, f"cuda {name} ({gb:.1f} GiB) → batch 1 (2000² VRAM)"
    except Exception as exc:
        return 1, f"torch probe failed ({exc}); batch 1"
    return 1, "no CUDA → batch 1 (CPU / iGPU shared mem)"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "weekend"),
        default="weekend",
        help="smoke: full build + 3-iter trains into *_smoke dirs; weekend: full runs.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--eval-period", type=int, default=None)
    parser.add_argument("--checkpoint-period", type=int, default=None)
    parser.add_argument(
        "--early-stop-patience-iters",
        type=int,
        default=None,
        help="Stop after this many iters without segm/AP improve (weekend default 1000).",
    )
    parser.add_argument("--image-size", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-area-m2", type=float, default=1.5)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument("--link-mode", default="auto", choices=("auto", "hard", "symlink", "copy"))
    parser.add_argument("--force", action="store_true", help="Rebuild COCO/aug/resample.")
    parser.add_argument(
        "--skip-leakage-check",
        action="store_true",
        help=(
            "Pass through to gpkg_to_coco: skip geographic footprint leakage "
            "check (known mild valid↔test cross-year overlap on stratified_coastal)."
        ),
    )
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="Only run balanced (resampled) training.",
    )
    parser.add_argument(
        "--skip-balanced",
        action="store_true",
        help="Only run control (no resample) training.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    # Resample knobs (exposed for overnight tuning).
    parser.add_argument("--positive-oversample", type=float, default=1.5)
    parser.add_argument("--keep-deposit", type=float, default=0.15)
    parser.add_argument("--keep-hilly", type=float, default=0.15)
    parser.add_argument("--keep-empty-coast", type=float, default=0.30)
    parser.add_argument("--keep-empty-other", type=float, default=0.15)
    parser.add_argument("--hilly-row-max", type=int, default=6)
    parser.add_argument("--resample-seed", type=int, default=42)
    args = parser.parse_args()

    if not SPLIT_YAML.is_file():
        raise SystemExit(f"Missing split config: {SPLIT_YAML}")

    if args.mode == "smoke":
        max_iter = 3 if args.max_iter is None else args.max_iter
        eval_period = 2 if args.eval_period is None else args.eval_period
        checkpoint_period = 2 if args.checkpoint_period is None else args.checkpoint_period
        early_stop = 0 if args.early_stop_patience_iters is None else args.early_stop_patience_iters
        run_suffix = "_smoke"
    else:
        max_iter = 5000 if args.max_iter is None else args.max_iter
        eval_period = 500 if args.eval_period is None else args.eval_period
        checkpoint_period = 1000 if args.checkpoint_period is None else args.checkpoint_period
        early_stop = (
            1000 if args.early_stop_patience_iters is None else args.early_stop_patience_iters
        )
        run_suffix = ""

    root = project_root_from_cwd()
    seg = root / "segmentation"
    # Prefer project venv when caller did not pin --python.
    if args.python == sys.executable:
        for cand in (
            root / ".venv_boulder" / "bin" / "python",
            root / "BoulderCalculator" / ".venv_boulder" / "bin" / "python",
        ):
            if cand.is_file():
                py = str(cand)
                break
        else:
            py = args.python
    else:
        py = args.python
    device = pick_device(args.device)
    batch_size, batch_reason = pick_batch_size(args.batch_size)
    if args.mode == "smoke" and args.batch_size is None:
        batch_size = 1
        batch_reason = "smoke default batch 1"

    print(f"Project root: {root}")
    print(f"Python: {py}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}  batch_size={batch_size} ({batch_reason})")
    if device == "cpu":
        print(
            "WARNING: CUDA not available — training Mask R-CNN @ 2000² on CPU "
            "is very slow. Prefer a CUDA machine, or smoke with --max-iter 3 first.",
            flush=True,
        )
    print(
        f"max_iter={max_iter} eval_period={eval_period} "
        f"early_stop={early_stop} image_size={args.image_size}"
    )
    print(f"min_area_m2={args.min_area_m2} jitter={args.jitter} no-rich-aug split={SETUP}")

    coco_raw = seg / COCO_RAW
    coco_aug = seg / COCO_AUG
    coco_bal = seg / COCO_BALANCED
    run_bal = seg / f"{RUN_BALANCED}{run_suffix}"
    run_ctrl = seg / f"{RUN_CONTROL}{run_suffix}"

    # --- 1) Raw geo-split COCO (RGB, 2000 tiles) ---
    gpkg_cmd = [
        py,
        str(SCRIPTS / "gpkg_to_coco.py"),
        "--segmentation-dir",
        str(seg),
        "--years",
        "24,25",
        "--split-config",
        str(SPLIT_YAML),
        "--output-dir",
        str(coco_raw),
        "--min-area-m2",
        str(args.min_area_m2),
        "--link-mode",
        args.link_mode,
    ]
    if args.force:
        gpkg_cmd.append("--force")
    if args.skip_leakage_check:
        gpkg_cmd.append("--skip-leakage-check")
    run(gpkg_cmd, label="gpkg_to_coco stratified_coastal RGB")

    # --- 2) Offline aug train-only ---
    aug_cmd = [
        py,
        str(SCRIPTS / "augment_coco_dataset.py"),
        "--input-dir",
        str(coco_raw),
        "--output-dir",
        str(coco_aug),
        "--splits",
        "train",
        "--jitter",
        str(args.jitter),
    ]
    if args.force:
        aug_cmd.append("--force")
    run(aug_cmd, label="offline 8x+jitter (train only; valid/test copied)")

    # Sanity: hold-out must not contain aug stems.
    for split, ann in (
        ("valid", "validation_annotations.json"),
        ("test", "testing_annotations.json"),
    ):
        data = json.loads((coco_aug / ann).read_text(encoding="utf-8"))
        bad = [
            im["file_name"]
            for im in data.get("images", [])
            if any(
                Path(im["file_name"]).stem.endswith(s)
                for s in (
                    "_hflip",
                    "_vflip",
                    "_rot90",
                    "_rot180",
                    "_rot270",
                    "_transpose",
                    "_antitranspose",
                )
            )
        ]
        if bad:
            raise SystemExit(
                f"Hold-out contamination in {ann}: {len(bad)} aug stems "
                f"(e.g. {bad[0]}). Augment must use --splits train only."
            )
        print(f"[check] {split}: {len(data.get('images', []))} unaugmented images OK")

    # --- 3) Resampled train for run A ---
    if not args.skip_balanced:
        res_cmd = [
            py,
            str(SCRIPTS / "resample_coco_train.py"),
            "--input-dir",
            str(coco_aug),
            "--output-dir",
            str(coco_bal),
            "--seed",
            str(args.resample_seed),
            "--positive-oversample",
            str(args.positive_oversample),
            "--keep-deposit",
            str(args.keep_deposit),
            "--keep-hilly",
            str(args.keep_hilly),
            "--keep-empty-coast",
            str(args.keep_empty_coast),
            "--keep-empty-other",
            str(args.keep_empty_other),
            "--hilly-row-max",
            str(args.hilly_row_max),
            "--link-mode",
            args.link_mode,
        ]
        extents = EXP_DIR / "tile_extents_stratified_coastal.geojson"
        if extents.is_file():
            res_cmd.extend(["--tile-extents", str(extents)])
        if args.force and coco_bal.exists():
            # resample always rewrites; no skip helper
            pass
        run(res_cmd, label="resample train (balanced negatives)")

    def train_one(dataset: Path, out_run: Path, label: str) -> None:
        if args.skip_train:
            print(f"[skip] train {label}")
            return
        cmd = [
            py,
            str(SCRIPTS / "train_boulder_local.py"),
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(out_run),
            "--no-rich-aug",
            "--max-iter",
            str(max_iter),
            "--batch-size",
            str(batch_size),
            "--image-size",
            str(args.image_size),
            "--checkpoint-period",
            str(checkpoint_period),
            "--eval-period",
            str(eval_period),
            "--num-workers",
            str(args.num_workers),
            "--device",
            device,
        ]
        if early_stop and early_stop > 0:
            cmd.extend(
                [
                    "--early-stop-patience-iters",
                    str(early_stop),
                    "--early-stop-metric",
                    "segm/AP",
                ]
            )
        run(cmd, label=label)

    # --- 4) Run A then Run B ---
    if not args.skip_balanced:
        train_one(
            coco_bal,
            run_bal,
            f"train A balanced → {run_bal.name}",
        )
    if not args.skip_control:
        train_one(
            coco_aug,
            run_ctrl,
            f"train B control → {run_ctrl.name}",
        )

    if args.mode == "smoke":
        print(
            "\nSmoke OK.\n"
            f"  Balanced: {run_bal}\n"
            f"  Control:  {run_ctrl}\n"
            "Re-run with --mode weekend for the full train.\n",
            flush=True,
        )
    else:
        print(
            "\nDone.\n"
            f"  Balanced: {run_bal}\n"
            f"  Control:  {run_ctrl}\n"
            "Compare metrics_valid.json / metrics_test.json to prior RGB geo runs.\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
