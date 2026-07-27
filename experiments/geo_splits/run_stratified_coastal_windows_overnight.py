#!/usr/bin/env python3
"""Windows overnight: stratified_coastal @ 2000×2000 — RGB A/B + local-relief A.

Pipeline
--------
1. RGB COCO via ``stratified_coastal.yaml``, ``--min-area-m2 1.5`` (iscrowd for
   deposits + sub-threshold boulders).
2. Offline 8× + ``--jitter 0.15`` on **train only**; valid/test stay unaugmented.
3. **Run A (RGB balanced):** resample train → train ``--no-rich-aug``.
4. **Run B (RGB control):** same aug dataset, no resample → train.
5. Build **local-relief** parent tiles (opening SE 5 m, fixed 0.5 m clip, …).
6. Clone RGB COCO → 4-band local-relief images; aug train-only; **same resample
   knobs as Run A** → **Run C (local-relief balanced)** with ``--four-band``.

Designed for a Windows box with admin + CUDA. Batch size auto: 2 if VRAM ≥10 GiB
else 1. Eval every 500; early-stop patience 1000 on ``segm/AP``.

**Smoke first** (builds COCO / local-relief / aug / resample + 3-iter trains)::

  python ...\\run_stratified_coastal_windows_overnight.py --mode smoke --device cuda

  BoulderCalculator\\experiments\\geo_splits\\smoke_stratified_coastal_windows.bat

Full overnight (after smoke succeeds)::

  python BoulderCalculator\\experiments\\geo_splits\\run_stratified_coastal_windows_overnight.py --device cuda

Or::

  BoulderCalculator\\experiments\\geo_splits\\run_stratified_coastal_windows_overnight.bat
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

# RGB
COCO_RGB = "coco_geo_stratified_coastal"
COCO_RGB_AUG = "coco_geo_stratified_coastal_aug"
COCO_RGB_BAL = "coco_geo_stratified_coastal_aug_balanced"
RUN_RGB_BAL = "training_run_geo_stratified_coastal_rgb_balanced"
RUN_RGB_CTRL = "training_run_geo_stratified_coastal_rgb"

# Local-relief (same geographic split + Run-A resample policy)
COCO_LR = "coco_geo_stratified_coastal_rgb_local_relief"
COCO_LR_AUG = "coco_geo_stratified_coastal_rgb_local_relief_aug"
COCO_LR_BAL = "coco_geo_stratified_coastal_rgb_local_relief_aug_balanced"
RUN_LR_BAL = "training_run_geo_stratified_coastal_rgb_local_relief_balanced"

TILE_LR_24 = "tiling_rgb_dsm_local_relief_24"
TILE_LR_25 = "tiling_rgb_dsm_local_relief_25"

# Local-relief encoding (match geo_splits_512 / relief_opening_study defaults)
RELIEF_BACKGROUND = "opening"
RELIEF_OPENING_SE_M = 5.0
RELIEF_RADIUS_M = 10.0
RELIEF_CLIP_M = 0.5
RELIEF_STRETCH = "fixed"
RELIEF_CONTEXT_BUFFER_M = 60.0

AUG_SUFFIXES = (
    "_hflip",
    "_vflip",
    "_rot90",
    "_rot180",
    "_rot270",
    "_transpose",
    "_antitranspose",
)


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
    return 1, "no CUDA → batch 1"


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


def assert_holdout_unaugmented(coco_dir: Path) -> None:
    for split, ann in (
        ("valid", "validation_annotations.json"),
        ("test", "testing_annotations.json"),
    ):
        path = coco_dir / ann
        data = json.loads(path.read_text(encoding="utf-8"))
        bad = [
            im["file_name"]
            for im in data.get("images", [])
            if any(Path(im["file_name"]).stem.endswith(s) for s in AUG_SUFFIXES)
        ]
        if bad:
            raise SystemExit(
                f"Hold-out contamination in {path}: {len(bad)} aug stems "
                f"(e.g. {bad[0]})."
            )
        print(f"[check] {coco_dir.name}/{split}: {len(data.get('images', []))} unaug OK")


def resample_cmd(
    py: str,
    inp: Path,
    out: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        py,
        str(SCRIPTS / "resample_coco_train.py"),
        "--input-dir",
        str(inp),
        "--output-dir",
        str(out),
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


def train_cmd(
    *,
    py: str,
    dataset: Path,
    out_run: Path,
    args: argparse.Namespace,
    device: str,
    batch_size: int,
    four_band: bool,
) -> list[str]:
    cmd = [
        py,
        str(SCRIPTS / "train_boulder_local.py"),
        "--dataset-dir",
        str(dataset),
        "--output-dir",
        str(out_run),
        "--no-rich-aug",
        "--max-iter",
        str(args.max_iter),
        "--batch-size",
        str(batch_size),
        "--image-size",
        str(args.image_size),
        "--checkpoint-period",
        str(args.checkpoint_period),
        "--eval-period",
        str(args.eval_period),
        "--num-workers",
        str(args.num_workers),
        "--device",
        device,
    ]
    if args.early_stop_patience_iters and args.early_stop_patience_iters > 0:
        cmd.extend(
            [
                "--early-stop-patience-iters",
                str(args.early_stop_patience_iters),
                "--early-stop-metric",
                "segm/AP",
            ]
        )
    if four_band:
        cmd.append("--four-band")
    return cmd


def ensure_local_relief_tiles(root: Path, py: str, force: bool) -> None:
    for year in (24, 25):
        cmd = [
            py,
            str(SCRIPTS / "build_rgb_dsm_tiles.py"),
            "--year",
            str(year),
            "--dsm-mode",
            "local_relief",
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
        if force:
            cmd.append("--force")
        run(cmd, label=f"build_rgb_dsm_tiles year={year} local_relief")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "weekend"),
        default="weekend",
        help="smoke: full build + 3-iter trains into *_smoke dirs; weekend: full runs.",
    )
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--eval-period", type=int, default=None)
    parser.add_argument("--checkpoint-period", type=int, default=None)
    parser.add_argument("--early-stop-patience-iters", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-area-m2", type=float, default=1.5)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument(
        "--link-mode",
        default="hard",
        choices=("auto", "hard", "symlink", "copy"),
        help="Default hard (Windows guest / admin friendly).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-rgb-balanced", action="store_true")
    parser.add_argument("--skip-rgb-control", action="store_true")
    parser.add_argument(
        "--skip-local-relief",
        action="store_true",
        help="Skip tile build + Run C (local-relief balanced).",
    )
    parser.add_argument("--python", default=sys.executable)
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

    # Mode defaults (CLI overrides win when explicitly set).
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

    # Stash resolved train knobs back onto args for train_cmd helpers.
    args.max_iter = max_iter
    args.eval_period = eval_period
    args.checkpoint_period = checkpoint_period
    args.early_stop_patience_iters = early_stop

    root = project_root_from_cwd()
    seg = root / "segmentation"
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
    print(
        f"max_iter={args.max_iter} eval={args.eval_period} "
        f"early_stop={args.early_stop_patience_iters} image_size={args.image_size}"
    )
    print(
        f"min_area_m2={args.min_area_m2} jitter={args.jitter} "
        f"no-rich-aug split={SETUP} link_mode={args.link_mode}"
    )

    coco_rgb = seg / COCO_RGB
    coco_rgb_aug = seg / COCO_RGB_AUG
    coco_rgb_bal = seg / COCO_RGB_BAL
    run_rgb_bal = seg / f"{RUN_RGB_BAL}{run_suffix}"
    run_rgb_ctrl = seg / f"{RUN_RGB_CTRL}{run_suffix}"
    run_lr_bal = seg / f"{RUN_LR_BAL}{run_suffix}"

    # --- RGB COCO ---
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
        str(coco_rgb),
        "--min-area-m2",
        str(args.min_area_m2),
        "--link-mode",
        args.link_mode,
    ]
    if args.force:
        gpkg_cmd.append("--force")
    run(gpkg_cmd, label="gpkg_to_coco stratified_coastal RGB")

    aug_rgb = [
        py,
        str(SCRIPTS / "augment_coco_dataset.py"),
        "--input-dir",
        str(coco_rgb),
        "--output-dir",
        str(coco_rgb_aug),
        "--splits",
        "train",
        "--jitter",
        str(args.jitter),
    ]
    if args.force:
        aug_rgb.append("--force")
    run(aug_rgb, label="RGB offline 8x+jitter (train only)")
    assert_holdout_unaugmented(coco_rgb_aug)

    if not args.skip_rgb_balanced:
        run(
            resample_cmd(py, coco_rgb_aug, coco_rgb_bal, args),
            label="RGB resample train (Run A policy)",
        )

    def do_train(dataset: Path, out_run: Path, label: str, *, four_band: bool) -> None:
        if args.skip_train:
            print(f"[skip] train {label}")
            return
        run(
            train_cmd(
                py=py,
                dataset=dataset,
                out_run=out_run,
                args=args,
                device=device,
                batch_size=batch_size,
                four_band=four_band,
            ),
            label=label,
        )

    # --- Train RGB A / B ---
    if not args.skip_rgb_balanced:
        do_train(
            coco_rgb_bal,
            run_rgb_bal,
            f"train A RGB balanced → {run_rgb_bal.name}",
            four_band=False,
        )
    if not args.skip_rgb_control:
        do_train(
            coco_rgb_aug,
            run_rgb_ctrl,
            f"train B RGB control → {run_rgb_ctrl.name}",
            four_band=False,
        )

    # --- Local-relief tiles + Run C (same as A) ---
    if not args.skip_local_relief:
        ensure_local_relief_tiles(root, py, args.force)
        tile_24 = seg / TILE_LR_24
        tile_25 = seg / TILE_LR_25
        if not tile_24.is_dir() or not tile_25.is_dir():
            raise SystemExit(
                f"Missing local-relief tiling dirs: {tile_24} / {tile_25}"
            )

        coco_lr = seg / COCO_LR
        coco_lr_aug = seg / COCO_LR_AUG
        coco_lr_bal = seg / COCO_LR_BAL

        lr_coco_cmd = [
            py,
            str(SCRIPTS / "build_coco_rgb_dsm.py"),
            "--source-coco",
            str(coco_rgb),
            "--tile-dirs",
            str(tile_24),
            str(tile_25),
            "--output-dir",
            str(coco_lr),
            "--link-mode",
            args.link_mode,
        ]
        if args.force:
            lr_coco_cmd.append("--force")
        run(lr_coco_cmd, label="build_coco_rgb_dsm → stratified local-relief")

        aug_lr = [
            py,
            str(SCRIPTS / "augment_coco_dataset.py"),
            "--input-dir",
            str(coco_lr),
            "--output-dir",
            str(coco_lr_aug),
            "--splits",
            "train",
            "--jitter",
            str(args.jitter),
        ]
        if args.force:
            aug_lr.append("--force")
        run(aug_lr, label="local-relief offline 8x+jitter (train only)")
        assert_holdout_unaugmented(coco_lr_aug)

        run(
            resample_cmd(py, coco_lr_aug, coco_lr_bal, args),
            label="local-relief resample train (same as Run A)",
        )
        do_train(
            coco_lr_bal,
            run_lr_bal,
            f"train C local-relief balanced → {run_lr_bal.name}",
            four_band=True,
        )

    if args.mode == "smoke":
        print(
            "\nSmoke OK — builds + short trains succeeded.\n"
            f"  A: {run_rgb_bal}\n"
            f"  B: {run_rgb_ctrl}\n"
            f"  C: {run_lr_bal}\n"
            "Re-run with --mode weekend (default) for the real overnight.\n"
            "Datasets are reused; only training dirs differ (*_smoke vs full).\n",
            flush=True,
        )
    else:
        print(
            "\nDone.\n"
            f"  A RGB balanced:          {run_rgb_bal}\n"
            f"  B RGB control:           {run_rgb_ctrl}\n"
            f"  C local-relief balanced: {run_lr_bal}\n"
            "All use stratified_coastal; A and C share resample knobs; "
            "hold-outs are unaugmented.\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
