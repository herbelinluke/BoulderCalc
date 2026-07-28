#!/usr/bin/env python3
"""Windows guest: stratified_coastal RGB chips @ 512 and 1024 (balanced train).

Builds chip stores from parent ``tiling/``, COCO with
``stratified_coastal.yaml`` + ``--min-area-m2 1.5`` (iscrowd deposits/small),
offline **8× + jitter 0.15 on train only**, then the same train resample as
Run A (oversample positives / thin deposit + green-hilly empties / keep some
coastal empties). Trains RGB only ``--no-rich-aug`` for each chip size.

Hold-outs stay unaugmented (aug script default + assert).

Modes
-----
* ``smoke`` — full build + 3-iter trains into ``*_smoke`` dirs
* ``weekend`` — full early-stopped trains

From project root (``BoulderCalculator\\`` + ``segmentation\\``)::

  python BoulderCalculator\\experiments\\geo_splits\\run_stratified_coastal_rgb_chips_windows.py --mode smoke --device cuda
  python BoulderCalculator\\experiments\\geo_splits\\run_stratified_coastal_rgb_chips_windows.py --mode weekend --device cuda

Or the ``.bat`` helpers in this folder.
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


def default_batch_for_chip(chip: int, requested: int | None) -> tuple[int, str]:
    if requested is not None:
        return requested, "cli"
    try:
        import torch

        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            name = torch.cuda.get_device_name(0)
            if chip <= 512:
                bs = 4 if gb >= 8 else 2
            elif chip <= 1024:
                bs = 2 if gb >= 8 else 1
            else:
                bs = 1
            return bs, f"cuda {name} ({gb:.1f} GiB) chip={chip} → batch {bs}"
    except Exception as exc:
        return (2 if chip <= 512 else 1), f"torch probe failed ({exc})"
    return (2 if chip <= 512 else 1), "no CUDA → conservative batch"


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


def paths_for_chip(seg: Path, chip: int) -> dict[str, Path]:
    return {
        "tiling": seg / f"tiling_{chip}",
        "coco": seg / f"coco_geo{chip}_{SETUP}",
        "coco_aug": seg / f"coco_geo{chip}_{SETUP}_aug",
        "coco_bal": seg / f"coco_geo{chip}_{SETUP}_aug_balanced",
        "run_bal": seg / f"training_run_geo{chip}_{SETUP}_rgb_balanced",
    }


def retile_rgb(root: Path, py: str, chip: int, force: bool, link_note: str) -> None:
    seg = root / "segmentation"
    out = seg / f"tiling_{chip}"
    cmd = [
        py,
        str(SCRIPTS / "retile_to_chips.py"),
        "--source-dir",
        str(seg / "tiling"),
        "--output-dir",
        str(out),
        "--chip-size",
        str(chip),
        "--years",
        "24,25",
        "--segmentation-dir",
        str(seg),
    ]
    if force:
        cmd.append("--force")
    run(cmd, label=f"retile RGB → tiling_{chip} ({link_note})")


def build_chip_coco(
    *,
    root: Path,
    py: str,
    chip: int,
    min_area_m2: float,
    link_mode: str,
    force: bool,
    skip_leakage_check: bool = False,
) -> Path:
    seg = root / "segmentation"
    p = paths_for_chip(seg, chip)
    cmd = [
        py,
        str(SCRIPTS / "gpkg_to_coco.py"),
        "--segmentation-dir",
        str(seg),
        "--tile-dir",
        str(p["tiling"]),
        "--years",
        "24,25",
        "--split-config",
        str(SPLIT_YAML),
        "--output-dir",
        str(p["coco"]),
        "--min-area-m2",
        str(min_area_m2),
        "--expand-chips",
        "--link-mode",
        link_mode,
    ]
    if force:
        cmd.append("--force")
    if skip_leakage_check:
        cmd.append("--skip-leakage-check")
    run(cmd, label=f"gpkg_to_coco {chip} stratified_coastal RGB")
    return p["coco"]


def augment_train_only(
    py: str, coco: Path, coco_aug: Path, jitter: float, force: bool
) -> None:
    cmd = [
        py,
        str(SCRIPTS / "augment_coco_dataset.py"),
        "--input-dir",
        str(coco),
        "--output-dir",
        str(coco_aug),
        "--splits",
        "train",
        "--jitter",
        str(jitter),
    ]
    if force:
        cmd.append("--force")
    run(cmd, label=f"offline 8x+jitter train-only → {coco_aug.name}")
    assert_holdout_unaugmented(coco_aug)


def resample_balanced(py: str, coco_aug: Path, coco_bal: Path, args) -> None:
    cmd = [
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
        cmd.extend(["--tile-extents", str(extents)])
    run(cmd, label=f"resample train → {coco_bal.name}")


def train_rgb(
    *,
    py: str,
    dataset: Path,
    out_run: Path,
    chip: int,
    device: str,
    batch_size: int,
    max_iter: int,
    eval_period: int,
    checkpoint_period: int,
    early_stop: int,
    num_workers: int,
) -> None:
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
        str(chip),
        "--checkpoint-period",
        str(checkpoint_period),
        "--eval-period",
        str(eval_period),
        "--num-workers",
        str(num_workers),
        "--device",
        device,
    ]
    if early_stop > 0:
        cmd.extend(
            [
                "--early-stop-patience-iters",
                str(early_stop),
                "--early-stop-metric",
                "segm/AP",
            ]
        )
    run(cmd, label=f"train RGB balanced chip={chip} → {out_run.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "weekend"), default="weekend")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--chips",
        default="512,1024",
        help="Comma-separated chip sizes (default 512,1024).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override for all chips.")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--eval-period", type=int, default=None)
    parser.add_argument("--checkpoint-period", type=int, default=None)
    parser.add_argument("--early-stop-patience-iters", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-area-m2", type=float, default=1.5)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument(
        "--link-mode",
        default="hard",
        choices=("auto", "hard", "symlink", "copy"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-leakage-check",
        action="store_true",
        help=(
            "Pass through to gpkg_to_coco: skip geographic footprint leakage "
            "check (known mild valid↔test cross-year overlap on stratified_coastal)."
        ),
    )
    parser.add_argument("--skip-retile", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
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
        raise SystemExit(f"Missing {SPLIT_YAML}")

    chips = [int(x.strip()) for x in args.chips.split(",") if x.strip()]
    chips = [c for c in chips if c > 0]
    if not chips:
        raise SystemExit("No valid --chips")

    if args.mode == "smoke":
        max_iter = 3 if args.max_iter is None else args.max_iter
        eval_period = 2 if args.eval_period is None else args.eval_period
        checkpoint_period = 2 if args.checkpoint_period is None else args.checkpoint_period
        early_stop = 0 if args.early_stop_patience_iters is None else args.early_stop_patience_iters
        run_suffix = "_smoke"
    else:
        max_iter = 3000 if args.max_iter is None else args.max_iter
        eval_period = 500 if args.eval_period is None else args.eval_period
        checkpoint_period = 1000 if args.checkpoint_period is None else args.checkpoint_period
        early_stop = (
            500 if args.early_stop_patience_iters is None else args.early_stop_patience_iters
        )
        run_suffix = ""

    root = project_root_from_cwd()
    seg = root / "segmentation"
    py = args.python
    device = pick_device(args.device)

    print(f"Project root: {root}")
    print(f"Python: {py}")
    print(f"Mode: {args.mode}  chips={chips}  device={device}")
    print(
        f"max_iter={max_iter} eval={eval_period} early_stop={early_stop} "
        f"min_area_m2={args.min_area_m2} jitter={args.jitter} link_mode={args.link_mode}"
    )
    # --- Retile ---
    if not args.skip_retile:
        for chip in chips:
            retile_rgb(root, py, chip, args.force, args.link_mode)
    else:
        print("[skip] retile")

    finished: list[str] = []
    for chip in chips:
        p = paths_for_chip(seg, chip)
        batch_size, batch_reason = default_batch_for_chip(chip, args.batch_size)
        if args.mode == "smoke" and args.batch_size is None:
            batch_size, batch_reason = 1, "smoke default batch 1"
        print(f"\n##### chip={chip}  batch_size={batch_size} ({batch_reason}) #####")

        build_chip_coco(
            root=root,
            py=py,
            chip=chip,
            min_area_m2=args.min_area_m2,
            link_mode=args.link_mode,
            force=args.force,
            skip_leakage_check=args.skip_leakage_check,
        )
        augment_train_only(py, p["coco"], p["coco_aug"], args.jitter, args.force)
        resample_balanced(py, p["coco_aug"], p["coco_bal"], args)

        out_run = Path(str(p["run_bal"]) + run_suffix)
        if args.skip_train:
            print(f"[skip] train {out_run.name}")
        else:
            train_rgb(
                py=py,
                dataset=p["coco_bal"],
                out_run=out_run,
                chip=chip,
                device=device,
                batch_size=batch_size,
                max_iter=max_iter,
                eval_period=eval_period,
                checkpoint_period=checkpoint_period,
                early_stop=early_stop,
                num_workers=args.num_workers,
            )
        finished.append(str(out_run))

    kind = "Smoke OK" if args.mode == "smoke" else "Done"
    print(f"\n{kind}. Runs:")
    for r in finished:
        print(f"  {r}")
    if args.mode == "smoke":
        print("Re-run with --mode weekend for full trains (datasets reused).")
    print(flush=True)


if __name__ == "__main__":
    main()
