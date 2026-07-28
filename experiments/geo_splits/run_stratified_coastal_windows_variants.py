#!/usr/bin/env python3
"""Windows guest: stratified_coastal @ 2000² — four balanced training variants.

Shared defaults (override only where a variant says otherwise)
------------------------------------------------------------
- ``stratified_coastal.yaml`` + ``--skip-leakage-check``
- ``--no-rich-aug``, offline **8×** train-only aug, ``--jitter 0.15``
- ``--min-area-m2 1.5`` with **iscrowd** for deposits + sub-threshold boulders
- train resample: positives ×1.5; thin deposit-heavy / green-hilly / empties

Runs
----
A. **RGB+DSM elevation** (4-band) — default annotation policy
B. **RGB** — ``--min-area-m2 1.0`` (still iscrowd deposits + small)
C. **RGB** — ``--drop-deposits`` (iscrowd remains for small @ 1.5 m²)
D. **RGB** — ``--drop-deposits --drop-below-min-area`` (omit both)

Each balanced dataset writes QGIS-ready resample audits under its COCO dir and
under ``segmentation/resample_audits_stratified_coastal/<variant>/``::

  resample_train_audit_parents.geojson   # joinable / style by status
  resample_train_audit_parents.csv
  resample_train_audit_images.csv
  resample_train_summary.json            # keep rates + oversample factor

Smoke first::

  smoke_stratified_coastal_windows_variants.bat

Overnight::

  run_stratified_coastal_windows_variants.bat
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SPLIT_YAML = EXP_DIR / "stratified_coastal.yaml"
EXTENTS_GEOJSON = EXP_DIR / "tile_extents_stratified_coastal.geojson"

SETUP = "stratified_coastal"
TILE_DSM_24 = "tiling_rgb_dsm_24"
TILE_DSM_25 = "tiling_rgb_dsm_25"
AUDIT_ROOT_NAME = "resample_audits_stratified_coastal"
AUG_SUFFIXES = (
    "_hflip",
    "_vflip",
    "_rot90",
    "_rot180",
    "_rot270",
    "_transpose",
    "_antitranspose",
)


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    coco: str
    coco_aug: str
    coco_bal: str
    run: str
    four_band: bool
    min_area_m2: float
    drop_deposits: bool
    drop_below_min_area: bool
    source_rgb_coco: str | None = None  # if set, clone RGB→4-band from this coco


VARIANTS: tuple[Variant, ...] = (
    Variant(
        key="A_rgb_dsm",
        label="A RGB+DSM elevation balanced",
        coco="coco_geo_stratified_coastal_rgb_dsm",
        coco_aug="coco_geo_stratified_coastal_rgb_dsm_aug",
        coco_bal="coco_geo_stratified_coastal_rgb_dsm_aug_balanced",
        run="training_run_geo_stratified_coastal_rgb_dsm_balanced",
        four_band=True,
        min_area_m2=1.5,
        drop_deposits=False,
        drop_below_min_area=False,
        source_rgb_coco="coco_geo_stratified_coastal",
    ),
    Variant(
        key="B_rgb_min1p0",
        label="B RGB min-area 1.0 m² balanced",
        coco="coco_geo_stratified_coastal_min1p0",
        coco_aug="coco_geo_stratified_coastal_min1p0_aug",
        coco_bal="coco_geo_stratified_coastal_min1p0_aug_balanced",
        run="training_run_geo_stratified_coastal_rgb_min1p0_balanced",
        four_band=False,
        min_area_m2=1.0,
        drop_deposits=False,
        drop_below_min_area=False,
    ),
    Variant(
        key="C_rgb_drop_deposits",
        label="C RGB drop deposits (iscrowd small) balanced",
        coco="coco_geo_stratified_coastal_drop_deposits",
        coco_aug="coco_geo_stratified_coastal_drop_deposits_aug",
        coco_bal="coco_geo_stratified_coastal_drop_deposits_aug_balanced",
        run="training_run_geo_stratified_coastal_rgb_drop_deposits_balanced",
        four_band=False,
        min_area_m2=1.5,
        drop_deposits=True,
        drop_below_min_area=False,
    ),
    Variant(
        key="D_rgb_drop_dep_small",
        label="D RGB drop deposits + small balanced",
        coco="coco_geo_stratified_coastal_drop_dep_small",
        coco_aug="coco_geo_stratified_coastal_drop_dep_small_aug",
        coco_bal="coco_geo_stratified_coastal_drop_dep_small_aug_balanced",
        run="training_run_geo_stratified_coastal_rgb_drop_dep_small_balanced",
        four_band=False,
        min_area_m2=1.5,
        drop_deposits=True,
        drop_below_min_area=True,
    ),
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


def ensure_elevation_dsm_tiles(root: Path, py: str, force: bool) -> None:
    seg = root / "segmentation"
    for year in (24, 25):
        tile_dir = seg / f"tiling_rgb_dsm_{year}"
        if tile_dir.is_dir() and any(tile_dir.glob("*.tif")) and not force:
            print(f"[skip] elevation DSM tiles exist: {tile_dir.name}")
            continue
        cmd = [
            py,
            str(SCRIPTS / "build_rgb_dsm_tiles.py"),
            "--year",
            str(year),
            "--dsm-mode",
            "elevation",
        ]
        if force:
            cmd.append("--force")
        run(cmd, label=f"build_rgb_dsm_tiles year={year} elevation")


def gpkg_cmd(
    *,
    py: str,
    seg: Path,
    out: Path,
    min_area_m2: float,
    drop_deposits: bool,
    drop_below_min_area: bool,
    link_mode: str,
    force: bool,
    skip_leakage_check: bool,
) -> list[str]:
    cmd = [
        py,
        str(SCRIPTS / "gpkg_to_coco.py"),
        "--segmentation-dir",
        str(seg),
        "--years",
        "24,25",
        "--split-config",
        str(SPLIT_YAML),
        "--output-dir",
        str(out),
        "--min-area-m2",
        str(min_area_m2),
        "--link-mode",
        link_mode,
    ]
    if drop_deposits:
        cmd.append("--drop-deposits")
    if drop_below_min_area:
        cmd.append("--drop-below-min-area")
    if force:
        cmd.append("--force")
    if skip_leakage_check:
        cmd.append("--skip-leakage-check")
    return cmd


def resample_cmd(
    *,
    py: str,
    inp: Path,
    out: Path,
    audit_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        py,
        str(SCRIPTS / "resample_coco_train.py"),
        "--input-dir",
        str(inp),
        "--output-dir",
        str(out),
        "--audit-dir",
        str(audit_dir),
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
    if EXTENTS_GEOJSON.is_file():
        cmd.extend(["--tile-extents", str(EXTENTS_GEOJSON)])
    return cmd


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


def build_variant_dataset(
    *,
    root: Path,
    py: str,
    variant: Variant,
    args: argparse.Namespace,
) -> Path:
    seg = root / "segmentation"
    audit_root = seg / AUDIT_ROOT_NAME / variant.key
    coco = seg / variant.coco
    coco_aug = seg / variant.coco_aug
    coco_bal = seg / variant.coco_bal

    if variant.source_rgb_coco:
        # A: RGB baseline annotations → swap in elevation RGB+DSM tiles.
        src = seg / variant.source_rgb_coco
        if not (src / "train_annotations.json").is_file() or args.force:
            run(
                gpkg_cmd(
                    py=py,
                    seg=seg,
                    out=src,
                    min_area_m2=variant.min_area_m2,
                    drop_deposits=variant.drop_deposits,
                    drop_below_min_area=variant.drop_below_min_area,
                    link_mode=args.link_mode,
                    force=args.force,
                    skip_leakage_check=args.skip_leakage_check,
                ),
                label=f"{variant.key}: gpkg baseline RGB (for DSM clone)",
            )
        ensure_elevation_dsm_tiles(root, py, args.force)
        tile_24 = seg / TILE_DSM_24
        tile_25 = seg / TILE_DSM_25
        if not tile_24.is_dir() or not tile_25.is_dir():
            raise SystemExit(f"Missing elevation DSM tilings: {tile_24} / {tile_25}")
        dsm_cmd = [
            py,
            str(SCRIPTS / "build_coco_rgb_dsm.py"),
            "--source-coco",
            str(src),
            "--tile-dirs",
            str(tile_24),
            str(tile_25),
            "--output-dir",
            str(coco),
            "--link-mode",
            args.link_mode,
            "--project-root",
            str(root),
        ]
        if args.force:
            dsm_cmd.append("--force")
        run(dsm_cmd, label=f"{variant.key}: build_coco_rgb_dsm elevation")
    else:
        run(
            gpkg_cmd(
                py=py,
                seg=seg,
                out=coco,
                min_area_m2=variant.min_area_m2,
                drop_deposits=variant.drop_deposits,
                drop_below_min_area=variant.drop_below_min_area,
                link_mode=args.link_mode,
                force=args.force,
                skip_leakage_check=args.skip_leakage_check,
            ),
            label=f"{variant.key}: gpkg_to_coco",
        )

    aug_cmd = [
        py,
        str(SCRIPTS / "augment_coco_dataset.py"),
        "--input-dir",
        str(coco),
        "--output-dir",
        str(coco_aug),
        "--splits",
        "train",
        "--jitter",
        str(args.jitter),
    ]
    if args.force:
        aug_cmd.append("--force")
    run(aug_cmd, label=f"{variant.key}: offline 8x+jitter (train only)")
    assert_holdout_unaugmented(coco_aug)

    run(
        resample_cmd(
            py=py,
            inp=coco_aug,
            out=coco_bal,
            audit_dir=audit_root,
            args=args,
        ),
        label=f"{variant.key}: resample train + QGIS audit → {audit_root.name}",
    )
    # Also keep a copy beside the balanced COCO (resample already writes there
    # when audit_dir == output; here we mirror into coco_bal via second write
    # by copying audit files if needed).
    for name in (
        "resample_train_audit_images.csv",
        "resample_train_audit_parents.csv",
        "resample_train_audit_parents.geojson",
        "resample_train_summary.json",
    ):
        src = audit_root / name
        dst = coco_bal / name
        if src.is_file() and (not dst.is_file() or args.force or src.resolve() != dst.resolve()):
            dst.write_bytes(src.read_bytes())

    return coco_bal


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
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument(
        "--link-mode",
        default="hard",
        choices=("auto", "hard", "symlink", "copy"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-leakage-check",
        dest="skip_leakage_check",
        action="store_true",
        default=True,
        help="Skip geographic leakage check (default: on for this runner).",
    )
    parser.add_argument(
        "--no-skip-leakage-check",
        dest="skip_leakage_check",
        action="store_false",
        help="Enforce geographic leakage check.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated variant keys to run (default: all). "
        "Keys: A_rgb_dsm,B_rgb_min1p0,C_rgb_drop_deposits,D_rgb_drop_dep_small",
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

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    variants = [v for v in VARIANTS if not only or v.key in only]
    if not variants:
        raise SystemExit(f"No variants selected (only={args.only!r})")

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
        f"jitter={args.jitter} no-rich-aug split={SETUP} "
        f"skip_leakage_check={args.skip_leakage_check} link_mode={args.link_mode}"
    )
    print(
        "Resample: "
        f"pos×{args.positive_oversample} keep_deposit={args.keep_deposit} "
        f"keep_hilly={args.keep_hilly} keep_empty_coast={args.keep_empty_coast} "
        f"keep_empty_other={args.keep_empty_other} seed={args.resample_seed}"
    )
    print("Variants:", ", ".join(v.key for v in variants))
    if EXTENTS_GEOJSON.is_file():
        print(f"Tile extents for audit join: {EXTENTS_GEOJSON}")
    else:
        print(f"WARNING: missing {EXTENTS_GEOJSON} — parent GeoJSON will lack geometries")

    finished: list[str] = []
    for variant in variants:
        print(f"\n########## {variant.label} ##########", flush=True)
        coco_bal = build_variant_dataset(root=root, py=py, variant=variant, args=args)
        out_run = seg / f"{variant.run}{run_suffix}"
        if args.skip_train:
            print(f"[skip] train {out_run.name}")
        else:
            run(
                train_cmd(
                    py=py,
                    dataset=coco_bal,
                    out_run=out_run,
                    args=args,
                    device=device,
                    batch_size=batch_size,
                    four_band=variant.four_band,
                ),
                label=f"train {variant.label} → {out_run.name}",
            )
        finished.append(str(out_run))

    kind = "Smoke OK" if args.mode == "smoke" else "Done"
    print(f"\n{kind}. Runs:")
    for r in finished:
        print(f"  {r}")
    print(f"\nQGIS resample audits: {seg / AUDIT_ROOT_NAME}/")
    print(
        "  Style parents GeoJSON by 'status' (removed / thinned / kept / oversampled)\n"
        "  or 'oversample_factor' / 'primary_bucket'."
    )
    if args.mode == "smoke":
        print("Re-run with --mode weekend for full trains (datasets reused unless --force).")
    print(flush=True)


if __name__ == "__main__":
    main()
