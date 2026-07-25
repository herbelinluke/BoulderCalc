#!/usr/bin/env python3
"""Build a COCO dataset dir whose images are 4-band RGB+DSM tiles.

Copies annotation JSON from an existing RGB COCO dataset and replaces each
split image with the matching 4-band GeoTIFF (same file_name) from one or more
tiling_rgb_dsm_* directories.

By default, coverage ignore is refreshed on the 4-band images (RGB bands +
year DSM nodata) so black-edge / gated DSM-gap iscrowd regions stay accurate
even if the source RGB COCO was built without DSM gating.

Example:
  python BoulderCalculator/scripts/build_coco_rgb_dsm.py \\
    --source-coco segmentation/coco_dataset \\
    --tile-dirs segmentation/tiling_rgb_dsm_25 \\
    --output-dir segmentation/coco_dataset_rgb_dsm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage_ignore import (  # noqa: E402
    DEFAULT_BLUR_M,
    DEFAULT_MIN_AREA_M2,
    DEFAULT_OVERLAP_FRAC,
    DEFAULT_RGB_MAX,
    default_dsm_path,
    refresh_split_coverage,
)
from file_link import add_link_mode_argument, link_or_copy  # noqa: E402
from skip_existing import (  # noqa: E402
    add_force_argument,
    should_skip_coco_dataset,
    should_skip_file,
)


def resolve_four_band(tile_dirs: list[Path], file_name: str) -> Path:
    """Find file_name in tile dirs; also try stripping a leading '24_' / '25_' prefix."""
    candidates = [file_name]
    for prefix in ("24_", "25_"):
        if file_name.startswith(prefix):
            candidates.append(file_name[len(prefix) :])
    for tile_dir in tile_dirs:
        for name in candidates:
            path = tile_dir / name
            if path.exists():
                return path
    raise FileNotFoundError(
        f"4-band tile not found for {file_name!r} under {[str(d) for d in tile_dirs]}"
    )


def assert_four_bands(path: Path) -> None:
    with rasterio.open(path) as ds:
        if ds.count < 4:
            raise ValueError(f"{path} has {ds.count} bands; expected 4")


def copy_split(
    source_coco: Path,
    output_dir: Path,
    split: str,
    ann_name: str,
    tile_dirs: list[Path],
    *,
    force: bool,
    link_mode: str,
    coverage_ignore: bool,
    project_root: Path,
    coverage_rgb_max: int,
    coverage_blur_m: float,
    coverage_min_area_m2: float,
    coverage_overlap_frac: float,
) -> dict:
    ann_src = source_coco / ann_name
    data = json.loads(ann_src.read_text(encoding="utf-8"))
    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    copied = []
    skipped = 0
    modes_used: dict[str, int] = {}
    for image in data["images"]:
        dst = split_out / image["file_name"]
        if should_skip_file(dst, force=force):
            skipped += 1
            copied.append(image["file_name"])
            continue
        src = resolve_four_band(tile_dirs, image["file_name"])
        assert_four_bands(src)
        used = link_or_copy(src, dst, link_mode)
        modes_used[used] = modes_used.get(used, 0) + 1
        copied.append(image["file_name"])

    coverage_stats = None
    if coverage_ignore:
        dsm_by_year: dict[int, Path] = {}
        for year in (24, 25):
            path = default_dsm_path(project_root, year)
            if path.is_file():
                dsm_by_year[year] = path
        data = refresh_split_coverage(
            data,
            split_out,
            dsm_by_year=dsm_by_year,
            project_root=project_root,
            rgb_max=coverage_rgb_max,
            blur_m=coverage_blur_m,
            min_area_m2=coverage_min_area_m2,
            overlap_frac=coverage_overlap_frac,
        )
        coverage_stats = data.pop("_coverage_refresh_stats", None)

    (output_dir / ann_name).write_text(json.dumps(data), encoding="utf-8")
    return {
        "split": split,
        "images": len(copied),
        "copied": len(copied) - skipped,
        "skipped": skipped,
        "link_modes": modes_used,
        "ann": ann_name,
        "coverage_refresh": coverage_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-coco",
        type=Path,
        default=Path("segmentation/coco_dataset"),
        help="Source RGB COCO dataset dir. Default: segmentation/coco_dataset",
    )
    parser.add_argument(
        "--tile-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more tiling_rgb_dsm_* directories containing 4-band GeoTIFFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmentation/coco_dataset_rgb_dsm"),
        help="Output 4-band COCO dataset dir. Default: segmentation/coco_dataset_rgb_dsm",
    )
    parser.add_argument(
        "--coverage-ignore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh coverage iscrowd on 4-band images (default: on).",
    )
    parser.add_argument("--coverage-rgb-max", type=int, default=DEFAULT_RGB_MAX)
    parser.add_argument("--coverage-blur-m", type=float, default=DEFAULT_BLUR_M)
    parser.add_argument(
        "--coverage-min-area-m2", type=float, default=DEFAULT_MIN_AREA_M2
    )
    parser.add_argument(
        "--coverage-overlap-frac", type=float, default=DEFAULT_OVERLAP_FRAC
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root with 2024/ and 2025/ DSMs (default: .).",
    )
    add_force_argument(parser)
    add_link_mode_argument(parser, default="hard")
    args = parser.parse_args()

    if should_skip_coco_dataset(
        args.output_dir,
        force=args.force,
        label="build_coco_rgb_dsm",
        expected_flags={
            "four_band": True,
            "coverage_ignore": bool(args.coverage_ignore),
            "coverage_rgb_max": int(args.coverage_rgb_max),
            "coverage_blur_m": float(args.coverage_blur_m),
            "coverage_min_area_m2": float(args.coverage_min_area_m2),
            "coverage_overlap_frac": float(args.coverage_overlap_frac),
        },
    ):
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_root = args.project_root.resolve()
    summary = []
    for split, ann in [
        ("train", "train_annotations.json"),
        ("valid", "validation_annotations.json"),
        ("test", "testing_annotations.json"),
    ]:
        summary.append(
            copy_split(
                args.source_coco,
                args.output_dir,
                split,
                ann,
                args.tile_dirs,
                force=args.force,
                link_mode=args.link_mode,
                coverage_ignore=bool(args.coverage_ignore),
                project_root=project_root,
                coverage_rgb_max=int(args.coverage_rgb_max),
                coverage_blur_m=float(args.coverage_blur_m),
                coverage_min_area_m2=float(args.coverage_min_area_m2),
                coverage_overlap_frac=float(args.coverage_overlap_frac),
            )
        )

    out = {
        "source": str(args.source_coco),
        "output": str(args.output_dir),
        "force": bool(args.force),
        "link_mode": args.link_mode,
        "coverage_ignore": bool(args.coverage_ignore),
        "splits": summary,
    }
    (args.output_dir / "build_coco_rgb_dsm_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))

    from run_provenance import write_dataset_provenance

    write_dataset_provenance(
        args.output_dir,
        tool="build_coco_rgb_dsm.py",
        flags={
            "source_coco": str(args.source_coco),
            "tile_dirs": [str(d) for d in args.tile_dirs],
            "four_band": True,
            "force": bool(args.force),
            "link_mode": args.link_mode,
            "coverage_ignore": bool(args.coverage_ignore),
            "coverage_rgb_max": int(args.coverage_rgb_max),
            "coverage_blur_m": float(args.coverage_blur_m),
            "coverage_min_area_m2": float(args.coverage_min_area_m2),
            "coverage_overlap_frac": float(args.coverage_overlap_frac),
        },
        splits_summary=summary,
        parents=[args.source_coco, *args.tile_dirs],
        notes=(
            "COCO annotations from source RGB dataset; images replaced with "
            "4-band RGB+DSM tiles; coverage ignore refreshed unless disabled."
        ),
        extra={"legacy_summary_file": "build_coco_rgb_dsm_summary.json"},
    )


if __name__ == "__main__":
    main()
