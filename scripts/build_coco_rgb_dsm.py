#!/usr/bin/env python3
"""Build a COCO dataset dir whose images are 4-band RGB+DSM tiles.

Copies annotation JSON from an existing RGB COCO dataset and replaces each
split image with the matching 4-band GeoTIFF (same file_name) from one or more
tiling_rgb_dsm_* directories.

Example:
  python BoulderCalculator/scripts/build_coco_rgb_dsm.py \\
    --source-coco segmentation/coco_dataset \\
    --tile-dirs segmentation/tiling_rgb_dsm_25 \\
    --output-dir segmentation/coco_dataset_rgb_dsm
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
) -> dict:
    ann_src = source_coco / ann_name
    data = json.loads(ann_src.read_text(encoding="utf-8"))
    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    copied = []
    skipped = 0
    for image in data["images"]:
        dst = split_out / image["file_name"]
        if should_skip_file(dst, force=force):
            skipped += 1
            copied.append(image["file_name"])
            continue
        src = resolve_four_band(tile_dirs, image["file_name"])
        assert_four_bands(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(image["file_name"])

    (output_dir / ann_name).write_text(json.dumps(data), encoding="utf-8")
    return {
        "split": split,
        "images": len(copied),
        "copied": len(copied) - skipped,
        "skipped": skipped,
        "ann": ann_name,
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
    add_force_argument(parser)
    args = parser.parse_args()

    if should_skip_coco_dataset(
        args.output_dir,
        force=args.force,
        label="build_coco_rgb_dsm",
        expected_flags={"four_band": True},
    ):
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
            )
        )

    out = {
        "source": str(args.source_coco),
        "output": str(args.output_dir),
        "force": bool(args.force),
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
        },
        splits_summary=summary,
        parents=[args.source_coco, *args.tile_dirs],
        notes="COCO annotations from source RGB dataset; images replaced with 4-band RGB+DSM tiles.",
        extra={"legacy_summary_file": "build_coco_rgb_dsm_summary.json"},
    )


if __name__ == "__main__":
    main()
