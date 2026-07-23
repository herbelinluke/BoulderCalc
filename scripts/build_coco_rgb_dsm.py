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
from pathlib import Path

import rasterio


def resolve_tile(tile_dirs: list[Path], file_name: str) -> Path:
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
        f"Tile not found for {file_name!r} under {[str(d) for d in tile_dirs]}"
    )


def assert_min_bands(path: Path, min_bands: int) -> None:
    with rasterio.open(path) as ds:
        if ds.count < min_bands:
            raise ValueError(f"{path} has {ds.count} bands; expected >={min_bands}")


def copy_split(
    source_coco: Path,
    output_dir: Path,
    split: str,
    ann_name: str,
    tile_dirs: list[Path],
    min_bands: int,
) -> dict:
    ann_src = source_coco / ann_name
    data = json.loads(ann_src.read_text())
    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    copied = []
    for image in data["images"]:
        src = resolve_tile(tile_dirs, image["file_name"])
        assert_min_bands(src, min_bands)
        dst = split_out / image["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(image["file_name"])

    (output_dir / ann_name).write_text(json.dumps(data))
    return {"split": split, "images": len(copied), "ann": ann_name}


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
        help="One or more tiling_rgb_dsm_* (or tiling_rgb_dsm_ddsm_*) directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmentation/coco_dataset_rgb_dsm"),
        help="Output multi-band COCO dataset dir.",
    )
    parser.add_argument(
        "--min-bands",
        type=int,
        default=4,
        help="Require at least this many bands per tile (4=RGB+DSM, 5=RGB+DSM+dDSM).",
    )
    args = parser.parse_args()

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
                args.min_bands,
            )
        )

    out = {
        "source": str(args.source_coco),
        "output": str(args.output_dir),
        "min_bands": args.min_bands,
        "splits": summary,
    }
    (args.output_dir / "build_coco_rgb_dsm_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    from run_provenance import write_dataset_provenance

    write_dataset_provenance(
        args.output_dir,
        tool="build_coco_rgb_dsm.py",
        flags={
            "source_coco": str(args.source_coco),
            "tile_dirs": [str(d) for d in args.tile_dirs],
            "min_bands": args.min_bands,
            "four_band": args.min_bands == 4,
            "five_band": args.min_bands >= 5,
        },
        splits_summary=summary,
        parents=[args.source_coco, *args.tile_dirs],
        notes=(
            "COCO annotations from source RGB dataset; images replaced with "
            f"{args.min_bands}-band tiles."
        ),
        extra={"legacy_summary_file": "build_coco_rgb_dsm_summary.json"},
    )


if __name__ == "__main__":
    main()
