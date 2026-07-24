#!/usr/bin/env python3
"""Retile large ortho / RGB+DSM GeoTIFFs into fixed-size chips without resampling.

Windows at ``chip_size`` (default 512) with the last window anchored at
``parent_size - chip_size`` so every pixel is covered exactly once in the
interior and edge strips may overlap slightly — GSD/transform unchanged.

Example:
  python BoulderCalculator/scripts/retile_to_chips.py \\
    --source-dir segmentation/tiling --output-dir segmentation/tiling_512 \\
    --chip-size 512 --years 24,25

  python BoulderCalculator/scripts/retile_to_chips.py \\
    --source-dir segmentation/tiling_rgb_dsm_24 \\
    --output-dir segmentation/tiling_512_rgb_dsm_24 \\
    --chip-size 512 --flat --year 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpkg_to_coco import (  # noqa: E402
    resolve_tile_path,
    resolve_tiles_by_year,
    tile_filename,
)
from skip_existing import add_force_argument, should_skip_file  # noqa: E402


def chip_starts(length: int, chip_size: int) -> list[int]:
    if length <= chip_size:
        return [0]
    starts = list(range(0, length - chip_size + 1, chip_size))
    last = length - chip_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def chip_windows(width: int, height: int, chip_size: int) -> list[tuple[int, int, int, int]]:
    """Return (row_off, col_off, chip_row_idx, chip_col_idx) for each chip."""
    out = []
    for ri, row_off in enumerate(chip_starts(height, chip_size)):
        for ci, col_off in enumerate(chip_starts(width, chip_size)):
            out.append((row_off, col_off, ri, ci))
    return out


def chip_name(parent_name: str, ri: int, ci: int) -> str:
    stem = Path(parent_name).stem
    suffix = Path(parent_name).suffix
    return f"{stem}_r{ri}_c{ci}{suffix}"


def write_chip(
    src_path: Path,
    out_path: Path,
    *,
    row_off: int,
    col_off: int,
    chip_size: int,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        window = Window(col_off, row_off, chip_size, chip_size)
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            {
                "height": chip_size,
                "width": chip_size,
                "transform": transform,
                "compress": profile.get("compress") or "deflate",
            }
        )
        # Pad if window hangs off edge (should not happen with chip_starts).
        if data.shape[-2] != chip_size or data.shape[-1] != chip_size:
            padded = np.zeros((src.count, chip_size, chip_size), dtype=data.dtype)
            padded[:, : data.shape[-2], : data.shape[-1]] = data
            data = padded
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
    return {
        "output": str(out_path),
        "parent": str(src_path),
        "row_off": row_off,
        "col_off": col_off,
        "chip_size": chip_size,
    }


def retile_one(
    src_path: Path,
    out_dir: Path,
    *,
    chip_size: int,
    force: bool,
) -> list[dict]:
    rows: list[dict] = []
    with rasterio.open(src_path) as src:
        width, height = src.width, src.height
    for row_off, col_off, ri, ci in chip_windows(width, height, chip_size):
        out_path = out_dir / chip_name(src_path.name, ri, ci)
        if should_skip_file(out_path, force=force):
            rows.append(
                {
                    "output": str(out_path),
                    "parent": str(src_path),
                    "row_off": row_off,
                    "col_off": col_off,
                    "chip_size": chip_size,
                    "skipped": True,
                }
            )
            continue
        rows.append(
            write_chip(
                src_path,
                out_path,
                row_off=row_off,
                col_off=col_off,
                chip_size=chip_size,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Parent tile root (tiling/ with year subdirs, or a flat tiling_rgb_dsm_YY).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Chip output root (mirrors source layout).",
    )
    parser.add_argument("--chip-size", type=int, default=512)
    parser.add_argument(
        "--years",
        type=str,
        default="24,25",
        help="Comma-separated years when source uses year subdirs (default 24,25).",
    )
    parser.add_argument(
        "--year",
        type=int,
        choices=[24, 25],
        default=None,
        help="Single year for --flat sources (tiling_rgb_dsm_YY).",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Source/output are flat dirs (no year subdirectory).",
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("segmentation"),
        help="Used to resolve tiles_used.txt parent keys (default: segmentation).",
    )
    parser.add_argument(
        "--all-tifs",
        action="store_true",
        help="Retile every .tif under source (ignore tiles_used).",
    )
    add_force_argument(parser)
    args = parser.parse_args()

    years = [args.year] if args.year is not None else [
        int(y.strip()) for y in args.years.split(",") if y.strip()
    ]
    if args.flat and len(years) != 1:
        raise SystemExit("--flat requires exactly one --year")

    summary: list[dict] = []
    built = skipped = 0

    if args.flat:
        year = years[0]
        src_dir = args.source_dir
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.all_tifs:
            parents = sorted(src_dir.glob("*.tif"))
        else:
            tiles_by_year = resolve_tiles_by_year(args.segmentation_dir, None)
            parents = []
            for short in tiles_by_year.get(year, []):
                try:
                    parents.append(resolve_tile_path(src_dir, short, year))
                except FileNotFoundError:
                    # Flat rgb_dsm dirs use un-prefixed names via tile_filename.
                    name = tile_filename(short, year)
                    path = src_dir / name
                    if path.is_file():
                        parents.append(path)
        for src in tqdm(parents, desc=f"retile flat/{year}"):
            rows = retile_one(src, out_dir, chip_size=args.chip_size, force=args.force)
            for row in rows:
                summary.append(row)
                if row.get("skipped"):
                    skipped += 1
                else:
                    built += 1
    else:
        tiles_by_year = resolve_tiles_by_year(args.segmentation_dir, None)
        for year in years:
            src_year = args.source_dir / str(year)
            out_year = args.output_dir / str(year)
            out_year.mkdir(parents=True, exist_ok=True)
            if args.all_tifs:
                parents = sorted(src_year.glob("*.tif"))
            else:
                parents = []
                for short in tiles_by_year.get(year, []):
                    parents.append(resolve_tile_path(args.source_dir, short, year))
            for src in tqdm(parents, desc=f"retile/{year}"):
                rows = retile_one(
                    src, out_year, chip_size=args.chip_size, force=args.force
                )
                for row in rows:
                    summary.append({**row, "year": year})
                    if row.get("skipped"):
                        skipped += 1
                    else:
                        built += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "retile_to_chips_manifest.json"
    payload = {
        "chip_size": args.chip_size,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "built": built,
        "skipped": skipped,
        "chips": len(summary),
        "force": bool(args.force),
    }
    manifest.write_text(json.dumps({**payload, "rows_sample": summary[:20]}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    from run_provenance import write_tiling_provenance

    write_tiling_provenance(
        args.output_dir,
        tool="retile_to_chips.py",
        flags={
            "chip_size": args.chip_size,
            "source_dir": str(args.source_dir),
            "years": years,
            "flat": bool(args.flat),
            "force": bool(args.force),
            "built": built,
            "skipped": skipped,
        },
        tiles_summary=summary,
        parents=[args.source_dir],
        extra={"legacy_manifest_file": "retile_to_chips_manifest.json"},
    )


if __name__ == "__main__":
    main()
