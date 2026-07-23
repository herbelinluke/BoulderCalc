#!/usr/bin/env python3
"""Build 5-band RGB+DSM+dDSM tiles for moved-boulder fine-tuning.

Band order:
  1-3  RGB (from existing 4-band tile, or from ortho)
  4    Year DSM (elevation or local_relief uint8, from existing 4-band tile)
  5    Difference DSM (site-wide 25−24), warped to the tile grid, symmetric
       uint8 stretch (127 ≈ no change)

Default input difference raster (project root):

  25_minus_24_dsm.tif

Fast path: stack dDSM onto existing ``tiling_rgb_dsm_{24,25}`` tiles
(``--from-four-band-dir``). Writes to ``segmentation/tiling_rgb_dsm_ddsm_{year}/``.

Example:
  python BoulderCalculator/scripts/build_rgb_dsm_ddsm_tiles.py --year 25
  python BoulderCalculator/scripts/build_rgb_dsm_ddsm_tiles.py --year 24 \\
    --from-coco segmentation/coco_dataset_both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_rgb_dsm_tiles import (  # noqa: E402
    keys_from_coco,
    parse_tile_keys,
    tile_filename,
)


def ddsm_to_uint8(dz: np.ndarray) -> np.ndarray:
    """Map signed Δz to uint8 with ~127 at zero change (symmetric 98% clip)."""
    finite = dz[np.isfinite(dz)]
    if finite.size == 0:
        return np.full(dz.shape, 127, dtype=np.uint8)
    lim = float(np.percentile(np.abs(finite), 98))
    lim = max(lim, 1e-6)
    scaled = (dz / lim + 1.0) * 0.5
    out = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    out = np.where(np.isfinite(dz), out, np.uint8(127))
    return out


def warp_ddsm_to_grid(
    ref_path: Path,
    ddsm_path: Path,
) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    with rasterio.open(ref_path) as ref:
        dz = np.zeros((ref.height, ref.width), dtype=np.float32)
        profile = ref.profile.copy()
        with rasterio.open(ddsm_path) as ddsm:
            reproject(
                source=rasterio.band(ddsm, 1),
                destination=dz,
                src_transform=ddsm.transform,
                src_crs=ddsm.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.bilinear,
            )
        return dz, profile


def stack_five_band(
    four_band_path: Path,
    ddsm_path: Path,
    output_path: Path,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(four_band_path) as src:
        if src.count < 4:
            raise ValueError(f"{four_band_path} has {src.count} bands; need 4")
        rgb_dsm = src.read([1, 2, 3, 4])
        profile = src.profile.copy()

    dz, _ = warp_ddsm_to_grid(four_band_path, ddsm_path)
    ddsm_u8 = ddsm_to_uint8(dz)

    profile.update(count=5, dtype="uint8", nodata=None, compress="deflate")
    profile.pop("photometric", None)
    with rasterio.open(output_path, "w", **profile) as out:
        for i in range(4):
            out.write(rgb_dsm[i], i + 1)
        out.write(ddsm_u8, 5)
        out.set_band_description(1, "red")
        out.set_band_description(2, "green")
        out.set_band_description(3, "blue")
        out.set_band_description(4, "dsm")
        out.set_band_description(5, "ddsm_25_minus_24")

    return {
        "source_four_band": str(four_band_path),
        "output": str(output_path),
        "bands": 5,
        "ddsm_p98_abs_m": float(
            np.percentile(np.abs(dz[np.isfinite(dz)]), 98)
            if np.any(np.isfinite(dz))
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=[24, 25], required=True)
    parser.add_argument(
        "--ddsm",
        type=Path,
        default=Path("25_minus_24_dsm.tif"),
        help="Site-wide difference DSM (default: project-root 25_minus_24_dsm.tif)",
    )
    parser.add_argument(
        "--from-four-band-dir",
        type=Path,
        default=None,
        help="Existing 4-band tile dir (default: segmentation/tiling_rgb_dsm_{year})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: segmentation/tiling_rgb_dsm_ddsm_{year}",
    )
    parser.add_argument(
        "--ortho-dir",
        type=Path,
        default=Path("segmentation/tiling"),
        help="Used only to resolve keys when --from-coco / --tile-keys set",
    )
    parser.add_argument("--tile-keys", type=str, default="")
    parser.add_argument(
        "--from-coco",
        type=Path,
        default=None,
        help="Only build tiles referenced by this COCO dataset for --year",
    )
    args = parser.parse_args()

    if not args.ddsm.exists():
        raise FileNotFoundError(
            f"Difference DSM not found: {args.ddsm} "
            "(export site-wide 25−24 GeoTIFF to the project root)"
        )

    seg_root = Path("segmentation")
    four_dir = args.from_four_band_dir or (seg_root / f"tiling_rgb_dsm_{args.year}")
    out_dir = args.output_dir or (seg_root / f"tiling_rgb_dsm_ddsm_{args.year}")
    if not four_dir.is_dir():
        raise FileNotFoundError(
            f"4-band tile dir not found: {four_dir}. "
            "Build elevation RGB+DSM tiles first, or pass --from-four-band-dir."
        )

    if args.from_coco is not None:
        keys = keys_from_coco(args.from_coco, args.year)
    else:
        keys = parse_tile_keys(args.tile_keys, args.year)

    summary = []
    missing = []
    for key in tqdm(keys, desc=f"rgb+dsm+ddsm/{args.year}"):
        name = tile_filename(key, args.year)
        src = four_dir / name
        if not src.is_file():
            # Also try year-prefixed COCO-style names inside four_dir
            alt = four_dir / f"{args.year}_{name}"
            src = alt if alt.is_file() else src
        if not src.is_file():
            missing.append(name)
            continue
        summary.append(stack_five_band(src, args.ddsm, out_dir / name))

    if missing:
        print(f"WARNING: {len(missing)} 4-band tiles missing (skipping), e.g. {missing[:5]}")

    manifest = out_dir / "build_rgb_dsm_ddsm_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "tiles": len(summary),
                "missing": len(missing),
                "output_dir": str(out_dir),
                "ddsm": str(args.ddsm),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
