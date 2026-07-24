#!/usr/bin/env python3
"""Clip DSM to each ortho tile grid and write 4-band RGB+DSM GeoTIFFs.

Band order:
  1-3  RGB from the orthomosaic tile (uint8)
  4    DSM warped to the ortho grid, scaled to uint8

DSM scaling modes:
  elevation     per-tile 2–98 percentile stretch of absolute elevation (default)
  local_relief  DSM minus Gaussian-smoothed DSM, then percentile stretch

Run from the project root (directory that contains ``segmentation/`` and
``2024/`` / ``2025/``). Paths are relative so the same commands work on
Linux and Windows.

Example:
  python BoulderCalculator/scripts/build_rgb_dsm_tiles.py --year 25
  python BoulderCalculator/scripts/build_rgb_dsm_tiles.py --year 24 --tile-keys 14_15,15_10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpkg_to_coco import TILES_24, TILES_25, resolve_tile_path, tile_filename  # noqa: E402


def fill_dem(dem: np.ndarray) -> np.ndarray:
    dem = dem.astype(np.float32)
    if not np.any(np.isfinite(dem)):
        return np.zeros_like(dem, dtype=np.float32)
    fill = float(np.nanmedian(dem[np.isfinite(dem)]))
    return np.where(np.isfinite(dem), dem, fill)


def compute_local_relief(
    dem: np.ndarray,
    pixel_size: float,
    radius_m: float = 10.0,
) -> np.ndarray:
    dem = fill_dem(dem)
    sigma_px = max(1.0, radius_m / pixel_size)
    smooth = gaussian_filter(dem, sigma=sigma_px)
    return dem - smooth


def relief_to_uint8(relief: np.ndarray) -> np.ndarray:
    finite = relief[np.isfinite(relief)]
    if finite.size == 0:
        return np.zeros(relief.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [2, 98])
    scaled = (relief - lo) / max(hi - lo, 1e-6)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def warp_dem_to_tile(ortho_path: Path, dsm_path: Path) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    with rasterio.open(ortho_path) as ortho:
        dem = np.zeros((ortho.height, ortho.width), dtype=np.float32)
        with rasterio.open(dsm_path) as dsm:
            reproject(
                source=rasterio.band(dsm, 1),
                destination=dem,
                src_transform=dsm.transform,
                src_crs=dsm.crs,
                dst_transform=ortho.transform,
                dst_crs=ortho.crs,
                resampling=Resampling.bilinear,
            )
        return dem, ortho.profile.copy()


def elevation_to_uint8(dem: np.ndarray) -> np.ndarray:
    dem = fill_dem(dem)
    finite = dem[np.isfinite(dem)]
    if finite.size == 0:
        return np.zeros(dem.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [2, 98])
    scaled = (dem - lo) / max(hi - lo, 1e-6)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def read_rgb_uint8(ortho_path: Path) -> np.ndarray:
    with rasterio.open(ortho_path) as ds:
        if ds.count < 3:
            raise ValueError(f"{ortho_path} has {ds.count} bands; need RGB")
        rgb = ds.read([1, 2, 3])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def build_rgb_dsm_tile(
    ortho_path: Path,
    dsm_path: Path,
    output_path: Path,
    dsm_mode: str,
    relief_radius_m: float,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = read_rgb_uint8(ortho_path)
    dem, profile = warp_dem_to_tile(ortho_path, dsm_path)
    pixel_size = abs(profile["transform"].a)

    if dsm_mode == "elevation":
        dsm_u8 = elevation_to_uint8(dem)
    elif dsm_mode == "local_relief":
        relief = compute_local_relief(dem, pixel_size=pixel_size, radius_m=relief_radius_m)
        dsm_u8 = relief_to_uint8(relief)
    else:
        raise ValueError(f"Unknown dsm_mode: {dsm_mode}")

    profile.update(
        count=4,
        dtype="uint8",
        nodata=None,
        compress="deflate",
    )
    profile.pop("photometric", None)
    with rasterio.open(output_path, "w", **profile) as out:
        out.write(rgb[0], 1)
        out.write(rgb[1], 2)
        out.write(rgb[2], 3)
        out.write(dsm_u8, 4)
        out.set_band_description(1, "red")
        out.set_band_description(2, "green")
        out.set_band_description(3, "blue")
        out.set_band_description(4, f"dsm_{dsm_mode}")

    return {
        "ortho": str(ortho_path),
        "output": str(output_path),
        "dsm_mode": dsm_mode,
        "shape": [int(dsm_u8.shape[0]), int(dsm_u8.shape[1])],
        "bands": 4,
    }


def parse_tile_keys(value: str, year: int) -> list[str]:
    if not value:
        return list(TILES_24 if year == 24 else TILES_25)
    return [k.strip() for k in value.split(",") if k.strip()]


def keys_from_coco(coco_dir: Path, year: int) -> list[str]:
    """Derive RR_CC keys from COCO image file_names for the given year.

    Accepts both raw ortho names and gpkg_to_coco year-prefixed copies, e.g.:
      25IniSouthOrt_04_34.tif
      25_25IniSouthOrt_04_34.tif
      Sites1and2_2024_Orthomosaic_11_07.tif
      24_Sites1and2_2024_Orthomosaic_11_07.tif
    """
    keys: list[str] = []
    seen: set[str] = set()
    year_prefix = f"{year}_"
    for ann_name in (
        "train_annotations.json",
        "validation_annotations.json",
        "testing_annotations.json",
    ):
        path = coco_dir / ann_name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for image in data["images"]:
            name = image["file_name"]
            stem = Path(name).stem
            # COCO RGB copies are named ``{year}_{ortho_basename}``.
            if stem.startswith(year_prefix):
                stem = stem[len(year_prefix) :]
            parts = stem.split("_")
            if year == 25 and stem.startswith("25IniSouthOrt_") and len(parts) >= 3:
                key = f"{int(parts[-2])}_{int(parts[-1])}"
            elif year == 24 and "2024_Orthomosaic" in stem and len(parts) >= 2:
                key = f"{int(parts[-2])}_{int(parts[-1])}"
            else:
                continue
            if key not in seen:
                seen.add(key)
                keys.append(key)
    if not keys:
        raise ValueError(f"No year={year} tile names found under {coco_dir}")
    return keys


def default_dsm(project_root: Path, year: int) -> Path:
    if year == 24:
        return project_root / "2024" / "Sites1and2_2024_DSM_30mm.tif"
    return project_root / "2025" / "25IniSouthDSM.tif"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsm",
        type=Path,
        default=None,
        help="DSM GeoTIFF (default: year-matched file under project 2024/ or 2025/)",
    )
    parser.add_argument(
        "--ortho-dir",
        type=Path,
        default=Path("segmentation/tiling"),
        help="Ortho tile root (layout: tiling/{24,25}/...). Default: segmentation/tiling",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: segmentation/tiling_rgb_dsm_{year}",
    )
    parser.add_argument("--year", type=int, choices=[24, 25], required=True)
    parser.add_argument(
        "--dsm-mode",
        choices=["elevation", "local_relief"],
        default="elevation",
        help="How to scale DSM into band 4 uint8 (default: elevation).",
    )
    parser.add_argument(
        "--tile-keys",
        type=str,
        default="",
        help="Comma-separated tile keys (e.g. 14_15,15_10). Default: all tiles for --year.",
    )
    parser.add_argument(
        "--from-coco",
        type=Path,
        default=None,
        help="If set, only build tiles referenced by this COCO dataset dir for --year.",
    )
    parser.add_argument(
        "--relief-radius-m",
        type=float,
        default=10.0,
        help="Gaussian radius (m) for local_relief mode.",
    )
    args = parser.parse_args()

    # CWD is the project root (same relative layout on Linux/Windows).
    project_root = Path(".")
    if args.dsm is None:
        args.dsm = default_dsm(project_root, args.year)
    if not args.dsm.exists():
        raise FileNotFoundError(
            f"DSM not found: {args.dsm} (run from project root, or pass --dsm)"
        )

    seg_root = args.ortho_dir.parent if args.ortho_dir.name == "tiling" else args.ortho_dir
    if args.output_dir is None:
        if args.dsm_mode == "local_relief":
            args.output_dir = seg_root / f"tiling_rgb_dsm_local_relief_{args.year}"
        else:
            args.output_dir = seg_root / f"tiling_rgb_dsm_{args.year}"

    if args.from_coco is not None:
        keys = keys_from_coco(args.from_coco, args.year)
    else:
        keys = parse_tile_keys(args.tile_keys, args.year)
    summary = []
    for key in tqdm(keys, desc=f"rgb+dsm/{args.year}"):
        ortho_path = resolve_tile_path(args.ortho_dir, key, args.year)
        out_path = args.output_dir / tile_filename(key, args.year)
        summary.append(
            build_rgb_dsm_tile(
                ortho_path,
                args.dsm,
                out_path,
                dsm_mode=args.dsm_mode,
                relief_radius_m=args.relief_radius_m,
            )
        )

    manifest = args.output_dir / "build_rgb_dsm_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"tiles": len(summary), "output_dir": str(args.output_dir), "manifest": str(manifest)}, indent=2))

    from run_provenance import write_tiling_provenance

    # Parents are path references only; DSM is a binary GeoTIFF — do not open it
    # as text when writing provenance (Windows cp1252 / charmap decode crash).
    write_tiling_provenance(
        args.output_dir,
        tool="build_rgb_dsm_tiles.py",
        flags={
            "year": args.year,
            "dsm": str(args.dsm),
            "ortho_dir": str(args.ortho_dir),
            "dsm_mode": args.dsm_mode,
            "relief_radius_m": args.relief_radius_m,
            "from_coco": str(args.from_coco) if args.from_coco else None,
            "tile_keys": args.tile_keys,
        },
        tiles_summary=summary,
        parents=[args.ortho_dir, args.dsm],
        extra={"legacy_manifest_file": "build_rgb_dsm_manifest.json"},
    )


if __name__ == "__main__":
    main()
