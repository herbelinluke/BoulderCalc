#!/usr/bin/env python3
"""Warp DSM to each ortho tile grid and write uint8 DSM-derived training tiles.

Modes:
  hillshade     - classic shaded relief (default)
  local_relief  - DSM minus Gaussian-smoothed DSM (highlights local bumps / boulders)
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
from gpkg_to_coco import ALL_TILES, tile_filename  # noqa: E402


def fill_dem(dem: np.ndarray) -> np.ndarray:
    dem = dem.astype(np.float32)
    if not np.any(np.isfinite(dem)):
        return np.zeros_like(dem, dtype=np.float32)
    fill = float(np.nanmedian(dem[np.isfinite(dem)]))
    return np.where(np.isfinite(dem), dem, fill)


def compute_hillshade(
    dem: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    dem = fill_dem(dem)
    dzdx = np.gradient(dem, pixel_size_x, axis=1)
    dzdy = np.gradient(dem, abs(pixel_size_y), axis=0)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdy, dzdx)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    shaded = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(
        az_rad - aspect
    )
    shaded = (shaded - shaded.min()) / max(shaded.max() - shaded.min(), 1e-6)
    return np.clip(shaded * 255.0, 0, 255).astype(np.uint8)


def compute_local_relief(
    dem: np.ndarray,
    pixel_size: float,
    radius_m: float = 10.0,
) -> np.ndarray:
    """Local relief = elevation minus large-scale smoothed surface (meters)."""
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


def build_tile_dsm_image(
    ortho_path: Path,
    dsm_path: Path,
    output_path: Path,
    mode: str,
    azimuth: float,
    altitude: float,
    relief_radius_m: float,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dem, profile = warp_dem_to_tile(ortho_path, dsm_path)
    pixel_size = abs(profile["transform"].a)

    if mode == "hillshade":
        gray = compute_hillshade(
            dem,
            pixel_size_x=pixel_size,
            pixel_size_y=abs(profile["transform"].e),
            azimuth=azimuth,
            altitude=altitude,
        )
    elif mode == "local_relief":
        relief = compute_local_relief(dem, pixel_size=pixel_size, radius_m=relief_radius_m)
        gray = relief_to_uint8(relief)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 3-band uint8 for Detectron2 / OpenCV compatibility
    profile.update(count=3, dtype="uint8", nodata=None, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as out:
        for band in (1, 2, 3):
            out.write(gray, band)

    return {
        "ortho": str(ortho_path),
        "output": str(output_path),
        "mode": mode,
        "shape": [int(gray.shape[0]), int(gray.shape[1])],
    }


def parse_tile_keys(value: str) -> list[str]:
    if not value:
        return list(ALL_TILES)
    return [k.strip() for k in value.split(",") if k.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsm",
        type=Path,
        default=Path("/home/herbs/Documents/tamucc/2025/25IniSouthDSM.tif"),
    )
    parser.add_argument(
        "--ortho-dir",
        type=Path,
        default=Path("/home/herbs/Documents/tamucc/segmentation/tiling"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: tiling_hillshade or tiling_local_relief depending on --mode",
    )
    parser.add_argument(
        "--mode",
        choices=["hillshade", "local_relief"],
        default="hillshade",
    )
    parser.add_argument(
        "--tile-keys",
        type=str,
        default="",
        help="Comma-separated tile keys (e.g. 08_22,07_24). Default: all v2 ALL_TILES.",
    )
    parser.add_argument("--azimuth", type=float, default=315.0)
    parser.add_argument("--altitude", type=float, default=45.0)
    parser.add_argument(
        "--relief-radius-m",
        type=float,
        default=10.0,
        help="Gaussian smoothing radius in meters for local_relief mode (default 10).",
    )
    args = parser.parse_args()

    if not args.dsm.exists():
        raise FileNotFoundError(args.dsm)

    seg_root = args.ortho_dir.parent
    if args.output_dir is None:
        subdir = "tiling_local_relief" if args.mode == "local_relief" else "tiling_hillshade"
        args.output_dir = seg_root / subdir

    keys = parse_tile_keys(args.tile_keys)
    summary = []
    for key in tqdm(keys, desc=args.mode):
        ortho_path = args.ortho_dir / tile_filename(key)
        out_path = args.output_dir / tile_filename(key)
        if not ortho_path.exists():
            raise FileNotFoundError(ortho_path)
        summary.append(
            build_tile_dsm_image(
                ortho_path,
                args.dsm,
                out_path,
                mode=args.mode,
                azimuth=args.azimuth,
                altitude=args.altitude,
                relief_radius_m=args.relief_radius_m,
            )
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
