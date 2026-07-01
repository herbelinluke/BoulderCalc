#!/usr/bin/env python3
"""Warp DSM to each ortho tile grid and write uint8 hillshade GeoTIFFs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geojson_tiles_to_coco import TILE_MAP  # noqa: E402


def compute_hillshade(
    dem: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    dem = dem.astype(np.float32)
    dem[~np.isfinite(dem)] = np.nan
    fill = float(np.nanmedian(dem)) if np.any(np.isfinite(dem)) else 0.0
    dem = np.where(np.isfinite(dem), dem, fill)

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


def build_tile_hillshade(
    ortho_path: Path,
    dsm_path: Path,
    output_path: Path,
    azimuth: float,
    altitude: float,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

        hillshade = compute_hillshade(
            dem,
            pixel_size_x=abs(ortho.transform.a),
            pixel_size_y=abs(ortho.transform.e),
            azimuth=azimuth,
            altitude=altitude,
        )

        profile = ortho.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=None, compress="deflate")
        with rasterio.open(output_path, "w", **profile) as out:
            out.write(hillshade, 1)

    return {
        "ortho": str(ortho_path),
        "output": str(output_path),
        "shape": [int(hillshade.shape[0]), int(hillshade.shape[1])],
    }


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
        default=Path("/home/herbs/Documents/tamucc/segmentation/tiling_hillshade"),
    )
    parser.add_argument(
        "--tile-keys",
        type=str,
        default="",
        help="Comma-separated tile keys (e.g. 25_05_31,25_05_32). Default: all TILE_MAP keys.",
    )
    parser.add_argument("--azimuth", type=float, default=315.0)
    parser.add_argument("--altitude", type=float, default=45.0)
    args = parser.parse_args()

    if not args.dsm.exists():
        raise FileNotFoundError(args.dsm)

    keys = [k.strip() for k in args.tile_keys.split(",") if k.strip()] or list(TILE_MAP.keys())
    summary = []
    for key in keys:
        if key not in TILE_MAP:
            raise KeyError(f"Unknown tile key: {key}")
        ortho_name = TILE_MAP[key]
        ortho_path = args.ortho_dir / ortho_name
        out_path = args.output_dir / ortho_name
        if not ortho_path.exists():
            raise FileNotFoundError(ortho_path)
        summary.append(
            build_tile_hillshade(
                ortho_path,
                args.dsm,
                out_path,
                azimuth=args.azimuth,
                altitude=args.altitude,
            )
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
