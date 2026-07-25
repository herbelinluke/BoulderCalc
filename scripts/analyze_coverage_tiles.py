#!/usr/bin/env python3
"""Analyze coverage-ignore + 4-band strip artifacts on selected tiles (GDAL I/O)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal, gdalconst
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Prefer array helpers from coverage_ignore (stub rasterio if needed).
try:
    from coverage_ignore import (  # noqa: E402
        DEFAULT_BLUR_M,
        DEFAULT_RGB_MAX,
        build_coverage_mask_from_arrays,
        dilate_mask_m,
        near_black_mask,
        border_connected_mask,
    )
except ImportError:
    import importlib.util
    import types

    for name in (
        "rasterio",
        "rasterio.features",
        "rasterio.warp",
        "shapely",
        "shapely.geometry",
        "shapely.ops",
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["rasterio.warp"].Resampling = types.SimpleNamespace(bilinear=1)
    sys.modules["rasterio.warp"].reproject = None
    sys.modules["shapely.geometry"].Polygon = object
    sys.modules["shapely.geometry"].shape = lambda g: g
    sys.modules["shapely.ops"].unary_union = lambda x: x
    spec = importlib.util.spec_from_file_location(
        "coverage_ignore", Path(__file__).resolve().parent / "coverage_ignore.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    build_coverage_mask_from_arrays = mod.build_coverage_mask_from_arrays
    DEFAULT_BLUR_M = mod.DEFAULT_BLUR_M
    DEFAULT_RGB_MAX = mod.DEFAULT_RGB_MAX


gdal.UseExceptions()

TILES = [
    (24, "04_46"),
    (24, "05_46"),
    (24, "06_45"),
    (24, "06_41"),
    (24, "07_42"),
    (24, "16_09"),
    (25, "05_34"),
    (25, "08_27"),
]


def ortho_path(root: Path, year: int, key: str) -> Path:
    if year == 24:
        return (
            root
            / "segmentation"
            / "tiling"
            / "24"
            / f"Sites1and2_2024_Orthomosaic_{key}.tif"
        )
    return root / "segmentation" / "tiling" / "25" / f"25IniSouthOrt_{key}.tif"


def fourband_path(root: Path, year: int, key: str, mode: str) -> Path | None:
    if mode == "elevation":
        d = root / "segmentation" / f"tiling_rgb_dsm_{year}"
    else:
        d = root / "segmentation" / f"tiling_rgb_dsm_local_relief_{year}"
    if year == 24:
        p = d / f"Sites1and2_2024_Orthomosaic_{key}.tif"
    else:
        p = d / f"25IniSouthOrt_{key}.tif"
    return p if p.is_file() else None


def dsm_path(root: Path, year: int) -> Path:
    if year == 24:
        return root / "2024" / "Sites1and2_2024_DSM_30mm.tif"
    return root / "2025" / "25IniSouthDSM.tif"


def read_rgb(path: Path) -> tuple[np.ndarray, float, list[float]]:
    ds = gdal.Open(str(path), gdalconst.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    bands = [ds.GetRasterBand(i).ReadAsArray() for i in (1, 2, 3)]
    rgb = np.stack(bands, axis=0).astype(np.float32)
    gt = ds.GetGeoTransform()
    pixel_size = abs(gt[1])
    ds = None
    return rgb, pixel_size, list(gt)


def warp_dsm_valid(ortho: Path, dsm: Path) -> np.ndarray:
    """Return bool valid mask on ortho grid (True = finite DSM)."""
    ods = gdal.Open(str(ortho), gdalconst.GA_ReadOnly)
    if ods is None:
        raise FileNotFoundError(ortho)
    w, h = ods.RasterXSize, ods.RasterYSize
    gt = ods.GetGeoTransform()
    proj = ods.GetProjection()
    ods = None

    # Warp DSM to ortho grid; nodata → NaN via float32 dest init.
    mem = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Float32)
    mem.SetGeoTransform(gt)
    mem.SetProjection(proj)
    band = mem.GetRasterBand(1)
    band.WriteArray(np.full((h, w), np.nan, dtype=np.float32))
    band.SetNoDataValue(np.nan)

    warp_opts = gdal.WarpOptions(
        format="MEM",
        outputType=gdal.GDT_Float32,
        dstSRS=proj,
        outputBounds=(
            gt[0],
            gt[3] + h * gt[5],
            gt[0] + w * gt[1],
            gt[3],
        ),
        width=w,
        height=h,
        resampleAlg="bilinear",
        dstNodata=np.nan,
        multithread=True,
    )
    out = gdal.Warp(destNameOrDestDS="", srcDSOrSrcDSTab=str(dsm), options=warp_opts)
    arr = out.GetRasterBand(1).ReadAsArray().astype(np.float32)
    src = gdal.Open(str(dsm), gdalconst.GA_ReadOnly)
    src_nodata = src.GetRasterBand(1).GetNoDataValue()
    src = None
    out = None
    mem = None
    if src_nodata is not None and np.isfinite(src_nodata):
        arr = np.where(np.isclose(arr, float(src_nodata)), np.nan, arr)
    # gdal Warp may leave 0 outside; treat exact 0 with no neighbors as suspect
    # only if source nodata wasn't set — keep NaN-only for validity.
    return np.isfinite(arr)


def read_band4(path: Path) -> np.ndarray:
    ds = gdal.Open(str(path), gdalconst.GA_ReadOnly)
    if ds is None or ds.RasterCount < 4:
        raise ValueError(f"need 4 bands: {path}")
    b4 = ds.GetRasterBand(4).ReadAsArray().astype(np.float32)
    ds = None
    return b4


def save_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    *,
    extra_red: np.ndarray | None = None,
) -> None:
    """RGB preview with coverage in cyan; optional extra_red (strips) in red."""
    # Downsample for preview if large
    h, w = mask.shape
    step = max(1, max(h, w) // 1024)
    r = rgb[0, ::step, ::step]
    g = rgb[1, ::step, ::step]
    b = rgb[2, ::step, ::step]
    m = mask[::step, ::step]
    vis = np.stack([r, g, b], axis=-1)
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    # Dim coverage and tint cyan
    cov = m.astype(bool)
    vis[cov] = (vis[cov].astype(np.float32) * 0.35 + np.array([0, 180, 200]) * 0.65).astype(
        np.uint8
    )
    if extra_red is not None:
        er = extra_red[::step, ::step].astype(bool)
        vis[er] = (vis[er].astype(np.float32) * 0.3 + np.array([220, 40, 40]) * 0.7).astype(
            np.uint8
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(out_path)


def analyze_strip_artifact(rgb: np.ndarray, b4: np.ndarray) -> dict:
    """Find regions where RGB has texture but band4 is flat/zero (shine-through)."""
    # Local RGB variance (approx texture)
    gray = np.mean(rgb, axis=0)
    # 8x8 block stats
    bh = bw = 32
    h, w = gray.shape
    shine = np.zeros((h, w), dtype=bool)
    n_blocks = 0
    n_shine = 0
    for y0 in range(0, h - bh + 1, bh):
        for x0 in range(0, w - bw + 1, bw):
            n_blocks += 1
            g = gray[y0 : y0 + bh, x0 : x0 + bw]
            z = b4[y0 : y0 + bh, x0 : x0 + bw]
            rgb_std = float(np.std(g))
            b4_std = float(np.std(z))
            b4_mean = float(np.mean(z))
            # Textured RGB but flat/near-zero band4
            if rgb_std > 8.0 and b4_std < 3.0 and b4_mean < 15.0:
                shine[y0 : y0 + bh, x0 : x0 + bw] = True
                n_shine += 1
    return {
        "shine_blocks": n_shine,
        "total_blocks": n_blocks,
        "shine_fraction": n_shine / max(n_blocks, 1),
        "shine_mask": shine,
        "band4_zero_frac": float(np.mean(b4 <= 1)),
        "band4_mean": float(np.mean(b4)),
        "band4_std": float(np.std(b4)),
    }


def main() -> None:
    root = Path(".").resolve()
    out_dir = root / "segmentation" / "coverage_debug_tiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for year, key in TILES:
        ortho = ortho_path(root, year, key)
        row: dict = {"year": year, "key": key, "ortho": str(ortho), "exists": ortho.is_file()}
        if not ortho.is_file():
            summary.append(row)
            print(f"[MISS] {year} {key}")
            continue

        rgb, pixel_size, _gt = read_rgb(ortho)
        dsm = dsm_path(root, year)
        valid = None
        if dsm.is_file():
            print(f"[warp] {year} {key} DSM…", flush=True)
            valid = warp_dsm_valid(ortho, dsm)
            row["dsm_valid_frac"] = float(np.mean(valid))
        else:
            row["dsm_valid_frac"] = None

        mask = build_coverage_mask_from_arrays(
            rgb.astype(np.uint8),
            pixel_size=pixel_size,
            dsm_valid=valid,
            rgb_max=DEFAULT_RGB_MAX,
            blur_m=DEFAULT_BLUR_M,
        )
        row["coverage_frac"] = float(np.mean(mask))
        row["coverage_pixels"] = int(np.count_nonzero(mask))
        row["rgb_nearblack_frac"] = float(
            np.mean(np.max(rgb, axis=0) <= DEFAULT_RGB_MAX)
        )
        row["pixel_size_m"] = pixel_size

        # Near-black border void alone (no dilate) for diagnostics
        black = np.max(rgb, axis=0) <= DEFAULT_RGB_MAX
        from coverage_ignore import border_connected_mask as bcm

        void0 = bcm(black)
        row["border_void_frac_no_dilate"] = float(np.mean(void0))

        overlay = out_dir / f"{year}_{key}_coverage_overlay.png"
        mask_png = out_dir / f"{year}_{key}_coverage_mask.png"
        save_overlay(rgb, mask, overlay)
        Image.fromarray((mask.astype(np.uint8) * 255)[::2, ::2]).save(mask_png)

        # 4-band strip analysis if present
        for mode in ("elevation", "local_relief"):
            fb = fourband_path(root, year, key, mode)
            if fb is None:
                row[f"fourband_{mode}"] = None
                continue
            b4 = read_band4(fb)
            art = analyze_strip_artifact(rgb, b4)
            shine = art.pop("shine_mask")
            row[f"fourband_{mode}"] = {k: v for k, v in art.items()}
            strip_path = out_dir / f"{year}_{key}_{mode}_shine_strips.png"
            save_overlay(rgb, mask, strip_path, extra_red=shine)
            # Also save band4 preview
            b4u = np.clip(b4, 0, 255).astype(np.uint8)
            Image.fromarray(b4u[::2, ::2]).save(out_dir / f"{year}_{key}_{mode}_band4.png")

        summary.append(row)
        print(
            f"[OK] {year} {key} coverage={row['coverage_frac']:.3f} "
            f"dsm_valid={row.get('dsm_valid_frac')} "
            f"elev={row.get('fourband_elevation')} lr={row.get('fourband_local_relief')}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir}/summary.json and overlays", flush=True)


if __name__ == "__main__":
    main()
