#!/usr/bin/env python3
"""Quantitative compare Gaussian vs morphological-opening local-relief backgrounds.

Does NOT modify the production pipeline. Writes metrics + crop previews under
``segmentation/relief_opening_study/``.

Run from project root (needs GDAL Python bindings + scipy/numpy/geopandas).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal, gdalconst
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_opening
from shapely.geometry import Point
from shapely import STRtree

gdal.UseExceptions()

ROOT = Path(".").resolve()
OUT = ROOT / "segmentation" / "relief_opening_study"

# Problem tiles from investigation (dense deposits) + natural-terrain controls.
DENSE_TILES = [
    (25, "04_35"),
    (25, "04_36"),
    (25, "05_32"),
    (25, "11_14"),
]
CONTROL_TILES = [
    (24, "15_32"),
    (24, "15_31"),
    (25, "7_39"),
    (25, "13_10"),
]

# SE diameters from measured dense diam/NN distribution (investigation.json).
# diam_p90≈1.54m, nn_p50≈1.73m → sweep above boulder footprint, below old 10m σ.
OPENING_SE_M = [2.0, 3.0, 5.0, 8.0]
GAUSS_SIGMA_M = 10.0
CLIP_M = 0.5


def ortho_path(year: int, key: str) -> Path:
    if year == 24:
        return ROOT / "segmentation/tiling/24" / f"Sites1and2_2024_Orthomosaic_{key}.tif"
    return ROOT / "segmentation/tiling/25" / f"25IniSouthOrt_{key}.tif"


def dsm_path(year: int) -> Path:
    if year == 24:
        return ROOT / "2024/Sites1and2_2024_DSM_30mm.tif"
    return ROOT / "2025/25IniSouthDSM.tif"


def disk_footprint(radius_px: int) -> np.ndarray:
    r = max(1, int(radius_px))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= (r * r)


def fill_dem(dem: np.ndarray) -> np.ndarray:
    dem = dem.astype(np.float32)
    finite = np.isfinite(dem)
    if not np.any(finite):
        return np.zeros_like(dem, dtype=np.float32)
    fill = float(np.nanmedian(dem[finite]))
    return np.where(finite, dem, fill)


def warp_dem(ortho: Path, dsm: Path, pad_m: float) -> tuple[np.ndarray, np.ndarray, float, int, int, int]:
    """Return dem_pad, valid_pad, pixel_size, pad_px, h, w (parent size)."""
    ods = gdal.Open(str(ortho), gdalconst.GA_ReadOnly)
    w, h = ods.RasterXSize, ods.RasterYSize
    gt = ods.GetGeoTransform()
    proj = ods.GetProjection()
    pixel_size = abs(gt[1])
    pad_px = int(math.ceil(pad_m / max(pixel_size, 1e-9)))
    new_w, new_h = w + 2 * pad_px, h + 2 * pad_px
    # Expanded geotransform (pad to NW)
    gt2 = (
        gt[0] - pad_px * gt[1],
        gt[1],
        0.0,
        gt[3] - pad_px * gt[5],
        0.0,
        gt[5],
    )
    minx = gt2[0]
    maxx = gt2[0] + new_w * gt2[1]
    maxy = gt2[3]
    miny = gt2[3] + new_h * gt2[5]
    warp_opts = gdal.WarpOptions(
        format="MEM",
        outputType=gdal.GDT_Float32,
        dstSRS=proj,
        outputBounds=(minx, miny, maxx, maxy),
        width=new_w,
        height=new_h,
        resampleAlg="bilinear",
        dstNodata=float("nan"),
        multithread=True,
    )
    out = gdal.Warp("", str(dsm), options=warp_opts)
    dem = out.GetRasterBand(1).ReadAsArray().astype(np.float32)
    src = gdal.Open(str(dsm))
    src_nodata = src.GetRasterBand(1).GetNoDataValue()
    if src_nodata is not None and np.isfinite(src_nodata):
        dem = np.where(np.isclose(dem, float(src_nodata)), np.nan, dem)
    valid = np.isfinite(dem)
    ods = None
    out = None
    src = None
    return dem, valid, pixel_size, pad_px, h, w


def relief_gaussian(dem_filled: np.ndarray, pixel_size: float, sigma_m: float) -> np.ndarray:
    sigma_px = max(1.0, sigma_m / pixel_size)
    smooth = gaussian_filter(dem_filled, sigma=sigma_px)
    return dem_filled - smooth


def relief_opening(dem_filled: np.ndarray, pixel_size: float, se_diam_m: float) -> np.ndarray:
    """Background = grey opening with flat square SE of given side length (meters).

    Square ``size=`` is used (not a disk footprint) so scipy can run the
    separable morphology path — disk footprints at multi-metre SE on 2k tiles
    are impractically slow for this study.
    """
    side_px = max(3, int(round(se_diam_m / pixel_size)))
    if side_px % 2 == 0:
        side_px += 1
    bg = grey_opening(dem_filled, size=(side_px, side_px))
    return dem_filled - bg


def to_u8_pos(relief: np.ndarray, clip_m: float = CLIP_M) -> np.ndarray:
    r = np.maximum(relief.astype(np.float32), 0.0)
    return np.clip(r / max(clip_m, 1e-6), 0, 1) * 255.0


def load_boulders(year: int) -> gpd.GeoDataFrame:
    path = ROOT / "segmentation/annotations" / f"july14_{year}.gpkg"
    gdf = gpd.read_file(path)
    cls_col = "Class" if "Class" in gdf.columns else "class"
    def parse_cls(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0
        if isinstance(v, (int, float)) and int(v) in (0, 1):
            return int(v)
        return 1 if "deposit" in str(v).lower() else 0
    gdf = gdf.copy()
    gdf["cls"] = gdf[cls_col].map(parse_cls)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs(25829)
    gdf = gdf[gdf["cls"] == 0].copy()
    gdf["cent_x"] = gdf.geometry.centroid.x
    gdf["cent_y"] = gdf.geometry.centroid.y
    return gdf


def dense_flags(boulders: gpd.GeoDataFrame, deposits: gpd.GeoDataFrame) -> np.ndarray:
    if len(deposits) == 0:
        from scipy.spatial import cKDTree

        coords = np.column_stack([boulders.cent_x.values, boulders.cent_y.values])
        tree = cKDTree(coords)
        neigh = tree.query_ball_point(coords, r=3.0)
        return np.array([len(ix) >= 5 for ix in neigh])
    geoms = list(deposits.geometry)
    tree = STRtree(geoms)
    flags = []
    for x, y in zip(boulders.cent_x, boulders.cent_y):
        p = Point(x, y)
        idxs = tree.query(p)
        flags.append(any(geoms[j].contains(p) or geoms[j].intersects(p) for j in idxs))
    return np.array(flags)


def sample_relief_at_points(
    relief: np.ndarray,
    transform: tuple,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Sample relief (meters) at map coords; NaN if outside."""
    gt = transform
    # col = (x - gt0) / gt1 ; row = (y - gt3) / gt5
    cols = np.floor((xs - gt[0]) / gt[1]).astype(int)
    rows = np.floor((ys - gt[3]) / gt[5]).astype(int)
    h, w = relief.shape
    out = np.full(len(xs), np.nan, dtype=np.float32)
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out[ok] = relief[rows[ok], cols[ok]]
    return out


def save_crop(rgb: np.ndarray, relief_u8: np.ndarray, cy: int, cx: int, half: int, path: Path) -> None:
    y0, y1 = max(0, cy - half), min(rgb.shape[1], cy + half)
    x0, x1 = max(0, cx - half), min(rgb.shape[2], cx + half)
    r = rgb[:, y0:y1, x0:x1]
    rel = relief_u8[y0:y1, x0:x1]
    # side-by-side RGB | relief
    rgb_img = np.clip(np.transpose(r, (1, 2, 0)), 0, 255).astype(np.uint8)
    rel_img = np.stack([rel, rel, rel], axis=-1).astype(np.uint8)
    # match heights
    h = min(rgb_img.shape[0], rel_img.shape[0])
    w = min(rgb_img.shape[1], rel_img.shape[1])
    combo = np.concatenate([rgb_img[:h, :w], rel_img[:h, :w]], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combo).save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inv = json.loads((OUT / "investigation.json").read_text()) if (OUT / "investigation.json").is_file() else {}

    methods = [("gaussian_sigma_m", GAUSS_SIGMA_M, "gaussian")]
    for se in OPENING_SE_M:
        methods.append((f"opening_se_m", se, "opening"))

    # Pad: Gaussian needs ~3σ; opening needs ~half SE + small margin.
    # Cap at 60 m for this study (same production default context).
    max_pad_m = min(
        60.0,
        max(3.0 * GAUSS_SIGMA_M, max(OPENING_SE_M) / 2.0 + 5.0),
    )

    boulder_rows = []
    control_rows = []
    crop_meta = []

    # Preload deposits for dense flags
    dep25 = gpd.read_file(ROOT / "segmentation/annotations/july14_25.gpkg").to_crs(25829)
    dep25 = dep25[dep25.geometry.notna() & ~dep25.geometry.is_empty]
    if "Class" in dep25.columns:
        dep25 = dep25[dep25["Class"].astype(str).str.lower().str.contains("deposit") | (dep25["Class"] == 1)]

    for year, key in DENSE_TILES + CONTROL_TILES:
        ortho = ortho_path(year, key)
        if not ortho.is_file():
            print(f"[skip missing] {year}_{key}")
            continue
        is_control = (year, key) in CONTROL_TILES
        print(f"[tile] {year}_{key} control={is_control} pad_m={max_pad_m:.1f}", flush=True)
        dem_pad, valid_pad, pixel_size, pad_px, h, w = warp_dem(
            ortho, dsm_path(year), max_pad_m
        )
        filled = fill_dem(dem_pad)
        # Parent-window geotransform for sampling
        ods = gdal.Open(str(ortho))
        gt = ods.GetGeoTransform()
        rgb = np.stack([ods.GetRasterBand(i).ReadAsArray() for i in (1, 2, 3)], 0)
        ods = None

        # Compute each method on padded DEM, then crop to parent
        reliefs: dict[str, np.ndarray] = {}
        for kind_key, param, kind in methods:
            if kind == "gaussian":
                name = f"gauss_s{param:g}"
                rel_pad = relief_gaussian(filled, pixel_size, param)
            else:
                name = f"open_d{param:g}"
                rel_pad = relief_opening(filled, pixel_size, param)
            rel = rel_pad[pad_px : pad_px + h, pad_px : pad_px + w]
            valid = valid_pad[pad_px : pad_px + h, pad_px : pad_px + w]
            rel = np.where(valid, rel, np.nan)
            reliefs[name] = rel

        if is_control:
            # False-positive proxy: mean/p90 of positive relief on valid pixels
            for name, rel in reliefs.items():
                pos = np.maximum(np.nan_to_num(rel, nan=0.0), 0.0)
                valid = np.isfinite(rel)
                if not np.any(valid):
                    continue
                control_rows.append(
                    {
                        "year": year,
                        "key": key,
                        "method": name,
                        "n_valid": int(valid.sum()),
                        "pos_mean_m": float(pos[valid].mean()),
                        "pos_p50_m": float(np.percentile(pos[valid], 50)),
                        "pos_p90_m": float(np.percentile(pos[valid], 90)),
                        "pos_p99_m": float(np.percentile(pos[valid], 99)),
                        "frac_gt_0_25m": float(np.mean(pos[valid] > 0.25)),
                        "frac_gt_0_5m": float(np.mean(pos[valid] >= 0.5)),
                    }
                )
            continue

        # Dense tile: sample at boulder centroids
        b = load_boulders(year)
        # Keep boulders whose centroid falls in this tile
        minx, maxy = gt[0], gt[3]
        maxx = minx + w * gt[1]
        miny = maxy + h * gt[5]
        in_tile = (
            (b.cent_x >= minx)
            & (b.cent_x <= maxx)
            & (b.cent_y >= miny)
            & (b.cent_y <= maxy)
        )
        b = b.loc[in_tile].copy()
        if len(b) == 0:
            print(f"  no boulders in tile")
            continue
        dens = dense_flags(b, dep25 if year == 25 else gpd.GeoDataFrame(geometry=[], crs=25829))
        for name, rel in reliefs.items():
            vals = sample_relief_at_points(rel, gt, b.cent_x.values, b.cent_y.values)
            for i, (v, dflag) in enumerate(zip(vals, dens)):
                if not np.isfinite(v):
                    continue
                boulder_rows.append(
                    {
                        "year": year,
                        "key": key,
                        "method": name,
                        "dense": bool(dflag),
                        "relief_m": float(max(v, 0.0)),
                        "sat_0_5": bool(v >= CLIP_M),
                    }
                )

        # Crops centered on densest boulder cluster centroid
        dens_b = b.loc[dens]
        if len(dens_b):
            cx_m = float(dens_b.cent_x.mean())
            cy_m = float(dens_b.cent_y.mean())
        else:
            cx_m = float(b.cent_x.mean())
            cy_m = float(b.cent_y.mean())
        col = int((cx_m - gt[0]) / gt[1])
        row = int((cy_m - gt[3]) / gt[5])
        for name, rel in reliefs.items():
            u8 = to_u8_pos(np.nan_to_num(rel, nan=0.0)).astype(np.uint8)
            crop_path = OUT / "crops" / f"{year}_{key}_{name}.png"
            save_crop(rgb, u8, row, col, half=400, path=crop_path)
            crop_meta.append(
                {"year": year, "key": key, "method": name, "crop": str(crop_path)}
            )

    bdf = pd.DataFrame(boulder_rows)
    cdf = pd.DataFrame(control_rows)
    summary = {"methods": [], "investigation_ref": "investigation.json"}

    if len(bdf):
        for method, g in bdf.groupby("method"):
            dense = g[g["dense"]]
            iso = g[~g["dense"]]
            summary["methods"].append(
                {
                    "method": method,
                    "dense_n": int(len(dense)),
                    "dense_relief_mean_m": float(dense.relief_m.mean()) if len(dense) else None,
                    "dense_relief_p50_m": float(dense.relief_m.median()) if len(dense) else None,
                    "dense_sat_frac": float(dense.sat_0_5.mean()) if len(dense) else None,
                    "iso_n": int(len(iso)),
                    "iso_relief_mean_m": float(iso.relief_m.mean()) if len(iso) else None,
                    "iso_relief_p50_m": float(iso.relief_m.median()) if len(iso) else None,
                    "iso_sat_frac": float(iso.sat_0_5.mean()) if len(iso) else None,
                }
            )
    if len(cdf):
        ctrl_sum = []
        for method, g in cdf.groupby("method"):
            ctrl_sum.append(
                {
                    "method": method,
                    "ctrl_pos_mean_m": float(g.pos_mean_m.mean()),
                    "ctrl_pos_p90_m": float(g.pos_p90_m.mean()),
                    "ctrl_frac_gt_0_25m": float(g.frac_gt_0_25m.mean()),
                    "ctrl_frac_gt_0_5m": float(g.frac_gt_0_5m.mean()),
                }
            )
        # merge into methods
        by_m = {m["method"]: m for m in summary["methods"]}
        for c in ctrl_sum:
            by_m.setdefault(c["method"], {"method": c["method"]}).update(c)
        summary["methods"] = list(by_m.values())

    # Rank: maximize dense relief, keep iso sat high, minimize control false relief
    for m in summary["methods"]:
        dense_p50 = m.get("dense_relief_p50_m") or 0
        iso_sat = m.get("iso_sat_frac") or 0
        ctrl_p90 = m.get("ctrl_pos_p90_m") or 0
        # Simple score
        m["score"] = float(dense_p50 + 0.25 * iso_sat - 0.5 * ctrl_p90)

    summary["methods"].sort(key=lambda m: -(m.get("score") or -1e9))
    summary["pad_m_used"] = max_pad_m
    summary["clip_m"] = CLIP_M
    summary["dense_tiles"] = [f"{y}_{k}" for y, k in DENSE_TILES]
    summary["control_tiles"] = [f"{y}_{k}" for y, k in CONTROL_TILES]
    summary["crops"] = crop_meta
    summary["recommendation_note"] = (
        "Prefer opening if dense_relief_p50 >> gaussian and iso_sat_frac >= gaussian "
        "and ctrl_pos_p90 not much worse. Fixed 0.5m clip retained for all methods."
    )

    (OUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    if len(bdf):
        bdf.to_csv(OUT / "boulder_samples.csv", index=False)
    if len(cdf):
        cdf.to_csv(OUT / "control_samples.csv", index=False)

    # Markdown table
    lines = [
        "# Local-relief background comparison",
        "",
        f"Pad used: **{max_pad_m:.1f} m**. Clip: **{CLIP_M} m** positive-only.",
        "",
        "| method | dense n | dense p50 (m) | dense sat≥0.5 | iso n | iso p50 (m) | iso sat | ctrl p90 (m) | ctrl frac>0.25 | score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in summary["methods"]:
        lines.append(
            "| {method} | {dense_n} | {dense_relief_p50_m} | {dense_sat_frac} | {iso_n} | "
            "{iso_relief_p50_m} | {iso_sat_frac} | {ctrl_pos_p90_m} | {ctrl_frac_gt_0_25m} | {score} |".format(
                method=m.get("method"),
                dense_n=m.get("dense_n", ""),
                dense_relief_p50_m=f"{m['dense_relief_p50_m']:.3f}" if m.get("dense_relief_p50_m") is not None else "",
                dense_sat_frac=f"{m['dense_sat_frac']:.2f}" if m.get("dense_sat_frac") is not None else "",
                iso_n=m.get("iso_n", ""),
                iso_relief_p50_m=f"{m['iso_relief_p50_m']:.3f}" if m.get("iso_relief_p50_m") is not None else "",
                iso_sat_frac=f"{m['iso_sat_frac']:.2f}" if m.get("iso_sat_frac") is not None else "",
                ctrl_pos_p90_m=f"{m['ctrl_pos_p90_m']:.3f}" if m.get("ctrl_pos_p90_m") is not None else "",
                ctrl_frac_gt_0_25m=f"{m['ctrl_frac_gt_0_25m']:.3f}" if m.get("ctrl_frac_gt_0_25m") is not None else "",
                score=f"{m['score']:.3f}" if m.get("score") is not None else "",
            )
        )
    lines += [
        "",
        "## Investigation anchors",
        f"- Dense NN p50: {inv.get('stats', {}).get('nn_dense_m', {}).get('p50')} m",
        f"- Dense diam p90: {inv.get('stats', {}).get('diam_dense_m', {}).get('p90')} m",
        f"- DSM GSD: {inv.get('dsm', {})}",
        "",
        "Crops under `crops/`.",
    ]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}/comparison.md")


if __name__ == "__main__":
    main()
