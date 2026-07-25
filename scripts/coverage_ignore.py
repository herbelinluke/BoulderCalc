#!/usr/bin/env python3
"""Ocean-safe coverage ignore for mosaic black edges and DSM gaps.

Detects border-connected near-black RGB voids (true empty ortho / soft fringe)
and DSM nodata that co-occurs with those voids — not dark textured ocean.

Emits COCO ``iscrowd=1`` polygons (``ignore_reason=coverage``) for train/val/test.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio import features as rio_features
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_dilation, label
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

# Defaults used by gpkg_to_coco / build_coco_rgb_dsm CLIs.
DEFAULT_RGB_MAX = 8
DEFAULT_BLUR_M = 0.5
DEFAULT_MIN_AREA_M2 = 1.0
DEFAULT_OVERLAP_FRAC = 0.5


def default_dsm_path(project_root: Path, year: int) -> Path:
    if year == 24:
        return project_root / "2024" / "Sites1and2_2024_DSM_30mm.tif"
    if year == 25:
        return project_root / "2025" / "25IniSouthDSM.tif"
    raise ValueError(f"Unsupported year for DSM: {year}")


def year_from_coco_file_name(file_name: str) -> int | None:
    stem = Path(file_name).name
    for prefix, year in (("24_", 24), ("25_", 25)):
        if stem.startswith(prefix):
            return year
    return None


def warp_dem_with_validity(
    ortho_path: Path,
    dsm_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp DSM onto ortho grid; invalid / nodata / out-of-footprint → NaN.

    Returns ``(dem float32 with NaNs, valid bool mask)``.
    """
    with rasterio.open(ortho_path) as ortho:
        h, w = ortho.height, ortho.width
        dem = np.full((h, w), np.nan, dtype=np.float32)
        with rasterio.open(dsm_path) as dsm:
            src_nodata = dsm.nodata
            reproject(
                source=rasterio.band(dsm, 1),
                destination=dem,
                src_transform=dsm.transform,
                src_crs=dsm.crs,
                dst_transform=ortho.transform,
                dst_crs=ortho.crs,
                resampling=Resampling.bilinear,
                src_nodata=src_nodata,
                dst_nodata=np.nan,
            )
            # Explicit nodata equality (some DSMs use a finite sentinel).
            if src_nodata is not None and np.isfinite(src_nodata):
                dem = np.where(np.isclose(dem, float(src_nodata)), np.nan, dem)
        valid = np.isfinite(dem)
        return dem, valid


def near_black_mask(rgb: np.ndarray, rgb_max: int) -> np.ndarray:
    """``rgb`` is (3, H, W) or (H, W, 3)."""
    arr = np.asarray(rgb)
    if arr.ndim != 3:
        raise ValueError(f"RGB must be 3D, got shape {arr.shape}")
    if arr.shape[0] == 3:
        mx = np.max(arr, axis=0)
    elif arr.shape[-1] == 3:
        mx = np.max(arr, axis=-1)
    else:
        raise ValueError(f"RGB channel axis not found in shape {arr.shape}")
    return mx <= int(rgb_max)


def border_connected_mask(seed: np.ndarray) -> np.ndarray:
    """Keep connected components of ``seed`` that touch the image border."""
    seed = np.asarray(seed, dtype=bool)
    if not np.any(seed):
        return np.zeros_like(seed, dtype=bool)
    labeled, _n = label(seed)
    border = np.zeros_like(seed, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    touch = set(int(v) for v in labeled[border & seed].ravel() if v != 0)
    if not touch:
        return np.zeros_like(seed, dtype=bool)
    return np.isin(labeled, list(touch))


def dilate_mask_m(mask: np.ndarray, pixel_size: float, blur_m: float) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if blur_m <= 0:
        return mask
    dilate_px = int(math.ceil(float(blur_m) / max(float(pixel_size), 1e-9)))
    if dilate_px <= 0:
        return mask
    return binary_dilation(mask, iterations=dilate_px)


def build_coverage_mask_from_arrays(
    rgb: np.ndarray,
    *,
    pixel_size: float,
    dsm_valid: np.ndarray | None = None,
    rgb_max: int = DEFAULT_RGB_MAX,
    blur_m: float = DEFAULT_BLUR_M,
) -> np.ndarray:
    """Core ocean-safe mask from RGB (+ optional DSM validity).

    - Border-connected near-black RGB void, dilated by ``blur_m``.
    - DSM gaps only where they also intersect near-black or the dilated void.
    """
    black = near_black_mask(rgb, rgb_max)
    void = border_connected_mask(black)
    void = dilate_mask_m(void, pixel_size=pixel_size, blur_m=blur_m)
    if dsm_valid is None:
        return void
    gap = ~np.asarray(dsm_valid, dtype=bool)
    gated_gap = gap & (black | void)
    return void | gated_gap


def build_coverage_mask(
    image_path: Path,
    *,
    dsm_path: Path | None = None,
    rgb_max: int = DEFAULT_RGB_MAX,
    blur_m: float = DEFAULT_BLUR_M,
) -> tuple[np.ndarray, Any]:
    """Read tile (+ optional DSM) and return ``(mask HxW bool, transform)``."""
    with rasterio.open(image_path) as ds:
        if ds.count < 3:
            raise ValueError(f"{image_path} has {ds.count} bands; need RGB")
        rgb = ds.read([1, 2, 3])
        transform = ds.transform
        pixel_size = abs(transform.a)

    dsm_valid = None
    if dsm_path is not None and Path(dsm_path).is_file():
        _dem, dsm_valid = warp_dem_with_validity(image_path, Path(dsm_path))

    mask = build_coverage_mask_from_arrays(
        rgb,
        pixel_size=pixel_size,
        dsm_valid=dsm_valid,
        rgb_max=rgb_max,
        blur_m=blur_m,
    )
    return mask, transform


def mask_to_polygons(
    mask: np.ndarray,
    transform,
    *,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> list[Polygon]:
    """Polygonize True pixels; drop parts smaller than ``min_area_m2`` (map units²)."""
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if not np.any(mask_u8):
        return []
    polys: list[Polygon] = []
    for geom, val in rio_features.shapes(mask_u8, mask=mask_u8.astype(bool), transform=transform):
        if int(val) != 1:
            continue
        poly = shape(geom)
        if not poly.is_valid:
            poly = poly.buffer(0)
        for part in _polygons_of(poly):
            if part.area >= float(min_area_m2):
                polys.append(part)
    return polys


def _polygons_of(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_polygons_of(g))
        return out
    return []


def _ring_to_seg(ring_coords, inv_transform) -> list[float]:
    coords = list(ring_coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    seg: list[float] = []
    for x, y in coords:
        col, row = inv_transform * (x, y)
        seg.extend([float(col), float(row)])
    return seg


def _bbox_from_seg(seg: list[float]) -> list[float]:
    xs = seg[0::2]
    ys = seg[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _poly_area_px(seg: list[float]) -> float:
    xs = seg[0::2]
    ys = seg[1::2]
    area = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def coverage_polygons_to_annotations(
    polys: list[Polygon],
    *,
    image_id: int,
    ann_start_id: int,
    transform,
    category_id: int = 1,
) -> tuple[list[dict], int]:
    """COCO iscrowd anns for coverage voids."""
    inv = ~transform
    anns: list[dict] = []
    ann_id = ann_start_id
    for poly in polys:
        seg = _ring_to_seg(poly.exterior.coords, inv)
        if len(seg) < 6:
            continue
        area = _poly_area_px(seg)
        if area <= 0:
            continue
        anns.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [seg],
                "area": area,
                "bbox": _bbox_from_seg(seg),
                "iscrowd": 1,
                "attributes": {"occluded": False, "ignore_reason": "coverage"},
            }
        )
        ann_id += 1
    return anns, ann_id


def mark_coverage_overlap_annotations(
    annotations: list[dict],
    coverage_union,
    transform,
    *,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> int:
    """Set iscrowd on anns whose map-space geometry is mostly inside coverage.

    Returns count newly marked ``coverage_overlap``.
    """
    if coverage_union is None or coverage_union.is_empty:
        return 0
    n_marked = 0
    for ann in annotations:
        if ann.get("iscrowd", 0):
            continue
        segs = ann.get("segmentation") or []
        if not segs:
            continue
        # Rebuild polygon in pixel space then to map coords via transform.
        geom_parts = []
        for seg in segs:
            if len(seg) < 6:
                continue
            coords = []
            for i in range(0, len(seg), 2):
                col, row = float(seg[i]), float(seg[i + 1])
                x, y = transform * (col, row)
                coords.append((x, y))
            if len(coords) < 3:
                continue
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                p = Polygon(coords)
            except Exception:
                continue
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                geom_parts.append(p)
        if not geom_parts:
            continue
        geom = unary_union(geom_parts)
        if geom.is_empty or geom.area <= 0:
            continue
        inter = geom.intersection(coverage_union)
        frac = inter.area / geom.area if geom.area > 0 else 0.0
        if frac >= float(overlap_frac):
            ann["iscrowd"] = 1
            attrs = dict(ann.get("attributes") or {})
            attrs["ignore_reason"] = "coverage_overlap"
            ann["attributes"] = attrs
            n_marked += 1
    return n_marked


def build_tile_coverage(
    image_path: Path,
    *,
    dsm_path: Path | None,
    image_id: int,
    ann_start_id: int,
    rgb_max: int = DEFAULT_RGB_MAX,
    blur_m: float = DEFAULT_BLUR_M,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
    existing_annotations: list[dict] | None = None,
) -> tuple[list[dict], int, dict]:
    """Build coverage iscrowd anns (+ optionally mark overlapping positives).

    Returns ``(new_or_updated_ann_list_delta, next_ann_id, stats)``.
    When ``existing_annotations`` is provided it is mutated in place for overlap
    marking; coverage polys are returned separately to append.
    """
    mask, transform = build_coverage_mask(
        image_path,
        dsm_path=dsm_path,
        rgb_max=rgb_max,
        blur_m=blur_m,
    )
    polys = mask_to_polygons(mask, transform, min_area_m2=min_area_m2)
    coverage_union = unary_union(polys) if polys else None
    cov_anns, next_id = coverage_polygons_to_annotations(
        polys,
        image_id=image_id,
        ann_start_id=ann_start_id,
        transform=transform,
    )
    n_overlap = 0
    if existing_annotations is not None and coverage_union is not None:
        n_overlap = mark_coverage_overlap_annotations(
            existing_annotations,
            coverage_union,
            transform,
            overlap_frac=overlap_frac,
        )
    stats = {
        "coverage_polys": len(cov_anns),
        "coverage_pixels": int(np.count_nonzero(mask)),
        "coverage_overlap_marked": n_overlap,
        "has_dsm": bool(dsm_path and Path(dsm_path).is_file()),
    }
    return cov_anns, next_id, stats


def strip_coverage_annotations(annotations: list[dict]) -> list[dict]:
    """Drop prior coverage / coverage_overlap anns before a refresh."""
    kept: list[dict] = []
    for ann in annotations:
        reason = (ann.get("attributes") or {}).get("ignore_reason")
        if reason in ("coverage", "coverage_overlap"):
            # coverage_overlap may have been a real boulder — restore trainable
            # only when we re-mark; for strip before refresh, drop coverage polys
            # and un-crowd coverage_overlap back to trainable.
            if reason == "coverage":
                continue
            # coverage_overlap: restore as trainable for re-evaluation
            ann = dict(ann)
            ann["iscrowd"] = 0
            attrs = dict(ann.get("attributes") or {})
            attrs.pop("ignore_reason", None)
            ann["attributes"] = attrs
        kept.append(ann)
    return kept


def refresh_split_coverage(
    data: dict,
    image_dir: Path,
    *,
    dsm_by_year: dict[int, Path],
    project_root: Path,
    rgb_max: int = DEFAULT_RGB_MAX,
    blur_m: float = DEFAULT_BLUR_M,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> dict:
    """Recompute coverage ignore for every image in a COCO dict."""
    images = data.get("images") or []
    anns = strip_coverage_annotations(list(data.get("annotations") or []))
    by_image: dict[int, list[dict]] = {}
    for ann in anns:
        by_image.setdefault(ann["image_id"], []).append(ann)

    next_id = 1
    for a in anns:
        next_id = max(next_id, int(a["id"]) + 1)

    new_anns: list[dict] = []
    stats_total = {
        "coverage_polys": 0,
        "coverage_overlap_marked": 0,
        "images": 0,
    }
    for im in images:
        image_id = int(im["id"])
        path = image_dir / im["file_name"]
        if not path.is_file():
            # Try without year prefix inside tile dirs is not needed; image_dir
            # should contain the COCO file_name as written.
            raise FileNotFoundError(f"Missing image for coverage refresh: {path}")
        year = year_from_coco_file_name(im["file_name"])
        dsm_path = None
        if year is not None:
            dsm_path = dsm_by_year.get(year) or default_dsm_path(project_root, year)
            if not Path(dsm_path).is_file():
                dsm_path = None
        img_anns = by_image.get(image_id, [])
        cov_anns, next_id, stats = build_tile_coverage(
            path,
            dsm_path=dsm_path,
            image_id=image_id,
            ann_start_id=next_id,
            rgb_max=rgb_max,
            blur_m=blur_m,
            min_area_m2=min_area_m2,
            overlap_frac=overlap_frac,
            existing_annotations=img_anns,
        )
        new_anns.extend(img_anns)
        new_anns.extend(cov_anns)
        stats_total["coverage_polys"] += stats["coverage_polys"]
        stats_total["coverage_overlap_marked"] += stats["coverage_overlap_marked"]
        stats_total["images"] += 1

    out = dict(data)
    out["annotations"] = new_anns
    out["_coverage_refresh_stats"] = stats_total
    return out


def write_coverage_debug_png(mask: np.ndarray, out_path: Path) -> None:
    """Write a simple grayscale PNG of the coverage mask (255 = ignore)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    try:
        import cv2

        cv2.imwrite(str(out_path), img)
    except ImportError:
        # Fallback via rasterio as single-band GeoTIFF if OpenCV missing.
        tif = out_path.with_suffix(".tif")
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            height=img.shape[0],
            width=img.shape[1],
            count=1,
            dtype="uint8",
        ) as dst:
            dst.write(img, 1)


def main_debug() -> None:
    """CLI: write a coverage debug mask for one tile.

    Example:
      python BoulderCalculator/scripts/coverage_ignore.py \\
        --image segmentation/tiling/25/25IniSouthOrt_01_01.tif \\
        --year 25 --output /tmp/coverage_debug.png
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description=main_debug.__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--year", type=int, choices=[24, 25], default=None)
    parser.add_argument("--dsm", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-max", type=int, default=DEFAULT_RGB_MAX)
    parser.add_argument("--blur-m", type=float, default=DEFAULT_BLUR_M)
    args = parser.parse_args()

    dsm = args.dsm
    if dsm is None and args.year is not None:
        dsm = default_dsm_path(args.project_root.resolve(), args.year)
    mask, _transform = build_coverage_mask(
        args.image,
        dsm_path=dsm if dsm and Path(dsm).is_file() else None,
        rgb_max=args.rgb_max,
        blur_m=args.blur_m,
    )
    write_coverage_debug_png(mask, args.output)
    print(
        json.dumps(
            {
                "image": str(args.image),
                "dsm": str(dsm) if dsm else None,
                "coverage_pixels": int(np.count_nonzero(mask)),
                "coverage_fraction": float(np.count_nonzero(mask) / max(mask.size, 1)),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main_debug()
