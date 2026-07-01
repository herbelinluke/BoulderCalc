#!/usr/bin/env python3
"""Python port of BoulderCalculator volume_extractor for environments without MATLAB."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import median_filter
from shapely.geometry import Polygon
from shapely.ops import unary_union


@dataclass
class GeoRaster:
    data: np.ndarray
    transform: rasterio.Affine
    crs: str | None
    width: int
    height: int

    @property
    def pixel_size_x(self) -> float:
        return abs(self.transform.a)

    @property
    def pixel_size_y(self) -> float:
        return abs(self.transform.e)

    def world_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        left = self.transform.c
        top = self.transform.f
        right = left + self.width * self.transform.a
        bottom = top + self.height * self.transform.e
        return (left, right), (bottom, top)


def load_raster(path: Path, bands: list[int] | None = None) -> GeoRaster:
    with rasterio.open(path) as ds:
        if bands is None:
            data = ds.read()
            if data.shape[0] == 1:
                data = data[0]
            else:
                data = np.transpose(data, (1, 2, 0))
        else:
            data = ds.read(bands)
            if len(bands) == 1:
                data = data[0]
            else:
                data = np.transpose(data, (1, 2, 0))
        return GeoRaster(
            data=data,
            transform=ds.transform,
            crs=str(ds.crs) if ds.crs else None,
            width=ds.width,
            height=ds.height,
        )


def pix_to_world(xy: np.ndarray, raster: GeoRaster) -> tuple[np.ndarray, np.ndarray]:
    xlim, ylim = raster.world_limits()
    px = xlim[0] + (xy[:, 0] - 0.5) * raster.pixel_size_x
    py = ylim[1] - (xy[:, 1] - 0.5) * raster.pixel_size_y
    return px, py


def world_to_pix(px: np.ndarray, py: np.ndarray, raster: GeoRaster) -> tuple[np.ndarray, np.ndarray]:
    xlim, ylim = raster.world_limits()
    x = 1 + np.round(np.abs(px - xlim[0]) / raster.pixel_size_x)
    y = 1 + np.round(np.abs(py - ylim[1]) / raster.pixel_size_y)
    return x.astype(int), y.astype(int)


def transfer_points(xy: np.ndarray, src: GeoRaster, dst: GeoRaster) -> tuple[np.ndarray, np.ndarray]:
    wx, wy = pix_to_world(xy, src)
    return world_to_pix(wx, wy, dst)


def seg_to_polygon(seg: list[float]) -> Polygon | None:
    if len(seg) < 6:
        return None
    coords = list(zip(seg[0::2], seg[1::2]))
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda p: p.area)
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None


def suppress_nested_large_polygons(
    items: list[dict],
    containment_threshold: float = 0.5,
    max_area_ratio: float = 0.25,
) -> list[dict]:
    """Drop oversized detections that engulf smaller ones (common false positives)."""
    keep = [True] * len(items)
    for i, large in enumerate(items):
        if not keep[i]:
            continue
        for j, small in enumerate(items):
            if i == j or not keep[j]:
                continue
            if large["area"] <= small["area"]:
                continue
            inter = large["poly"].intersection(small["poly"]).area
            if inter <= 0:
                continue
            if inter / small["area"] >= containment_threshold:
                if small["area"] / large["area"] < max_area_ratio:
                    keep[i] = False
                    break
    return [item for item, ok in zip(items, keep) if ok]


def polygon_iou(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return inter / min(a.area, b.area)


def should_merge_polygons(
    base: Polygon,
    cand: Polygon,
    overlap_threshold: float,
    min_area_ratio: float | None,
) -> bool:
    inter = base.intersection(cand).area
    if inter <= 0:
        return False
    iou_min = inter / min(base.area, cand.area)
    if iou_min <= overlap_threshold:
        return False
    if min_area_ratio is None:
        return True
    area_ratio = min(base.area, cand.area) / max(base.area, cand.area)
    return area_ratio >= min_area_ratio


def merge_objects(
    annotations: list[dict],
    overlap_threshold: float = 0.15,
    overlap_num_threshold: int = 1,
    area_threshold: float = 0.0,
    min_area_ratio: float | None = 0.25,
) -> list[dict]:
    items = []
    for ann in annotations:
        seg = ann["segmentation"][0]
        poly = seg_to_polygon(seg)
        if poly is None or poly.area <= area_threshold:
            continue
        cx, cy = poly.centroid.x, poly.centroid.y
        items.append(
            {
                "name": ann.get("category_name", "Boulder"),
                "bbox": ann["bbox"],
                "seg": seg,
                "poly": poly,
                "area": poly.area,
                "cx": cx,
                "cy": cy,
                "score": float(ann.get("score", 0)),
            }
        )

    items.sort(key=lambda x: x["area"], reverse=True)
    items = suppress_nested_large_polygons(items, max_area_ratio=min_area_ratio or 0.25)
    merged: list[dict] = []

    while items:
        base = items.pop(0)
        overlaps = [base]
        remaining = []
        for cand in items:
            if should_merge_polygons(
                base["poly"], cand["poly"], overlap_threshold, min_area_ratio
            ):
                overlaps.append(cand)
            else:
                remaining.append(cand)

        if len(overlaps) == 1:
            merged.append(
                {
                    "name": base["name"],
                    "bbox": base["bbox"],
                    "seg": base["seg"],
                    "poly": base["poly"],
                    "area": base["area"],
                    "cx": base["cx"],
                    "cy": base["cy"],
                    "duplicate": 1,
                    "score": base["score"],
                }
            )
        else:
            union_geom = unary_union([g["poly"] for g in overlaps])
            if union_geom.geom_type == "MultiPolygon":
                for g in overlaps:
                    merged.append(
                        {
                            "name": g["name"],
                            "bbox": g["bbox"],
                            "seg": g["seg"],
                            "poly": g["poly"],
                            "area": g["area"],
                            "cx": g["cx"],
                            "cy": g["cy"],
                            "duplicate": 1,
                            "score": g["score"],
                        }
                    )
                items = remaining
                continue
            union_poly = _largest_polygon(union_geom)
            if union_poly is None:
                continue
            coords = list(union_poly.exterior.coords)
            seg = []
            for x, y in coords[:-1]:
                seg.extend([x, y])
            merged.append(
                {
                    "name": base["name"],
                    "bbox": union_poly.bounds,
                    "seg": seg,
                    "poly": union_poly,
                    "area": union_poly.area,
                    "cx": union_poly.centroid.x,
                    "cy": union_poly.centroid.y,
                    "duplicate": len(overlaps),
                    "score": max(g["score"] for g in overlaps),
                }
            )
        items = remaining

    return merged


def xy_expansion(seg: list[float], pixels: float) -> tuple[np.ndarray, np.ndarray]:
    poly = seg_to_polygon(seg)
    if poly is None:
        return np.array([]), np.array([])
    expanded = _largest_polygon(poly.buffer(pixels))
    if expanded is None:
        return np.array([]), np.array([])
    coords = np.asarray(expanded.exterior.coords)
    return coords[:, 0], coords[:, 1]


def fit_poly33_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, float]:
    # MATLAB fit(..., 'poly33')
    A = np.column_stack(
        [
            np.ones_like(x),
            x,
            y,
            x * x,
            x * y,
            y * y,
            x**3,
            (x**2) * y,
            x * (y**2),
            y**3,
        ]
    )
    coeff, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    zhat = A @ coeff
    rmse = float(np.sqrt(np.mean((z - zhat) ** 2)))
    return coeff, rmse


def eval_poly33(coeff: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return (
        coeff[0]
        + coeff[1] * X
        + coeff[2] * Y
        + coeff[3] * X * X
        + coeff[4] * X * Y
        + coeff[5] * Y * Y
        + coeff[6] * X**3
        + coeff[7] * (X**2) * Y
        + coeff[8] * X * (Y**2)
        + coeff[9] * Y**3
    )


def fill_polygon_mask(shape: tuple[int, int], seg: list[float]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] -= 1
    pts[:, 1] -= 1
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def regionprops_axes(mask: np.ndarray, pixel_size: float) -> tuple[float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return np.nan, np.nan, np.nan
    coords = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (width, height), angle = cv2.minAreaRect(coords)
    major = max(width, height) * pixel_size
    minor = min(width, height) * pixel_size
    return major, minor, angle


def extract_volumes(
    ortho: GeoRaster,
    dsm: GeoRaster,
    merged: list[dict],
    height_threshold: tuple[float, float] = (0.3, 20.0),
) -> pd.DataFrame:
    rows = []
    dem_scale_x = dsm.pixel_size_x
    dem_scale_y = dsm.pixel_size_y
    im_scale_x = ortho.pixel_size_x
    im_scale_y = ortho.pixel_size_y

    for i, obj in enumerate(merged, start=1):
        seg = obj["seg"]
        description = []

        xe, ye = xy_expansion(seg, 5)
        if len(xe) == 0:
            continue

        im_xmin = int(np.floor(min(xe)))
        im_xmax = int(np.ceil(max(xe)))
        im_ymin = int(np.floor(min(ye)))
        im_ymax = int(np.ceil(max(ye)))

        wx, wy = pix_to_world(np.column_stack([obj["cx"], obj["cy"]]), ortho)
        dem_xe, dem_ye = transfer_points(np.column_stack([xe, ye]), ortho, dsm)

        dem_xmin = int(np.min(dem_xe))
        dem_xmax = int(np.max(dem_xe))
        dem_ymin = int(np.min(dem_ye))
        dem_ymax = int(np.max(dem_ye))

        dem_pad = max(1, int(round(2 / dem_scale_x)))
        dem_xfrom = dem_xmin - dem_pad
        dem_xto = dem_xmax + dem_pad
        dem_yfrom = dem_ymin - dem_pad
        dem_yto = dem_ymax + dem_pad

        dem_xfrom = max(1, dem_xfrom)
        dem_yfrom = max(1, dem_yfrom)
        dem_xto = min(dsm.width, dem_xto)
        dem_yto = min(dsm.height, dem_yto)
        if dem_yfrom >= dem_yto or dem_xfrom >= dem_xto:
            continue

        dem_trim = dsm.data[dem_yfrom - 1 : dem_yto, dem_xfrom - 1 : dem_xto].astype(float)
        dem_trim[dem_trim < -100] = np.nan

        dem_smooth_cm = 20.0
        dem_smooth_pix = max(1, int(round(dem_smooth_cm / (dem_scale_x * 100))))
        dem_trim_rd = median_filter(np.nan_to_num(dem_trim, nan=np.nanmedian(dem_trim)), size=dem_smooth_pix)
        if np.any(dem_trim_rd == 0):
            continue

        seg_temp = list(seg)
        dem_xtemp, dem_ytemp = transfer_points(
            np.column_stack([seg[0::2], seg[1::2]]), ortho, dsm
        )
        seg_temp = list(seg)
        for j in range(0, len(seg_temp), 2):
            seg_temp[j] = dem_xtemp[j // 2] - (dem_xmin - dem_pad)
            seg_temp[j + 1] = dem_ytemp[j // 2] - (dem_yfrom - dem_pad)

        shape_mask = fill_polygon_mask(dem_trim_rd.shape, seg_temp)
        major, minor, orientation = regionprops_axes(shape_mask, (dem_scale_x + dem_scale_y) / 2)

        mask_seg: list[float] = []
        dem_mask_x, dem_mask_y = transfer_points(
            np.column_stack([seg[0::2], seg[1::2]]), ortho, dsm
        )
        for x, y in zip(dem_mask_x, dem_mask_y):
            mask_seg.extend([x - (dem_xmin - dem_pad), y - (dem_yfrom - dem_pad)])
        obj_mask = fill_polygon_mask(dem_trim_rd.shape, mask_seg)

        mask_pad = max(1, int(round(dem_scale_x * 100 * 1.5)))
        basement_mask = np.ones(dem_trim_rd.shape, dtype=bool)
        xe2, ye2 = xy_expansion(seg, mask_pad)
        if len(xe2):
            dem_xe2, dem_ye2 = transfer_points(np.column_stack([xe2, ye2]), ortho, dsm)
            ring_seg = []
            for j in range(len(dem_xe2)):
                ring_seg.extend(
                    [
                        dem_xe2[j] - (dem_xmin - dem_pad),
                        dem_ye2[j] - (dem_yfrom - dem_pad),
                    ]
                )
            ring = fill_polygon_mask(dem_trim_rd.shape, ring_seg)
            basement_mask[ring] = False

        basement_vals = dem_trim_rd[basement_mask]
        yy, xx = np.mgrid[0 : dem_trim_rd.shape[0], 0 : dem_trim_rd.shape[1]]
        sample_idx = basement_mask
        if sample_idx.sum() < 10:
            continue

        coeff, fit_rmse = fit_poly33_surface(
            xx[sample_idx].ravel(),
            yy[sample_idx].ravel(),
            dem_trim_rd[sample_idx].ravel(),
        )
        height_basement = eval_poly33(coeff, xx, yy)
        height_obj = dem_trim_rd - height_basement
        height_obj[height_obj < 0] = 0

        vol_mask = ~basement_mask
        dem_obj = height_obj * vol_mask
        meanh_obj = float(np.nanmedian(dem_obj[dem_obj > 0])) if np.any(dem_obj > 0) else 0.0
        maxh_obj = float(np.max(dem_obj)) if np.any(dem_obj > 0) else 0.0

        if fit_rmse > 1:
            base_height_obj = np.nan
            description.append("Height is set as NaN because it exceeds the fitting rmse.")
        else:
            base_height_obj = float(np.nanmedian(height_basement[vol_mask]))

        isboulder = 0
        if height_threshold[0] < maxh_obj <= height_threshold[1]:
            dsm_vol_m3 = float(np.nansum(dem_obj) * dem_scale_x * dem_scale_y)
            if dsm_vol_m3 > 0:
                isboulder = 1
            else:
                dsm_vol_m3 = 0.0
        else:
            dsm_vol_m3 = np.nan
            description.append("Vdsm is set as NaN because it exceeds the height thresholds.")

        if fit_rmse > 1:
            isboulder = 0

        ob_area_m2 = obj["area"] * im_scale_x * im_scale_y
        abc_vol_m3 = major * minor * maxh_obj
        eli_vol_m3 = (4 / 3) * np.pi * abc_vol_m3 / 8

        rows.append(
            {
                "calc_id": i,
                "ob_id": f"ob_{i:03d}",
                "ob_name": obj["name"],
                "isboulder": isboulder,
                "centroid_world_x": float(wx[0]),
                "centroid_world_y": float(wy[0]),
                "base_height_obj": base_height_obj,
                "MajorAxis_m": major,
                "MinorAxis_m": minor,
                "maxh_obj_m": maxh_obj,
                "meanh_obj_m": meanh_obj,
                "Orientation_deg": orientation,
                "ob_area_m2": ob_area_m2,
                "abc_vol_m3": abc_vol_m3,
                "eli_vol_m3": eli_vol_m3,
                "dsm_vol_m3": dsm_vol_m3,
                "fit_rmse_m": fit_rmse,
                "detection_score": obj.get("score", np.nan),
                "duplicate_count": obj.get("duplicate", 1),
                "description": "; ".join(description),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ortho", required=True, type=Path)
    parser.add_argument("--dsm", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ortho = load_raster(args.ortho, bands=[1, 2, 3])
    dsm = load_raster(args.dsm, bands=[1])
    coco = json.loads(args.json.read_text())

    annotations = coco["annotations"]
    for ann in annotations:
        ann["category_name"] = coco["categories"][0]["name"]

    merged = merge_objects(
        annotations,
        overlap_threshold=0.15,
        overlap_num_threshold=1,
        area_threshold=0,
        min_area_ratio=0.25,
    )
    results = extract_volumes(ortho, dsm, merged)

    csv_path = args.output_dir / f"{args.ortho.stem}_results.csv"
    summary_path = args.output_dir / f"{args.ortho.stem}_volume_summary.json"
    results.to_csv(csv_path, index=False)

    summary = {
        "raw_detections": len(annotations),
        "merged_objects": len(merged),
        "objects_with_volume": int(results["isboulder"].sum()) if not results.empty else 0,
        "csv": str(csv_path),
        "note": "Python port of BoulderCalculator volume_extractor; MATLAB not available on this machine.",
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not results.empty:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
