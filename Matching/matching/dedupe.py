"""Collapse duplicate boulder polygons from overlapping inference tiles.

Sliding-window / multi-tile inference often emits several masks for the same
physical boulder.  ``dedupe_polygons`` keeps the highest-score instance and
suppresses others whose IoU (or centroid proximity) exceeds a threshold.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np


def _iou(a, b) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def dedupe_polygons(
    gdf: gpd.GeoDataFrame,
    iou_thresh: float = 0.4,
    centroid_dist_m: float | None = 0.75,
    score_col: str = "score",
) -> gpd.GeoDataFrame:
    """Non-maximum suppression over polygon instances.

    Parameters
    ----------
    gdf
        Detection polygons (any CRS). Empty input is returned unchanged.
    iou_thresh
        Suppress a candidate when IoU with a kept polygon is >= this value.
    centroid_dist_m
        If set, also suppress when centroid distance is below this (metres /
        CRS units).  Helps when two masks barely overlap but mark the same
        rock.  Pass ``None`` to disable.
    score_col
        Column used as ranking priority (higher kept).  Missing → all equal.

    Returns
    -------
    GeoDataFrame with duplicates removed, index reset.  Adds ``kept_from_n``
    counting how many raw detections were merged into each survivor (1 = unique).
    """
    if gdf is None or gdf.empty:
        return gdf.copy() if gdf is not None else gpd.GeoDataFrame(geometry=[], crs="EPSG:25829")

    work = gdf.copy().reset_index(drop=True)
    if score_col not in work.columns:
        work[score_col] = 1.0
    work[score_col] = work[score_col].fillna(0.0).astype(float)

    order = np.argsort(-work[score_col].to_numpy())
    geoms = list(work.geometry)
    centroids = [g.centroid for g in geoms]
    suppressed = np.zeros(len(work), dtype=bool)
    keep_idx: list[int] = []
    merge_counts: dict[int, int] = {}

    for i in order:
        if suppressed[i]:
            continue
        keep_idx.append(int(i))
        merge_counts[int(i)] = 1
        gi = geoms[i]
        ci = centroids[i]
        for j in order:
            if j == i or suppressed[j]:
                continue
            gj = geoms[j]
            overlap = _iou(gi, gj) >= iou_thresh
            near = False
            if centroid_dist_m is not None:
                near = ci.distance(centroids[j]) <= centroid_dist_m
            if overlap or near:
                suppressed[j] = True
                merge_counts[int(i)] += 1

    out = work.iloc[keep_idx].copy().reset_index(drop=True)
    out["kept_from_n"] = [merge_counts[i] for i in keep_idx]
    return out
