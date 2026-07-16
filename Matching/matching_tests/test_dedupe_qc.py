# test_dedupe_qc.py
#
# Unit tests for overlapping-detection dedupe and DoD QC helpers.
# Run from Matching/: pytest matching_tests/ -v

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from matching.dedupe import dedupe_polygons
from matching.qc import (
    _classify_appeared,
    _classify_disappeared,
    _classify_match,
    build_dod,
    run_dod_qc,
)


def make_square(cx, cy, size=1.0):
    h = size / 2
    return Polygon(
        [
            (cx - h, cy - h),
            (cx + h, cy - h),
            (cx + h, cy + h),
            (cx - h, cy + h),
        ]
    )


def test_dedupe_keeps_highest_score_on_iou():
    # Two heavily overlapping boxes; higher score should win
    gdf = gpd.GeoDataFrame(
        {
            "score": [0.6, 0.95, 0.4],
            "geometry": [
                make_square(0, 0, 1.0),
                make_square(0.1, 0.05, 1.0),  # overlaps first
                make_square(10, 10, 1.0),  # far away
            ],
        },
        crs="EPSG:25829",
    )
    out = dedupe_polygons(gdf, iou_thresh=0.4, centroid_dist_m=None)
    assert len(out) == 2
    assert set(out["score"]) == {0.95, 0.4}
    assert int(out.loc[out["score"] == 0.95, "kept_from_n"].iloc[0]) == 2


def test_dedupe_centroid_near_without_iou():
    # Same center, tiny polygons that barely overlap if size differs — use centroid rule
    gdf = gpd.GeoDataFrame(
        {
            "score": [0.5, 0.9],
            "geometry": [
                make_square(0, 0, 0.4),
                make_square(0.2, 0.0, 0.4),  # centroids 0.2 m apart
            ],
        },
        crs="EPSG:25829",
    )
    out = dedupe_polygons(gdf, iou_thresh=0.99, centroid_dist_m=0.5)
    assert len(out) == 1
    assert float(out.iloc[0]["score"]) == pytest.approx(0.9)


def test_dedupe_empty():
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:25829")
    out = dedupe_polygons(empty)
    assert out.empty


def test_classify_match_labels():
    src_ok = {"v_neg_m3": 0.2, "v_pos_m3": 0.0}
    sink_ok = {"v_neg_m3": 0.0, "v_pos_m3": 0.2}
    weak = {"v_neg_m3": 0.0, "v_pos_m3": 0.0}
    assert _classify_match(src_ok, sink_ok, 1.0, 0.35, 0.05) == "consistent_move"
    assert _classify_match(weak, weak, 0.1, 0.35, 0.05) == "stationary_ok"
    assert _classify_match(src_ok, weak, 1.0, 0.35, 0.05) == "move_source_only"
    assert _classify_disappeared(src_ok, 0.05) == "likely_missed_mover_source"
    assert _classify_appeared(sink_ok, 0.05) == "likely_missed_mover_sink"


def _write_dsm(path: Path, data: np.ndarray, x0=0.0, y0=10.0, res=1.0):
    transform = from_origin(x0, y0, res, res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:25829",
        transform=transform,
    ) as ds:
        ds.write(data.astype(np.float32), 1)


def test_dod_qc_end_to_end(tmp_path: Path):
    # 10x10 grid, res=1 m. Boulder leaves (2,2)-(4,4) and arrives (6,2)-(8,4).
    before = np.full((10, 10), 5.0, dtype=np.float32)
    after = np.full((10, 10), 5.0, dtype=np.float32)
    before[2:5, 2:5] = 6.5  # source elevated in before
    after[2:5, 6:9] = 6.5  # sink elevated in after

    b_path = tmp_path / "before_dsm.tif"
    a_path = tmp_path / "after_dsm.tif"
    _write_dsm(b_path, before)
    _write_dsm(a_path, after)

    dod, transform, crs, pixel_area = build_dod(b_path, a_path)
    assert pixel_area == pytest.approx(1.0)
    # Source should be negative, sink positive
    assert float(np.nanmean(dod[2:5, 2:5])) < -0.5
    assert float(np.nanmean(dod[2:5, 6:9])) > 0.5

    before_poly = gpd.GeoDataFrame(
        {"before_id": [0], "geometry": [box(2, 5, 5, 8)]},  # world: x 2-5, y 5-8
        crs="EPSG:25829",
    )
    # from_origin(0,10): row 2 → y=8, row 5 → y=5; col 2 → x=2, col 5 → x=5
    # Wait: box(2,5,5,8) is minx,miny,maxx,maxy → covers source footprint
    after_poly = gpd.GeoDataFrame(
        {"after_id": [0], "geometry": [box(6, 5, 9, 8)]},
        crs="EPSG:25829",
    )
    matches = gpd.GeoDataFrame(
        {
            "before_id": [0],
            "after_id": [0],
            "match_score": [0.9],
            "distance_m": [4.0],
            "geometry": [after_poly.iloc[0].geometry],
        },
        crs="EPSG:25829",
    )
    results = {
        "matches": matches,
        "appeared": gpd.GeoDataFrame(geometry=[], crs="EPSG:25829"),
        "disappeared": gpd.GeoDataFrame(geometry=[], crs="EPSG:25829"),
        "vectors": gpd.GeoDataFrame(geometry=[], crs="EPSG:25829"),
    }
    qc = run_dod_qc(
        results,
        before_polygons=before_poly,
        after_polygons=after_poly,
        before_dsm=b_path,
        after_dsm=a_path,
        lod_m=0.1,
        min_change_m3=0.5,
        stationary_dist_m=0.35,
    )
    assert len(qc["match_qc"]) == 1
    assert qc["match_qc"].iloc[0]["qc_label"] == "consistent_move"
    assert qc["match_qc"].iloc[0]["dod_src_v_neg_m3"] > 0.5
    assert qc["match_qc"].iloc[0]["dod_sink_v_pos_m3"] > 0.5
