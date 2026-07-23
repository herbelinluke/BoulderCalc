# test_evaluate_matches.py — unit tests for label record / annotation hits (no GUI)

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from matching.evaluate_matches import (
    LABEL_CONFIRMED,
    annotation_hits,
    build_pair_record,
    load_labels_db,
    match_key,
    save_labels_db,
)


def test_match_key():
    assert match_key(2, 62) == "b2_a62"


def test_annotation_hits_intersect_and_fid():
    ann = gpd.GeoDataFrame(
        {"ann_fid": [10, 20], "geometry": [box(0, 0, 2, 2), box(10, 10, 12, 12)]},
        crs="EPSG:25829",
    )
    hit = annotation_hits(box(0.5, 0.5, 1.5, 1.5), ann, min_iou=0.1)
    assert hit["intersects"] is True
    assert 10 in hit["fids"]
    assert hit["best_fid"] == 10
    assert hit["best_iou"] > 0.2

    miss = annotation_hits(box(50, 50, 51, 51), ann)
    assert miss["intersects"] is False
    assert miss["fids"] == []


def test_build_pair_record_and_save(tmp_path: Path):
    before_g = box(0, 0, 1, 1)
    after_g = box(0.2, 0.1, 1.2, 1.1)
    ann24 = gpd.GeoDataFrame(
        {"ann_fid": [1], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:25829"
    )
    ann25 = gpd.GeoDataFrame(
        {"ann_fid": [5], "geometry": [box(0.2, 0.1, 1.2, 1.1)]}, crs="EPSG:25829"
    )
    row = {
        "before_id": 3,
        "after_id": 7,
        "match_score": 0.9,
        "distance_m": 0.22,
        "dx": 0.2,
        "dy": 0.1,
        "before_area": 1.0,
        "after_area": 1.0,
        "before_volume": 0.5,
        "after_volume": 0.55,
    }
    rec = build_pair_record(
        row, before_g, after_g, ann24, ann25, label=LABEL_CONFIRMED
    )
    assert rec["label_id"] == "b3_a7"
    assert rec["label"] == LABEL_CONFIRMED
    assert rec["manual_ann_24"]["intersects"] is True
    assert rec["manual_ann_25"]["intersects"] is True
    assert 1 in rec["manual_ann_24"]["fids"]
    assert 5 in rec["manual_ann_25"]["fids"]
    assert rec["qgis_extent"] is not None
    assert rec["before"]["centroid_x"] == pytest.approx(0.5)

    path = tmp_path / "labels.json"
    db = load_labels_db(path)
    db["labels"] = {rec["label_id"]: rec}
    save_labels_db(db, path, also_geojson=True)
    assert path.exists()
    assert path.with_suffix(".geojson").exists()
    reloaded = load_labels_db(path)
    assert "b3_a7" in reloaded["labels"]
    assert reloaded["labels"]["b3_a7"]["label"] == LABEL_CONFIRMED
