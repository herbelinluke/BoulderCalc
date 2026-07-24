# test_classes.py

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from matching.classes import keep_boulders_only, parse_class_value
from matching.matcher import BoulderMatcher


class FakeSurvey:
    def __init__(self, polygons):
        self.polygons = polygons


def test_parse_class_value():
    assert parse_class_value(0) == 0
    assert parse_class_value(1) == 1
    assert parse_class_value("1") == 1
    assert parse_class_value("BoulderDeposit") == 1
    assert parse_class_value("boulder") == 0


def test_keep_boulders_only_drops_deposits():
    gdf = gpd.GeoDataFrame(
        {
            "Class": [0, 1, "0", "1"],
            "geometry": [box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5), box(6, 6, 7, 7)],
        },
        crs="EPSG:25829",
    )
    out = keep_boulders_only(gdf)
    assert len(out) == 2
    assert all(parse_class_value(c) == 0 for c in out["Class"])


def test_keep_boulders_only_noop_without_class():
    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:25829")
    out = keep_boulders_only(gdf)
    assert len(out) == 1


def test_matcher_excludes_deposits_by_default():
    before = gpd.GeoDataFrame(
        {
            "Class": [0, 1],
            "volume": [1.0, 10.0],
            "geometry": [box(0, 0, 1, 1), box(10, 10, 12, 12)],
        },
        crs="EPSG:25829",
    )
    after = gpd.GeoDataFrame(
        {
            "Class": [0, 1],
            "volume": [1.1, 10.0],
            "geometry": [box(0.2, 0.1, 1.2, 1.1), box(10.2, 10.1, 12.2, 12.1)],
        },
        crs="EPSG:25829",
    )
    results = BoulderMatcher(
        FakeSurvey(before), FakeSurvey(after), search_radius=5.0, min_score=0.3
    ).match()
    # Only the boulder pair should match; deposit pair is dropped before scoring.
    assert len(results["matches"]) == 1
    assert len(results["appeared"]) == 0
    assert len(results["disappeared"]) == 0
