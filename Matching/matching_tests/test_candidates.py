# test_candidates.py

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from matching.attributes import compute_basic_attributes
from matching.candidates import build_missed_candidates
from matching.matcher import BoulderMatcher


class FakeSurvey:
    def __init__(self, polygons):
        self.polygons = polygons


def _gdf(squares, crs="EPSG:25829"):
    rows = []
    for cx, cy, size, vol in squares:
        h = size / 2
        rows.append({"geometry": box(cx - h, cy - h, cx + h, cy + h), "volume": vol})
    return compute_basic_attributes(gpd.GeoDataFrame(rows, crs=crs))


def test_default_search_radius_is_extended():
    assert BoulderMatcher.DEFAULT_SEARCH_RADIUS >= 15.0


def test_missed_candidates_finds_far_pair():
    # Matcher at 5 m would miss a 12 m move; candidate radius 20 should see it.
    before = _gdf([(0, 0, 1.0, 2.0)])
    after = _gdf([(12, 0, 1.0, 2.1)])
    before["before_id"] = before.index
    after["after_id"] = after.index

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=5.0)
    results = matcher.match()
    assert results["matches"].empty
    assert len(results["disappeared"]) == 1
    assert len(results["appeared"]) == 1

    missed = build_missed_candidates(
        results["disappeared"],
        results["appeared"],
        candidate_radius=20.0,
        min_score=0.2,
    )
    assert len(missed) == 1
    assert missed.iloc[0]["distance_m"] == pytest.approx(12.0, abs=0.1)
    assert missed.iloc[0]["before_id"] == 0
    assert missed.iloc[0]["after_id"] == 0
