# test_matcher.py
#
# Tests the matching logic using small synthetic GeoDataFrames.
# No DSM / rasterio / Detectron2 needed for most of these -- volume
# is injected directly as a column so we can control it precisely.
#
# Run with:  pytest tests/test_matcher.py -v

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from attributes import angle_difference_deg, compute_basic_attributes, safe_log_ratio
from matcher import BoulderMatcher


def make_square(cx, cy, size=1.0):
    """Square polygon centered at (cx, cy) with given side length."""
    h = size / 2
    return Polygon(
        [
            (cx - h, cy - h),
            (cx + h, cy - h),
            (cx + h, cy + h),
            (cx - h, cy + h),
        ]
    )


class FakeSurvey:
    """Minimal stand-in for BoulderSurvey -- just needs a .polygons attribute."""

    def __init__(self, polygons):
        self.polygons = polygons


def gdf_from_boulders(boulders, crs="EPSG:32633"):
    """
    boulders: list of dicts, e.g.
        {"cx": 0, "cy": 0, "size": 1.0, "volume": 2.0}
    """
    rows = []
    for b in boulders:
        geom = make_square(b["cx"], b["cy"], b.get("size", 1.0))
        rows.append({"geometry": geom, "volume": b.get("volume", np.nan)})
    gdf = gpd.GeoDataFrame(rows, crs=crs)
    return compute_basic_attributes(gdf)


# ---------------------------------------------------------------------
# attributes.py unit tests
# ---------------------------------------------------------------------

def test_safe_log_ratio_identical_values_is_zero():
    assert safe_log_ratio(5.0, 5.0) == pytest.approx(0.0, abs=1e-6)


def test_safe_log_ratio_is_symmetric():
    assert safe_log_ratio(2.0, 8.0) == pytest.approx(safe_log_ratio(8.0, 2.0))


def test_angle_difference_wraps_correctly():
    assert angle_difference_deg(10, 170) == pytest.approx(20.0)
    assert angle_difference_deg(5, 5) == pytest.approx(0.0)
    assert angle_difference_deg(0, 90) == pytest.approx(90.0)


def test_compute_basic_attributes_fills_defaults():
    gdf = gpd.GeoDataFrame(
        {"geometry": [make_square(0, 0)]}, crs="EPSG:32633"
    )
    out = compute_basic_attributes(gdf)
    assert out.loc[0, "area"] == pytest.approx(1.0)
    assert np.isnan(out.loc[0, "orientation"])
    assert np.isnan(out.loc[0, "volume"])
    assert out.loc[0, "confidence"] == 1.0


# ---------------------------------------------------------------------
# matcher.py integration tests (synthetic data, no DSM)
# ---------------------------------------------------------------------

def test_simple_one_to_one_match():
    """A boulder that barely moved should match with high confidence."""
    before = gdf_from_boulders([{"cx": 0, "cy": 0, "volume": 2.0}])
    after = gdf_from_boulders([{"cx": 0.2, "cy": 0.1, "volume": 2.1}])

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=5.0)
    result = matcher.match()

    assert len(result["matches"]) == 1
    assert len(result["appeared"]) == 0
    assert len(result["disappeared"]) == 0
    assert result["matches"].iloc[0]["match_score"] > 0.8


def test_boulder_outside_search_radius_is_not_matched():
    """Two boulders too far apart should show up as disappeared + appeared."""
    before = gdf_from_boulders([{"cx": 0, "cy": 0, "volume": 2.0}])
    after = gdf_from_boulders([{"cx": 50, "cy": 50, "volume": 2.0}])

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=5.0)
    result = matcher.match()

    assert len(result["matches"]) == 0
    assert len(result["appeared"]) == 1
    assert len(result["disappeared"]) == 1


def test_no_candidates_does_not_crash_and_has_geometry_column():
    """
    Regression test for the empty-GeoDataFrame bug: matches/vectors must
    have a usable geometry column even when there are zero matches, so
    that .to_file() downstream doesn't raise.
    """
    before = gdf_from_boulders([{"cx": 0, "cy": 0, "volume": 2.0}])
    after = gdf_from_boulders([{"cx": 100, "cy": 100, "volume": 2.0}])

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=1.0)
    result = matcher.match()

    assert result["matches"].empty
    assert "geometry" in result["matches"].columns
    assert result["vectors"].empty
    assert "geometry" in result["vectors"].columns
    # this is the line that used to raise if the geometry column was missing
    result["matches"].set_geometry("geometry")


def test_new_boulder_appears():
    """After has an extra boulder with nothing nearby in before -> appeared."""
    before = gdf_from_boulders([{"cx": 0, "cy": 0, "volume": 2.0}])
    after = gdf_from_boulders(
        [
            {"cx": 0, "cy": 0, "volume": 2.0},
            {"cx": 20, "cy": 20, "volume": 1.5},
        ]
    )

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=5.0)
    result = matcher.match()

    assert len(result["matches"]) == 1
    assert len(result["appeared"]) == 1


def test_boulder_disappears():
    """Before has an extra boulder with nothing nearby in after -> disappeared."""
    before = gdf_from_boulders(
        [
            {"cx": 0, "cy": 0, "volume": 2.0},
            {"cx": 20, "cy": 20, "volume": 1.5},
        ]
    )
    after = gdf_from_boulders([{"cx": 0, "cy": 0, "volume": 2.0}])

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=5.0)
    result = matcher.match()

    assert len(result["matches"]) == 1
    assert len(result["disappeared"]) == 1


def test_hungarian_prefers_globally_best_assignment():
    """
    Two before-boulders are each near two after-boulders, but a naive
    'greedy nearest neighbor' would give a worse total assignment than
    the Hungarian algorithm. This checks the assignment is optimal,
    not just locally greedy.
    """
    before = gdf_from_boulders(
        [
            {"cx": 0, "cy": 0, "volume": 2.0},
            {"cx": 3, "cy": 0, "volume": 5.0},
        ]
    )
    after = gdf_from_boulders(
        [
            {"cx": 0.5, "cy": 0, "volume": 2.0},   # close match for before #1
            {"cx": 3.5, "cy": 0, "volume": 5.0},   # close match for before #2
        ]
    )

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=5.0)
    result = matcher.match()

    matches = result["matches"].sort_values("before_id")
    assert len(matches) == 2
    # before #1 (volume 2.0) should match after #1 (volume 2.0), not after #2 (volume 5.0)
    assert matches.iloc[0]["before_volume"] == pytest.approx(2.0)
    assert matches.iloc[0]["after_volume"] == pytest.approx(2.0)


def test_min_score_threshold_rejects_weak_match():
    """A pair that's technically within radius but very dissimilar should be rejected."""
    before = gdf_from_boulders([{"cx": 0, "cy": 0, "size": 1.0, "volume": 1.0}])
    after = gdf_from_boulders([{"cx": 4.9, "cy": 0, "size": 10.0, "volume": 50.0}])

    matcher = BoulderMatcher(
        FakeSurvey(before), FakeSurvey(after), search_radius=5.0, min_score=0.55
    )
    result = matcher.match()

    assert len(result["matches"]) == 0
    assert len(result["appeared"]) == 1
    assert len(result["disappeared"]) == 1


def test_volume_missing_falls_back_to_neutral_score():
    """When volume is NaN on both sides, volume_score should default to 0.5, not crash."""
    before = gdf_from_boulders([{"cx": 0, "cy": 0, "volume": np.nan}])
    after = gdf_from_boulders([{"cx": 0.1, "cy": 0.1, "volume": np.nan}])

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=5.0)
    result = matcher.match()

    assert len(result["matches"]) == 1
    assert not np.isnan(result["matches"].iloc[0]["match_score"])
