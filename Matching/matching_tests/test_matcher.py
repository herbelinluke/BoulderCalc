# test_matcher.py
#
# Tests the matching logic using small synthetic GeoDataFrames.
# No DSM / rasterio / Detectron2 needed for most of these -- volume
# is injected directly as a column so we can control it precisely.
#
# Run with:  pytest matching_tests/test_matcher.py -v

import geopandas as gpd
import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon

from matching.attributes import angle_difference_deg, compute_basic_attributes, safe_log_ratio
from matching.matcher import BoulderMatcher
from matching.survey import filter_boulder_class, filter_min_volume


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


def make_triangle(cx, cy, size=2.0):
    return Polygon(
        [
            (cx, cy + size * 0.7),
            (cx - size * 0.6, cy - size * 0.5),
            (cx + size * 0.6, cy - size * 0.5),
        ]
    )


class FakeSurvey:
    """Minimal stand-in for BoulderSurvey -- just needs a .polygons attribute."""

    def __init__(self, polygons):
        self.polygons = polygons


def gdf_from_boulders(boulders, crs="EPSG:32633"):
    """
    boulders: list of dicts, e.g.
        {"cx": 0, "cy": 0, "size": 1.0, "volume": 2.0, "shape": "square"|"triangle"}
    """
    rows = []
    for b in boulders:
        kind = b.get("shape", "square")
        if kind == "triangle":
            geom = make_triangle(b["cx"], b["cy"], b.get("size", 2.0))
        else:
            geom = make_square(b["cx"], b["cy"], b.get("size", 1.0))
        if b.get("rotate"):
            geom = rotate(geom, b["rotate"], origin="centroid")
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
# filter helpers
# ---------------------------------------------------------------------

def test_filter_boulder_class():
    gdf = gpd.GeoDataFrame(
        {
            "Class": [0, 1, 0],
            "geometry": [make_square(0, 0), make_square(2, 0), make_square(4, 0)],
        },
        crs="EPSG:32633",
    )
    out = filter_boulder_class(gdf)
    assert len(out) == 2


def test_filter_boulder_class_string_labels():
    """july14_24 stores Class as object strings; must still filter."""
    gdf = gpd.GeoDataFrame(
        {
            "Class": ["0", "1", "0"],
            "geometry": [make_square(0, 0), make_square(2, 0), make_square(4, 0)],
        },
        crs="EPSG:32633",
    )
    out = filter_boulder_class(gdf)
    assert len(out) == 2


def test_filter_min_volume():
    gdf = gpd.GeoDataFrame(
        {
            "volume": [0.2, 1.5, np.nan],
            "geometry": [make_square(0, 0, 2), make_square(3, 0, 2), make_square(6, 0, 2)],
        },
        crs="EPSG:32633",
    )
    gdf = compute_basic_attributes(gdf)
    out = filter_min_volume(gdf, min_volume=0.5, min_area_fallback=0.5)
    # 0.2 dropped; 1.5 kept; NaN kept via area fallback (area=4)
    assert len(out) == 2
    assert set(out["volume"].fillna(-1).tolist()) == {1.5, -1.0}


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


def test_far_mover_within_200m_gate_is_matched():
    """Storm transport: same shape ~150 m away should still match at 200 m gate."""
    before = gdf_from_boulders(
        [{"cx": 0, "cy": 0, "size": 2.0, "shape": "triangle", "volume": 3.0}]
    )
    after = gdf_from_boulders(
        [{"cx": 150, "cy": 20, "size": 2.0, "shape": "triangle", "volume": 3.1, "rotate": 25}]
    )

    matcher = BoulderMatcher(
        FakeSurvey(before), FakeSurvey(after), search_radius=200.0, min_score=0.5
    )
    result = matcher.match()

    assert len(result["matches"]) == 1
    assert result["matches"].iloc[0]["distance_m"] > 100


def test_shape_prefers_true_match_over_nearby_wrong_shape():
    """Nearby wrong silhouette should lose to a farther same-shape twin."""
    before = gdf_from_boulders(
        [{"cx": 0, "cy": 0, "size": 2.5, "shape": "triangle", "volume": 4.0}]
    )
    after = gdf_from_boulders(
        [
            {"cx": 2, "cy": 0, "size": 2.5, "shape": "square", "volume": 4.0},
            {"cx": 30, "cy": 5, "size": 2.5, "shape": "triangle", "volume": 4.1},
        ]
    )

    matcher = BoulderMatcher(
        FakeSurvey(before), FakeSurvey(after), search_radius=50.0, min_score=0.45
    )
    result = matcher.match()

    assert len(result["matches"]) == 1
    assert result["matches"].iloc[0]["after_id"] == 1


def test_no_candidates_does_not_crash_and_has_geometry_column():
    before = gdf_from_boulders([{"cx": 0, "cy": 0, "volume": 2.0}])
    after = gdf_from_boulders([{"cx": 100, "cy": 100, "volume": 2.0}])

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=1.0)
    result = matcher.match()

    assert result["matches"].empty
    assert "geometry" in result["matches"].columns
    assert result["vectors"].empty
    assert "geometry" in result["vectors"].columns
    result["matches"].set_geometry("geometry")


def test_new_boulder_appears():
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
    before = gdf_from_boulders(
        [
            {"cx": 0, "cy": 0, "volume": 2.0, "size": 1.0},
            {"cx": 3, "cy": 0, "volume": 5.0, "size": 2.0},
        ]
    )
    after = gdf_from_boulders(
        [
            {"cx": 0.5, "cy": 0, "volume": 2.0, "size": 1.0},
            {"cx": 3.5, "cy": 0, "volume": 5.0, "size": 2.0},
        ]
    )

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=5.0)
    result = matcher.match()

    matches = result["matches"].sort_values("before_id")
    assert len(matches) == 2
    assert matches.iloc[0]["before_volume"] == pytest.approx(2.0)
    assert matches.iloc[0]["after_volume"] == pytest.approx(2.0)


def test_min_score_threshold_rejects_weak_match():
    """Dissimilar shape + size/volume should fall below min_score."""
    before = gdf_from_boulders(
        [{"cx": 0, "cy": 0, "size": 1.0, "shape": "triangle", "volume": 1.0}]
    )
    after = gdf_from_boulders(
        [{"cx": 4.9, "cy": 0, "size": 10.0, "shape": "square", "volume": 50.0}]
    )

    matcher = BoulderMatcher(
        FakeSurvey(before), FakeSurvey(after), search_radius=5.0, min_score=0.55
    )
    result = matcher.match()

    assert len(result["matches"]) == 0
    assert len(result["appeared"]) == 1
    assert len(result["disappeared"]) == 1


def test_volume_missing_falls_back_to_neutral_score():
    before = gdf_from_boulders([{"cx": 0, "cy": 0, "volume": np.nan}])
    after = gdf_from_boulders([{"cx": 0.1, "cy": 0.1, "volume": np.nan}])

    matcher = BoulderMatcher(FakeSurvey(before), FakeSurvey(after), search_radius=5.0)
    result = matcher.match()

    assert len(result["matches"]) == 1
    assert not np.isnan(result["matches"].iloc[0]["match_score"])
