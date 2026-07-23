"""Propose missed-match candidates from unmatched (appeared / disappeared) polygons.

The Hungarian matcher only considers pairs inside ``search_radius`` that pass
``min_score``.  Boulders that moved farther, or that scored just below the
threshold, land in appeared/disappeared.  This module builds a second queue of
candidate pairs so humans can label true missed matches for the eval dataset.
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .attributes import compute_basic_attributes, safe_log_ratio
from .matcher import BoulderMatcher


def _row_score(
    before_row,
    after_row,
    search_radius: float,
    weights: dict | None = None,
) -> float:
    """Same feature mix as BoulderMatcher.score_pair (standalone for gdf rows)."""
    weights = weights or {
        "distance": 0.30,
        "area": 0.25,
        "volume": 0.35,
        "orientation": 0.10,
    }
    distance = before_row.geometry.centroid.distance(after_row.geometry.centroid)
    distance_score = max(0.0, 1.0 - distance / max(search_radius, 1e-6))
    area_score = math.exp(-safe_log_ratio(float(before_row["area"]), float(after_row["area"])))

    bv = before_row["volume"] if "volume" in before_row.index else np.nan
    av = after_row["volume"] if "volume" in after_row.index else np.nan
    if not (isinstance(bv, float) and np.isnan(bv)) and not (
        isinstance(av, float) and np.isnan(av)
    ) and pd.notna(bv) and pd.notna(av):
        volume_score = math.exp(-safe_log_ratio(float(bv), float(av)))
    else:
        volume_score = 0.5

    bo = before_row["orientation"] if "orientation" in before_row.index else np.nan
    ao = after_row["orientation"] if "orientation" in after_row.index else np.nan
    if pd.notna(bo) and pd.notna(ao):
        from .attributes import angle_difference_deg

        angle_diff = angle_difference_deg(float(bo), float(ao))
        orientation_score = max(0.0, 1.0 - angle_diff / 90.0)
    else:
        orientation_score = 0.5

    return (
        weights["distance"] * distance_score
        + weights["area"] * area_score
        + weights["volume"] * volume_score
        + weights["orientation"] * orientation_score
    )


def build_missed_candidates(
    disappeared: gpd.GeoDataFrame,
    appeared: gpd.GeoDataFrame,
    candidate_radius: float = 20.0,
    min_score: float = 0.35,
    max_per_before: int = 5,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> gpd.GeoDataFrame:
    """Rank appeared↔disappeared pairs the matcher did not accept.

    Parameters
    ----------
    candidate_radius
        Wider than the matcher search radius so long movers can be reviewed.
    min_score
        Soft floor (usually lower than matcher ``min_score``) so near-misses
        still appear in the review queue.
    max_per_before
        Cap candidates per disappeared boulder (nearest / highest score first).
    exclude_pairs
        ``{(before_id, after_id), ...}`` already accepted by the matcher.
    """
    exclude_pairs = exclude_pairs or set()
    if disappeared is None or disappeared.empty or appeared is None or appeared.empty:
        return gpd.GeoDataFrame(geometry=[], crs=getattr(disappeared, "crs", "EPSG:25829"))

    before_gdf = compute_basic_attributes(disappeared.copy()).reset_index(drop=True)
    after_gdf = compute_basic_attributes(appeared.copy()).reset_index(drop=True)

    if "before_id" not in before_gdf.columns:
        before_gdf["before_id"] = before_gdf.index
    if "after_id" not in after_gdf.columns:
        after_gdf["after_id"] = after_gdf.index

    if before_gdf.crs != after_gdf.crs and after_gdf.crs is not None:
        after_gdf = after_gdf.to_crs(before_gdf.crs)

    records = []
    for _, brow in before_gdf.iterrows():
        dists = after_gdf.geometry.centroid.distance(brow.geometry.centroid)
        nearby_idx = after_gdf.index[dists <= candidate_radius].tolist()
        scored = []
        for j in nearby_idx:
            arow = after_gdf.loc[j]
            bid, aid = int(brow["before_id"]), int(arow["after_id"])
            if (bid, aid) in exclude_pairs:
                continue
            score = _row_score(brow, arow, search_radius=candidate_radius)
            if score < min_score:
                continue
            dist = float(dists.loc[j])
            scored.append((score, dist, arow))
        scored.sort(key=lambda t: (-t[0], t[1]))
        for score, dist, arow in scored[:max_per_before]:
            dx = float(arow["centroid_x"] - brow["centroid_x"])
            dy = float(arow["centroid_y"] - brow["centroid_y"])
            records.append(
                {
                    "before_id": int(brow["before_id"]),
                    "after_id": int(arow["after_id"]),
                    "match_score": float(score),
                    "dx": dx,
                    "dy": dy,
                    "distance_m": float(dist),
                    "rotation_deg": np.nan,
                    "before_area": float(brow["area"]),
                    "after_area": float(arow["area"]),
                    "before_volume": _maybe_vol(brow),
                    "after_volume": _maybe_vol(arow),
                    "candidate_source": "appeared_disappeared",
                    "geometry": arow.geometry,
                    "before_geometry": brow.geometry,
                }
            )

    if not records:
        return gpd.GeoDataFrame(geometry=[], crs=before_gdf.crs)

    out = gpd.GeoDataFrame(records, crs=before_gdf.crs)
    out = out.sort_values("match_score", ascending=False).reset_index(drop=True)
    return out


def _maybe_vol(row) -> float:
    if "volume" not in row.index:
        return np.nan
    v = row["volume"]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f


def candidates_to_match_rows(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop helper columns so rows look like ``matched_boulders`` for the eval UI."""
    if candidates.empty:
        return candidates
    cols = [
        c
        for c in candidates.columns
        if c
        in {
            "before_id",
            "after_id",
            "match_score",
            "dx",
            "dy",
            "distance_m",
            "rotation_deg",
            "before_area",
            "after_area",
            "before_volume",
            "after_volume",
            "candidate_source",
            "geometry",
        }
    ]
    return candidates[cols].copy()


def write_missed_candidates(candidates: gpd.GeoDataFrame, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # GeoJSON can't store before_geometry easily alongside; drop it for file IO
    out = candidates.drop(columns=["before_geometry"], errors="ignore")
    if out.empty:
        gpd.GeoDataFrame({"geometry": []}, crs="EPSG:25829").to_file(path, driver="GeoJSON")
    else:
        out.to_file(path, driver="GeoJSON")


def run_matcher_with_candidates(
    before_survey,
    after_survey,
    search_radius: float = 15.0,
    min_score: float = 0.55,
    candidate_radius: float | None = None,
    candidate_min_score: float = 0.35,
) -> dict:
    """Match, then attach a ``missed_candidates`` layer for eval review."""
    matcher = BoulderMatcher(
        before=before_survey,
        after=after_survey,
        search_radius=search_radius,
        min_score=min_score,
    )
    results = matcher.match()
    exclude = set()
    if not results["matches"].empty:
        exclude = {
            (int(r.before_id), int(r.after_id))
            for r in results["matches"].itertuples()
        }
    cand_r = candidate_radius if candidate_radius is not None else max(search_radius * 1.5, search_radius + 5.0)
    missed = build_missed_candidates(
        results["disappeared"],
        results["appeared"],
        candidate_radius=cand_r,
        min_score=candidate_min_score,
        exclude_pairs=exclude,
    )
    results["missed_candidates"] = missed
    results["candidate_radius"] = cand_r
    return results


def candidate_vectors(candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if candidates.empty or "before_geometry" not in candidates.columns:
        return gpd.GeoDataFrame(geometry=[], crs=getattr(candidates, "crs", None))
    rows = []
    for _, row in candidates.iterrows():
        bg, ag = row["before_geometry"], row.geometry
        if bg is None or ag is None:
            continue
        rows.append(
            {
                "before_id": row["before_id"],
                "after_id": row["after_id"],
                "match_score": row["match_score"],
                "distance_m": row["distance_m"],
                "geometry": LineString([bg.centroid, ag.centroid]),
            }
        )
    return gpd.GeoDataFrame(rows, crs=candidates.crs)
