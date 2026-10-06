# matcher.py

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point

from .attributes import compute_basic_attributes, safe_log_ratio
from .shape import attach_shape_signatures, shape_similarity


class BoulderMatcher:
    def __init__(
        self,
        before,
        after,
        search_radius: float = 200.0,
        min_score: float = 0.55,
        n_contour_points: int = 64,
    ):
        self.before = before
        self.after = after
        self.search_radius = search_radius
        self.min_score = min_score
        self.n_contour_points = n_contour_points

        # Shape-first: contour similarity dominates; position is a weak prior.
        self.weights = {
            "shape": 0.70,
            "volume": 0.20,
            "distance": 0.10,
        }

    def score_pair(self, before_row, after_row) -> float:
        distance = before_row.geometry.centroid.distance(after_row.geometry.centroid)
        distance_score = max(0.0, 1.0 - distance / self.search_radius)

        sig_a = (before_row["shape_sig"], before_row.get("orientation", float("nan")))
        sig_b = (after_row["shape_sig"], after_row.get("orientation", float("nan")))
        contour_score = shape_similarity(
            before_row.geometry,
            after_row.geometry,
            n=self.n_contour_points,
            sig_a=sig_a,
            sig_b=sig_b,
        )
        area_score = math.exp(
            -safe_log_ratio(before_row["area"], after_row["area"])
        )
        # Fold area into the shape term so wildly different sizes lose
        # (contours are scale-normalized, so size must come from area/volume).
        shape_score = 0.70 * contour_score + 0.30 * area_score

        if not np.isnan(before_row["volume"]) and not np.isnan(after_row["volume"]):
            volume_score = math.exp(
                -safe_log_ratio(before_row["volume"], after_row["volume"])
            )
        else:
            volume_score = 0.5

        return (
            self.weights["shape"] * shape_score
            + self.weights["volume"] * volume_score
            + self.weights["distance"] * distance_score
        )

    def match(self):
        before_gdf = self.before.polygons.copy()
        after_gdf = self.after.polygons.copy()

        if before_gdf.crs != after_gdf.crs:
            after_gdf = after_gdf.to_crs(before_gdf.crs)

        before_gdf = compute_basic_attributes(before_gdf).reset_index(drop=True)
        after_gdf = compute_basic_attributes(after_gdf).reset_index(drop=True)

        before_gdf = attach_shape_signatures(before_gdf, n=self.n_contour_points)
        after_gdf = attach_shape_signatures(after_gdf, n=self.n_contour_points)

        before_gdf["before_id"] = before_gdf.index
        after_gdf["after_id"] = after_gdf.index

        candidate_rows = []

        if len(before_gdf) and len(after_gdf):
            # Buffer-based spatial join for candidate generation (meters in projected CRS).
            before_pts = before_gdf.copy()
            before_pts["geometry"] = before_pts.geometry.centroid
            after_pts = after_gdf.copy()
            after_pts["geometry"] = after_pts.geometry.centroid

            before_buf = before_pts.copy()
            before_buf["geometry"] = before_buf.geometry.buffer(self.search_radius)

            joined = gpd.sjoin(
                before_buf[["before_id", "geometry"]],
                after_pts[["after_id", "geometry"]],
                how="inner",
                predicate="intersects",
            )

            before_by_id = before_gdf.set_index("before_id", drop=False)
            after_by_id = after_gdf.set_index("after_id", drop=False)

            for before_id, after_id in zip(
                joined["before_id"].to_numpy(),
                joined["after_id"].to_numpy(),
            ):
                before_row = before_by_id.loc[before_id]
                after_row = after_by_id.loc[after_id]
                # Exact centroid distance gate (buffer is slightly loose).
                d = Point(before_row["centroid_x"], before_row["centroid_y"]).distance(
                    Point(after_row["centroid_x"], after_row["centroid_y"])
                )
                if d > self.search_radius:
                    continue
                score = self.score_pair(before_row, after_row)
                if score >= self.min_score:
                    candidate_rows.append(
                        {
                            "before_id": int(before_id),
                            "after_id": int(after_id),
                            "score": score,
                        }
                    )

        candidates = pd.DataFrame(candidate_rows)

        if candidates.empty:
            empty_matches = gpd.GeoDataFrame({"geometry": []}, crs=before_gdf.crs)
            empty_vectors = gpd.GeoDataFrame({"geometry": []}, crs=before_gdf.crs)
            return {
                "matches": empty_matches,
                "appeared": after_gdf,
                "disappeared": before_gdf,
                "vectors": empty_vectors,
            }

        before_ids = sorted(candidates["before_id"].unique())
        after_ids = sorted(candidates["after_id"].unique())

        before_id_to_row = {v: i for i, v in enumerate(before_ids)}
        after_id_to_col = {v: i for i, v in enumerate(after_ids)}

        cost_matrix = np.ones((len(before_ids), len(after_ids))) * 9999

        for _, row in candidates.iterrows():
            i = before_id_to_row[row["before_id"]]
            j = after_id_to_col[row["after_id"]]
            cost_matrix[i, j] = 1.0 - row["score"]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        match_records = []
        vector_records = []
        matched_before = set()
        matched_after = set()

        for i, j in zip(row_ind, col_ind):
            cost = cost_matrix[i, j]

            if cost >= 9999:
                continue

            score = 1.0 - cost

            if score < self.min_score:
                continue

            before_id = before_ids[i]
            after_id = after_ids[j]

            before_row = before_gdf.loc[before_gdf["before_id"] == before_id].iloc[0]
            after_row = after_gdf.loc[after_gdf["after_id"] == after_id].iloc[0]

            dx = after_row["centroid_x"] - before_row["centroid_x"]
            dy = after_row["centroid_y"] - before_row["centroid_y"]
            distance = math.sqrt(dx**2 + dy**2)

            rotation = np.nan
            if not np.isnan(before_row["orientation"]) and not np.isnan(
                after_row["orientation"]
            ):
                # PCA orientations are in [0, 180); difference wraps at 180.
                diff = abs(before_row["orientation"] - after_row["orientation"]) % 180
                rotation = min(diff, 180 - diff)

            record = {
                "before_id": before_id,
                "after_id": after_id,
                "match_score": score,
                "dx": dx,
                "dy": dy,
                "distance_m": distance,
                "rotation_deg": rotation,
                "before_area": before_row["area"],
                "after_area": after_row["area"],
                "before_volume": before_row["volume"],
                "after_volume": after_row["volume"],
                "geometry": after_row.geometry,
            }
            if "source_fid" in before_row.index and pd.notna(before_row["source_fid"]):
                record["before_fid"] = int(before_row["source_fid"])
            if "source_fid" in after_row.index and pd.notna(after_row["source_fid"]):
                record["after_fid"] = int(after_row["source_fid"])

            match_records.append(record)

            vector_records.append(
                {
                    "before_id": before_id,
                    "after_id": after_id,
                    "match_score": score,
                    "distance_m": distance,
                    "rotation_deg": rotation,
                    "geometry": LineString(
                        [
                            before_row.geometry.centroid,
                            after_row.geometry.centroid,
                        ]
                    ),
                }
            )

            matched_before.add(before_id)
            matched_after.add(after_id)

        matches = gpd.GeoDataFrame(match_records, crs=before_gdf.crs)
        vectors = gpd.GeoDataFrame(vector_records, crs=before_gdf.crs)

        disappeared = before_gdf[~before_gdf["before_id"].isin(matched_before)].copy()
        appeared = after_gdf[~after_gdf["after_id"].isin(matched_after)].copy()

        # Drop bulky signature arrays before returning layers for export.
        for gdf in (appeared, disappeared):
            if "shape_sig" in gdf.columns:
                gdf.drop(columns=["shape_sig"], inplace=True)

        return {
            "matches": matches,
            "appeared": appeared,
            "disappeared": disappeared,
            "vectors": vectors,
        }
