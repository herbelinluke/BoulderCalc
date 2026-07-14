# matcher.py

import math

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString

from .attributes import angle_difference_deg, compute_basic_attributes, safe_log_ratio


class BoulderMatcher:
    def __init__(
        self,
        before,
        after,
        search_radius: float = 5.0,
        min_score: float = 0.55,
    ):
        self.before = before
        self.after = after
        self.search_radius = search_radius
        self.min_score = min_score

        self.weights = {
            "distance": 0.30,
            "area": 0.25,
            "volume": 0.35,
            "orientation": 0.10,
        }

    def score_pair(self, before_row, after_row) -> float:
        distance = before_row.geometry.centroid.distance(after_row.geometry.centroid)

        distance_score = max(0.0, 1.0 - distance / self.search_radius)
        area_score = math.exp(-safe_log_ratio(before_row["area"], after_row["area"]))

        if not np.isnan(before_row["volume"]) and not np.isnan(after_row["volume"]):
            volume_score = math.exp(
                -safe_log_ratio(before_row["volume"], after_row["volume"])
            )
        else:
            volume_score = 0.5

        if not np.isnan(before_row["orientation"]) and not np.isnan(after_row["orientation"]):
            angle_diff = angle_difference_deg(
                before_row["orientation"],
                after_row["orientation"],
            )
            orientation_score = max(0.0, 1.0 - angle_diff / 90.0)
        else:
            orientation_score = 0.5

        return (
            self.weights["distance"] * distance_score
            + self.weights["area"] * area_score
            + self.weights["volume"] * volume_score
            + self.weights["orientation"] * orientation_score
        )

    def match(self):
        before_gdf = self.before.polygons.copy()
        after_gdf = self.after.polygons.copy()

        if before_gdf.crs != after_gdf.crs:
            after_gdf = after_gdf.to_crs(before_gdf.crs)

        before_gdf = compute_basic_attributes(before_gdf).reset_index(drop=True)
        after_gdf = compute_basic_attributes(after_gdf).reset_index(drop=True)

        before_gdf["before_id"] = before_gdf.index
        after_gdf["after_id"] = after_gdf.index

        candidate_rows = []

        for _, before_row in before_gdf.iterrows():
            nearby = after_gdf[
                after_gdf.geometry.centroid.distance(before_row.geometry.centroid)
                <= self.search_radius
            ]

            for _, after_row in nearby.iterrows():
                score = self.score_pair(before_row, after_row)

                if score >= self.min_score:
                    candidate_rows.append(
                        {
                            "before_id": before_row["before_id"],
                            "after_id": after_row["after_id"],
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

            before_row = before_gdf.loc[
                before_gdf["before_id"] == before_id
            ].iloc[0]

            after_row = after_gdf.loc[
                after_gdf["after_id"] == after_id
            ].iloc[0]

            dx = after_row["centroid_x"] - before_row["centroid_x"]
            dy = after_row["centroid_y"] - before_row["centroid_y"]
            distance = math.sqrt(dx**2 + dy**2)

            rotation = np.nan
            if not np.isnan(before_row["orientation"]) and not np.isnan(after_row["orientation"]):
                rotation = angle_difference_deg(
                    before_row["orientation"],
                    after_row["orientation"],
                )

            match_records.append(
                {
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
            )

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

        disappeared = before_gdf[
            ~before_gdf["before_id"].isin(matched_before)
        ].copy()

        appeared = after_gdf[
            ~after_gdf["after_id"].isin(matched_after)
        ].copy()

        return {
            "matches": matches,
            "appeared": appeared,
            "disappeared": disappeared,
            "vectors": vectors,
        }