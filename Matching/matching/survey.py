# survey.py

import geopandas as gpd

from .attributes import compute_basic_attributes, estimate_volume_from_dsm


class BoulderSurvey:
    def __init__(self, name: str, polygon_path: str, dsm_path: str | None = None):
        self.name = name
        self.polygon_path = polygon_path
        self.dsm_path = dsm_path
        self.polygons = gpd.read_file(polygon_path)

    def compute_attributes(self):
        self.polygons = compute_basic_attributes(self.polygons)
        return self

    def compute_volume(self, buffer_distance: float = 0.5):
        if self.dsm_path is None:
            raise ValueError(f"No DSM path provided for {self.name}")

        self.polygons = estimate_volume_from_dsm(
            self.polygons,
            self.dsm_path,
            buffer_distance=buffer_distance,
        )
        return self