# survey.py

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .attributes import compute_basic_attributes, estimate_volume_from_dsm


def load_polygons(polygon_path: str | Path, target_crs: str | None = "EPSG:25829"):
    """Load GPKG/GeoJSON, preserving GeoPackage fid as ``source_fid`` when possible."""
    path = Path(polygon_path)
    try:
        gdf = gpd.read_file(path, engine="pyogrio", fid_as_index=True)
        gdf = gdf.reset_index()
        if "fid" in gdf.columns:
            gdf = gdf.rename(columns={"fid": "source_fid"})
        elif "source_fid" not in gdf.columns:
            gdf["source_fid"] = np.arange(1, len(gdf) + 1)
    except Exception:
        gdf = gpd.read_file(path)
        gdf = gdf.copy()
        gdf["source_fid"] = np.arange(1, len(gdf) + 1)

    if target_crs is not None:
        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs)
        elif str(gdf.crs) != target_crs:
            gdf = gdf.to_crs(target_crs)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf.reset_index(drop=True)


def filter_boulder_class(gdf, class_value: int = 0, class_column: str = "Class"):
    """Keep rows where Class == class_value when the column exists.

    Handles mixed dtypes across years (e.g. july14_24 uses string ``'0'``/``'1'``,
    july14_25 uses int).
    """
    if class_column not in gdf.columns:
        # Case-insensitive fallback
        matches = [c for c in gdf.columns if c.lower() == class_column.lower()]
        if not matches:
            return gdf
        class_column = matches[0]

    series = gdf[class_column]
    # Prefer numeric compare; fall back to string equality for object columns.
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        mask = numeric == class_value
        # Also keep exact string matches when numeric parse failed for some rows
        mask = mask | (series.astype(str) == str(class_value))
    else:
        mask = series.astype(str) == str(class_value)
    return gdf.loc[mask].copy().reset_index(drop=True)


def filter_min_volume(
    gdf,
    min_volume: float | None = 0.5,
    min_area_fallback: float | None = 0.5,
):
    """
    Drop polygons below ``min_volume`` (m³) when volume is available.

    When volume is NaN for a row, optionally keep rows with plan area ≥
    ``min_area_fallback`` (m²). If ``min_area_fallback`` is None, NaN-volume
    rows are dropped when ``min_volume`` is set.
    """
    if min_volume is None:
        return gdf

    gdf = gdf.copy()
    if "volume" not in gdf.columns:
        gdf["volume"] = np.nan
    if "area" not in gdf.columns:
        gdf["area"] = gdf.geometry.area

    vol = gdf["volume"].to_numpy(dtype=float)
    area = gdf["area"].to_numpy(dtype=float)
    keep = np.zeros(len(gdf), dtype=bool)
    has_vol = ~np.isnan(vol)
    keep[has_vol] = vol[has_vol] >= min_volume
    if min_area_fallback is not None:
        keep[~has_vol] = area[~has_vol] >= min_area_fallback
    return gdf.loc[keep].reset_index(drop=True)


class BoulderSurvey:
    def __init__(self, name: str, polygon_path: str, dsm_path: str | None = None):
        self.name = name
        self.polygon_path = polygon_path
        self.dsm_path = dsm_path
        self.polygons = load_polygons(polygon_path)

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
