"""Boulder vs boulder-deposit Class helpers for matching.

Manual GPKGs use ``Class`` 0 = boulder, 1 = boulder deposit (same convention
as ``gpkg_to_coco.parse_class_value``). Matching never needs deposits, so the
matcher drops them by default when a Class column is present. Inference
layers without Class are left unchanged.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np


CLASS_BOULDER = 0
CLASS_DEPOSIT = 1

# Common column names seen on manual / converted layers
CLASS_COLUMNS = ("Class", "class", "class_id", "CLASS")


def parse_class_value(raw) -> int:
    """Map Class attribute to 0=boulder, 1=boulder deposit."""
    if raw is None:
        return CLASS_BOULDER
    try:
        if isinstance(raw, (int, np.integer)):
            return CLASS_DEPOSIT if int(raw) == 1 else CLASS_BOULDER
        if isinstance(raw, float) and np.isnan(raw):
            return CLASS_BOULDER
        if isinstance(raw, (float, np.floating)):
            return CLASS_DEPOSIT if int(raw) == 1 else CLASS_BOULDER
    except (TypeError, ValueError):
        pass
    text = str(raw).strip().lower()
    if not text or text in ("0", "boulder", "boulders"):
        return CLASS_BOULDER
    if "deposit" in text or text == "1":
        return CLASS_DEPOSIT
    return CLASS_BOULDER


def class_column(gdf: gpd.GeoDataFrame | None) -> str | None:
    if gdf is None or gdf.empty:
        return None
    for col in CLASS_COLUMNS:
        if col in gdf.columns:
            return col
    return None


def keep_boulders_only(
    gdf: gpd.GeoDataFrame | None,
    *,
    class_col: str | None = None,
    drop_unknown_as_boulder: bool = True,
) -> gpd.GeoDataFrame:
    """Drop boulder-deposit polygons when a Class column is present.

    Layers without Class (typical inference output) are returned unchanged.
    """
    if gdf is None:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:25829")
    if gdf.empty:
        return gdf.copy()

    col = class_col or class_column(gdf)
    if col is None:
        return gdf.copy()

    out = gdf.copy()
    parsed = out[col].map(parse_class_value)
    if not drop_unknown_as_boulder:
        # reserved for future stricter modes; default treats unknown as boulder
        pass
    mask = parsed == CLASS_BOULDER
    n_drop = int((~mask).sum())
    if n_drop:
        print(f"Excluding {n_drop} boulder-deposit polygons (Class≠0) from matching")
    return out.loc[mask].reset_index(drop=True)
