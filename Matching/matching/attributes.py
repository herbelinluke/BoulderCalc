# attributes.py

import math
import numpy as np
import rasterio
from rasterio.mask import mask


def safe_log_ratio(a: float, b: float, eps: float = 1e-6) -> float:
    return abs(math.log((a + eps) / (b + eps)))


def angle_difference_deg(a: float, b: float) -> float:
    diff = (a - b) % 180
    return min(diff, 180 - diff)


def compute_basic_attributes(gdf):
    gdf = gdf.copy()

    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length
    gdf["centroid_x"] = gdf.geometry.centroid.x
    gdf["centroid_y"] = gdf.geometry.centroid.y

    if "orientation" not in gdf.columns:
        gdf["orientation"] = np.nan

    if "volume" not in gdf.columns:
        gdf["volume"] = np.nan

    if "confidence" not in gdf.columns:
        gdf["confidence"] = 1.0

    return gdf


def estimate_volume_from_dsm(gdf, dsm_path, buffer_distance=0.5):
    gdf = gdf.copy()

    volumes = []
    mean_heights = []
    max_heights = []

    with rasterio.open(dsm_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        pixel_area = abs(src.res[0] * src.res[1])

        for geom in gdf.geometry:
            boulder_img, _ = mask(src, [geom], crop=True, filled=False)
            boulder_vals = boulder_img[0].compressed()

            outer = geom.buffer(buffer_distance)
            ring = outer.difference(geom)

            ring_img, _ = mask(src, [ring], crop=True, filled=False)
            ring_vals = ring_img[0].compressed()

            if len(boulder_vals) == 0 or len(ring_vals) == 0:
                volumes.append(np.nan)
                mean_heights.append(np.nan)
                max_heights.append(np.nan)
                continue

            base = np.median(ring_vals)
            heights = boulder_vals - base
            heights = heights[heights > 0]

            volumes.append(np.sum(heights) * pixel_area)
            mean_heights.append(np.mean(heights) if len(heights) else 0)
            max_heights.append(np.max(heights) if len(heights) else 0)

    gdf["volume"] = volumes
    gdf["mean_height"] = mean_heights
    gdf["max_height"] = max_heights

    return gdf