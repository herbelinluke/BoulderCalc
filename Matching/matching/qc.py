"""DSM-of-Difference (DoD) QC layer for boulder matching.

After polygon matching, compare elevation change under before/after footprints
to flag:

- consistent movers (source depletion + sink deposition)
- likely missed movers among appeared/disappeared
- stationary / weak / inconsistent DoD signals

Does not change match assignment — it annotates results for review.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from shapely.geometry import mapping

from .attributes import _same_horizontal_crs


def _ensure_crs(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs(target_crs)
    if _same_horizontal_crs(gdf.crs, target_crs):
        return gdf
    return gdf.to_crs(target_crs)


def build_dod(
    before_dsm: str | Path,
    after_dsm: str | Path,
    bounds: tuple[float, float, float, float] | None = None,
    pad_m: float = 5.0,
) -> tuple[np.ndarray, rasterio.Affine, object, float]:
    """Warp after DSM onto before DSM grid; return (DoD, transform, crs, pixel_area).

    DoD = after − before (positive = deposition / arrival).

    If ``bounds`` (minx, miny, maxx, maxy) is given, only that window (+ pad)
    is loaded — important for large site DSMs.
    """
    before_dsm = Path(before_dsm)
    after_dsm = Path(after_dsm)

    with rasterio.open(before_dsm) as src_b, rasterio.open(after_dsm) as src_a:
        if bounds is not None:
            from rasterio.windows import from_bounds, Window

            minx, miny, maxx, maxy = bounds
            window = from_bounds(
                minx - pad_m,
                miny - pad_m,
                maxx + pad_m,
                maxy + pad_m,
                transform=src_b.transform,
            ).round_offsets().round_lengths()
            # Clamp to raster
            row_off = max(0, int(window.row_off))
            col_off = max(0, int(window.col_off))
            height = min(int(window.height), src_b.height - row_off)
            width = min(int(window.width), src_b.width - col_off)
            if height <= 0 or width <= 0:
                raise ValueError("DoD bounds do not intersect before DSM")
            window = Window(col_off, row_off, width, height)
            before = src_b.read(1, window=window, masked=True).astype(np.float32)
            transform = src_b.window_transform(window)
        else:
            before = src_b.read(1, masked=True).astype(np.float32)
            transform = src_b.transform

        after = np.full(before.shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src_a, 1),
            destination=after,
            src_transform=src_a.transform,
            src_crs=src_a.crs,
            dst_transform=transform,
            dst_crs=src_b.crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        crs = src_b.crs
        pixel_area = abs(src_b.res[0] * src_b.res[1])
        nodata_b = src_b.nodata

    before_f = np.ma.filled(before, np.nan).astype(np.float32)
    if nodata_b is not None:
        before_f = np.where(before_f == nodata_b, np.nan, before_f)

    dod = after - before_f
    return dod, transform, crs, float(pixel_area)


def _union_bounds(*gdfs: gpd.GeoDataFrame) -> tuple[float, float, float, float] | None:
    bounds_list = [gdf.total_bounds for gdf in gdfs if gdf is not None and not gdf.empty]
    if not bounds_list:
        return None
    minx = min(b[0] for b in bounds_list)
    miny = min(b[1] for b in bounds_list)
    maxx = max(b[2] for b in bounds_list)
    maxy = max(b[3] for b in bounds_list)
    return (float(minx), float(miny), float(maxx), float(maxy))


def _polygon_change_stats(
    dod: np.ndarray,
    transform: rasterio.Affine,
    geom,
    lod_m: float,
    pixel_area: float,
) -> dict:
    """Integrate signed / positive / negative change under one polygon."""
    empty = {
        "n_pixels": 0,
        "mean_dz": np.nan,
        "v_pos_m3": 0.0,
        "v_neg_m3": 0.0,
        "v_signed_m3": 0.0,
        "frac_above_lod": 0.0,
        "frac_below_lod": 0.0,
    }
    if geom is None or geom.is_empty:
        return empty

    minx, miny, maxx, maxy = geom.bounds
    # row/col from affine (north-up: e < 0)
    inv = ~transform
    c0, r0 = inv * (minx, maxy)
    c1, r1 = inv * (maxx, miny)
    row_start = max(0, int(np.floor(min(r0, r1))) - 1)
    row_stop = min(dod.shape[0], int(np.ceil(max(r0, r1))) + 2)
    col_start = max(0, int(np.floor(min(c0, c1))) - 1)
    col_stop = min(dod.shape[1], int(np.ceil(max(c0, c1))) + 2)
    if row_stop <= row_start or col_stop <= col_start:
        return empty

    window = dod[row_start:row_stop, col_start:col_stop]
    win_transform = transform * rasterio.Affine.translation(col_start, row_start)
    try:
        outside = geometry_mask(
            [mapping(geom)],
            out_shape=window.shape,
            transform=win_transform,
            invert=False,
            all_touched=True,
        )
    except Exception:
        return empty

    vals = window[~outside]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return empty

    pos = vals[vals > lod_m]
    neg = vals[vals < -lod_m]
    v_pos = float(np.sum(pos) * pixel_area) if pos.size else 0.0
    v_neg = float(np.abs(np.sum(neg)) * pixel_area) if neg.size else 0.0
    return {
        "n_pixels": int(vals.size),
        "mean_dz": float(np.nanmean(vals)),
        "v_pos_m3": v_pos,
        "v_neg_m3": v_neg,
        "v_signed_m3": float(np.nansum(vals) * pixel_area),
        "frac_above_lod": float(np.mean(vals > lod_m)),
        "frac_below_lod": float(np.mean(vals < -lod_m)),
    }


def _classify_match(
    src: dict,
    sink: dict,
    distance_m: float,
    stationary_dist_m: float,
    min_change_m3: float,
) -> str:
    moved = distance_m > stationary_dist_m
    src_ok = src["v_neg_m3"] >= min_change_m3
    sink_ok = sink["v_pos_m3"] >= min_change_m3
    both_weak = (not src_ok) and (not sink_ok)

    if not moved:
        if both_weak:
            return "stationary_ok"
        if src_ok or sink_ok:
            return "stationary_dod_noise"
        return "stationary_ok"

    if src_ok and sink_ok:
        return "consistent_move"
    if src_ok and not sink_ok:
        return "move_source_only"
    if sink_ok and not src_ok:
        return "move_sink_only"
    return "move_weak_dod"


def _classify_disappeared(src: dict, min_change_m3: float) -> str:
    if src["v_neg_m3"] >= min_change_m3:
        return "likely_missed_mover_source"
    if src["v_pos_m3"] >= min_change_m3:
        return "unexpected_deposition"
    return "weak_or_true_loss"


def _classify_appeared(sink: dict, min_change_m3: float) -> str:
    if sink["v_pos_m3"] >= min_change_m3:
        return "likely_missed_mover_sink"
    if sink["v_neg_m3"] >= min_change_m3:
        return "unexpected_depletion"
    return "weak_or_true_gain"


def run_dod_qc(
    results: dict[str, gpd.GeoDataFrame],
    before_polygons: gpd.GeoDataFrame,
    after_polygons: gpd.GeoDataFrame,
    before_dsm: str | Path,
    after_dsm: str | Path,
    lod_m: float = 0.08,
    min_change_m3: float = 0.05,
    stationary_dist_m: float = 0.35,
) -> dict[str, gpd.GeoDataFrame]:
    """Annotate match / appeared / disappeared layers with DoD QC fields.

    Returns a dict with:
      - match_qc
      - disappeared_qc
      - appeared_qc
      - summary (single-row GeoDataFrame with counts; empty geometry)
    """
    with rasterio.open(before_dsm) as src:
        dsm_crs = src.crs

    before = _ensure_crs(before_polygons.copy(), dsm_crs)
    after = _ensure_crs(after_polygons.copy(), dsm_crs)
    matches = results.get("matches", gpd.GeoDataFrame(geometry=[], crs=dsm_crs))
    disappeared = results.get("disappeared", gpd.GeoDataFrame(geometry=[], crs=dsm_crs))
    appeared = results.get("appeared", gpd.GeoDataFrame(geometry=[], crs=dsm_crs))

    if not matches.empty:
        matches = _ensure_crs(matches, dsm_crs)
    if not disappeared.empty:
        disappeared = _ensure_crs(disappeared, dsm_crs)
    if not appeared.empty:
        appeared = _ensure_crs(appeared, dsm_crs)

    bounds = _union_bounds(before, after, matches, disappeared, appeared)
    dod, transform, crs, pixel_area = build_dod(
        before_dsm, after_dsm, bounds=bounds, pad_m=10.0
    )

    def _before_geom(before_id):
        if "before_id" in before.columns:
            hit = before[before["before_id"] == before_id]
            if not hit.empty:
                return hit.iloc[0].geometry
        idx = int(before_id)
        if 0 <= idx < len(before):
            return before.iloc[idx].geometry
        return None

    def _after_geom(after_id):
        if "after_id" in after.columns:
            hit = after[after["after_id"] == after_id]
            if not hit.empty:
                return hit.iloc[0].geometry
        idx = int(after_id)
        if 0 <= idx < len(after):
            return after.iloc[idx].geometry
        return None

    match_rows = []
    for _, row in matches.iterrows():
        b_geom = _before_geom(row.get("before_id"))
        a_geom = _after_geom(row.get("after_id"))
        if a_geom is None:
            a_geom = row.geometry
        src = _polygon_change_stats(dod, transform, b_geom, lod_m, pixel_area)
        sink = _polygon_change_stats(dod, transform, a_geom, lod_m, pixel_area)
        dist = float(row.get("distance_m", np.nan))
        if np.isnan(dist) and b_geom is not None and a_geom is not None:
            dist = float(b_geom.centroid.distance(a_geom.centroid))
        label = _classify_match(src, sink, dist, stationary_dist_m, min_change_m3)
        ratio = np.nan
        if src["v_neg_m3"] > 0 and sink["v_pos_m3"] > 0:
            ratio = float(min(src["v_neg_m3"], sink["v_pos_m3"]) / max(src["v_neg_m3"], sink["v_pos_m3"]))
        match_rows.append(
            {
                **{k: row[k] for k in row.index if k != "geometry"},
                "qc_label": label,
                "dod_src_v_neg_m3": src["v_neg_m3"],
                "dod_src_mean_dz": src["mean_dz"],
                "dod_sink_v_pos_m3": sink["v_pos_m3"],
                "dod_sink_mean_dz": sink["mean_dz"],
                "dod_volume_ratio": ratio,
                "geometry": a_geom if a_geom is not None else row.geometry,
            }
        )

    disc_rows = []
    for _, row in disappeared.iterrows():
        geom = row.geometry
        src = _polygon_change_stats(dod, transform, geom, lod_m, pixel_area)
        label = _classify_disappeared(src, min_change_m3)
        disc_rows.append(
            {
                **{k: row[k] for k in row.index if k != "geometry"},
                "qc_label": label,
                "dod_src_v_neg_m3": src["v_neg_m3"],
                "dod_src_v_pos_m3": src["v_pos_m3"],
                "dod_src_mean_dz": src["mean_dz"],
                "geometry": geom,
            }
        )

    app_rows = []
    for _, row in appeared.iterrows():
        geom = row.geometry
        sink = _polygon_change_stats(dod, transform, geom, lod_m, pixel_area)
        label = _classify_appeared(sink, min_change_m3)
        app_rows.append(
            {
                **{k: row[k] for k in row.index if k != "geometry"},
                "qc_label": label,
                "dod_sink_v_pos_m3": sink["v_pos_m3"],
                "dod_sink_v_neg_m3": sink["v_neg_m3"],
                "dod_sink_mean_dz": sink["mean_dz"],
                "geometry": geom,
            }
        )

    match_qc = gpd.GeoDataFrame(match_rows, crs=crs) if match_rows else gpd.GeoDataFrame(geometry=[], crs=crs)
    disappeared_qc = (
        gpd.GeoDataFrame(disc_rows, crs=crs) if disc_rows else gpd.GeoDataFrame(geometry=[], crs=crs)
    )
    appeared_qc = gpd.GeoDataFrame(app_rows, crs=crs) if app_rows else gpd.GeoDataFrame(geometry=[], crs=crs)

    def _counts(gdf, col="qc_label"):
        if gdf.empty or col not in gdf.columns:
            return {}
        return {str(k): int(v) for k, v in gdf[col].value_counts().items()}

    summary = {
        "lod_m": lod_m,
        "min_change_m3": min_change_m3,
        "stationary_dist_m": stationary_dist_m,
        "n_match_qc": len(match_qc),
        "n_disappeared_qc": len(disappeared_qc),
        "n_appeared_qc": len(appeared_qc),
        "match_labels": _counts(match_qc),
        "disappeared_labels": _counts(disappeared_qc),
        "appeared_labels": _counts(appeared_qc),
        "likely_missed_mover_sources": int(
            (disappeared_qc["qc_label"] == "likely_missed_mover_source").sum()
            if not disappeared_qc.empty
            else 0
        ),
        "likely_missed_mover_sinks": int(
            (appeared_qc["qc_label"] == "likely_missed_mover_sink").sum()
            if not appeared_qc.empty
            else 0
        ),
        "consistent_moves": int(
            (match_qc["qc_label"] == "consistent_move").sum() if not match_qc.empty else 0
        ),
    }

    return {
        "match_qc": match_qc,
        "disappeared_qc": disappeared_qc,
        "appeared_qc": appeared_qc,
        "summary": summary,
    }


def write_dod_qc(qc: dict, outdir: Path) -> Path:
    """Write QC GeoJSONs + summary JSON under ``outdir``."""
    import json

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    qc["match_qc"].to_file(outdir / "match_dod_qc.geojson", driver="GeoJSON")
    qc["disappeared_qc"].to_file(outdir / "disappeared_dod_qc.geojson", driver="GeoJSON")
    qc["appeared_qc"].to_file(outdir / "appeared_dod_qc.geojson", driver="GeoJSON")
    summary_path = outdir / "dod_qc_summary.json"
    summary_path.write_text(json.dumps(qc["summary"], indent=2))
    return summary_path
