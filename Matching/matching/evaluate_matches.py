"""Interactive labeling UI for building a matcher evaluation dataset.

Flip through inferred matches (same layout as ``visualize.run_gui``), label each
as a confirmed match / not-a-match / unsure, and append records to a JSON file.

Each record stores centroids, bbox, WKT polygons, matcher score, and whether
the detections intersect the manual annotation GPKGs (july14_24 / july14_25),
including GeoPackage ``fid`` values when available (stable under appends).

Example:
  python -m matching.evaluate_matches \\
    --outdir ../../segmentation/training_run_rgb_dsm_4000/matching \\
    --labels-json ../../segmentation/training_run_rgb_dsm_4000/matching/eval/match_labels.json

Keys:
  n / →     next match
  p / ←     previous match
  y         label CONFIRMED MATCH
  x         label NOT A MATCH
  ? / u     label UNSURE
  backspace clear label for current match
  o         toggle overview zoom
  j         jump to next unlabeled
  c         print QGIS-friendly extent / centroids to terminal
  s / Ctrl+s save JSON (+ companion GeoJSON for QGIS)
  q         save and quit
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from .visualize import (
    COLORS,
    _draw_displacement_vector,
    _match_pair_geoms,
    load_inputs,
    load_results,
    plot_match_detail,
    plot_overview,
)

LABEL_CONFIRMED = "confirmed_match"
LABEL_NOT = "not_match"
LABEL_UNSURE = "unsure"
VALID_LABELS = {LABEL_CONFIRMED, LABEL_NOT, LABEL_UNSURE}

LABEL_COLORS = {
    LABEL_CONFIRMED: "#2ecc71",
    LABEL_NOT: "#e74c3c",
    LABEL_UNSURE: "#f1c40f",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_manual_annotations(path: Path, target_crs: str = "EPSG:25829") -> gpd.GeoDataFrame:
    """Load a GPKG/GeoJSON, keep GeoPackage ``fid`` when present, reproject."""
    if not path.exists():
        raise FileNotFoundError(path)

    # Prefer pyogrio so GPKG fid is preserved as the index.
    try:
        gdf = gpd.read_file(path, engine="pyogrio", fid_as_index=True)
        gdf = gdf.reset_index()
        if "fid" in gdf.columns:
            gdf = gdf.rename(columns={"fid": "ann_fid"})
        elif "ann_fid" not in gdf.columns:
            gdf["ann_fid"] = np.arange(1, len(gdf) + 1)
    except Exception:
        gdf = gpd.read_file(path)
        gdf = gdf.copy()
        gdf["ann_fid"] = np.arange(1, len(gdf) + 1)

    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    elif str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.map(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g
    )
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    gdf["ann_fid"] = gdf["ann_fid"].astype(int)
    return gdf


def _iou(a, b) -> float:
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    return float(inter / union) if union > 0 else 0.0


def annotation_hits(
    geom,
    ann_gdf: gpd.GeoDataFrame,
    min_iou: float = 0.05,
) -> dict:
    """Return intersect flag + overlapping annotation fids / best IoU."""
    empty = {
        "intersects": False,
        "fids": [],
        "best_iou": 0.0,
        "best_fid": None,
    }
    if geom is None or geom.is_empty or ann_gdf is None or ann_gdf.empty:
        return empty

    try:
        cand = ann_gdf[ann_gdf.intersects(geom)]
    except Exception:
        return empty

    if cand.empty:
        return empty

    hits = []
    best_iou = 0.0
    best_fid = None
    for _, row in cand.iterrows():
        iou = _iou(geom, row.geometry)
        fid = int(row["ann_fid"]) if "ann_fid" in row.index and row["ann_fid"] is not None else None
        if iou >= min_iou or geom.intersects(row.geometry):
            if fid is not None and iou >= min_iou:
                hits.append({"fid": fid, "iou": round(iou, 4)})
            if iou > best_iou:
                best_iou = iou
                best_fid = fid

    # Always mark intersects if any spatial intersect, even with low IoU
    intersects = True
    # Prefer fids that meet min_iou; if none, still list any intersecting fid
    if not hits:
        for _, row in cand.iterrows():
            fid = int(row["ann_fid"]) if "ann_fid" in row.index and row["ann_fid"] is not None else None
            if fid is not None:
                hits.append({"fid": fid, "iou": round(_iou(geom, row.geometry), 4)})

    hits = sorted(hits, key=lambda h: h["iou"], reverse=True)
    return {
        "intersects": intersects,
        "fids": [h["fid"] for h in hits],
        "fid_ious": hits,
        "best_iou": round(best_iou, 4),
        "best_fid": best_fid,
    }


def match_key(before_id, after_id) -> str:
    return f"b{int(before_id)}_a{int(after_id)}"


def geom_record(geom, crs: str = "EPSG:25829") -> dict | None:
    if geom is None or geom.is_empty:
        return None
    c = geom.centroid
    minx, miny, maxx, maxy = geom.bounds
    return {
        "centroid_x": float(c.x),
        "centroid_y": float(c.y),
        "bbox": [float(minx), float(miny), float(maxx), float(maxy)],
        "area_m2": float(geom.area),
        "wkt": geom.wkt,
        "geojson": mapping(geom),
        "crs": crs,
    }


def qgis_extent_string(bounds, pad_m: float = 5.0) -> str:
    minx, miny, maxx, maxy = bounds
    return (
        f"{minx - pad_m:.3f},{miny - pad_m:.3f},"
        f"{maxx + pad_m:.3f},{maxy + pad_m:.3f} [EPSG:25829]"
    )


def build_pair_record(
    match_row,
    before_geom,
    after_geom,
    ann24: gpd.GeoDataFrame | None,
    ann25: gpd.GeoDataFrame | None,
    label: str | None,
    note: str = "",
    min_iou: float = 0.05,
) -> dict:
    before_id = int(match_row["before_id"])
    after_id = int(match_row["after_id"])
    key = match_key(before_id, after_id)

    before_rec = geom_record(before_geom)
    after_rec = geom_record(after_geom)

    geoms = [g for g in (before_geom, after_geom) if g is not None]
    if geoms:
        union = unary_union(geoms)
        bbox = list(union.bounds)
    else:
        bbox = [np.nan, np.nan, np.nan, np.nan]

    hit24 = (
        annotation_hits(before_geom, ann24, min_iou=min_iou)
        if ann24 is not None
        else {
            "intersects": None,
            "fids": [],
            "best_iou": None,
            "best_fid": None,
            "fid_ious": [],
        }
    )
    hit25 = (
        annotation_hits(after_geom, ann25, min_iou=min_iou)
        if ann25 is not None
        else {
            "intersects": None,
            "fids": [],
            "best_iou": None,
            "best_fid": None,
            "fid_ious": [],
        }
    )

    return {
        "label_id": key,
        "before_id": before_id,
        "after_id": after_id,
        "label": label,
        "note": note,
        "labeled_at": _utc_now() if label else None,
        "match_score": float(match_row.get("match_score", np.nan)),
        "distance_m": float(match_row.get("distance_m", np.nan)),
        "dx": float(match_row.get("dx", np.nan)) if match_row.get("dx") is not None else None,
        "dy": float(match_row.get("dy", np.nan)) if match_row.get("dy") is not None else None,
        "before_area": float(match_row.get("before_area", np.nan))
        if match_row.get("before_area") is not None
        else None,
        "after_area": float(match_row.get("after_area", np.nan))
        if match_row.get("after_area") is not None
        else None,
        "before_volume": _maybe_float(match_row.get("before_volume")),
        "after_volume": _maybe_float(match_row.get("after_volume")),
        "before": before_rec,
        "after": after_rec,
        "pair_bbox": bbox,
        "qgis_extent": qgis_extent_string(bbox) if geoms else None,
        "manual_ann_24": {
            "intersects": hit24["intersects"],
            "fids": hit24.get("fids", []),
            "best_fid": hit24.get("best_fid"),
            "best_iou": hit24.get("best_iou"),
            "note": (
                "fid values come from GeoPackage and stay stable when appending "
                "features; they can change if features are deleted/recreated."
            ),
        },
        "manual_ann_25": {
            "intersects": hit25["intersects"],
            "fids": hit25.get("fids", []),
            "best_fid": hit25.get("best_fid"),
            "best_iou": hit25.get("best_iou"),
        },
    }


def _maybe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def load_labels_db(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text())
        if "labels" not in data:
            data["labels"] = {}
        # Support list or dict form
        if isinstance(data["labels"], list):
            data["labels"] = {r["label_id"]: r for r in data["labels"] if "label_id" in r}
        return data
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "description": (
            "Human labels for boulder matcher evaluation. "
            "confirmed_match / not_match / unsure."
        ),
        "labels": {},
    }


def save_labels_db(db: dict, path: Path, also_geojson: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = deepcopy(db)
    db["updated_at"] = _utc_now()
    # Persist labels as a sorted list for readability / diffs
    labels_dict = db.get("labels", {})
    if isinstance(labels_dict, dict):
        db["labels"] = [labels_dict[k] for k in sorted(labels_dict.keys())]
        db["n_labels"] = len(db["labels"])
        db["counts"] = {
            LABEL_CONFIRMED: sum(1 for r in db["labels"] if r.get("label") == LABEL_CONFIRMED),
            LABEL_NOT: sum(1 for r in db["labels"] if r.get("label") == LABEL_NOT),
            LABEL_UNSURE: sum(1 for r in db["labels"] if r.get("label") == LABEL_UNSURE),
        }

    path.write_text(json.dumps(db, indent=2))
    print(f"Saved {path}  ({db.get('n_labels', 0)} labels  {db.get('counts')})")

    if also_geojson:
        _export_qgis_geojson(db, path.with_suffix(".geojson"))


def _export_qgis_geojson(db: dict, path: Path) -> None:
    """Write labeled pairs as after-centroid points + properties for QGIS."""
    features = []
    labels = db.get("labels", [])
    if isinstance(labels, dict):
        labels = list(labels.values())
    for rec in labels:
        after = rec.get("after") or {}
        before = rec.get("before") or {}
        cx = after.get("centroid_x")
        cy = after.get("centroid_y")
        if cx is None or cy is None:
            cx = before.get("centroid_x")
            cy = before.get("centroid_y")
        if cx is None or cy is None:
            continue
        props = {
            "label_id": rec.get("label_id"),
            "label": rec.get("label"),
            "before_id": rec.get("before_id"),
            "after_id": rec.get("after_id"),
            "match_score": rec.get("match_score"),
            "distance_m": rec.get("distance_m"),
            "before_cx": before.get("centroid_x"),
            "before_cy": before.get("centroid_y"),
            "after_cx": after.get("centroid_x"),
            "after_cy": after.get("centroid_y"),
            "qgis_extent": rec.get("qgis_extent"),
            "intersects_manual_24": (rec.get("manual_ann_24") or {}).get("intersects"),
            "intersects_manual_25": (rec.get("manual_ann_25") or {}).get("intersects"),
            "manual_24_fids": ",".join(
                str(f) for f in (rec.get("manual_ann_24") or {}).get("fids", [])
            ),
            "manual_25_fids": ",".join(
                str(f) for f in (rec.get("manual_ann_25") or {}).get("fids", [])
            ),
            "labeled_at": rec.get("labeled_at"),
        }
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [cx, cy]},
            }
        )

    fc = {
        "type": "FeatureCollection",
        "name": "match_eval_labels",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::25829"}},
        "features": features,
    }
    path.write_text(json.dumps(fc, indent=2))
    print(f"Wrote QGIS layer {path}")


def _ensure_ids(gdf: gpd.GeoDataFrame | None, col: str) -> gpd.GeoDataFrame | None:
    if gdf is None:
        return None
    if col not in gdf.columns:
        gdf = gdf.copy()
        gdf[col] = gdf.index
    return gdf


def run_eval_gui(
    results: dict[str, gpd.GeoDataFrame],
    before: gpd.GeoDataFrame | None,
    after: gpd.GeoDataFrame | None,
    labels_path: Path,
    ann24: gpd.GeoDataFrame | None = None,
    ann25: gpd.GeoDataFrame | None = None,
    before_raster: Path | None = None,
    after_raster: Path | None = None,
    pair_tiles: list[tuple[str, str]] | None = None,
    pad_m: float = 8.0,
    overview_pad_m: float = 25.0,
    meta: dict | None = None,
    min_iou: float = 0.05,
):
    matches = results["matches"].sort_values("match_score", ascending=False).reset_index(drop=True)
    if matches.empty:
        raise SystemExit("No matches to evaluate.")

    before = _ensure_ids(before, "before_id")
    after = _ensure_ids(after, "after_id")

    db = load_labels_db(labels_path)
    if meta:
        db.update({k: v for k, v in meta.items() if v is not None})
    labels: dict = db.setdefault("labels", {})
    if isinstance(labels, list):
        labels = {r["label_id"]: r for r in labels}
        db["labels"] = labels

    # Precompute annotation hits for status display speed
    cache: dict[str, dict] = {}

    def _record_for(row, label=None, note=""):
        before_geom, after_geom = _match_pair_geoms(row, before, after)
        return build_pair_record(
            row,
            before_geom,
            after_geom,
            ann24,
            ann25,
            label=label,
            note=note,
            min_iou=min_iou,
        )

    for _, row in matches.iterrows():
        key = match_key(row["before_id"], row["after_id"])
        if key not in cache:
            # Keep existing label if present
            existing = labels.get(key)
            lab = existing.get("label") if existing else None
            note = existing.get("note", "") if existing else ""
            cache[key] = _record_for(row, label=lab, note=note)
            if existing and existing.get("label"):
                # Preserve original labeled_at if already labeled
                if existing.get("labeled_at"):
                    cache[key]["labeled_at"] = existing["labeled_at"]
                labels[key] = cache[key]

    state = {"idx": 0, "overview_zoomed": True, "dirty": False}

    fig = plt.figure(figsize=(17, 8))
    ax_overview = fig.add_subplot(1, 3, 1)
    ax_before = fig.add_subplot(1, 3, 2)
    ax_after = fig.add_subplot(1, 3, 3)
    detail_axes = [ax_before, ax_after]
    fig.canvas.manager.set_window_title("Matcher evaluation — label matches")

    status_ax = fig.add_axes([0.02, 0.01, 0.96, 0.07])
    status_ax.set_axis_off()
    status_text = status_ax.text(
        0.0,
        0.5,
        "",
        va="center",
        ha="left",
        family="monospace",
        fontsize=9,
        wrap=True,
        transform=status_ax.transAxes,
    )

    def _full_extent():
        geoms = []
        for key in ("matches", "appeared", "disappeared", "vectors"):
            gdf = results.get(key)
            if gdf is not None and not gdf.empty:
                geoms.extend(list(gdf.geometry))
        if not geoms:
            return None
        minx = min(g.bounds[0] for g in geoms)
        miny = min(g.bounds[1] for g in geoms)
        maxx = max(g.bounds[2] for g in geoms)
        maxy = max(g.bounds[3] for g in geoms)
        pad = max(5.0, 0.05 * max(maxx - minx, maxy - miny))
        return (minx - pad, miny - pad, maxx + pad, maxy + pad)

    full_extent = _full_extent()

    def _current_row():
        return matches.iloc[state["idx"]]

    def _current_key():
        row = _current_row()
        return match_key(row["before_id"], row["after_id"])

    def _status_lines(row, rec) -> str:
        lab = rec.get("label") or "UNLABELED"
        n_done = sum(1 for r in labels.values() if r.get("label") in VALID_LABELS)
        m24 = rec.get("manual_ann_24") or {}
        m25 = rec.get("manual_ann_25") or {}
        b = rec.get("before") or {}
        a = rec.get("after") or {}
        f24 = ",".join(str(f) for f in m24.get("fids", [])[:6]) or "-"
        f25 = ",".join(str(f) for f in m25.get("fids", [])[:6]) or "-"
        return (
            f"[{state['idx']+1}/{len(matches)}]  {rec['label_id']}  LABEL={lab}  "
            f"scored={n_done}/{len(matches)}  score={rec.get('match_score', float('nan')):.3f}  "
            f"dist={rec.get('distance_m', float('nan')):.2f}m\n"
            f"before centroid: {b.get('centroid_x', float('nan')):.3f}, "
            f"{b.get('centroid_y', float('nan')):.3f}   "
            f"after: {a.get('centroid_x', float('nan')):.3f}, "
            f"{a.get('centroid_y', float('nan')):.3f}\n"
            f"QGIS extent: {rec.get('qgis_extent')}\n"
            f"manual24 intersect={m24.get('intersects')} fids=[{f24}] iou={m24.get('best_iou')}   "
            f"manual25 intersect={m25.get('intersects')} fids=[{f25}] iou={m25.get('best_iou')}\n"
            f"keys: y=confirm  x=not-match  ?=unsure  Bksp=clear  j=next unlabeled  "
            f"c=print coords  s=save  n/p=nav  o=zoom  q=quit"
        )

    def _apply_overview_zoom(row):
        if state["overview_zoomed"]:
            before_geom, after_geom = _match_pair_geoms(row, before, after)
            geoms = [g for g in (before_geom, after_geom, row.geometry) if g is not None]
            if geoms:
                minx = min(g.bounds[0] for g in geoms) - overview_pad_m
                miny = min(g.bounds[1] for g in geoms) - overview_pad_m
                maxx = max(g.bounds[2] for g in geoms) + overview_pad_m
                maxy = max(g.bounds[3] for g in geoms) + overview_pad_m
                ax_overview.set_xlim(minx, maxx)
                ax_overview.set_ylim(miny, maxy)
                return
        if full_extent is not None:
            ax_overview.set_xlim(full_extent[0], full_extent[2])
            ax_overview.set_ylim(full_extent[1], full_extent[3])

    def redraw():
        row = _current_row()
        key = _current_key()
        rec = labels.get(key) or cache.get(key) or _record_for(row)
        cache[key] = rec

        ax_overview.cla()
        lab = rec.get("label")
        title = f"Overview  label={lab or 'UNLABELED'}"
        plot_overview(results, ax=ax_overview, title=title)
        before_geom, after_geom = _match_pair_geoms(row, before, after)
        edge = LABEL_COLORS.get(lab, "yellow")
        if before_geom is not None:
            gpd.GeoSeries([before_geom], crs="EPSG:25829").plot(
                ax=ax_overview, facecolor="none", edgecolor=edge, linewidth=2.5, zorder=6
            )
        if after_geom is not None:
            gpd.GeoSeries([after_geom], crs="EPSG:25829").plot(
                ax=ax_overview, facecolor="none", edgecolor="cyan", linewidth=2.5, zorder=6
            )
        if before_geom is not None and after_geom is not None:
            _draw_displacement_vector(ax_overview, before_geom, after_geom, color=edge)
        _apply_overview_zoom(row)

        for a in detail_axes:
            a.cla()
        plot_match_detail(
            row,
            before=before,
            after=after,
            before_raster=before_raster,
            after_raster=after_raster,
            pad_m=pad_m,
            side_by_side=True,
            pair_tiles=pair_tiles,
            axes=detail_axes,
            draw_vector=True,
        )
        # Overlay manual annotation outlines (thin) if present in view
        bounds = None
        if before_geom is not None or after_geom is not None:
            from .visualize import _match_bounds

            bounds = _match_bounds(before_geom, after_geom, pad_m)
        if bounds is not None:
            box = shape(
                {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [bounds[0], bounds[1]],
                            [bounds[2], bounds[1]],
                            [bounds[2], bounds[3]],
                            [bounds[0], bounds[3]],
                            [bounds[0], bounds[1]],
                        ]
                    ],
                }
            )
            if ann24 is not None and not ann24.empty:
                near24 = ann24[ann24.intersects(box)]
                if not near24.empty:
                    near24.plot(
                        ax=detail_axes[0],
                        facecolor="none",
                        edgecolor="#9b59b6",
                        linewidth=1.0,
                        alpha=0.8,
                        zorder=4,
                    )
            if ann25 is not None and not ann25.empty:
                near25 = ann25[ann25.intersects(box)]
                if not near25.empty:
                    near25.plot(
                        ax=detail_axes[1],
                        facecolor="none",
                        edgecolor="#9b59b6",
                        linewidth=1.0,
                        alpha=0.8,
                        zorder=4,
                    )

        detail_axes[1].set_xlabel(
            f"Match {state['idx'] + 1}/{len(matches)}  |  y confirm · x reject · ? unsure"
        )
        status_text.set_text(_status_lines(row, rec))
        if lab in LABEL_COLORS:
            status_text.set_color(LABEL_COLORS[lab])
        else:
            status_text.set_color("0.15")
        fig.canvas.draw_idle()

    def set_label(label: str | None):
        row = _current_row()
        key = _current_key()
        if label is None:
            if key in labels:
                # Keep geometry cache but clear label
                rec = cache.get(key) or _record_for(row)
                rec = dict(rec)
                rec["label"] = None
                rec["labeled_at"] = None
                cache[key] = rec
                labels.pop(key, None)
            state["dirty"] = True
            redraw()
            return

        note = (labels.get(key) or {}).get("note", "")
        rec = _record_for(row, label=label, note=note)
        cache[key] = rec
        labels[key] = rec
        state["dirty"] = True
        redraw()
        # Auto-advance to next unlabeled after labeling
        _goto_next_unlabeled(prefer_advance=True)

    def _goto_next_unlabeled(prefer_advance: bool = False):
        start = state["idx"]
        n = len(matches)
        begin = (start + 1) % n if prefer_advance else start
        for k in range(n):
            i = (begin + k) % n
            row = matches.iloc[i]
            key = match_key(row["before_id"], row["after_id"])
            lab = (labels.get(key) or {}).get("label")
            if lab not in VALID_LABELS:
                state["idx"] = i
                redraw()
                return
        print("All matches labeled.")
        redraw()

    def do_save():
        db["labels"] = labels
        save_labels_db(db, labels_path, also_geojson=True)
        state["dirty"] = False

    def print_coords():
        key = _current_key()
        rec = labels.get(key) or cache.get(key)
        if not rec:
            return
        b = rec.get("before") or {}
        a = rec.get("after") or {}
        print("\n—— QGIS lookup ——")
        print(f"label_id: {rec['label_id']}  label={rec.get('label')}")
        print(f"extent:   {rec.get('qgis_extent')}")
        print(
            f"before:   {b.get('centroid_x'):.3f}, {b.get('centroid_y'):.3f}  "
            f"(EPSG:25829)"
        )
        print(
            f"after:    {a.get('centroid_x'):.3f}, {a.get('centroid_y'):.3f}  "
            f"(EPSG:25829)"
        )
        print(f"manual24: intersects={rec['manual_ann_24'].get('intersects')} "
              f"fids={rec['manual_ann_24'].get('fids')}")
        print(f"manual25: intersects={rec['manual_ann_25'].get('intersects')} "
              f"fids={rec['manual_ann_25'].get('fids')}")
        print("Paste extent into QGIS: View → Set Extent… (or paste into Python console)\n")

    def on_key(event):
        if event.key in ("n", "right"):
            state["idx"] = (state["idx"] + 1) % len(matches)
            redraw()
        elif event.key in ("p", "left"):
            state["idx"] = (state["idx"] - 1) % len(matches)
            redraw()
        elif event.key in ("y", "Y"):
            set_label(LABEL_CONFIRMED)
        elif event.key in ("x", "X"):
            set_label(LABEL_NOT)
        elif event.key in ("?", "u", "U"):
            set_label(LABEL_UNSURE)
        elif event.key in ("backspace", "delete"):
            set_label(None)
        elif event.key in ("o", "O"):
            state["overview_zoomed"] = not state["overview_zoomed"]
            redraw()
        elif event.key in ("j", "J"):
            _goto_next_unlabeled(prefer_advance=True)
        elif event.key in ("c", "C"):
            print_coords()
        elif event.key in ("s", "S", "ctrl+s"):
            do_save()
        elif event.key in ("q", "Q"):
            do_save()
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    # Start at first unlabeled if any
    _goto_next_unlabeled(prefer_advance=False)
    print(
        f"Evaluating {len(matches)} matches → {labels_path}\n"
        "y=confirm match  x=not a match  ?=unsure  j=next unlabeled  "
        "c=print coords  s=save  q=save+quit"
    )
    plt.show()
    if state["dirty"]:
        do_save()


def main():
    root = _project_root()
    default_outdir = root / "segmentation" / "training_run_rgb_dsm_4000" / "matching"
    default_ann24 = root / "segmentation" / "annotations" / "july14_24.gpkg"
    default_ann25 = root / "segmentation" / "annotations" / "july14_25.gpkg"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=default_outdir,
        help="Matching run directory (contains results/ + predictions/)",
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=None,
        help="Output JSON database (default: <outdir>/eval/match_labels.json)",
    )
    parser.add_argument("--ann-24", type=Path, default=default_ann24)
    parser.add_argument("--ann-25", type=Path, default=default_ann25)
    parser.add_argument("--no-ann", action="store_true", help="Skip manual annotation intersect checks")
    parser.add_argument("--pad-m", type=float, default=8.0)
    parser.add_argument("--min-iou", type=float, default=0.05, help="Min IoU to list an ann fid")
    args = parser.parse_args()

    outdir = args.outdir
    results_dir = outdir / "results"
    summary_path = outdir / "match_summary.json"
    before_path = outdir / "predictions" / "before_inferred_boulders.geojson"
    after_path = outdir / "predictions" / "after_inferred_boulders.geojson"
    labels_path = args.labels_json or (outdir / "eval" / "match_labels.json")

    if not (results_dir / "matched_boulders.geojson").exists():
        raise SystemExit(f"No matched_boulders.geojson under {results_dir}")

    results = load_results(results_dir)
    before, after = load_inputs(before_path, after_path)

    pair_tiles = None
    before_raster = after_raster = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        tiles = summary.get("tiles") or []
        if tiles:
            pair_tiles = [(t["tile_24"], t["tile_25"]) for t in tiles]
            before_raster = Path(tiles[0]["tile_24"])
            after_raster = Path(tiles[0]["tile_25"])

    ann24 = ann25 = None
    if not args.no_ann:
        if args.ann_24.exists():
            print(f"Loading manual 2024 annotations: {args.ann_24}")
            ann24 = load_manual_annotations(args.ann_24)
            print(f"  {len(ann24)} polygons")
        else:
            print(f"Warning: missing {args.ann_24}")
        if args.ann_25.exists():
            print(f"Loading manual 2025 annotations: {args.ann_25}")
            ann25 = load_manual_annotations(args.ann_25)
            print(f"  {len(ann25)} polygons")
        else:
            print(f"Warning: missing {args.ann_25}")

    meta = {
        "source_matching_outdir": str(outdir.resolve()),
        "manual_ann_24": str(args.ann_24.resolve()) if args.ann_24 else None,
        "manual_ann_25": str(args.ann_25.resolve()) if args.ann_25 else None,
        "crs": "EPSG:25829",
        "min_iou_for_fid": args.min_iou,
    }

    run_eval_gui(
        results=results,
        before=before,
        after=after,
        labels_path=labels_path,
        ann24=ann24,
        ann25=ann25,
        before_raster=before_raster,
        after_raster=after_raster,
        pair_tiles=pair_tiles,
        pad_m=args.pad_m,
        meta=meta,
        min_iou=args.min_iou,
    )


if __name__ == "__main__":
    main()
