"""Interactive labeling UI for building a matcher evaluation dataset.

Flip through inferred matches and missed-match candidates (same layout as
``visualize.run_gui``), label each as confirmed / not-a-match / unsure, tag
boulder/deposit/isolated flags, and append records to a JSON file.

Each record stores centroids, bbox, WKT polygons, matcher score, volumes,
``source_queue`` (``matcher`` vs ``missed_candidate``), and whether the
detections intersect the manual annotation GPKGs (july14_24 / july14_25),
including GeoPackage ``fid`` values when available (stable under appends).

Example:
  python -m matching.evaluate_matches \\
    --outdir ../../segmentation/training_run_rgb_dsm_4000/matching \\
    --labels-json ../../segmentation/training_run_rgb_dsm_4000/matching/eval/match_labels.json

Keys:
  y         confirm match
  x         not match
  ?         unsure
  b         boulder
  z         not boulder
  d         deposit
  i         isolated
  m         cycle queue mode: matcher matches ↔ missed candidates (appeared↔disappeared)
  j         next unlabeled
  c         print coords / volumes
  s         save
  q         quit
  n / p     next / previous
  o         toggle overview zoom
  Shift+b   clear boulder flag
  Shift+d   clear deposit flag
  backspace clear match label only (keeps boulder/deposit/isolated flags)
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
    OrthoPreview,
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

BOULDER_YES = "boulder"
BOULDER_NO = "not_boulder"
VALID_BOULDER_FLAGS = {BOULDER_YES, BOULDER_NO}

SOURCE_MATCHER = "matcher"
SOURCE_MISSED = "missed_candidate"

LABEL_COLORS = {
    LABEL_CONFIRMED: "#2ecc71",
    LABEL_NOT: "#e74c3c",
    LABEL_UNSURE: "#f1c40f",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _fmt_vol(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(f):
        return "—"
    return f"{f:.3f} m³"


def load_manual_annotations(path: Path, target_crs: str = "EPSG:25829") -> gpd.GeoDataFrame:
    """Load a GPKG/GeoJSON, keep GeoPackage ``fid`` when present, reproject."""
    if not path.exists():
        raise FileNotFoundError(path)

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

    intersects = True
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


def label_id_for(before_id, after_id, source_queue: str = SOURCE_MATCHER) -> str:
    base = match_key(before_id, after_id)
    if source_queue == SOURCE_MISSED:
        return f"missed_{base}"
    return base


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


def _row_get(row, key, default=None):
    """Safe field access for pandas Series / mapping rows."""
    try:
        if hasattr(row, "index") and key in row.index:
            return row[key]
    except Exception:  # noqa: BLE001
        pass
    try:
        return row.get(key, default)
    except Exception:  # noqa: BLE001
        return default


def build_pair_record(
    match_row,
    before_geom,
    after_geom,
    ann24: gpd.GeoDataFrame | None,
    ann25: gpd.GeoDataFrame | None,
    label: str | None,
    note: str = "",
    min_iou: float = 0.05,
    source_queue: str = SOURCE_MATCHER,
    boulder_flag: str | None = None,
    deposit_flag: bool | None = None,
    isolated: bool | None = None,
) -> dict:
    before_id = int(match_row["before_id"])
    after_id = int(match_row["after_id"])
    lid = label_id_for(before_id, after_id, source_queue)

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
        "label_id": lid,
        "before_id": before_id,
        "after_id": after_id,
        "label": label,
        "boulder_flag": boulder_flag,
        "deposit_flag": deposit_flag,
        "isolated": isolated,
        "source_queue": source_queue,
        "note": note,
        "labeled_at": _utc_now() if label else None,
        "match_score": float(_row_get(match_row, "match_score", np.nan)),
        "distance_m": float(_row_get(match_row, "distance_m", np.nan)),
        "dx": _maybe_float(_row_get(match_row, "dx")),
        "dy": _maybe_float(_row_get(match_row, "dy")),
        "before_area": _maybe_float(_row_get(match_row, "before_area")),
        "after_area": _maybe_float(_row_get(match_row, "after_area")),
        "before_volume": _maybe_float(_row_get(match_row, "before_volume")),
        "after_volume": _maybe_float(_row_get(match_row, "after_volume")),
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


def load_labels_db(path: Path) -> dict:
    if path.exists():
        data = json.loads(path.read_text())
        if "labels" not in data:
            data["labels"] = {}
        if isinstance(data["labels"], list):
            data["labels"] = {r["label_id"]: r for r in data["labels"] if "label_id" in r}
        return data
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "description": (
            "Human labels for boulder matcher evaluation. "
            "confirmed_match / not_match / unsure; boulder / not_boulder / deposit / isolated."
        ),
        "labels": {},
    }


def _label_counts(labels: list[dict]) -> dict:
    return {
        LABEL_CONFIRMED: sum(1 for r in labels if r.get("label") == LABEL_CONFIRMED),
        LABEL_NOT: sum(1 for r in labels if r.get("label") == LABEL_NOT),
        LABEL_UNSURE: sum(1 for r in labels if r.get("label") == LABEL_UNSURE),
        BOULDER_YES: sum(1 for r in labels if r.get("boulder_flag") == BOULDER_YES),
        BOULDER_NO: sum(1 for r in labels if r.get("boulder_flag") == BOULDER_NO),
        "in_deposit": sum(1 for r in labels if r.get("deposit_flag") is True),
        "isolated": sum(1 for r in labels if r.get("isolated") is True),
    }


def save_labels_db(db: dict, path: Path, also_geojson: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = deepcopy(db)
    db["updated_at"] = _utc_now()
    labels_dict = db.get("labels", {})
    if isinstance(labels_dict, dict):
        db["labels"] = [labels_dict[k] for k in sorted(labels_dict.keys())]
        db["n_labels"] = len(db["labels"])
        db["counts"] = _label_counts(db["labels"])

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
            "boulder_flag": rec.get("boulder_flag"),
            "deposit_flag": rec.get("deposit_flag"),
            "isolated": rec.get("isolated"),
            "source_queue": rec.get("source_queue"),
            "before_id": rec.get("before_id"),
            "after_id": rec.get("after_id"),
            "match_score": rec.get("match_score"),
            "distance_m": rec.get("distance_m"),
            "before_volume": rec.get("before_volume"),
            "after_volume": rec.get("after_volume"),
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


def _volumes_mostly_missing(gdf: gpd.GeoDataFrame | None, cols: tuple[str, ...]) -> bool:
    """True when every listed volume column is absent or all-null/non-finite."""
    if gdf is None or gdf.empty:
        return False
    saw = False
    for col in cols:
        if col not in gdf.columns:
            continue
        saw = True
        vals = gdf[col]
        finite = vals.notna() & np.isfinite(vals.astype(float))
        if bool(finite.any()):
            return False
    return saw or True


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf is None or gdf.empty:
        gpd.GeoDataFrame({"geometry": []}, crs="EPSG:25829").to_file(path, driver="GeoJSON")
    else:
        gdf.to_file(path, driver="GeoJSON")


def enrich_pair_volumes(
    before: gpd.GeoDataFrame | None,
    after: gpd.GeoDataFrame | None,
    matches: gpd.GeoDataFrame,
    missed: gpd.GeoDataFrame,
    before_dsm: Path,
    after_dsm: Path,
    *,
    appeared: gpd.GeoDataFrame | None = None,
    disappeared: gpd.GeoDataFrame | None = None,
    persist_dir: Path | None = None,
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Compute DSM volumes on before/after polygons and map onto match rows.

    Manual july14 datasets were often built without ``--compute-volume``, so
    ``before_volume`` / ``after_volume`` are all null. This fills them in-place
    for the eval UI and optionally rewrites prediction/result GeoJSONs.
    """
    from .attributes import estimate_volume_from_dsm_progress

    if before is None or after is None or before.empty or after.empty:
        print("Warning: cannot enrich volumes — missing before/after polygons")
        return before, after, matches, missed

    before = before.copy().reset_index(drop=True)
    after = after.copy().reset_index(drop=True)
    before["before_id"] = before.index.astype(int)
    after["after_id"] = after.index.astype(int)

    need_b: set[int] = set()
    need_a: set[int] = set()
    for gdf in (matches, missed):
        if gdf is None or gdf.empty:
            continue
        need_b.update(int(x) for x in gdf["before_id"].tolist())
        need_a.update(int(x) for x in gdf["after_id"].tolist())

    if not need_b and not need_a:
        print("No match/missed IDs to volume-enrich")
        return before, after, matches, missed

    before_sub = before[before["before_id"].isin(need_b)].copy()
    after_sub = after[after["after_id"].isin(need_a)].copy()
    print(
        f"Volume fill for {len(before_sub)} before + {len(after_sub)} after "
        f"polygons referenced by matches/missed "
        f"(of {len(before)}/{len(after)} total)"
    )

    before_sub = estimate_volume_from_dsm_progress(before_sub, before_dsm, label="2024")
    after_sub = estimate_volume_from_dsm_progress(after_sub, after_dsm, label="2025")

    if "volume" not in before.columns:
        before["volume"] = np.nan
    if "volume" not in after.columns:
        after["volume"] = np.nan
    before.loc[before_sub.index, "volume"] = before_sub["volume"].to_numpy()
    after.loc[after_sub.index, "volume"] = after_sub["volume"].to_numpy()
    # height diagnostics when present
    for col in ("mean_height", "max_height"):
        if col in before_sub.columns:
            if col not in before.columns:
                before[col] = np.nan
            before.loc[before_sub.index, col] = before_sub[col].to_numpy()
        if col in after_sub.columns:
            if col not in after.columns:
                after[col] = np.nan
            after.loc[after_sub.index, col] = after_sub[col].to_numpy()

    vol_b = before.set_index("before_id")["volume"]
    vol_a = after.set_index("after_id")["volume"]

    def _map_pair_vols(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf is None or gdf.empty:
            return gdf
        out = gdf.copy()
        out["before_volume"] = out["before_id"].map(vol_b)
        out["after_volume"] = out["after_id"].map(vol_a)
        return out

    matches = _map_pair_vols(matches)
    missed = _map_pair_vols(missed)

    if appeared is not None and not appeared.empty and "after_id" in appeared.columns:
        appeared = appeared.copy()
        appeared["volume"] = appeared["after_id"].map(vol_a)
    if disappeared is not None and not disappeared.empty and "before_id" in disappeared.columns:
        disappeared = disappeared.copy()
        disappeared["volume"] = disappeared["before_id"].map(vol_b)

    if persist_dir is not None:
        pred = persist_dir / "predictions"
        res = persist_dir / "results"
        _write_geojson(before, pred / "before_inferred_boulders.geojson")
        _write_geojson(after, pred / "after_inferred_boulders.geojson")
        _write_geojson(matches, res / "matched_boulders.geojson")
        if missed is not None and not missed.empty:
            from .candidates import write_missed_candidates

            write_missed_candidates(missed, res / "missed_candidates.geojson")
        if appeared is not None:
            _write_geojson(appeared, res / "appeared_boulders.geojson")
        if disappeared is not None:
            _write_geojson(disappeared, res / "disappeared_boulders.geojson")
        print(f"Persisted DSM volumes under {persist_dir}")

    return before, after, matches, missed


def _ensure_ids(gdf: gpd.GeoDataFrame | None, col: str) -> gpd.GeoDataFrame | None:
    if gdf is None:
        return None
    if col not in gdf.columns:
        gdf = gdf.copy()
        gdf[col] = gdf.index
    return gdf


def _record_has_data(rec: dict) -> bool:
    if rec.get("label") in VALID_LABELS:
        return True
    if rec.get("boulder_flag") in VALID_BOULDER_FLAGS:
        return True
    if rec.get("deposit_flag") is True:
        return True
    if rec.get("isolated") is True:
        return True
    return False


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
    missed_candidates: gpd.GeoDataFrame | None = None,
    pad_m: float = 12.0,
    overview_pad_m: float = 25.0,
    meta: dict | None = None,
    min_iou: float = 0.05,
    ortho: OrthoPreview | None = None,
    raster_pad_m: float = 40.0,
):
    matches = results["matches"].sort_values("match_score", ascending=False).reset_index(drop=True)
    missed = (
        missed_candidates.sort_values("match_score", ascending=False).reset_index(drop=True)
        if missed_candidates is not None and not missed_candidates.empty
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:25829")
    )

    if matches.empty and missed.empty:
        raise SystemExit("No matcher matches or missed candidates to evaluate.")

    before = _ensure_ids(before, "before_id")
    after = _ensure_ids(after, "after_id")

    db = load_labels_db(labels_path)
    if meta:
        db.update({k: v for k, v in meta.items() if v is not None})
    labels: dict = db.setdefault("labels", {})
    if isinstance(labels, list):
        labels = {r["label_id"]: r for r in labels}
        db["labels"] = labels

    cache: dict[str, dict] = {}

    def _record_for(
        row,
        label=None,
        note="",
        source_queue=SOURCE_MATCHER,
        boulder_flag=None,
        deposit_flag=None,
        isolated=None,
    ):
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
            source_queue=source_queue,
            boulder_flag=boulder_flag,
            deposit_flag=deposit_flag,
            isolated=isolated,
        )

    def _merge_existing(row, source_queue: str, existing: dict | None) -> dict:
        lab = existing.get("label") if existing else None
        note = existing.get("note", "") if existing else ""
        boulder_flag = existing.get("boulder_flag") if existing else None
        deposit_flag = existing.get("deposit_flag") if existing else None
        isolated = existing.get("isolated") if existing else None
        rec = _record_for(
            row,
            label=lab,
            note=note,
            source_queue=source_queue,
            boulder_flag=boulder_flag,
            deposit_flag=deposit_flag,
            isolated=isolated,
        )
        if existing:
            if existing.get("labeled_at") and lab:
                rec["labeled_at"] = existing["labeled_at"]
            if existing.get("source_queue"):
                rec["source_queue"] = existing["source_queue"]
        return rec

    for _, row in matches.iterrows():
        lid = label_id_for(row["before_id"], row["after_id"], SOURCE_MATCHER)
        existing = labels.get(lid)
        rec = _merge_existing(row, SOURCE_MATCHER, existing)
        cache[lid] = rec
        if _record_has_data(rec):
            labels[lid] = rec

    for _, row in missed.iterrows():
        lid = label_id_for(row["before_id"], row["after_id"], SOURCE_MISSED)
        existing = labels.get(lid)
        rec = _merge_existing(row, SOURCE_MISSED, existing)
        cache[lid] = rec
        if _record_has_data(rec):
            labels[lid] = rec

    initial_mode = "matches" if not matches.empty else "missed"
    state = {
        "idx": 0,
        "mode": initial_mode,
        "overview_zoomed": True,
        "dirty": False,
    }

    fig = plt.figure(figsize=(17, 9))
    # Leave room for panel titles (volumes) at top and a multi-line status band.
    fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.18, wspace=0.18)
    ax_overview = fig.add_subplot(1, 3, 1)
    ax_before = fig.add_subplot(1, 3, 2)
    ax_after = fig.add_subplot(1, 3, 3)
    detail_axes = [ax_before, ax_after]
    fig.canvas.manager.set_window_title("Matcher evaluation — label matches")

    status_ax = fig.add_axes([0.02, 0.01, 0.96, 0.14])
    status_ax.set_axis_off()
    status_text = status_ax.text(
        0.0,
        1.0,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        wrap=True,
        transform=status_ax.transAxes,
    )

    def _source_queue() -> str:
        return SOURCE_MISSED if state["mode"] == "missed" else SOURCE_MATCHER

    def _current_queue() -> gpd.GeoDataFrame:
        return missed if state["mode"] == "missed" else matches

    def _queue_len() -> int:
        return len(_current_queue())

    def _full_extent():
        geoms = []
        for key in ("matches", "appeared", "disappeared", "vectors", "missed_candidates"):
            gdf = results.get(key)
            if gdf is not None and not gdf.empty:
                geoms.extend(list(gdf.geometry))
        if missed is not None and not missed.empty and "missed_candidates" not in results:
            geoms.extend(list(missed.geometry))
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
        return _current_queue().iloc[state["idx"]]

    def _current_label_id() -> str:
        row = _current_row()
        return label_id_for(row["before_id"], row["after_id"], _source_queue())

    def _flag_summary(rec: dict) -> str:
        parts = []
        bf = rec.get("boulder_flag")
        if bf == BOULDER_YES:
            parts.append("BOULDER")
        elif bf == BOULDER_NO:
            parts.append("NOT_BOULDER")
        if rec.get("deposit_flag") is True:
            parts.append("DEPOSIT")
        if rec.get("isolated") is True:
            parts.append("ISOLATED")
        return "  ".join(parts) if parts else "—"

    def _status_lines(row, rec) -> str:
        lab = rec.get("label") or "UNLABELED"
        n_done = sum(
            1
            for _, r in _current_queue().iterrows()
            if (labels.get(label_id_for(r["before_id"], r["after_id"], _source_queue())) or {}).get(
                "label"
            )
            in VALID_LABELS
        )
        m24 = rec.get("manual_ann_24") or {}
        m25 = rec.get("manual_ann_25") or {}
        b = rec.get("before") or {}
        a = rec.get("after") or {}
        f24 = ",".join(str(f) for f in m24.get("fids", [])[:6]) or "-"
        f25 = ",".join(str(f) for f in m25.get("fids", [])[:6]) or "-"
        mode_name = "MISSED CANDIDATES" if state["mode"] == "missed" else "MATCHER MATCHES"
        # Keep volumes on line 1 — status band is short and lower lines clip easily.
        return (
            f"MODE={mode_name}  [{state['idx']+1}/{_queue_len()}]  {rec['label_id']}  "
            f"LABEL={lab}  flags={_flag_summary(rec)}  "
            f"vol24={_fmt_vol(rec.get('before_volume'))}  "
            f"vol25={_fmt_vol(rec.get('after_volume'))}  "
            f"score={rec.get('match_score', float('nan')):.3f}  "
            f"dist={rec.get('distance_m', float('nan')):.2f}m  "
            f"scored={n_done}/{_queue_len()}\n"
            f"centroids 24: {b.get('centroid_x', float('nan')):.3f}, "
            f"{b.get('centroid_y', float('nan')):.3f}   "
            f"25: {a.get('centroid_x', float('nan')):.3f}, "
            f"{a.get('centroid_y', float('nan')):.3f}   "
            f"QGIS: {rec.get('qgis_extent')}\n"
            f"manual24 intersect={m24.get('intersects')} fids=[{f24}] iou={m24.get('best_iou')}   "
            f"manual25 intersect={m25.get('intersects')} fids=[{f25}] iou={m25.get('best_iou')}\n"
            f"keys: y=confirm  x=not-match  ?=unsure  b=boulder  z=not-boulder  d=deposit  "
            f"i=isolated  m=mode  Bksp=clear  j=next unlabeled  c=print  s=save  n/p  o  q"
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
        key = _current_label_id()
        rec = labels.get(key) or cache.get(key) or _record_for(row, source_queue=_source_queue())
        cache[key] = rec

        ax_overview.cla()
        lab = rec.get("label")
        mode_tag = "missed" if state["mode"] == "missed" else "match"
        title = f"Overview ({mode_tag})  label={lab or 'UNLABELED'}"
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
            ortho=ortho,
            raster_pad_m=raster_pad_m,
        )
        vol24 = _fmt_vol(rec.get("before_volume"))
        vol25 = _fmt_vol(rec.get("after_volume"))
        score = rec.get("match_score", float("nan"))
        dist = rec.get("distance_m", float("nan"))
        fig.suptitle(
            f"{rec.get('label_id', '')}   score={score:.3f}   dist={dist:.2f}m   "
            f"[{state['idx'] + 1}/{_queue_len()}]",
            fontsize=12,
            y=0.98,
        )
        detail_axes[0].set_title(f"2024 (before)  vol={vol24}", fontsize=11)
        detail_axes[1].set_title(f"2025 (after)  vol={vol25}", fontsize=11)
        # Also stamp volumes on the image so they stay visible if titles clip.
        for ax, vol in ((detail_axes[0], vol24), (detail_axes[1], vol25)):
            ax.text(
                0.02,
                0.98,
                f"V={vol}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
                zorder=20,
            )

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

        status_text.set_text(_status_lines(row, rec))
        if lab in LABEL_COLORS:
            status_text.set_color(LABEL_COLORS[lab])
        else:
            status_text.set_color("0.15")
        fig.canvas.draw_idle()

    def _existing_flags(key: str) -> tuple:
        existing = labels.get(key) or cache.get(key) or {}
        return (
            existing.get("boulder_flag"),
            existing.get("deposit_flag"),
            existing.get("isolated"),
            existing.get("note", ""),
            existing.get("labeled_at"),
        )

    def _store_record(rec: dict):
        key = rec["label_id"]
        cache[key] = rec
        if _record_has_data(rec):
            labels[key] = rec
        else:
            labels.pop(key, None)
        state["dirty"] = True

    def set_label(label: str | None):
        row = _current_row()
        key = _current_label_id()
        boulder_flag, deposit_flag, isolated, note, labeled_at = _existing_flags(key)

        if label is None:
            rec = _record_for(
                row,
                label=None,
                note=note,
                source_queue=_source_queue(),
                boulder_flag=boulder_flag,
                deposit_flag=deposit_flag,
                isolated=isolated,
            )
            _store_record(rec)
            redraw()
            return

        rec = _record_for(
            row,
            label=label,
            note=note,
            source_queue=_source_queue(),
            boulder_flag=boulder_flag,
            deposit_flag=deposit_flag,
            isolated=isolated,
        )
        if labeled_at and (labels.get(key) or {}).get("label") == label:
            rec["labeled_at"] = labeled_at
        _store_record(rec)
        redraw()
        _goto_next_unlabeled(prefer_advance=True)

    def _update_flags(
        boulder_flag=..., deposit_flag=..., isolated=...
    ):
        row = _current_row()
        key = _current_label_id()
        existing = labels.get(key) or cache.get(key) or {}
        lab = existing.get("label")
        note = existing.get("note", "")
        bf = existing.get("boulder_flag") if boulder_flag is ... else boulder_flag
        df = existing.get("deposit_flag") if deposit_flag is ... else deposit_flag
        iso = existing.get("isolated") if isolated is ... else isolated
        rec = _record_for(
            row,
            label=lab,
            note=note,
            source_queue=_source_queue(),
            boulder_flag=bf,
            deposit_flag=df,
            isolated=iso,
        )
        if existing.get("labeled_at"):
            rec["labeled_at"] = existing["labeled_at"]
        _store_record(rec)
        redraw()

    def _goto_next_unlabeled(prefer_advance: bool = False):
        queue = _current_queue()
        start = state["idx"]
        n = len(queue)
        if n == 0:
            return
        begin = (start + 1) % n if prefer_advance else start
        sq = _source_queue()
        for k in range(n):
            i = (begin + k) % n
            row = queue.iloc[i]
            lid = label_id_for(row["before_id"], row["after_id"], sq)
            lab = (labels.get(lid) or {}).get("label")
            if lab not in VALID_LABELS:
                state["idx"] = i
                redraw()
                return
        print(f"All items in {state['mode']} queue labeled.")
        redraw()

    def do_save():
        db["labels"] = labels
        save_labels_db(db, labels_path, also_geojson=True)
        state["dirty"] = False

    def print_coords():
        key = _current_label_id()
        rec = labels.get(key) or cache.get(key)
        if not rec:
            return
        b = rec.get("before") or {}
        a = rec.get("after") or {}
        print("\n—— QGIS lookup ——")
        print(
            f"label_id: {rec['label_id']}  label={rec.get('label')}  "
            f"source={rec.get('source_queue')}  flags={_flag_summary(rec)}"
        )
        print(f"extent:   {rec.get('qgis_extent')}")
        print(
            f"before:   {b.get('centroid_x'):.3f}, {b.get('centroid_y'):.3f}  "
            f"vol={_fmt_vol(rec.get('before_volume'))}  (EPSG:25829)"
        )
        print(
            f"after:    {a.get('centroid_x'):.3f}, {a.get('centroid_y'):.3f}  "
            f"vol={_fmt_vol(rec.get('after_volume'))}  (EPSG:25829)"
        )
        print(
            f"manual24: intersects={rec['manual_ann_24'].get('intersects')} "
            f"fids={rec['manual_ann_24'].get('fids')}"
        )
        print(
            f"manual25: intersects={rec['manual_ann_25'].get('intersects')} "
            f"fids={rec['manual_ann_25'].get('fids')}"
        )
        print("Paste extent into QGIS: View → Set Extent… (or paste into Python console)\n")

    def toggle_mode():
        if missed.empty:
            print("No missed candidates queue (empty).")
            return
        state["mode"] = "missed" if state["mode"] == "matches" else "matches"
        state["idx"] = 0
        mode_name = "missed candidates" if state["mode"] == "missed" else "matcher matches"
        print(f"Switched to {mode_name} queue ({_queue_len()} items).")
        redraw()

    def on_key(event):
        if event.key in ("n", "right"):
            if _queue_len():
                state["idx"] = (state["idx"] + 1) % _queue_len()
                redraw()
        elif event.key in ("p", "left"):
            if _queue_len():
                state["idx"] = (state["idx"] - 1) % _queue_len()
                redraw()
        elif event.key in ("y", "Y"):
            set_label(LABEL_CONFIRMED)
        elif event.key in ("x", "X"):
            set_label(LABEL_NOT)
        elif event.key in ("?", "u", "U"):
            set_label(LABEL_UNSURE)
        elif event.key == "backspace":
            set_label(None)
        elif event.key == "b":
            _update_flags(boulder_flag=BOULDER_YES)
        elif event.key == "z":
            _update_flags(boulder_flag=BOULDER_NO)
        elif event.key == "d":
            _update_flags(deposit_flag=True)
        elif event.key == "i":
            _update_flags(isolated=True)
        elif event.key == "B":
            _update_flags(boulder_flag=None)
        elif event.key == "D":
            _update_flags(deposit_flag=None)
        elif event.key in ("m", "M"):
            toggle_mode()
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
    _goto_next_unlabeled(prefer_advance=False)
    missed_note = f", {len(missed)} missed candidates (m toggles)" if not missed.empty else ""
    ortho_note = (
        " [full orthos]" if ortho is not None and ortho.ready else " [tile/fallback rasters]"
    )
    print(
        f"Evaluating {len(matches)} matcher matches{missed_note} → {labels_path}{ortho_note}\n"
        "y=confirm  x=not-match  ?=unsure  b/z/d/i=flags  m=queue  j=next unlabeled  "
        "c=print  s=save  q=save+quit  (pan/zoom toolbar on larger ortho crop)"
    )
    try:
        plt.show()
    finally:
        if state["dirty"]:
            do_save()
        if ortho is not None:
            ortho.close()


def _load_missed_candidates(
    results: dict[str, gpd.GeoDataFrame],
    results_dir: Path,
    candidate_radius: float,
) -> gpd.GeoDataFrame:
    missed = results.get("missed_candidates")
    if missed is not None and not missed.empty:
        print(f"Loaded {len(missed)} missed candidates from {results_dir / 'missed_candidates.geojson'}")
        return missed

    disappeared = results.get("disappeared")
    appeared = results.get("appeared")
    if (
        disappeared is None
        or appeared is None
        or disappeared.empty
        or appeared.empty
    ):
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:25829")

    from .candidates import build_missed_candidates

    exclude: set[tuple[int, int]] = set()
    matches = results.get("matches")
    if matches is not None and not matches.empty:
        exclude = {
            (int(r["before_id"]), int(r["after_id"]))
            for _, r in matches.iterrows()
        }

    missed = build_missed_candidates(
        disappeared,
        appeared,
        candidate_radius=candidate_radius,
        exclude_pairs=exclude,
    )
    if not missed.empty:
        print(
            f"Built {len(missed)} missed candidates on the fly "
            f"(candidate_radius={candidate_radius}m)"
        )
    return missed


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
    parser.add_argument("--pad-m", type=float, default=12.0, help="Initial view pad around boulder (m)")
    parser.add_argument(
        "--raster-pad-m",
        type=float,
        default=40.0,
        help="Ortho crop pad (m); larger than --pad-m so you can pan in the UI",
    )
    parser.add_argument("--min-iou", type=float, default=0.05, help="Min IoU to list an ann fid")
    parser.add_argument(
        "--candidate-radius",
        type=float,
        default=25.0,
        help="Radius (m) for on-the-fly missed-candidate pairing when geojson is missing",
    )
    parser.add_argument(
        "--before-ortho",
        type=Path,
        default=root / "2024" / "Sites1and2_2024_Orthomosaic.tif",
        help="Full 2024 orthomosaic (preferred over inference tiles)",
    )
    parser.add_argument(
        "--after-ortho",
        type=Path,
        default=root / "2025" / "25IniSouthOrt.tif",
        help="Full 2025 orthomosaic (preferred over inference tiles)",
    )
    parser.add_argument(
        "--use-tiles",
        action="store_true",
        help="Force inference-tile previews instead of full orthos",
    )
    parser.add_argument(
        "--before-dsm",
        type=Path,
        default=root / "2024" / "Sites1and2_2024_DSM_30mm.tif",
        help="2024 DSM used to fill missing volumes",
    )
    parser.add_argument(
        "--after-dsm",
        type=Path,
        default=root / "2025" / "25IniSouthDSM.tif",
        help="2025 DSM used to fill missing volumes",
    )
    parser.add_argument(
        "--compute-volume",
        action="store_true",
        help="Force DSM volume computation even if some volumes already exist",
    )
    parser.add_argument(
        "--no-volume",
        action="store_true",
        help="Skip DSM volume fill (leave — when volumes were never computed)",
    )
    parser.add_argument(
        "--no-persist-volumes",
        action="store_true",
        help="Compute volumes in-memory only (do not rewrite GeoJSONs under --outdir)",
    )
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
    missed = _load_missed_candidates(results, results_dir, args.candidate_radius)

    volumes_missing = _volumes_mostly_missing(
        results.get("matches"), ("before_volume", "after_volume")
    )
    if args.no_volume:
        if volumes_missing:
            print("Volumes missing; skipping DSM fill (--no-volume).")
    elif volumes_missing or args.compute_volume:
        if not args.before_dsm.exists() or not args.after_dsm.exists():
            print(
                "Warning: volumes are missing but DSM(s) not found:\n"
                f"  before: {args.before_dsm} exists={args.before_dsm.exists()}\n"
                f"  after:  {args.after_dsm} exists={args.after_dsm.exists()}\n"
                "Pass --before-dsm / --after-dsm, or rebuild with "
                "build_gt_dataset --compute-volume."
            )
        else:
            reason = "forced" if args.compute_volume and not volumes_missing else "missing on match rows"
            print(f"Filling DSM volumes ({reason}) …")
            before, after, matches, missed = enrich_pair_volumes(
                before,
                after,
                results["matches"],
                missed,
                args.before_dsm,
                args.after_dsm,
                appeared=results.get("appeared"),
                disappeared=results.get("disappeared"),
                persist_dir=None if args.no_persist_volumes else outdir,
            )
            results["matches"] = matches
            results["missed_candidates"] = missed

    pair_tiles = None
    before_raster = after_raster = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        tiles = summary.get("tiles") or []
        if tiles:
            pair_tiles = [(t["tile_24"], t["tile_25"]) for t in tiles]
            before_raster = Path(tiles[0]["tile_24"])
            after_raster = Path(tiles[0]["tile_25"])

    ortho = None
    if not args.use_tiles:
        ortho = OrthoPreview(args.before_ortho, args.after_ortho)
        if ortho.ready:
            # Prefer full orthos — disable tile picking (tile edges → black)
            pair_tiles = None
            before_raster = args.before_ortho if args.before_ortho.exists() else before_raster
            after_raster = args.after_ortho if args.after_ortho.exists() else after_raster
        else:
            print(
                "Warning: full orthos not opened; falling back to inference tiles. "
                "Check --before-ortho / --after-ortho paths."
            )
            ortho = None
    else:
        print("Using inference tiles for previews (--use-tiles).")

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
        "candidate_radius": args.candidate_radius,
        "before_ortho": str(args.before_ortho) if args.before_ortho else None,
        "after_ortho": str(args.after_ortho) if args.after_ortho else None,
        "before_dsm": str(args.before_dsm) if args.before_dsm else None,
        "after_dsm": str(args.after_dsm) if args.after_dsm else None,
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
        missed_candidates=missed,
        pad_m=args.pad_m,
        meta=meta,
        min_iou=args.min_iou,
        ortho=ortho,
        raster_pad_m=args.raster_pad_m,
    )


if __name__ == "__main__":
    main()
