"""Export eye-confirmed matches into a clean verified dataset package.

Reads ``<outdir>/eval/match_labels.json`` (from ``evaluate_matches``) and writes:

- ``verified/verified_matches.geojson`` — after-geometry + pair attributes
- ``verified/verified_pairs.csv`` — flat table for spreadsheets
- ``verified/match_labels_confirmed.json`` — audit copy of confirmed records only

Example:
  python -m matching.export_verified \\
    --outdir ../../segmentation/manual_match_2024_2025
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import shape

from .evaluate_matches import LABEL_CONFIRMED, load_labels_db
from .paths import load_paths


def export_verified(outdir: Path, labels_json: Path | None = None) -> Path:
    outdir = Path(outdir)
    labels_path = labels_json or (outdir / "eval" / "match_labels.json")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"No labels at {labels_path}. Run matching.evaluate_matches first."
        )

    db = load_labels_db(labels_path)
    labels = db.get("labels") or {}
    if isinstance(labels, list):
        labels = {r["label_id"]: r for r in labels if "label_id" in r}

    confirmed = [
        r for r in labels.values() if r.get("label") == LABEL_CONFIRMED
    ]
    confirmed.sort(key=lambda r: (-(r.get("match_score") or 0), r.get("label_id", "")))

    verified_dir = outdir / "verified"
    verified_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    geoms = []
    for rec in confirmed:
        after = rec.get("after") or {}
        geom = None
        if after.get("geojson"):
            geom = shape(after["geojson"])
        elif after.get("wkt"):
            from shapely import wkt as shapely_wkt

            geom = shapely_wkt.loads(after["wkt"])
        geoms.append(geom)
        rows.append(
            {
                "label_id": rec.get("label_id"),
                "before_id": rec.get("before_id"),
                "after_id": rec.get("after_id"),
                "before_fid": rec.get("before_fid"),
                "after_fid": rec.get("after_fid"),
                "match_score": rec.get("match_score"),
                "distance_m": rec.get("distance_m"),
                "dx": rec.get("dx"),
                "dy": rec.get("dy"),
                "before_area": rec.get("before_area"),
                "after_area": rec.get("after_area"),
                "before_volume": rec.get("before_volume"),
                "after_volume": rec.get("after_volume"),
                "labeled_at": rec.get("labeled_at"),
                "note": rec.get("note") or "",
            }
        )

    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:25829")
    gdf = gdf[gdf.geometry.notna()].copy()
    geojson_path = verified_dir / "verified_matches.geojson"
    csv_path = verified_dir / "verified_pairs.csv"
    json_path = verified_dir / "match_labels_confirmed.json"

    if len(gdf):
        gdf.to_file(geojson_path, driver="GeoJSON")
    else:
        gpd.GeoDataFrame({"geometry": []}, crs="EPSG:25829").to_file(
            geojson_path, driver="GeoJSON"
        )

    fieldnames = list(rows[0].keys()) if rows else [
        "label_id",
        "before_id",
        "after_id",
        "before_fid",
        "after_fid",
        "match_score",
        "distance_m",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    audit = {
        "schema_version": db.get("schema_version", 1),
        "source_labels": str(labels_path.resolve()),
        "n_confirmed": len(confirmed),
        "n_total_labeled": sum(1 for r in labels.values() if r.get("label")),
        "labels": confirmed,
    }
    json_path.write_text(json.dumps(audit, indent=2))

    print(f"Confirmed matches: {len(confirmed)}")
    print(f"  GeoJSON: {geojson_path}")
    print(f"  CSV:     {csv_path}")
    print(f"  Audit:   {json_path}")
    return verified_dir


def main():
    try:
        paths = load_paths()
        default_outdir = paths["project_root"] / "segmentation" / "manual_match_2024_2025"
    except Exception:
        default_outdir = Path("../../segmentation/manual_match_2024_2025")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=default_outdir)
    parser.add_argument("--labels-json", type=Path, default=None)
    args = parser.parse_args()
    export_verified(args.outdir, args.labels_json)


if __name__ == "__main__":
    main()
