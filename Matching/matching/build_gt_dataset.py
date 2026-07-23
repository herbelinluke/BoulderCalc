"""Build a matcher-eval dataset folder from manual annotations and/or inference.

Creates a matching-outdir-like layout so ``evaluate_matches`` / the matcher CLI
can run on a larger set of boulders (full july14 GPKGs, optional tile filter).

Example (manual polygons only — no Detectron2):
  python -m matching.build_gt_dataset \\
    --ann-24 ../../segmentation/annotations/july14_24.gpkg \\
    --ann-25 ../../segmentation/annotations/july14_25.gpkg \\
    --outdir ../../segmentation/match_datasets/july14_manual \\
    --search-radius 15 --candidate-radius 25

Example (reuse an existing inference matching run's predictions, rematch):
  python -m matching.build_gt_dataset \\
    --before-polygons .../matching/predictions/before_inferred_boulders.geojson \\
    --after-polygons .../matching/predictions/after_inferred_boulders.geojson \\
    --outdir .../match_datasets/rematch_r15 \\
    --search-radius 15
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd

from .candidates import run_matcher_with_candidates, write_missed_candidates
from .dedupe import dedupe_polygons
from .evaluate_matches import load_manual_annotations
from .matcher import BoulderMatcher
from .survey import BoulderSurvey


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf is None or gdf.empty:
        gpd.GeoDataFrame({"geometry": []}, crs="EPSG:25829").to_file(path, driver="GeoJSON")
    else:
        gdf.to_file(path, driver="GeoJSON")


def _as_survey(name: str, gdf: gpd.GeoDataFrame, dsm: Path | None) -> BoulderSurvey:
    """Build a BoulderSurvey without re-reading from disk (inject polygons)."""
    survey = BoulderSurvey.__new__(BoulderSurvey)
    survey.name = name
    survey.polygon_path = f"<in-memory:{name}>"
    survey.dsm_path = str(dsm) if dsm else None
    survey.polygons = gdf.copy()
    return survey.compute_attributes()


def filter_by_bbox(
    gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float] | None,
) -> gpd.GeoDataFrame:
    if bbox is None or gdf.empty:
        return gdf
    minx, miny, maxx, maxy = bbox
    return gdf.cx[minx:maxx, miny:maxy].copy()


def build_dataset(
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    outdir: Path,
    search_radius: float = 15.0,
    min_score: float = 0.55,
    candidate_radius: float = 25.0,
    candidate_min_score: float = 0.35,
    dedupe: bool = True,
    before_dsm: Path | None = None,
    after_dsm: Path | None = None,
    compute_volume: bool = False,
    source_meta: dict | None = None,
) -> dict:
    outdir = Path(outdir)
    pred_dir = outdir / "predictions"
    results_dir = outdir / "results"
    pred_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    before = before.copy()
    after = after.copy()
    if before.crs is None:
        before = before.set_crs("EPSG:25829")
    else:
        before = before.to_crs("EPSG:25829")
    if after.crs is None:
        after = after.set_crs("EPSG:25829")
    else:
        after = after.to_crs("EPSG:25829")

    n_b_raw, n_a_raw = len(before), len(after)
    if dedupe:
        before = dedupe_polygons(before, iou_thresh=0.4, centroid_dist_m=0.75)
        after = dedupe_polygons(after, iou_thresh=0.4, centroid_dist_m=0.75)

    _write_geojson(before, pred_dir / "before_inferred_boulders_raw.geojson")
    _write_geojson(after, pred_dir / "after_inferred_boulders_raw.geojson")
    _write_geojson(before, pred_dir / "before_inferred_boulders.geojson")
    _write_geojson(after, pred_dir / "after_inferred_boulders.geojson")

    before_s = _as_survey("before", before, before_dsm)
    after_s = _as_survey("after", after, after_dsm)
    if compute_volume and before_dsm and after_dsm:
        print("Computing DSM volumes …")
        before_s.compute_volume()
        after_s.compute_volume()
        # Persist volumes back into prediction layers for the eval UI
        _write_geojson(before_s.polygons, pred_dir / "before_inferred_boulders.geojson")
        _write_geojson(after_s.polygons, pred_dir / "after_inferred_boulders.geojson")

    results = run_matcher_with_candidates(
        before_s,
        after_s,
        search_radius=search_radius,
        min_score=min_score,
        candidate_radius=candidate_radius,
        candidate_min_score=candidate_min_score,
    )

    _write_geojson(results["matches"], results_dir / "matched_boulders.geojson")
    _write_geojson(results["appeared"], results_dir / "appeared_boulders.geojson")
    _write_geojson(results["disappeared"], results_dir / "disappeared_boulders.geojson")
    _write_geojson(results["vectors"], results_dir / "movement_vectors.geojson")
    write_missed_candidates(
        results["missed_candidates"], results_dir / "missed_candidates.geojson"
    )

    summary = {
        "created_at": _utc_now(),
        "kind": "match_dataset",
        "search_radius": search_radius,
        "min_score": min_score,
        "candidate_radius": results.get("candidate_radius", candidate_radius),
        "candidate_min_score": candidate_min_score,
        "n_before_raw": n_b_raw,
        "n_after_raw": n_a_raw,
        "n_before": len(before_s.polygons),
        "n_after": len(after_s.polygons),
        "matches": len(results["matches"]),
        "appeared": len(results["appeared"]),
        "disappeared": len(results["disappeared"]),
        "missed_candidates": len(results["missed_candidates"]),
        "source": source_meta or {},
        "tiles": [],  # no inference tiles; eval UI falls back to optional orthos
    }
    (outdir / "match_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nDataset ready for labeling:\n  python -m matching.evaluate_matches --outdir {outdir}")
    return summary


def main():
    root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--ann-24",
        type=Path,
        default=root / "segmentation" / "annotations" / "july14_24.gpkg",
    )
    parser.add_argument(
        "--ann-25",
        type=Path,
        default=root / "segmentation" / "annotations" / "july14_25.gpkg",
    )
    parser.add_argument(
        "--before-polygons",
        type=Path,
        default=None,
        help="Override before layer (GeoJSON/GPKG), e.g. inference predictions",
    )
    parser.add_argument(
        "--after-polygons",
        type=Path,
        default=None,
        help="Override after layer (GeoJSON/GPKG)",
    )
    parser.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    parser.add_argument("--search-radius", type=float, default=BoulderMatcher.DEFAULT_SEARCH_RADIUS)
    parser.add_argument("--min-score", type=float, default=BoulderMatcher.DEFAULT_MIN_SCORE)
    parser.add_argument("--candidate-radius", type=float, default=25.0)
    parser.add_argument("--candidate-min-score", type=float, default=0.35)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--before-dsm", type=Path, default=None)
    parser.add_argument("--after-dsm", type=Path, default=None)
    parser.add_argument("--compute-volume", action="store_true")
    args = parser.parse_args()

    if args.before_polygons and args.after_polygons:
        before = gpd.read_file(args.before_polygons)
        after = gpd.read_file(args.after_polygons)
        source = {
            "type": "polygons",
            "before": str(args.before_polygons.resolve()),
            "after": str(args.after_polygons.resolve()),
        }
    else:
        print(f"Loading manual annotations:\n  {args.ann_24}\n  {args.ann_25}")
        before = load_manual_annotations(args.ann_24)
        after = load_manual_annotations(args.ann_25)
        source = {
            "type": "manual_gpkg",
            "ann_24": str(args.ann_24.resolve()),
            "ann_25": str(args.ann_25.resolve()),
        }

    bbox = tuple(args.bbox) if args.bbox else None
    if bbox:
        before = filter_by_bbox(before, bbox)
        after = filter_by_bbox(after, bbox)
        source["bbox"] = list(bbox)
        print(f"BBox filter → before={len(before)} after={len(after)}")

    build_dataset(
        before=before,
        after=after,
        outdir=args.outdir,
        search_radius=args.search_radius,
        min_score=args.min_score,
        candidate_radius=args.candidate_radius,
        candidate_min_score=args.candidate_min_score,
        dedupe=not args.no_dedupe,
        before_dsm=args.before_dsm,
        after_dsm=args.after_dsm,
        compute_volume=args.compute_volume,
        source_meta=source,
    )


if __name__ == "__main__":
    main()
