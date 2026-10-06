# cli.py

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .dedupe import dedupe_polygons
from .matcher import BoulderMatcher
from .qc import run_dod_qc, write_dod_qc
from .survey import (
    BoulderSurvey,
    filter_boulder_class,
    filter_min_volume,
)


def main():
    parser = argparse.ArgumentParser(
        description="Match boulder polygons between two surveys (shape-first)."
    )

    parser.add_argument("--before", default=None, help="Before survey polygon file")
    parser.add_argument("--after", default=None, help="After survey polygon file")
    parser.add_argument("--before-dsm", default=None, help="Before survey DSM")
    parser.add_argument("--after-dsm", default=None, help="After survey DSM")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--use-path-config",
        action="store_true",
        help="Fill missing before/after/DSM paths from paths.local.yaml",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=200.0,
        help="Candidate centroid gate in meters (default 200)",
    )
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--compute-volume", action="store_true")
    parser.add_argument(
        "--min-volume",
        type=float,
        default=None,
        help="Drop boulders with DSM volume below this (m³). "
        "When volume is NaN, uses --min-area-fallback.",
    )
    parser.add_argument(
        "--min-area-fallback",
        type=float,
        default=0.5,
        help="Plan-area (m²) gate used when volume is missing (default 0.5)",
    )
    parser.add_argument(
        "--boulder-class-only",
        action="store_true",
        help="Keep only Class==0 (boulder) features when Class column exists",
    )
    parser.add_argument(
        "--boulder-class-value",
        type=int,
        default=0,
        help="Class value for boulders (default 0)",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        default=False,
        help="Collapse overlapping detections (off by default; for Mask R-CNN)",
    )
    parser.add_argument("--no-dedupe", action="store_true", help="Disable dedupe")
    parser.add_argument("--dedupe-iou", type=float, default=0.4)
    parser.add_argument("--dedupe-centroid-m", type=float, default=0.75)
    parser.add_argument(
        "--dod-qc",
        action="store_true",
        default=True,
        help="Write DSM-of-Difference QC layers when both DSMs are given (default on)",
    )
    parser.add_argument("--no-dod-qc", action="store_true")
    parser.add_argument("--dod-lod-m", type=float, default=0.08)
    parser.add_argument("--dod-min-change-m3", type=float, default=0.05)

    args = parser.parse_args()
    if args.no_dedupe:
        args.dedupe = False
    if args.no_dod_qc:
        args.dod_qc = False

    if args.use_path_config:
        from .paths import load_paths

        paths = load_paths()
        if args.before is None:
            args.before = str(paths["annotations_24"])
        if args.after is None:
            args.after = str(paths["annotations_25"])
        if args.before_dsm is None:
            args.before_dsm = str(paths["dsm_24"]) if paths["dsm_24"] else None
        if args.after_dsm is None:
            args.after_dsm = str(paths["dsm_25"]) if paths["dsm_25"] else None
        print(f"Path config: {paths.get('config_path')}")
        print(f"  before: {args.before}")
        print(f"  after:  {args.after}")
        print(f"  dsm24:  {args.before_dsm}")
        print(f"  dsm25:  {args.after_dsm}")

    if not args.before or not args.after:
        raise SystemExit("Need --before/--after or --use-path-config with annotations set")

    outdir = Path(args.outdir)
    # CLI historically wrote GeoJSON into outdir root; also write results/ for
    # evaluate_matches / visualize which expect <outdir>/results/.
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    before = BoulderSurvey(
        name="before",
        polygon_path=args.before,
        dsm_path=args.before_dsm,
    ).compute_attributes()

    after = BoulderSurvey(
        name="after",
        polygon_path=args.after,
        dsm_path=args.after_dsm,
    ).compute_attributes()

    if args.boulder_class_only:
        n_b, n_a = len(before.polygons), len(after.polygons)
        before.polygons = filter_boulder_class(
            before.polygons, class_value=args.boulder_class_value
        )
        after.polygons = filter_boulder_class(
            after.polygons, class_value=args.boulder_class_value
        )
        print(
            f"Class filter ({args.boulder_class_value}): "
            f"before {n_b} → {len(before.polygons)}, "
            f"after {n_a} → {len(after.polygons)}"
        )

    if args.dedupe:
        n_b, n_a = len(before.polygons), len(after.polygons)
        before.polygons = dedupe_polygons(
            before.polygons,
            iou_thresh=args.dedupe_iou,
            centroid_dist_m=args.dedupe_centroid_m,
        )
        after.polygons = dedupe_polygons(
            after.polygons,
            iou_thresh=args.dedupe_iou,
            centroid_dist_m=args.dedupe_centroid_m,
        )
        print(
            f"Dedupe: before {n_b} → {len(before.polygons)}, "
            f"after {n_a} → {len(after.polygons)}"
        )
        before.polygons.to_file(outdir / "before_deduped.geojson", driver="GeoJSON")
        after.polygons.to_file(outdir / "after_deduped.geojson", driver="GeoJSON")

    if args.compute_volume:
        if not args.before_dsm or not args.after_dsm:
            raise SystemExit("--compute-volume requires --before-dsm and --after-dsm")
        # Cheap prefilter: tiny plan area cannot yield ≥ min_volume for typical heights.
        if args.min_volume is not None and args.min_area_fallback is not None:
            n_b, n_a = len(before.polygons), len(after.polygons)
            before.polygons = before.polygons[
                before.polygons.geometry.area >= args.min_area_fallback
            ].reset_index(drop=True)
            after.polygons = after.polygons[
                after.polygons.geometry.area >= args.min_area_fallback
            ].reset_index(drop=True)
            print(
                f"Prefilter area ≥ {args.min_area_fallback} m² before volume: "
                f"before {n_b} → {len(before.polygons)}, "
                f"after {n_a} → {len(after.polygons)}"
            )
        print("Computing DSM volumes (this can take a while) …")
        before.compute_volume()
        after.compute_volume()

    if args.min_volume is not None:
        n_b, n_a = len(before.polygons), len(after.polygons)
        before.polygons = filter_min_volume(
            before.polygons,
            min_volume=args.min_volume,
            min_area_fallback=args.min_area_fallback,
        )
        after.polygons = filter_min_volume(
            after.polygons,
            min_volume=args.min_volume,
            min_area_fallback=args.min_area_fallback,
        )
        print(
            f"Min volume {args.min_volume} m³ "
            f"(area fallback {args.min_area_fallback} m²): "
            f"before {n_b} → {len(before.polygons)}, "
            f"after {n_a} → {len(after.polygons)}"
        )

    # Persist filtered inputs for the eval GUI.
    pred_dir = outdir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    before.polygons.to_file(pred_dir / "before_inferred_boulders.geojson", driver="GeoJSON")
    after.polygons.to_file(pred_dir / "after_inferred_boulders.geojson", driver="GeoJSON")

    matcher = BoulderMatcher(
        before=before,
        after=after,
        search_radius=args.search_radius,
        min_score=args.min_score,
    )

    results = matcher.match()

    for name, key in (
        ("matched_boulders.geojson", "matches"),
        ("appeared_boulders.geojson", "appeared"),
        ("disappeared_boulders.geojson", "disappeared"),
        ("movement_vectors.geojson", "vectors"),
    ):
        # Drop non-serializable columns if any slipped through.
        gdf = results[key]
        drop_cols = [c for c in gdf.columns if c == "shape_sig"]
        if drop_cols:
            gdf = gdf.drop(columns=drop_cols)
        gdf.to_file(results_dir / name, driver="GeoJSON")
        # Also copy to outdir root for backward compatibility.
        shutil.copy2(results_dir / name, outdir / name)

    summary = {
        "before": str(Path(args.before).resolve()),
        "after": str(Path(args.after).resolve()),
        "before_dsm": args.before_dsm,
        "after_dsm": args.after_dsm,
        "search_radius": args.search_radius,
        "min_score": args.min_score,
        "min_volume": args.min_volume,
        "boulder_class_only": args.boulder_class_only,
        "n_before": len(before.polygons),
        "n_after": len(after.polygons),
        "n_matches": len(results["matches"]),
        "n_appeared": len(results["appeared"]),
        "n_disappeared": len(results["disappeared"]),
        "weights": matcher.weights,
        "mode": "manual_gt_gt" if args.use_path_config or args.boulder_class_only else "generic",
    }
    (outdir / "match_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Matches: {len(results['matches'])}")
    print(f"Appeared: {len(results['appeared'])}")
    print(f"Disappeared: {len(results['disappeared'])}")
    print(f"Movement vectors: {len(results['vectors'])}")
    print(f"Wrote results to {results_dir}")

    if args.dod_qc and args.before_dsm and args.after_dsm:
        print("Running DoD QC …")
        qc = run_dod_qc(
            results,
            before_polygons=before.polygons,
            after_polygons=after.polygons,
            before_dsm=args.before_dsm,
            after_dsm=args.after_dsm,
            lod_m=args.dod_lod_m,
            min_change_m3=args.dod_min_change_m3,
        )
        qc_dir = outdir / "dod_qc"
        write_dod_qc(qc, qc_dir)
        print(json.dumps(qc["summary"], indent=2))
        print(f"DoD QC written to {qc_dir}")
    elif args.dod_qc:
        print("Skipping DoD QC (need --before-dsm and --after-dsm).")


if __name__ == "__main__":
    main()
