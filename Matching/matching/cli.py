# cli.py

import argparse
import json
from pathlib import Path

from .candidates import run_matcher_with_candidates, write_missed_candidates
from .dedupe import dedupe_polygons
from .matcher import BoulderMatcher
from .qc import run_dod_qc, write_dod_qc
from .survey import BoulderSurvey


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--before", required=True, help="Before survey polygon file")
    parser.add_argument("--after", required=True, help="After survey polygon file")
    parser.add_argument("--before-dsm", default=None, help="Before survey DSM")
    parser.add_argument("--after-dsm", default=None, help="After survey DSM")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--search-radius",
        type=float,
        default=BoulderMatcher.DEFAULT_SEARCH_RADIUS,
        help=f"Centroid search radius in metres (default {BoulderMatcher.DEFAULT_SEARCH_RADIUS})",
    )
    parser.add_argument("--min-score", type=float, default=BoulderMatcher.DEFAULT_MIN_SCORE)
    parser.add_argument(
        "--candidate-radius",
        type=float,
        default=None,
        help="Radius for missed appeared↔disappeared candidates (default 1.5× search-radius)",
    )
    parser.add_argument(
        "--candidate-min-score",
        type=float,
        default=0.35,
        help="Softer score floor for missed-match review candidates",
    )
    parser.add_argument("--compute-volume", action="store_true")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        default=True,
        help="Collapse overlapping detections (default on)",
    )
    parser.add_argument("--no-dedupe", action="store_true")
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

    outdir = Path(args.outdir)
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
        before.compute_volume()
        after.compute_volume()

    results = run_matcher_with_candidates(
        before,
        after,
        search_radius=args.search_radius,
        min_score=args.min_score,
        candidate_radius=args.candidate_radius,
        candidate_min_score=args.candidate_min_score,
    )

    results["matches"].to_file(outdir / "matched_boulders.geojson", driver="GeoJSON")
    results["appeared"].to_file(outdir / "appeared_boulders.geojson", driver="GeoJSON")
    results["disappeared"].to_file(outdir / "disappeared_boulders.geojson", driver="GeoJSON")
    results["vectors"].to_file(outdir / "movement_vectors.geojson", driver="GeoJSON")
    write_missed_candidates(results["missed_candidates"], outdir / "missed_candidates.geojson")

    print(f"Matches: {len(results['matches'])}")
    print(f"Appeared: {len(results['appeared'])}")
    print(f"Disappeared: {len(results['disappeared'])}")
    print(f"Missed candidates (review): {len(results['missed_candidates'])}")
    print(f"Movement vectors: {len(results['vectors'])}")

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
