# cli.py

import argparse
from pathlib import Path

from .matcher import BoulderMatcher
from .survey import BoulderSurvey


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--before", required=True, help="Before survey polygon file")
    parser.add_argument("--after", required=True, help="After survey polygon file")
    parser.add_argument("--before-dsm", default=None, help="Before survey DSM")
    parser.add_argument("--after-dsm", default=None, help="After survey DSM")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--search-radius", type=float, default=5.0)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--compute-volume", action="store_true")

    args = parser.parse_args()

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

    if args.compute_volume:
        before.compute_volume()
        after.compute_volume()

    matcher = BoulderMatcher(
        before=before,
        after=after,
        search_radius=args.search_radius,
        min_score=args.min_score,
    )

    results = matcher.match()

    results["matches"].to_file(outdir / "matched_boulders.geojson", driver="GeoJSON")
    results["appeared"].to_file(outdir / "appeared_boulders.geojson", driver="GeoJSON")
    results["disappeared"].to_file(outdir / "disappeared_boulders.geojson", driver="GeoJSON")
    results["vectors"].to_file(outdir / "movement_vectors.geojson", driver="GeoJSON")

    print(f"Matches: {len(results['matches'])}")
    print(f"Appeared: {len(results['appeared'])}")
    print(f"Disappeared: {len(results['disappeared'])}")
    print(f"Movement vectors: {len(results['vectors'])}")


if __name__ == "__main__":
    main()