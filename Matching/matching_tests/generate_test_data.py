# generate_test_data.py
#
# Creates a synthetic before/after pair of boulder polygon layers with a
# KNOWN ground truth, so you can run the real matcher CLI against them
# and check whether it recovers the correct matches.
#
# Usage:
#   python generate_test_data.py --outdir test_data
#
# Produces:
#   test_data/before.geojson
#   test_data/after.geojson
#   test_data/ground_truth.csv   <- the answer key

import argparse
import csv
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon


def make_square(cx, cy, size=1.0):
    h = size / 2
    return Polygon(
        [
            (cx - h, cy - h),
            (cx + h, cy - h),
            (cx + h, cy + h),
            (cx - h, cy + h),
        ]
    )


def build_dataset():
    """
    Define the scenario by hand. Each entry is one "real" boulder with:
      - id: a label for the ground truth
      - before: (cx, cy, size) or None if it didn't exist yet (appears)
      - after:  (cx, cy, size) or None if it's gone (disappears)

    Feel free to edit these values directly to construct scenarios you
    want to stress-test (tight clusters, big size changes, boulders that
    swap positions, etc).
    """
    scenario = [
        # -- Easy cases: small realistic movement, should match cleanly --
        {"id": "B01", "before": (0, 0, 1.0), "after": (0.3, 0.1, 1.0)},
        {"id": "B02", "before": (10, 0, 1.5), "after": (10.2, -0.2, 1.5)},
        {"id": "B03", "before": (20, 5, 0.8), "after": (19.8, 5.3, 0.8)},

        # -- Stationary control: shouldn't move at all --
        {"id": "B04", "before": (30, 0, 1.2), "after": (30, 0, 1.2)},

        # -- Size change (e.g. rockfall breaking off a chunk) --
        {"id": "B05", "before": (0, 20, 2.0), "after": (0.2, 19.9, 1.4)},

        # -- Larger displacement, still within a generous search radius --
        {"id": "B06", "before": (15, 15, 1.0), "after": (17.5, 15.5, 1.0)},

        # -- Two boulders close together (tests the matcher doesn't cross-match) --
        {"id": "B07", "before": (40, 0, 1.0), "after": (40.3, 0.2, 1.0)},
        {"id": "B08", "before": (42, 0, 1.0), "after": (42.3, -0.2, 1.0)},

        # -- Disappeared: present before, gone after (e.g. buried or removed) --
        {"id": "B09", "before": (5, 30, 1.0), "after": None},
        {"id": "B10", "before": (25, 30, 1.3), "after": None},

        # -- Appeared: new boulder, not present before (e.g. new rockfall) --
        {"id": "B11", "before": None, "after": (12, 30, 1.1)},
        {"id": "B12", "before": None, "after": (35, 30, 0.9)},

        # -- Moved far enough that it should NOT match (bigger than search radius).
        # This is intentionally split into two separate ground-truth entries:
        # from the matcher's perspective (given search_radius=5), this SHOULD be
        # resolved as one boulder disappearing and an unrelated one appearing --
        # not treated as a single "match", even though a human reviewing the
        # scenario file might know they're "the same" boulder.
        {"id": "B13_disappeared", "before": (0, 40, 1.0), "after": None},
        {"id": "B13_appeared", "before": None, "after": (10, 40, 1.0)},
    ]
    return scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--crs", default="EPSG:32633", help="Projected CRS (meters)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scenario = build_dataset()

    before_rows = []
    after_rows = []
    ground_truth_rows = []

    for entry in scenario:
        before_geom = None
        after_geom = None

        if entry["before"] is not None:
            cx, cy, size = entry["before"]
            before_geom = make_square(cx, cy, size)
            before_rows.append({"geometry": before_geom, "gt_id": entry["id"]})

        if entry["after"] is not None:
            cx, cy, size = entry["after"]
            after_geom = make_square(cx, cy, size)
            after_rows.append({"geometry": after_geom, "gt_id": entry["id"]})

        if before_geom is not None and after_geom is not None:
            expected = "match"
        elif before_geom is not None:
            expected = "disappeared"
        else:
            expected = "appeared"

        ground_truth_rows.append({"gt_id": entry["id"], "expected_outcome": expected})

    before_gdf = gpd.GeoDataFrame(before_rows, crs=args.crs)
    after_gdf = gpd.GeoDataFrame(after_rows, crs=args.crs)

    # gt_id is our own bookkeeping column, not something the real pipeline
    # would have -- drop it from the actual files fed to the CLI, but keep
    # a lookup by position so we can trace results back to ground truth.
    before_lookup = before_gdf[["gt_id"]].copy()
    before_lookup["before_id"] = range(len(before_lookup))
    after_lookup = after_gdf[["gt_id"]].copy()
    after_lookup["after_id"] = range(len(after_lookup))

    before_gdf = before_gdf.drop(columns=["gt_id"])
    after_gdf = after_gdf.drop(columns=["gt_id"])

    before_path = outdir / "before.geojson"
    after_path = outdir / "after.geojson"
    before_gdf.to_file(before_path, driver="GeoJSON")
    after_gdf.to_file(after_path, driver="GeoJSON")

    gt_path = outdir / "ground_truth.csv"
    with open(gt_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gt_id", "expected_outcome"])
        writer.writeheader()
        writer.writerows(ground_truth_rows)

    before_lookup.to_csv(outdir / "before_lookup.csv", index=False)
    after_lookup.to_csv(outdir / "after_lookup.csv", index=False)

    print(f"Wrote {len(before_gdf)} before-polygons to {before_path}")
    print(f"Wrote {len(after_gdf)} after-polygons to {after_path}")
    print(f"Wrote ground truth ({len(ground_truth_rows)} entries) to {gt_path}")
    print("\nNext step: run your CLI, e.g.")
    print(
        f"  python -m your_package.cli --before {before_path} --after {after_path} "
        f"--outdir {outdir / 'results'} --search-radius 5"
    )
    print("\nThen check the results with:")
    print(f"  python check_results.py --outdir {outdir}")


if __name__ == "__main__":
    main()
