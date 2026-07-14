# check_results.py
#
# Compares the REAL matcher CLI output against the ground truth produced
# by generate_test_data.py, and reports precision/recall/errors.
#
# Usage:
#   python check_results.py --outdir test_data
#
# Expects test_data/ to contain:
#   before_lookup.csv, after_lookup.csv, ground_truth.csv   (from generator)
#   results/matched_boulders.geojson
#   results/appeared_boulders.geojson
#   results/disappeared_boulders.geojson                    (from the CLI run)

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    results_dir = outdir / "results"

    ground_truth = pd.read_csv(outdir / "ground_truth.csv")
    before_lookup = pd.read_csv(outdir / "before_lookup.csv")
    after_lookup = pd.read_csv(outdir / "after_lookup.csv")

    matched = gpd.read_file(results_dir / "matched_boulders.geojson")
    appeared = gpd.read_file(results_dir / "appeared_boulders.geojson")
    disappeared = gpd.read_file(results_dir / "disappeared_boulders.geojson")

    # Map the CLI's before_id / after_id back to our gt_id labels.
    before_id_to_gt = dict(zip(before_lookup["before_id"], before_lookup["gt_id"]))
    after_id_to_gt = dict(zip(after_lookup["after_id"], after_lookup["gt_id"]))

    predicted = {}  # gt_id -> "match" / "appeared" / "disappeared"
    mismatched_ids = []  # matches where before_id and after_id map to DIFFERENT gt_ids

    for _, row in matched.iterrows():
        before_gt = before_id_to_gt.get(row["before_id"])
        after_gt = after_id_to_gt.get(row["after_id"])

        if before_gt != after_gt:
            # The matcher paired two boulders that are NOT the same real
            # boulder according to ground truth -- this is a cross-match
            # error, more serious than a simple miss.
            mismatched_ids.append((before_gt, after_gt, row["match_score"]))
            predicted[before_gt] = f"WRONG_MATCH(->{after_gt})"
        else:
            predicted[before_gt] = "match"

    if "gt_id" in appeared.columns:
        pass  # appeared/disappeared don't carry gt_id directly; use lookups instead

    matched_after_ids = set(matched["after_id"])
    matched_before_ids = set(matched["before_id"])

    for _, row in after_lookup.iterrows():
        if row["after_id"] not in matched_after_ids:
            predicted[row["gt_id"]] = "appeared"

    for _, row in before_lookup.iterrows():
        if row["before_id"] not in matched_before_ids:
            predicted[row["gt_id"]] = "disappeared"

    # ---- Compare against ground truth ----
    n_correct = 0
    n_total = 0
    errors = []

    for _, row in ground_truth.iterrows():
        gt_id = row["gt_id"]
        expected = row["expected_outcome"]
        actual = predicted.get(gt_id, "MISSING")
        n_total += 1

        if actual == expected:
            n_correct += 1
        else:
            errors.append((gt_id, expected, actual))

    print(f"Overall accuracy: {n_correct}/{n_total} ({100 * n_correct / n_total:.1f}%)\n")

    if mismatched_ids:
        print("Cross-matches (matcher paired two DIFFERENT real boulders):")
        for before_gt, after_gt, score in mismatched_ids:
            print(f"  before={before_gt} matched to after={after_gt} (score={score:.2f})")
        print()

    if errors:
        print("Mismatches (gt_id: expected -> actual):")
        for gt_id, expected, actual in errors:
            print(f"  {gt_id}: expected={expected!r}, got={actual!r}")
    else:
        print("No mismatches. All boulders resolved as expected.")

    # ---- Movement accuracy for successfully matched boulders ----
    print("\nMatch score summary for correctly matched boulders:")
    correct_matches = matched[
        matched.apply(
            lambda r: before_id_to_gt.get(r["before_id"]) == after_id_to_gt.get(r["after_id"]),
            axis=1,
        )
    ]
    if not correct_matches.empty:
        print(correct_matches[["before_id", "after_id", "match_score", "distance_m"]])
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
