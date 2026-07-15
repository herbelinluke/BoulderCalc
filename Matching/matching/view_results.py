"""Open the match browser or rewrite screenshots from an existing matching run.

Does NOT re-run inference. Expects ``outdir`` to already contain:
  results/*.geojson
  predictions/before_inferred_boulders.geojson
  predictions/after_inferred_boulders.geojson
  match_summary.json  (pair tile paths)
  inference_tiles/…
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .visualize import export_screenshots, load_inputs, load_results, run_gui


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Matching outdir (e.g. …/training_run_rgb_dsm_4000/matching)",
    )
    parser.add_argument("--gui", action="store_true", help="Open interactive browser")
    parser.add_argument("--screenshots", action="store_true", help="Rewrite PNGs")
    parser.add_argument("--max-matches", type=int, default=40)
    parser.add_argument("--pad-m", type=float, default=8.0)
    parser.add_argument("--overlay", action="store_true")
    args = parser.parse_args()

    outdir = args.outdir
    results_dir = outdir / "results"
    summary_path = outdir / "match_summary.json"
    before_path = outdir / "predictions" / "before_inferred_boulders.geojson"
    after_path = outdir / "predictions" / "after_inferred_boulders.geojson"

    if not (results_dir / "matched_boulders.geojson").exists():
        raise SystemExit(f"No results under {results_dir} — run inference match first.")

    results = load_results(results_dir)
    before, after = load_inputs(before_path, after_path)
    if before is not None and "before_id" not in before.columns:
        before = before.copy()
        before["before_id"] = before.index
    if after is not None and "after_id" not in after.columns:
        after = after.copy()
        after["after_id"] = after.index

    pair_tiles = None
    before_raster = after_raster = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        tiles = summary.get("tiles") or []
        if tiles:
            pair_tiles = [(t["tile_24"], t["tile_25"]) for t in tiles]
            before_raster = Path(tiles[0]["tile_24"])
            after_raster = Path(tiles[0]["tile_25"])

    side_by_side = not args.overlay
    if not args.gui and not args.screenshots:
        args.gui = True  # default to GUI when nothing specified

    if args.screenshots:
        export_screenshots(
            results,
            outdir=outdir / "screenshots",
            before=before,
            after=after,
            before_raster=before_raster,
            after_raster=after_raster,
            max_matches=args.max_matches,
            pad_m=args.pad_m,
            side_by_side=side_by_side,
            pair_tiles=pair_tiles,
        )

    if args.gui:
        run_gui(
            results,
            before=before,
            after=after,
            before_raster=before_raster,
            after_raster=after_raster,
            pad_m=args.pad_m,
            side_by_side=side_by_side,
            pair_tiles=pair_tiles,
        )


if __name__ == "__main__":
    main()
