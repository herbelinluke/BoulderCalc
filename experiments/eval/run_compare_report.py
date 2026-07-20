#!/usr/bin/env python3
"""Compare geo training runs and write a report — no Jupyter required.

Run from the project root (folder with BoulderCalculator/ and segmentation/):

  python BoulderCalculator/experiments/eval/run_compare_report.py

Or from anywhere:

  python BoulderCalculator/experiments/eval/run_compare_report.py --root /path/to/tamucc

Writes under ``segmentation/eval_compare_geo/`` by default:
  - metrics_valid_comparison.csv / ranked CSV
  - curve_*.png learning curves
  - report.md summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _bootstrap_imports(root: Path):
    scripts = root / "BoulderCalculator" / "scripts"
    if not (scripts / "eval_utils.py").is_file():
        raise SystemExit(f"eval_utils.py not found under {scripts}")
    sys.path.insert(0, str(scripts))
    from eval_utils import (  # noqa: E402
        compare_metrics_valid,
        discover_geo_runs,
        extract_eval_curve,
        load_metrics_jsonl,
        plot_learning_curves,
        resolve_project_root,
    )

    return (
        compare_metrics_valid,
        discover_geo_runs,
        extract_eval_curve,
        load_metrics_jsonl,
        plot_learning_curves,
        resolve_project_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (auto-detected if omitted).",
    )
    parser.add_argument(
        "--geo-prefix",
        default="training_run_geo_",
        help="Discover segmentation/<prefix>*/metrics_valid.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <root>/segmentation/eval_compare_geo",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["bbox/AP50", "segm/AP50", "bbox/AR100", "segm/AR100"],
        help="Learning-curve metrics to plot.",
    )
    args = parser.parse_args()

    # Resolve root before importing so we can find eval_utils.
    if args.root is not None:
        root = args.root.resolve()
    else:
        # Lightweight walk so we can import resolve_project_root next.
        cur = Path.cwd().resolve()
        root = None
        for cand in [cur, *cur.parents]:
            if (cand / "BoulderCalculator" / "scripts" / "eval_utils.py").is_file():
                root = cand
                break
        if root is None:
            raise SystemExit(
                "Could not find project root. Pass --root /path/to/tamucc "
                "(must contain BoulderCalculator/scripts/eval_utils.py)."
            )

    (
        compare_metrics_valid,
        discover_geo_runs,
        extract_eval_curve,
        load_metrics_jsonl,
        plot_learning_curves,
        _resolve_project_root,
    ) = _bootstrap_imports(root)

    seg = root / "segmentation"
    out = args.output_dir or (seg / "eval_compare_geo")
    out.mkdir(parents=True, exist_ok=True)

    runs = discover_geo_runs(seg, prefix=args.geo_prefix)
    if not runs:
        raise SystemExit(f"No runs found under {seg}/{args.geo_prefix}*/metrics_valid.json")

    print(f"ROOT: {root}")
    print(f"Found {len(runs)} runs: {', '.join(runs)}")

    df = compare_metrics_valid(runs)
    csv_path = out / "metrics_valid_comparison.csv"
    df.to_csv(csv_path)
    print(f"Wrote {csv_path}")

    rank_cols = [c for c in ("bbox/AP50", "segm/AP50", "bbox/AR100", "segm/AR100") if c in df.columns]
    ranked = df[rank_cols].sort_values(rank_cols[0], ascending=False)
    ranked_path = out / "metrics_valid_ranked.csv"
    ranked.to_csv(ranked_path)
    print(f"Wrote {ranked_path}")
    print("\nRanked by bbox/AP50:" if "bbox/AP50" in rank_cols else "\nComparison:")
    print(ranked.round(2).to_string())

    curves = {}
    for name, valid_path in runs.items():
        mpath = Path(valid_path).parent / "metrics.json"
        if mpath.exists():
            curves[name] = extract_eval_curve(load_metrics_jsonl(mpath))

    for metric in args.metrics:
        if not any(metric in p for curve in curves.values() for p in curve):
            print(f"Skip curve {metric} (not present in metrics.json)")
            continue
        fig = plot_learning_curves(curves, metric=metric)
        png = out / f"curve_{metric.replace('/', '_')}.png"
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {png}")

    lines = [
        "# Geo training-run comparison",
        "",
        f"Root: `{root}`",
        f"Runs discovered with prefix `{args.geo_prefix}`:",
        "",
        ranked.round(2).to_markdown(),
        "",
        "## Notes",
        "",
        "- Whole-split `metrics_valid.json` is the headline comparison.",
        "- Per-tile heatmaps need images on disk; see `experiments/eval/README.md`.",
        "- On this laptop, `coco_geo_*_rgb_dsm` may be an empty stub — use RGB",
        "  annotations + `--image-dir tiling_rgb_dsm_*` or rebuild the 4-band COCO.",
        "",
    ]
    report = out / "report.md"
    try:
        report.write_text("\n".join(lines))
    except Exception:
        # pandas to_markdown needs tabulate; fall back to plain text.
        plain = [
            "# Geo training-run comparison",
            "",
            ranked.round(2).to_string(),
            "",
        ]
        report.write_text("\n".join(plain))
    print(f"Wrote {report}")
    print(f"\nDone. Open {out}/ for CSVs, PNGs, and report.md")


if __name__ == "__main__":
    main()
