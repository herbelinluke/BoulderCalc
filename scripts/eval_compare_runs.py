#!/usr/bin/env python3
"""Compare multiple training runs via metrics_valid.json and metrics.json curves.

Examples
--------
Compare the geo-split weekend experiment::

    python BoulderCalculator/scripts/eval_compare_runs.py \\
      --segmentation-dir segmentation \\
      --geo-prefix training_run_geo_ \\
      --output-dir segmentation/eval_compare_geo

Or pass explicit runs::

    python BoulderCalculator/scripts/eval_compare_runs.py \\
      --run baseline=segmentation/training_run_geo_baseline/metrics_valid.json \\
      --run alt_a=segmentation/training_run_geo_blocks_alt_a/metrics_valid.json \\
      --output-dir segmentation/eval_compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (  # noqa: E402
    compare_metrics_valid,
    discover_geo_runs,
    extract_eval_curve,
    load_metrics_jsonl,
    plot_learning_curves,
)


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.parent.name, path
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="name=path/to/metrics_valid.json (repeatable). "
        "If omitted with --geo-prefix, auto-discovers geo runs.",
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("segmentation"),
    )
    parser.add_argument(
        "--geo-prefix",
        type=str,
        default="",
        help="If set (e.g. training_run_geo_), discover metrics_valid under segmentation-dir.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--curve-metric",
        action="append",
        default=[],
        help="metrics.json key to plot (default: bbox/AP50 and segm/AP50). Repeatable.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, Path] = {}
    for spec in args.run:
        name, path = parse_run(spec)
        runs[name] = path
    if args.geo_prefix:
        runs.update(discover_geo_runs(args.segmentation_dir, prefix=args.geo_prefix))
    if not runs:
        # Sensible default for the geo experiment.
        runs = discover_geo_runs(args.segmentation_dir, prefix="training_run_geo_")
    if not runs:
        raise SystemExit("No runs found. Pass --run name=path or --geo-prefix training_run_geo_.")

    missing = [f"{n}: {p}" for n, p in runs.items() if not p.exists()]
    if missing:
        raise SystemExit("Missing metrics files:\n  " + "\n  ".join(missing))

    df = compare_metrics_valid(runs)
    csv_path = args.output_dir / "metrics_valid_comparison.csv"
    df.to_csv(csv_path)
    print(df.to_string(float_format=lambda x: f"{x:6.2f}"))
    print(f"\nWrote {csv_path}")

    # Also dump a ranked view on the headline metrics.
    rank_cols = [c for c in ("bbox/AP50", "segm/AP50", "bbox/AR100", "segm/AR100") if c in df.columns]
    if rank_cols:
        ranked = df[rank_cols].sort_values(rank_cols[0], ascending=False)
        ranked_path = args.output_dir / "metrics_valid_ranked.csv"
        ranked.to_csv(ranked_path)
        print(f"Wrote {ranked_path}")

    curve_metrics = args.curve_metric or ["bbox/AP50", "segm/AP50", "bbox/AR100", "segm/AR100"]
    curves: dict[str, list] = {}
    for name, valid_path in runs.items():
        metrics_json = valid_path.parent / "metrics.json"
        if not metrics_json.exists():
            continue
        curves[name] = extract_eval_curve(load_metrics_jsonl(metrics_json))

    for metric in curve_metrics:
        if not any(metric in (p for p in curve) for curve in curves.values() for p in curve):
            # Skip metrics never present (e.g. AR on older runs).
            if not any(metric in p for curve in curves.values() for p in curve):
                continue
        fig = plot_learning_curves(curves, metric=metric)
        safe = metric.replace("/", "_")
        out = args.output_dir / f"curve_{safe}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Wrote {out}")

    summary = {
        "runs": {n: str(p) for n, p in runs.items()},
        "best_bbox_AP50": None,
        "best_segm_AP50": None,
    }
    if "bbox/AP50" in df.columns:
        best = df["bbox/AP50"].idxmax()
        summary["best_bbox_AP50"] = {"run": best, "value": float(df.loc[best, "bbox/AP50"])}
    if "segm/AP50" in df.columns:
        best = df["segm/AP50"].idxmax()
        summary["best_segm_AP50"] = {"run": best, "value": float(df.loc[best, "segm/AP50"])}
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
