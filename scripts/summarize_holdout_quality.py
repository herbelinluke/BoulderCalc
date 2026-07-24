#!/usr/bin/env python3
"""Re-summarize per-tile CSVs: empty-GT effect + with-GT-only means (no GPU).

Reads one or more ``per_tile_metrics.csv`` dirs (from ``eval_per_tile.py``) and
writes ``holdout_quality*.json`` / ``empty_gt_tiles*.csv`` plus a combined
comparison table.

Example::

    python BoulderCalculator/scripts/summarize_holdout_quality.py \\
      --eval-dir segmentation/eval_per_tile_rgb_dsm \\
      --eval-dir segmentation/eval_per_tile_local_relief \\
      --eval-dir segmentation/eval_per_tile_geo_split_all \\
      --output-dir segmentation/eval_holdout_quality
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import plot_tile_heatmaps, write_holdout_quality_reports  # noqa: E402


def _load_rows(eval_dir: Path) -> tuple[str, list[dict]]:
    csv_path = eval_dir / "per_tile_metrics.csv"
    if not csv_path.is_file():
        # multi-split layout
        rows = []
        for sub in sorted(eval_dir.glob("*/per_tile_metrics.csv")):
            df = pd.read_csv(sub)
            if "split" not in df.columns:
                df["split"] = sub.parent.name
            rows.extend(df.to_dict(orient="records"))
        if not rows:
            raise FileNotFoundError(f"No per_tile_metrics.csv under {eval_dir}")
        return eval_dir.name, rows
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        df["split"] = "test"
    return eval_dir.name, df.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing per_tile_metrics.csv (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmentation/eval_holdout_quality"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = []
    for eval_dir in args.eval_dir:
        name, rows = _load_rows(eval_dir)
        run_out = args.output_dir / name
        run_out.mkdir(parents=True, exist_ok=True)
        # Copy/annotate split if missing
        for r in rows:
            r.setdefault("split", "test")
        paths = write_holdout_quality_reports(rows, run_out)
        summary = json.loads(paths["holdout_quality"].read_text())

        # Per-split reports + heatmaps when mixed
        by_split: dict[str, list] = {}
        for r in rows:
            by_split.setdefault(str(r.get("split") or "test"), []).append(r)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for split_name, split_rows in by_split.items():
            write_holdout_quality_reports(split_rows, run_out, split_label=split_name)
            figs = plot_tile_heatmaps(
                split_rows,
                metrics=("coco_AP50", "coco_AR100", "recall", "precision"),
                title_prefix=f"{name} {split_name} ",
            )
            for metric, fig in figs:
                out = run_out / f"heatmap_{split_name}_{metric}.png"
                fig.savefig(out, dpi=140)
                plt.close(fig)
            rows_gt = [r for r in split_rows if int(r.get("gt_count") or 0) > 0]
            if rows_gt:
                figs_gt = plot_tile_heatmaps(
                    rows_gt,
                    metrics=("coco_AP50", "coco_AR100", "recall", "precision"),
                    title_prefix=f"{name} {split_name} with-GT ",
                )
                for metric, fig in figs_gt:
                    out = run_out / f"heatmap_{split_name}_with_gt_{metric}.png"
                    fig.savefig(out, dpi=140)
                    plt.close(fig)
                    print(f"Wrote {out}")

        comparison.append(
            {
                "run": name,
                "eval_dir": str(eval_dir),
                "n_tiles": summary["n_tiles"],
                "n_with_gt": summary["n_with_gt"],
                "n_empty_gt": summary["n_empty_gt"],
                "n_empty_clean": summary["n_empty_clean"],
                "n_empty_false_positives": summary["n_empty_false_positives"],
                "recall_all": summary["means_all_tiles"].get("recall"),
                "recall_with_gt": summary["means_with_gt_only"].get("recall"),
                "precision_all": summary["means_all_tiles"].get("precision"),
                "precision_with_gt": summary["means_with_gt_only"].get("precision"),
                "coco_AP50_with_gt": summary["means_with_gt_only"].get("coco_AP50"),
                "coco_AR100_with_gt": summary["means_with_gt_only"].get("coco_AR100"),
            }
        )
        print(f"\n{name}:")
        print(
            f"  tiles={summary['n_tiles']} with_gt={summary['n_with_gt']} "
            f"empty={summary['n_empty_gt']} "
            f"(clean={summary['n_empty_clean']}, fp={summary['n_empty_false_positives']})"
        )
        print(
            f"  recall all={summary['means_all_tiles'].get('recall'):.3f} "
            f"→ with_gt={summary['means_with_gt_only'].get('recall'):.3f}"
        )
        print(
            f"  AP50 with_gt={summary['means_with_gt_only'].get('coco_AP50'):.2f} "
            f"AR100 with_gt={summary['means_with_gt_only'].get('coco_AR100'):.2f}"
        )

    df = pd.DataFrame(comparison)
    out_csv = args.output_dir / "holdout_quality_comparison.csv"
    df.to_csv(out_csv, index=False)
    out_json = args.output_dir / "holdout_quality_comparison.json"
    out_json.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
