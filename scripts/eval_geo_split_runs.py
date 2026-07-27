#!/usr/bin/env python3
# RUNS ON: Windows (needs checkpoints + COCO data + image tiles)
"""Batch ``eval_per_tile`` across all discovered geo512 (and optional parent) runs.

Discovers ``training_run_geo512_{setup}_{modality}/`` via glob (no hardcoded
matrix). For each run with a checkpoint + matching COCO dir, runs the shared
``run_per_tile_evaluation`` logic (per-tile + overall dual size buckets) and
writes results under::

    <output-dir>/<setup>__<modality>__<tiling>/<checkpoint_stem>/<split>/

Resumable: skips a run when ``overall_metrics.json`` already exists unless
``--force``.

Example::

    python BoulderCalculator/scripts/eval_geo_split_runs.py \\
      --segmentation-dir segmentation \\
      --output-dir segmentation/eval_geo512_all \\
      --split test --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_per_tile import run_per_tile_evaluation  # noqa: E402
from geo_split_run_index import (  # noqa: E402
    discover_geo512_runs,
    discover_parent_geo_runs,
    glob_report,
    prefer_checkpoint,
    resolve_dataset_dir,
)


def _done_marker(run_out: Path, split: str) -> Path:
    return run_out / split / "overall_metrics.json"


def _build_args(
    *,
    dataset_dir: Path,
    model: Path,
    split: str,
    output_dir: Path,
    device: str,
    four_band: bool,
    image_size: int,
    score_thresh: float,
    merge_iou: float | None,
    match_iou: float,
    iou_type: str,
    require_gt: bool,
    class_names: str,
) -> Namespace:
    return Namespace(
        gt_json=None,
        dataset_dir=dataset_dir,
        split=split,
        splits=None,
        image_dir=[],
        skip_missing=True,
        require_gt=require_gt,
        predictions_dir=None,
        model=model,
        device=device,
        four_band=four_band,
        score_thresh=score_thresh,
        image_size=image_size,
        class_names=class_names,
        merge_iou=merge_iou,
        match_iou=match_iou,
        iou_type=iou_type,
        output_dir=output_dir,
        extents=None,
        split_config=[],
        difficulty_split=None,
        overall_metrics=True,
    )


def _eval_one(
    run,
    *,
    seg: Path,
    output_root: Path,
    split: str,
    device: str,
    score_thresh: float,
    merge_iou: float | None,
    match_iou: float,
    iou_type: str,
    require_gt: bool,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    dataset = resolve_dataset_dir(run, seg)
    ckpt = run.checkpoint or prefer_checkpoint(run.run_dir)
    status: dict[str, Any] = {
        "setup": run.setup,
        "modality": run.modality,
        "tiling_regime": run.tiling_regime,
        "run_dir": str(run.run_dir),
        "dataset_dir": str(dataset) if dataset else None,
        "checkpoint": str(ckpt) if ckpt else None,
    }
    if dataset is None:
        status["status"] = "skip_no_dataset"
        return status
    if ckpt is None:
        status["status"] = "skip_no_checkpoint"
        return status

    image_size = run.image_size
    if image_size is None:
        image_size = 512 if run.tiling_regime == "512" else 2000

    run_key = f"{run.setup}__{run.modality}__{run.tiling_regime}"
    run_out = output_root / run_key / ckpt.stem
    marker = _done_marker(run_out, split)
    status["output_dir"] = str(run_out / split)
    if marker.is_file() and not force:
        status["status"] = "skip_exists"
        status["marker"] = str(marker)
        return status

    if dry_run:
        status["status"] = "dry_run"
        return status

    args = _build_args(
        dataset_dir=dataset,
        model=ckpt,
        split=split,
        output_dir=run_out / split,
        device=device,
        four_band=run.four_band,
        image_size=int(image_size),
        score_thresh=score_thresh,
        merge_iou=merge_iou,
        match_iou=match_iou,
        iou_type=iou_type,
        require_gt=require_gt,
        class_names="Boulder",
    )
    print(
        f"\n##### {run_key}  ckpt={ckpt.name}  four_band={run.four_band} "
        f"image_size={image_size} #####"
    )
    rows, extras = run_per_tile_evaluation(
        args,
        split,
        run_out / split,
        overall_metrics=True,
        image_size_for_buckets=int(image_size),
    )
    summary = {
        "setup": run.setup,
        "modality": run.modality,
        "tiling_regime": run.tiling_regime,
        "checkpoint": str(ckpt),
        "checkpoint_stem": ckpt.stem,
        "dataset_dir": str(dataset),
        "split": split,
        "n_tiles": len(rows),
        "image_size": int(image_size),
        "four_band": run.four_band,
        "overall": extras.get("overall_dual_buckets"),
        "run_provenance_flags": (run.provenance or {}).get("flags"),
    }
    summary_path = run_out / split / "run_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    status["status"] = "ok"
    status["summary"] = str(summary_path)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=Path("segmentation"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmentation/eval_geo512_all"),
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "test"],
        help="COCO split to evaluate (default: test).",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--merge-iou", type=float, default=None)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--iou-type", choices=["bbox", "segm"], default="bbox")
    parser.add_argument("--require-gt", action="store_true")
    parser.add_argument(
        "--include-parent-2000",
        action="store_true",
        help="Also evaluate training_run_geo_* (2000-tile) runs.",
    )
    parser.add_argument(
        "--setups",
        default="",
        help="Optional comma filter (default: all discovered).",
    )
    parser.add_argument(
        "--modalities",
        default="",
        help="Optional comma filter (default: all discovered).",
    )
    parser.add_argument("--force", action="store_true", help="Re-eval even if outputs exist.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and print plan only (no inference).",
    )
    args = parser.parse_args()

    seg = args.segmentation_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    inventory = glob_report(seg)
    (out / "disk_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Inventory: geo512_runs={inventory['n_geo512_runs']} "
        f"coco={inventory['n_geo512_coco']} parent={inventory['n_parent_runs']}"
    )

    runs = discover_geo512_runs(seg)
    if args.include_parent_2000:
        runs.extend(discover_parent_geo_runs(seg))

    setup_filter = {s.strip() for s in args.setups.split(",") if s.strip()}
    mod_filter = {m.strip() for m in args.modalities.split(",") if m.strip()}
    if setup_filter:
        runs = [r for r in runs if r.setup in setup_filter]
    if mod_filter:
        runs = [r for r in runs if r.modality in mod_filter]

    if not runs:
        print(
            "WARNING: no training runs discovered. "
            "Confirm naming training_run_geo512_{setup}_{modality} on Windows."
        )
        return

    results = []
    for run in runs:
        results.append(
            _eval_one(
                run,
                seg=seg,
                output_root=out,
                split=args.split,
                device=args.device,
                score_thresh=args.score_thresh,
                merge_iou=args.merge_iou,
                match_iou=args.match_iou,
                iou_type=args.iou_type,
                require_gt=args.require_gt,
                force=args.force,
                dry_run=args.dry_run,
            )
        )

    summary_path = out / "batch_eval_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nWrote {summary_path}")
    print("Status counts:", counts)


if __name__ == "__main__":
    main()
