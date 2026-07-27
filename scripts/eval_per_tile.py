#!/usr/bin/env python3
"""Per-tile AP / AR / precision / recall evaluation with heatmaps.

Two prediction sources
----------------------
1. **Existing predictions** (fast): a directory of ``*_detections.coco.json``
   or ``*_inference_summary.json`` from ``run_tile_inference.py``.
2. **Model inference** (slower): run a Detectron2 checkpoint on every image in
   a COCO split (``--model`` + ``--dataset-dir`` + ``--split``).

Merging overlapping detections
------------------------------
Sliding-window / multi-pass inference can emit several boxes for one boulder.
Pass ``--merge-iou 0.4`` to NMS-merge within each tile before scoring, or omit
it to score raw detections. You can run both modes and compare.

Geographic difficulty check
---------------------------
After scoring a common tile pool, pass several ``--split-config`` YAMLs. The
script reports mean per-tile AP50/AR100 on each setup's test tiles — useful to
see whether the baseline hold-out is intrinsically easier than alternate splits.

Examples
--------
::

    # From existing tile predictions
    python BoulderCalculator/scripts/eval_per_tile.py \\
      --gt-json segmentation/coco_geo_baseline/testing_annotations.json \\
      --predictions-dir segmentation/visualizations/test_inference_full \\
      --output-dir segmentation/eval_per_tile_baseline_test \\
      --extents BoulderCalculator/experiments/geo_splits/tile_extents_baseline.geojson

    # With model (RGB+DSM), test split, with and without merge
    python BoulderCalculator/scripts/eval_per_tile.py \\
      --dataset-dir segmentation/coco_geo_baseline_rgb_dsm \\
      --split test --model segmentation/training_run_geo_baseline/model_final.pth \\
      --four-band --device cuda --merge-iou 0.4 \\
      --output-dir segmentation/eval_per_tile_baseline_test_merged \\
      --split-config BoulderCalculator/experiments/geo_splits/baseline.yaml \\
      --split-config BoulderCalculator/experiments/geo_splits/blocks_alt_a.yaml \\
      --split-config BoulderCalculator/experiments/geo_splits/blocks_alt_b.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (  # noqa: E402
    collect_prediction_map,
    evaluate_coco_split_per_tile,
    filter_coco_to_existing_images,
    load_coco,
    load_prediction_file,
    load_split_yaml,
    plot_tile_heatmaps,
    resolve_gt_annotations,
    resolve_image_file,
    rows_to_geojson,
    summarize_by_split_membership,
    write_holdout_quality_reports,
)


def _build_predictor(args):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    from multiband_io import FOUR_BAND_PIXEL_MEAN, FOUR_BAND_PIXEL_STD

    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = str(args.model)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.DEVICE = args.device
    cfg.INPUT.MAX_SIZE_TEST = args.image_size
    cfg.INPUT.MIN_SIZE_TEST = args.image_size
    cfg.INPUT.FORMAT = "BGR"
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    if args.four_band:
        cfg.MODEL.PIXEL_MEAN = FOUR_BAND_PIXEL_MEAN
        cfg.MODEL.PIXEL_STD = FOUR_BAND_PIXEL_STD
    return DefaultPredictor(cfg), class_names


def _load_image_for_model(path: Path, four_band: bool) -> np.ndarray:
    if four_band:
        from multiband_io import load_bgrd_uint8

        return load_bgrd_uint8(path)
    import rasterio

    with rasterio.open(path) as ds:
        arr = ds.read()
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        rgb = np.stack([arr[0]] * 3, axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb[:, :, ::-1]  # BGR


def _band_count(path: Path) -> int:
    import rasterio

    with rasterio.open(path) as ds:
        return int(ds.count)


def predict_split(args, gt_coco: dict, image_dirs: list[Path]) -> dict[str, list[dict]]:
    predictor, class_names = _build_predictor(args)
    preds_by_stem: dict[str, list[dict]] = {}
    images = gt_coco["images"]
    print(
        f"Running inference on {len(images)} images "
        f"({args.device}, four_band={args.four_band}, image_dirs={len(image_dirs)})…"
    )
    for i, img in enumerate(images, start=1):
        path = resolve_image_file(img["file_name"], image_dirs)
        if path is None:
            print(f"  [{i}/{len(images)}] missing {img['file_name']}")
            if not args.skip_missing:
                preds_by_stem[Path(img["file_name"]).stem] = []
            continue
        if args.four_band:
            try:
                n_bands = _band_count(path)
            except Exception as exc:
                print(f"  [{i}/{len(images)}] unreadable {path}: {exc}")
                continue
            if n_bands < 4:
                print(
                    f"  [{i}/{len(images)}] skip {path.name}: {n_bands} bands "
                    f"(need 4; pass --image-dir tiling_rgb_dsm_*)"
                )
                continue
        image = _load_image_for_model(path, args.four_band)
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        dets = []
        if len(instances):
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            for box, score, cls_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(float, box.tolist())
                dets.append(
                    {
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "bbox_format": "xywh",
                        "score": float(score),
                        "category_id": int(cls_id) + 1,
                        "source": "coco",
                    }
                )
        stem = Path(img["file_name"]).stem
        preds_by_stem[stem] = dets
        if i % 10 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] {stem}: {len(dets)} dets")
    return preds_by_stem


def predictions_from_dir(predictions_dir: Path, gt_coco: dict) -> dict[str, list[dict]]:
    mapping = collect_prediction_map(predictions_dir)
    images = {img["file_name"]: img for img in gt_coco["images"]}
    # Also index by stem for flexible filename matching.
    by_stem_img = {Path(fn).stem: img for fn, img in images.items()}
    preds_by_stem: dict[str, list[dict]] = {}
    for stem, img in by_stem_img.items():
        # Try exact stem, or stem without year prefix / with year prefix variants.
        candidates = [stem]
        if stem.startswith(("24_", "25_")):
            # 24_Sites1and2_... → Sites1and2_...
            parts = stem.split("_", 1)
            if len(parts) == 2:
                candidates.append(parts[1])
        path = None
        for c in candidates:
            if c in mapping:
                path = mapping[c]
                break
            # Also try matching mapping keys that end with this stem.
            for k, p in mapping.items():
                if k == c or k.endswith(c) or c.endswith(k):
                    path = p
                    break
            if path:
                break
        if path is None:
            preds_by_stem[stem] = []
            continue
        preds_by_stem[stem] = load_prediction_file(
            path, img["file_name"], int(img["width"]), int(img["height"])
        )
    return preds_by_stem


def _filter_require_gt(gt_coco: dict) -> dict:
    """Keep images that have ≥1 non-crowd annotation."""
    trainable_ids = {
        int(a["image_id"])
        for a in gt_coco.get("annotations", [])
        if not a.get("iscrowd", 0)
    }
    images = [im for im in gt_coco.get("images", []) if int(im["id"]) in trainable_ids]
    keep = {int(im["id"]) for im in images}
    return {
        **gt_coco,
        "images": images,
        "annotations": [a for a in gt_coco.get("annotations", []) if int(a["image_id"]) in keep],
    }


def _resolve_image_dirs(args, gt_path: Path, split_name: str) -> list[Path]:
    image_dirs: list[Path] = list(args.image_dir)
    if image_dirs:
        return image_dirs

    if args.four_band:
        seg_root = None
        for base in (
            Path(args.dataset_dir).parent if args.dataset_dir else None,
            gt_path.parent.parent,
            Path("segmentation"),
        ):
            if base is None:
                continue
            base = Path(base)
            if base.name == "segmentation" and base.is_dir():
                seg_root = base
                break
            if (base / "segmentation").is_dir():
                seg_root = base / "segmentation"
                break
        if seg_root is not None:
            for year in ("24", "25"):
                td = seg_root / f"tiling_rgb_dsm_{year}"
                if td.is_dir():
                    image_dirs.append(td)
            # Prefer dataset split folder when it already has 4-band tiles.
            if args.dataset_dir is not None:
                split_dir = Path(args.dataset_dir) / split_name
                if split_dir.is_dir() and any(split_dir.glob("*.tif")):
                    image_dirs.insert(0, split_dir)
            if image_dirs:
                print(f"Using image dirs: {[str(d) for d in image_dirs]}")
                return image_dirs

    if args.dataset_dir is not None:
        split_dir = Path(args.dataset_dir) / split_name
        if split_dir.is_dir() and any(split_dir.iterdir()):
            image_dirs.append(split_dir)
        if (
            not args.four_band
            and gt_path.parent.resolve() != Path(args.dataset_dir).resolve()
        ):
            sib_split = gt_path.parent / split_name
            if sib_split.is_dir():
                image_dirs.append(sib_split)
    if not image_dirs:
        parent = gt_path.parent
        split_dir = parent / split_name
        if split_dir.is_dir():
            image_dirs.append(split_dir)
        elif not args.four_band:
            for cand in ("test", "valid", "train"):
                if (parent / cand).is_dir():
                    image_dirs.append(parent / cand)
        if not image_dirs:
            image_dirs.append(parent)
    return image_dirs


def run_per_tile_evaluation(
    args,
    split_name: str,
    out_dir: Path,
    *,
    overall_metrics: bool = False,
    gsd_m: float | None = None,
    image_size_for_buckets: int | None = None,
) -> tuple[list[dict], dict]:
    """Core per-tile (+ optional whole-split) evaluation. Importable by batch drivers.

    Returns ``(per_tile_rows, extras)`` where ``extras`` may contain
    ``overall_dual_buckets`` when ``overall_metrics`` is True.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Split: {split_name} → {out_dir} ===")
    extras: dict = {}

    try:
        gt_path = resolve_gt_annotations(args.dataset_dir, split_name, args.gt_json)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    # When looping multiple splits, --gt-json only applies to the first/sole split.
    if args.gt_json is not None and args.splits and "," in (args.splits or ""):
        # Force dataset-dir resolution for subsequent splits.
        if args.dataset_dir is None:
            raise SystemExit("Pass --dataset-dir when using --splits with multiple values.")
        gt_path = resolve_gt_annotations(args.dataset_dir, split_name, None)

    gt_coco = load_coco(gt_path)
    if args.require_gt:
        before = len(gt_coco.get("images", []))
        gt_coco = _filter_require_gt(gt_coco)
        print(
            f"require-gt: kept {len(gt_coco['images'])}/{before} tiles with trainable boulders"
        )
        if not gt_coco["images"]:
            raise SystemExit(f"No trainable-GT tiles in {split_name}.")

    image_dirs = _resolve_image_dirs(args, gt_path, split_name)

    if args.model is not None and args.skip_missing and image_dirs:
        before = len(gt_coco.get("images", []))
        gt_coco, missing = filter_coco_to_existing_images(gt_coco, image_dirs)
        if missing:
            print(
                f"Skipping {len(missing)}/{before} tiles with no image under "
                f"{[str(d) for d in image_dirs]}"
            )
        if not gt_coco.get("images"):
            raise SystemExit(
                "No images found for scoring. Pass --image-dir or use a dataset "
                "dir whose split folders contain GeoTIFFs."
            )

    if args.model is not None:
        if not image_dirs:
            raise SystemExit("--model requires a resolvable --image-dir / dataset split folder.")
        preds_by_stem = predict_split(args, gt_coco, image_dirs)
        if args.skip_missing:
            keep = set(preds_by_stem)
            before = len(gt_coco.get("images", []))
            gt_coco = {
                **gt_coco,
                "images": [im for im in gt_coco["images"] if Path(im["file_name"]).stem in keep],
            }
            keep_ids = {int(im["id"]) for im in gt_coco["images"]}
            gt_coco["annotations"] = [
                a for a in gt_coco.get("annotations", []) if int(a["image_id"]) in keep_ids
            ]
            dropped = before - len(gt_coco["images"])
            if dropped:
                print(f"Scoring {len(gt_coco['images'])} tiles ({dropped} skipped).")
            if not gt_coco["images"]:
                raise SystemExit("No tiles scored after missing-image filter.")
        pred_out = out_dir / "predictions"
        pred_out.mkdir(exist_ok=True)
        for stem, dets in preds_by_stem.items():
            (pred_out / f"{stem}_detections.coco.json").write_text(
                json.dumps(
                    {
                        "annotations": [
                            {**d, "id": i, "image_id": 1}
                            for i, d in enumerate(dets, start=1)
                        ]
                    },
                    indent=2,
                )
            )
    elif args.predictions_dir is not None:
        preds_by_stem = predictions_from_dir(args.predictions_dir, gt_coco)
    else:
        raise SystemExit("Pass --predictions-dir or --model.")

    rows = evaluate_coco_split_per_tile(
        gt_coco,
        preds_by_stem,
        iou_type=args.iou_type,
        match_iou=args.match_iou,
        merge_iou=args.merge_iou,
    )
    for r in rows:
        r["split"] = split_name

    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_tile_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} tiles)")

    agg_cols = [
        c
        for c in (
            "coco_AP",
            "coco_AP50",
            "coco_AR100",
            "precision",
            "recall",
            "f1",
            "gt_count",
            "pred_count",
        )
        if c in df.columns
    ]
    print("\nPer-tile means (all tiles):")
    print(df[agg_cols].mean(numeric_only=True).to_string())
    with_gt = df[df["gt_count"] > 0] if "gt_count" in df.columns else df.iloc[0:0]
    if len(with_gt):
        print(f"\nPer-tile means (with GT only, n={len(with_gt)}):")
        print(with_gt[agg_cols].mean(numeric_only=True).to_string())

    hq_paths = write_holdout_quality_reports(rows, out_dir, split_label=split_name)
    for label, path in hq_paths.items():
        print(f"Wrote {path}")

    prefix = f"{split_name} "
    if args.merge_iou is not None:
        prefix = f"{split_name} merged "
    figs = plot_tile_heatmaps(
        rows,
        metrics=("coco_AP50", "coco_AR100", "recall", "precision"),
        title_prefix=prefix,
    )
    for metric, fig in figs:
        out = out_dir / f"heatmap_{split_name}_{metric}.png"
        fig.savefig(out, dpi=140)
        fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=140)
        plt.close(fig)
        print(f"Wrote {out}")

    # With-GT-only heatmaps (easier to parse when many empty tiles).
    rows_gt = [r for r in rows if int(r.get("gt_count") or 0) > 0]
    if rows_gt:
        figs_gt = plot_tile_heatmaps(
            rows_gt,
            metrics=("coco_AP50", "coco_AR100", "recall", "precision"),
            title_prefix=f"{split_name} with-GT ",
        )
        for metric, fig in figs_gt:
            out = out_dir / f"heatmap_{split_name}_with_gt_{metric}.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            print(f"Wrote {out}")

    if args.extents is not None and args.extents.exists():
        gj = rows_to_geojson(rows, args.extents)
        gj_path = out_dir / "per_tile_metrics.geojson"
        gj_path.write_text(json.dumps(gj))
        print(f"Wrote {gj_path}")

    difficulty = {}
    membership = args.difficulty_split if args.difficulty_split else split_name
    for cfg_path in args.split_config:
        cfg = load_split_yaml(cfg_path)
        setup_id = cfg.get("id") or Path(cfg_path).stem
        difficulty[setup_id] = {
            "config": str(cfg_path),
            "membership_split": membership,
            "coco_AP50": summarize_by_split_membership(
                rows, cfg, metric="coco_AP50", membership_split=membership
            ),
            "coco_AR100": summarize_by_split_membership(
                rows, cfg, metric="coco_AR100", membership_split=membership
            ),
            "recall": summarize_by_split_membership(
                rows, cfg, metric="recall", membership_split=membership
            ),
        }
    if difficulty:
        diff_path = out_dir / "split_difficulty_summary.json"
        diff_path.write_text(json.dumps(difficulty, indent=2))
        print(f"\nSplit difficulty (membership={membership}):")
        for setup_id, block in difficulty.items():
            ap = block["coco_AP50"]
            ar = block["coco_AR100"]
            print(
                f"  {setup_id:20s}  n={ap['n']:3d}  "
                f"AP50={ap['mean']:.2f}±{ap['std']:.2f}  "
                f"AR100={ar['mean']:.2f}±{ar['std']:.2f}"
            )
        print(f"Wrote {diff_path}")

    if overall_metrics:
        from eval_utils import evaluate_coco_split_dual_buckets
        from geo_split_run_index import DEFAULT_GSD_M, read_gsd_m, resolve_split_image

        gsd = gsd_m
        gsd_src = "cli"
        if gsd is None:
            gsd = DEFAULT_GSD_M
            gsd_src = "default_fallback"
            if args.dataset_dir is not None and gt_coco.get("images"):
                fn = gt_coco["images"][0]["file_name"]
                img_path = resolve_split_image(Path(args.dataset_dir), split_name, fn)
                if img_path is None and image_dirs:
                    from eval_utils import resolve_image_file

                    img_path = resolve_image_file(fn, image_dirs)
                if img_path is not None:
                    gsd, gsd_src = read_gsd_m(img_path)
        img_size = image_size_for_buckets
        if img_size is None:
            img_size = int(getattr(args, "image_size", 0) or 0) or None
        dual = evaluate_coco_split_dual_buckets(
            gt_coco,
            preds_by_stem,
            gsd_m=float(gsd),
            image_size=img_size,
            iou_type=args.iou_type,
            merge_iou=args.merge_iou,
        )
        dual["gsd_source"] = gsd_src
        extras["overall_dual_buckets"] = dual
        overall_path = out_dir / "overall_metrics.json"
        overall_path.write_text(json.dumps(dual, indent=2) + "\n")
        print(f"Wrote {overall_path}")
        g = dual["overall_ground_m2"]
        print(
            f"Overall (ground-m² buckets primary): "
            f"AP={g.get('AP', float('nan')):.2f} "
            f"AP50={g.get('AP50', float('nan')):.2f} "
            f"AR100={g.get('AR100', float('nan')):.2f} "
            f"ARs/m/l={g.get('ARs', float('nan')):.1f}/"
            f"{g.get('ARm', float('nan')):.1f}/{g.get('ARl', float('nan')):.1f}"
        )

    meta = {
        "split": split_name,
        "gt_images": len(gt_coco["images"]),
        "require_gt": bool(args.require_gt),
        "merge_iou": args.merge_iou,
        "match_iou": args.match_iou,
        "iou_type": args.iou_type,
        "device": args.device,
        "model": str(args.model) if args.model else None,
        "predictions_dir": str(args.predictions_dir) if args.predictions_dir else None,
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir else None,
        "gt_json": str(gt_path),
        "overall_metrics": bool(overall_metrics),
    }
    (out_dir / "eval_meta.json").write_text(json.dumps(meta, indent=2))
    return rows, extras


def _run_one_split(args, split_name: str, out_dir: Path) -> list[dict]:
    rows, _extras = run_per_tile_evaluation(
        args,
        split_name,
        out_dir,
        overall_metrics=bool(getattr(args, "overall_metrics", False)),
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-json", type=Path, default=None, help="COCO GT JSON (one split).")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="COCO dataset dir.")
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default=None,
        help="Single split under --dataset-dir (default: test if --splits unset).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help=(
            "Comma-separated splits to evaluate (e.g. test,valid). "
            "Writes each split under output-dir/<split>/ with separate heatmaps."
        ),
    )
    parser.add_argument(
        "--image-dir",
        action="append",
        default=[],
        type=Path,
        help="Image search dir (repeatable). Defaults to dataset split folder / tiling dirs.",
    )
    parser.add_argument(
        "--skip-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop tiles whose images are not found (default: true).",
    )
    parser.add_argument(
        "--require-gt",
        action="store_true",
        help="Only score tiles with ≥1 non-crowd GT boulder.",
    )
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None, help="Run inference with this checkpoint.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device. CPU works without a GPU (slower).",
    )
    parser.add_argument("--four-band", action="store_true")
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--image-size", type=int, default=2000)
    parser.add_argument("--class-names", default="Boulder")
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=None,
        help="If set, NMS-merge overlapping predictions per tile before scoring.",
    )
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU for precision/recall.")
    parser.add_argument("--iou-type", choices=["bbox", "segm"], default="bbox")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--extents",
        type=Path,
        default=None,
        help="tile_extents_*.geojson to join metrics onto for QGIS.",
    )
    parser.add_argument(
        "--split-config",
        action="append",
        default=[],
        help="Geo-split YAML (repeatable). Summarizes mean metrics on that setup's membership.",
    )
    parser.add_argument(
        "--difficulty-split",
        default=None,
        choices=["train", "valid", "test"],
        help="Membership split for --split-config (default: the split being evaluated).",
    )
    parser.add_argument(
        "--overall-metrics",
        action="store_true",
        help=(
            "Also write overall_metrics.json with whole-split AP/AR and "
            "size-bucket breakdown (ground-m² primary + pixel-area)."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.splits:
        split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    elif args.split:
        split_names = [args.split]
    elif args.gt_json is not None:
        split_names = ["test"]
    else:
        split_names = ["test"]

    all_rows: list[dict] = []
    multi = len(split_names) > 1
    for split_name in split_names:
        out_dir = args.output_dir / split_name if multi else args.output_dir
        # Clear gt-json after first split when multi so later splits use dataset-dir.
        rows = _run_one_split(args, split_name, out_dir)
        all_rows.extend(rows)
        if multi and args.gt_json is not None:
            args.gt_json = None

    if multi and all_rows:
        combined = args.output_dir / "per_tile_metrics_all_splits.csv"
        pd.DataFrame(all_rows).to_csv(combined, index=False)
        print(f"\nWrote combined {combined}")
        hq = write_holdout_quality_reports(all_rows, args.output_dir, split_label="all_splits")
        print(f"Wrote {hq['holdout_quality']}")


if __name__ == "__main__":
    main()
