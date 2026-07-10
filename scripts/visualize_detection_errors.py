#!/usr/bin/env python3
"""Visualize true positives, false positives, and false negatives on ortho tiles.

Uses existing COCO prediction JSON or run_tile_inference summary JSON (no model
inference). Matches predictions to ground truth with greedy mask IoU (COCO-style,
default IoU >= 0.5). Inference summaries use bbox rectangles as prediction masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pycocotools import mask as mask_utils


def filter_coco_annotations(coco: dict, exclude_category_ids: set[int]) -> dict:
    if not exclude_category_ids:
        return coco
    filtered = dict(coco)
    filtered["annotations"] = [
        ann for ann in coco.get("annotations", []) if ann.get("category_id") not in exclude_category_ids
    ]
    return filtered


def category_ids_by_name(coco: dict, exclude_classes: set[str]) -> set[int]:
    if not exclude_classes:
        return set()
    return {
        cat["id"]
        for cat in coco.get("categories", [])
        if cat.get("name") in exclude_classes
    }


def stem_from_pred_path(pred_path: Path) -> str:
    name = pred_path.name
    for suffix in ("_detections.coco.json", "_inference_summary.json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return pred_path.stem


def bbox_xyxy_to_seg(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def inference_summary_to_coco(summary: dict, image_name: str, width: int, height: int) -> dict:
    annotations = []
    for det in summary.get("detections", []):
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        annotations.append(
            {
                "id": det.get("id", len(annotations) + 1),
                "image_id": 1,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": [bbox_xyxy_to_seg(bbox)],
                "score": float(det["score"]),
            }
        )
    return {
        "images": [{"id": 1, "file_name": image_name, "width": width, "height": height}],
        "categories": [{"id": 1, "name": "Boulder"}],
        "annotations": annotations,
    }


def load_pred_coco(pred_path: Path, gt_coco: dict, image_name: str) -> dict:
    data = json.loads(pred_path.read_text())
    if pred_path.name.endswith("_inference_summary.json"):
        gt_image = next(img for img in gt_coco["images"] if img["file_name"] == image_name)
        return inference_summary_to_coco(
            data, image_name, gt_image["width"], gt_image["height"]
        )
    return data


def collect_pred_files(predictions_dir: Path, tiles: list[str] | None) -> list[Path]:
    files = sorted(
        predictions_dir.glob("*_detections.coco.json")
    ) + sorted(predictions_dir.glob("*_inference_summary.json"))
    if not tiles:
        return files
    wanted = set(tiles)
    return [p for p in files if stem_from_pred_path(p) in wanted]


def load_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read()
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        rgb = np.stack([arr[0]] * 3, axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def seg_to_mask(segmentation: list | dict, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        return mask_utils.decode(segmentation)
    rles = mask_utils.frPyObjects(segmentation, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / union)


def greedy_match(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    pred_scores: list[float],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Return (matches as gt_idx, pred_idx, iou), unmatched_gt, unmatched_pred."""
    order = sorted(range(len(pred_masks)), key=lambda i: pred_scores[i], reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for pred_idx in order:
        best_gt = -1
        best_iou = 0.0
        for gt_idx, gt_mask in enumerate(gt_masks):
            if gt_idx in matched_gt:
                continue
            iou = mask_iou(gt_mask, pred_masks[pred_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            matches.append((best_gt, pred_idx, best_iou))
            matched_gt.add(best_gt)
            matched_pred.add(pred_idx)

    unmatched_gt = [i for i in range(len(gt_masks)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_masks)) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def contour_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def draw_mask_set(
    image: np.ndarray,
    masks: list[np.ndarray],
    color: tuple[int, int, int],
    labels: list[str],
    title: str,
) -> np.ndarray:
    out = image.copy()
    for mask, label in zip(masks, labels):
        contours = contour_from_mask(mask)
        if not contours:
            continue
        cv2.drawContours(out, contours, -1, color, 2)
        overlay = out.copy()
        cv2.drawContours(overlay, contours, -1, color, thickness=-1)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        ys, xs = np.where(mask)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(
                out,
                label,
                (max(4, cx - 30), max(14, cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        out,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        1,
        cv2.LINE_AA,
    )
    return out


def analyze_tile(
    image_path: Path,
    gt_coco: dict,
    pred_coco: dict,
    output_dir: Path,
    iou_threshold: float,
) -> dict:
    image_name = image_path.name
    gt_image = next(img for img in gt_coco["images"] if img["file_name"] == image_name)
    pred_image = next(
        (img for img in pred_coco["images"] if img["file_name"] == image_name),
        gt_image,
    )
    height, width = gt_image["height"], gt_image["width"]

    gt_anns = [a for a in gt_coco["annotations"] if a["image_id"] == gt_image["id"]]
    pred_anns = [a for a in pred_coco["annotations"] if a["image_id"] == pred_image["id"]]

    rgb = load_rgb(image_path)
    gt_masks = [seg_to_mask(a["segmentation"], height, width) for a in gt_anns]
    pred_masks = [seg_to_mask(a["segmentation"], height, width) for a in pred_anns]
    pred_scores = [float(a.get("score", 0.0)) for a in pred_anns]

    matches, unmatched_gt, unmatched_pred = greedy_match(
        gt_masks, pred_masks, pred_scores, iou_threshold
    )

    tp_masks: list[np.ndarray] = []
    tp_labels: list[str] = []
    for gt_idx, pred_idx, iou in matches:
        tp_masks.append(pred_masks[pred_idx])
        score = pred_scores[pred_idx]
        tp_labels.append(f"GT{gt_anns[gt_idx]['id']} IoU={iou:.2f} s={score:.2f}")

    fp_masks = [pred_masks[i] for i in unmatched_pred]
    fp_labels = [f"P{pred_anns[i].get('id', i)} s={pred_scores[i]:.2f}" for i in unmatched_pred]

    fn_masks = [gt_masks[i] for i in unmatched_gt]
    fn_labels = [f"GT{gt_anns[i]['id']}" for i in unmatched_gt]

    stem = image_path.stem
    tp_path = output_dir / f"{stem}_true_positives.jpg"
    fp_path = output_dir / f"{stem}_false_positives.jpg"
    fn_path = output_dir / f"{stem}_false_negatives.jpg"

    tp_img = draw_mask_set(
        rgb,
        tp_masks,
        (60, 220, 80),
        tp_labels,
        f"True positives ({len(tp_masks)})",
    )
    fp_img = draw_mask_set(
        rgb,
        fp_masks,
        (60, 80, 255),
        fp_labels,
        f"False positives ({len(fp_masks)})",
    )
    fn_img = draw_mask_set(
        rgb,
        fn_masks,
        (40, 220, 255),
        fn_labels,
        f"False negatives / missed ({len(fn_masks)})",
    )

    cv2.imwrite(str(tp_path), cv2.cvtColor(tp_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(fp_path), cv2.cvtColor(fp_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(fn_path), cv2.cvtColor(fn_img, cv2.COLOR_RGB2BGR))

    recall = len(matches) / len(gt_anns) if gt_anns else 0.0
    precision = len(matches) / len(pred_anns) if pred_anns else 0.0

    return {
        "image": str(image_path),
        "ground_truth_count": len(gt_anns),
        "prediction_count": len(pred_anns),
        "iou_threshold": iou_threshold,
        "true_positives": len(matches),
        "false_positives": len(unmatched_pred),
        "false_negatives": len(unmatched_gt),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_positives_image": str(tp_path),
        "false_positives_image": str(fp_path),
        "false_negatives_image": str(fn_path),
        "matches": [
            {
                "gt_id": gt_anns[gt_idx]["id"],
                "pred_score": pred_scores[pred_idx],
                "iou": round(iou, 4),
            }
            for gt_idx, pred_idx, iou in matches
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-json",
        type=Path,
        default=Path("segmentation/coco_dataset/testing_annotations.json"),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory with {stem}_detections.coco.json or {stem}_inference_summary.json.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("segmentation/tiling"),
        help="Ortho tile directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--tiles",
        nargs="*",
        default=None,
        help="Optional tile stems (e.g. 25IniSouthOrt_05_34). Default: all preds in dir.",
    )
    parser.add_argument(
        "--exclude-classes",
        type=str,
        default="",
        help="Comma-separated GT category names to omit (e.g. 'BoulderDeposit').",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_coco = json.loads(args.gt_json.read_text())
    exclude_classes = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}
    exclude_category_ids = category_ids_by_name(gt_coco, exclude_classes)
    gt_coco = filter_coco_annotations(gt_coco, exclude_category_ids)

    pred_files = collect_pred_files(args.predictions_dir, args.tiles)
    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files found in {args.predictions_dir} "
            "(expected *_detections.coco.json or *_inference_summary.json)"
        )

    summaries = []
    for pred_path in pred_files:
        stem = stem_from_pred_path(pred_path)
        image_path = args.image_dir / f"{stem}.tif"
        if not image_path.exists():
            image_path = args.gt_json.parent / "test" / f"{stem}.tif"
        if not image_path.exists():
            raise FileNotFoundError(f"Ortho tile not found for {stem}")

        pred_coco = load_pred_coco(pred_path, gt_coco, image_path.name)
        summary = analyze_tile(
            image_path,
            gt_coco,
            pred_coco,
            args.output_dir,
            args.iou_threshold,
        )
        summary["predictions_json"] = str(pred_path)
        summary["prediction_format"] = (
            "inference_summary" if pred_path.name.endswith("_inference_summary.json") else "coco"
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    report = {
        "gt_json": str(args.gt_json),
        "predictions_dir": str(args.predictions_dir),
        "iou_threshold": args.iou_threshold,
        "excluded_classes": sorted(exclude_classes),
        "tiles": summaries,
        "totals": {
            "ground_truth": sum(s["ground_truth_count"] for s in summaries),
            "predictions": sum(s["prediction_count"] for s in summaries),
            "true_positives": sum(s["true_positives"] for s in summaries),
            "false_positives": sum(s["false_positives"] for s in summaries),
            "false_negatives": sum(s["false_negatives"] for s in summaries),
        },
    }
    report_path = args.output_dir / "error_analysis_summary.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
