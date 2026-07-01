#!/usr/bin/env python3
"""Tile inference at a score threshold, DSM volume filter, and valid/invalid visualization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_volume_extraction import extract_volumes, load_raster, merge_objects  # noqa: E402


def load_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read()
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        rgb = np.stack([arr[0]] * 3, axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def mask_to_segmentation(mask: np.ndarray) -> list[float]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return []
    pts = contour.reshape(-1, 2).astype(float)
    seg: list[float] = []
    for x, y in pts:
        seg.extend([float(x), float(y)])
    return seg


def poly_area(seg: list[float]) -> float:
    xs = seg[0::2]
    ys = seg[1::2]
    area = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def instances_to_coco_annotations(instances: Instances, image_id: int = 1) -> list[dict]:
    annotations: list[dict] = []
    if len(instances) == 0:
        return annotations
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy()
    for idx, (box, score, mask) in enumerate(zip(boxes, scores, masks), start=1):
        seg = mask_to_segmentation(mask)
        if len(seg) < 6:
            continue
        x0, y0, x1, y1 = box.tolist()
        annotations.append(
            {
                "id": idx,
                "image_id": image_id,
                "category_id": 1,
                "category_name": "Boulder",
                "segmentation": [seg],
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "area": poly_area(seg),
                "score": float(score),
                "iscrowd": 0,
            }
        )
    return annotations


def draw_merged_on_rgb(rgb: np.ndarray, merged: list[dict]) -> np.ndarray:
    out = rgb.copy()
    for obj in merged:
        seg = obj["seg"]
        pts = np.round(np.asarray(seg, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
        if len(pts) < 3:
            continue
        cv2.fillPoly(out, [pts], (48, 48, 48))
        cv2.polylines(out, [pts], True, (0, 220, 255), 2)
    return out


def label_panel(image: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(
        out,
        title,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def run_pipeline(
    image: Path,
    model: Path,
    dsm: Path,
    output_dir: Path,
    score_thresh: float,
    device: str,
    hillshade_image: Path | None = None,
    ortho_image: Path | None = None,
    merge_overlap: float = 0.35,
    merge_min_area_ratio: float = 0.25,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(image)
    ortho_path = ortho_image or image
    ortho_rgb = load_rgb(ortho_path)
    h, w = rgb.shape[:2]

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = str(model)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device
    cfg.INPUT.MAX_SIZE_TEST = 2000
    cfg.INPUT.MIN_SIZE_TEST = 2000
    cfg.TEST.DETECTIONS_PER_IMAGE = 300

    predictor = DefaultPredictor(cfg)
    instances = predictor(rgb[:, :, ::-1])["instances"].to("cpu")

    annotations = instances_to_coco_annotations(instances)
    coco = {
        "images": [{"id": 1, "file_name": image.name, "width": w, "height": h}],
        "categories": [{"id": 1, "name": "Boulder"}],
        "annotations": annotations,
    }
    stem = image.stem
    coco_path = output_dir / f"{stem}_detections.coco.json"
    coco_path.write_text(json.dumps(coco))

    ortho = load_raster(ortho_path, bands=[1, 2, 3])
    dsm_raster = load_raster(dsm, bands=[1])
    merged = merge_objects(
        annotations,
        overlap_threshold=merge_overlap,
        overlap_num_threshold=1,
        area_threshold=0,
        min_area_ratio=merge_min_area_ratio,
    )
    volume_df = extract_volumes(ortho, dsm_raster, merged)

    valid_merged: list[dict] = []
    invalid_merged: list[dict] = []
    if not volume_df.empty:
        for obj, is_boulder in zip(merged, volume_df["isboulder"].tolist()):
            if int(is_boulder) == 1:
                valid_merged.append(obj)
            else:
                invalid_merged.append(obj)

    panel_all = label_panel(draw_merged_on_rgb(rgb, merged), f"All detections ({len(merged)})")
    panel_valid = label_panel(
        draw_merged_on_rgb(rgb, valid_merged), f"DSM-valid boulders ({len(valid_merged)})"
    )
    panel_invalid = label_panel(
        draw_merged_on_rgb(rgb, invalid_merged), f"DSM-rejected ({len(invalid_merged)})"
    )
    comparison = np.hstack([panel_all, panel_valid, panel_invalid])

    compare_path = output_dir / f"{stem}_volume_filter_comparison.jpg"
    ortho_compare_path = None
    if ortho_image and ortho_image != image:
        ortho_panel = label_panel(
            draw_merged_on_rgb(ortho_rgb, merged),
            f"Ortho + all detections ({len(merged)})",
        )
        ortho_compare_path = output_dir / f"{stem}_ortho_all_detections.jpg"
        cv2.imwrite(str(ortho_compare_path), cv2.cvtColor(ortho_panel, cv2.COLOR_RGB2BGR))
    csv_path = output_dir / f"{stem}_volume_results.csv"
    cv2.imwrite(str(compare_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    if not volume_df.empty:
        volume_df.to_csv(csv_path, index=False)

    hillshade_compare_path = None
    if hillshade_image and hillshade_image.exists():
        hillshade_rgb = load_rgb(hillshade_image)
        hs_panel = label_panel(
            draw_merged_on_rgb(hillshade_rgb, merged),
            f"Hillshade + all ({len(merged)})",
        )
        hillshade_compare_path = output_dir / f"{stem}_hillshade_all_detections.jpg"
        cv2.imwrite(str(hillshade_compare_path), cv2.cvtColor(hs_panel, cv2.COLOR_RGB2BGR))

    summary = {
        "image": str(image),
        "ortho_image": str(ortho_path),
        "model": str(model),
        "dsm": str(dsm),
        "score_thresh": score_thresh,
        "raw_instance_detections": len(instances),
        "coco_annotations": len(annotations),
        "merged_objects": len(merged),
        "dsm_valid": len(valid_merged),
        "dsm_rejected": len(invalid_merged),
        "merge_overlap": merge_overlap,
        "merge_min_area_ratio": merge_min_area_ratio,
        "detections_coco_json": str(coco_path),
        "volume_csv": str(csv_path) if not volume_df.empty else None,
        "comparison_image": str(compare_path),
        "ortho_overlay_image": str(ortho_compare_path) if ortho_compare_path else None,
        "hillshade_image": str(hillshade_compare_path) if hillshade_compare_path else None,
    }
    summary_path = output_dir / f"{stem}_volume_viz_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dsm", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hillshade-image", type=Path, default=None)
    parser.add_argument(
        "--ortho-image",
        type=Path,
        default=None,
        help="Matching ortho tile for DSM volume extraction when --image is hillshade.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument(
        "--merge-overlap",
        type=float,
        default=0.35,
        help="Min intersection/min-area IoU to merge two detections (default 0.35).",
    )
    parser.add_argument(
        "--merge-min-area-ratio",
        type=float,
        default=0.25,
        help="Min area ratio between pair to merge; blocks small-on-large absorption (default 0.25).",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = run_pipeline(
        args.image,
        args.model,
        args.dsm,
        args.output_dir,
        args.score_thresh,
        args.device,
        args.hillshade_image,
        args.ortho_image,
        args.merge_overlap,
        args.merge_min_area_ratio,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
