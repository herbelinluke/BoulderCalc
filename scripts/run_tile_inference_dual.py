#!/usr/bin/env python3
"""Run inference with ortho + optional hillshade comparison views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


def load_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read()
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        gray = arr[0]
        lo, hi = np.percentile(gray, (2, 98))
        if hi > lo:
            gray = np.clip((gray - lo) / (hi - lo) * 255.0, 0, 255)
        rgb = np.stack([gray] * 3, axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def draw_gt(image_rgb: np.ndarray, coco_path: Path, image_name: str) -> np.ndarray:
    if not coco_path.exists():
        return image_rgb
    coco = json.loads(coco_path.read_text())
    image_id = next(img["id"] for img in coco["images"] if img["file_name"] == image_name)
    out = image_rgb.copy()
    for ann in coco["annotations"]:
        if ann["image_id"] != image_id:
            continue
        seg = ann["segmentation"][0]
        pts = np.round(np.asarray(seg, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 128), 2)
    return out


def draw_predictions(rgb: np.ndarray, instances) -> np.ndarray:
    visualizer = Visualizer(
        rgb,
        metadata={"thing_classes": ["Boulder"]},
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW,
    )
    return visualizer.draw_instance_predictions(instances).get_image()


def label_panel(image: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(
        out,
        title,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Tile used for inference (ortho for ortho-trained models).",
    )
    parser.add_argument(
        "--hillshade-image",
        type=Path,
        default=None,
        help="Matching hillshade tile for the comparison panel.",
    )
    parser.add_argument(
        "--ortho-image",
        type=Path,
        default=None,
        help="Ortho tile for GT/pred panels when --image is hillshade (legacy hillshade runs).",
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gt-json", type=Path, default=None)
    parser.add_argument(
        "--gt-image-name",
        type=str,
        default=None,
        help="COCO file_name for GT lookup (default: ortho tile basename).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    infer_rgb = load_rgb(args.image)
    ortho_rgb = load_rgb(args.ortho_image) if args.ortho_image else infer_rgb
    hillshade_rgb = load_rgb(args.hillshade_image) if args.hillshade_image else None
    gt_name = args.gt_image_name or (args.ortho_image or args.image).name

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = str(args.model)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.DEVICE = args.device
    cfg.INPUT.MAX_SIZE_TEST = 2000
    cfg.INPUT.MIN_SIZE_TEST = 2000
    cfg.TEST.DETECTIONS_PER_IMAGE = 300

    predictor = DefaultPredictor(cfg)
    outputs = predictor(infer_rgb[:, :, ::-1])
    instances = outputs["instances"].to("cpu")

    gt_ortho = draw_gt(ortho_rgb, args.gt_json, gt_name) if args.gt_json else ortho_rgb
    pred_on_ortho = draw_predictions(ortho_rgb, instances)

    panels = [label_panel(gt_ortho, "Ortho + ground truth")]
    if hillshade_rgb is not None:
        panels.append(label_panel(draw_predictions(hillshade_rgb, instances), "Hillshade + predictions"))
    panels.append(label_panel(pred_on_ortho, "Ortho + predictions"))
    comparison = np.hstack(panels)

    stem = args.image.stem
    compare_path = args.output_dir / f"{stem}_ortho_hillshade_comparison.jpg"
    pred_path = args.output_dir / f"{stem}_predictions.jpg"
    cv2.imwrite(str(pred_path), cv2.cvtColor(pred_on_ortho, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(compare_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    detections = []
    boxes = instances.pred_boxes.tensor.numpy() if len(instances) else np.zeros((0, 4))
    scores = instances.scores.numpy() if len(instances) else np.zeros((0,))
    for idx, (box, score) in enumerate(zip(boxes, scores), start=1):
        detections.append(
            {
                "id": idx,
                "score": float(score),
                "bbox": [float(v) for v in box.tolist()],
            }
        )

    summary = {
        "image": str(args.image),
        "hillshade_image": str(args.hillshade_image) if args.hillshade_image else None,
        "ortho_image": str(args.ortho_image or args.image),
        "model": str(args.model),
        "device": args.device,
        "score_thresh": args.score_thresh,
        "num_detections": len(detections),
        "prediction_image": str(pred_path),
        "comparison_image": str(compare_path),
        "detections": detections,
    }
    summary_path = args.output_dir / f"{stem}_inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
