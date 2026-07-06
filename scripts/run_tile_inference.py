#!/usr/bin/env python3
"""Run a trained Detectron2 model on one tile and save a visualization."""

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
        rgb = np.stack([arr[0]] * 3, axis=-1)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gt-json", type=Path, default=None)
    parser.add_argument(
        "--class-names",
        type=str,
        default="Boulder",
        help="Comma-separated class names matching the trained model (e.g. 'Boulder,BoulderDeposit').",
    )
    args = parser.parse_args()
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(args.image)
    bgr = rgb[:, :, ::-1]

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = str(args.model)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.MODEL.DEVICE = args.device
    cfg.INPUT.MAX_SIZE_TEST = 2000
    cfg.INPUT.MIN_SIZE_TEST = 2000
    cfg.TEST.DETECTIONS_PER_IMAGE = 300

    predictor = DefaultPredictor(cfg)
    outputs = predictor(bgr)
    instances = outputs["instances"].to("cpu")

    visualizer = Visualizer(
        rgb,
        metadata={"thing_classes": class_names},
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW,
    )
    pred_vis = visualizer.draw_instance_predictions(instances).get_image()

    gt_vis = draw_gt(rgb, args.gt_json, args.image.name) if args.gt_json else rgb
    comparison = np.hstack([gt_vis, pred_vis])
    cv2.putText(
        comparison,
        "LEFT: ground truth   RIGHT: model predictions",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    stem = args.image.stem
    pred_path = args.output_dir / f"{stem}_predictions.jpg"
    compare_path = args.output_dir / f"{stem}_gt_vs_pred.jpg"
    cv2.imwrite(str(pred_path), cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR))
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
