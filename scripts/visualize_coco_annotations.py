#!/usr/bin/env python3
"""Overlay COCO ground-truth polygons on tile images for QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio


def load_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read()
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        rgb = np.stack([arr[0]] * 3, axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def draw_annotations(
    image: np.ndarray,
    annotations: list[dict],
    image_id: int,
    color: tuple[int, int, int] = (0, 255, 128),
) -> np.ndarray:
    out = image.copy()
    count = 0
    for ann in annotations:
        if ann["image_id"] != image_id:
            continue
        seg = ann["segmentation"][0]
        pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.polylines(out, [pts], True, color, 2)
        cv2.fillPoly(out, [pts], (color[0] // 4, color[1] // 4, color[2] // 4))
        x, y, w, h = ann["bbox"]
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (255, 200, 0), 1)
        cv2.putText(
            out,
            str(ann["id"]),
            (int(x), max(12, int(y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        count += 1
    cv2.putText(
        out,
        f"annotations: {count}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def visualize_split(
    dataset_dir: Path,
    split_name: str,
    ann_file: str,
    image_dir: str,
    output_dir: Path,
) -> list[dict]:
    coco = json.loads((dataset_dir / ann_file).read_text())
    split_out = output_dir / split_name
    split_out.mkdir(parents=True, exist_ok=True)

    results = []
    for image in coco["images"]:
        image_path = dataset_dir / image_dir / image["file_name"]
        rgb = load_rgb(image_path)
        overlay = draw_annotations(rgb, coco["annotations"], image["id"])
        out_path = split_out / f"{Path(image['file_name']).stem}_coco_gt.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        n_ann = sum(1 for a in coco["annotations"] if a["image_id"] == image["id"])
        results.append(
            {
                "split": split_name,
                "image": image["file_name"],
                "annotations": n_ann,
                "preview": str(out_path),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("segmentation/coco_dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmentation/visualizations/coco_gt"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    summary.extend(
        visualize_split(args.dataset_dir, "train", "train_annotations.json", "train", args.output_dir)
    )
    summary.extend(
        visualize_split(args.dataset_dir, "valid", "validation_annotations.json", "valid", args.output_dir)
    )
    summary.extend(
        visualize_split(args.dataset_dir, "test", "testing_annotations.json", "test", args.output_dir)
    )

    grid_path = args.output_dir / "all_tiles_montage.jpg"
    previews = []
    for split in ("train", "valid", "test"):
        for jpg in sorted((args.output_dir / split).glob("*.jpg")):
            previews.append(cv2.imread(str(jpg)))
    if previews:
        thumb_w = 900
        thumbs = []
        for img in previews:
            h, w = img.shape[:2]
            scale = thumb_w / w
            thumbs.append(cv2.resize(img, (thumb_w, int(h * scale))))
        montage = np.vstack(thumbs)
        cv2.imwrite(str(grid_path), montage)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"montage": str(grid_path), "tiles": summary}, indent=2))


if __name__ == "__main__":
    main()
