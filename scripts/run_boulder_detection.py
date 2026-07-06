#!/usr/bin/env python3
"""Run BoulderCalculator Detectron2 sliding-window detection on a georeferenced ortho."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import rasterio
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tifffile import TiffFile
from tqdm import tqdm


def poly_area(seg: list[float]) -> float:
    if len(seg) < 6:
        return 0.0
    x = np.asarray(seg[0::2], dtype=float)
    y = np.asarray(seg[1::2], dtype=float)
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def mask_to_seg(mask: np.ndarray, shift: tuple[int, int]) -> list[float]:
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    boundaries = np.reshape(contours[0], (1, -1))[0].astype(float)
    boundaries[0::2] += shift[0]
    boundaries[1::2] += shift[1]
    return boundaries.tolist()


def make_starts(length: int, window: int, step: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, step))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def coco_initialize(class_names: list[str] | None = None) -> dict:
    if not class_names:
        class_names = ["Boulder"]
    return {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "BoulderCalculator detection output",
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "none"}
            for i, name in enumerate(class_names)
        ],
        "images": [],
        "annotations": [],
    }


def ensure_rgb_uint8(tile: np.ndarray) -> np.ndarray:
    if tile.ndim == 2:
        tile = np.stack([tile, tile, tile], axis=-1)
    if tile.shape[2] == 4:
        tile = tile[:, :, :3]
    if tile.dtype != np.uint8:
        tile = np.clip(tile, 0, 255).astype(np.uint8)
    return tile


def pixel_to_world(
    xs: np.ndarray,
    ys: np.ndarray,
    origin_x: float,
    origin_y: float,
    pixel_width: float,
    pixel_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    world_x = origin_x + xs * pixel_width
    world_y = origin_y + ys * pixel_height
    return world_x, world_y


def write_geojson(
    coco: dict,
    out_path: Path,
    origin_x: float,
    origin_y: float,
    pixel_width: float,
    pixel_height: float,
    epsg: int,
) -> None:
    features = []
    for ann in coco["annotations"]:
        seg = ann["segmentation"][0]
        if len(seg) < 6:
            continue
        xs = np.asarray(seg[0::2], dtype=float)
        ys = np.asarray(seg[1::2], dtype=float)
        wx, wy = pixel_to_world(xs, ys, origin_x, origin_y, pixel_width, pixel_height)
        ring = [[float(x), float(y)] for x, y in zip(wx, wy)]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": ann["id"],
                    "score": ann.get("score", ""),
                    "area_px": ann.get("area", 0),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "name": out_path.stem,
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
        },
        "features": features,
    }
    out_path.write_text(json.dumps(geojson))


def write_preview_png(
    ortho_path: Path,
    coco: dict,
    out_path: Path,
    max_dim: int = 4096,
) -> None:
    with TiffFile(ortho_path) as tif:
        arr = tif.asarray()
    rgb = ensure_rgb_uint8(arr)
    h, w = rgb.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        preview = cv2.resize(
            rgb,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = rgb.copy()

    for ann in coco["annotations"]:
        seg = ann["segmentation"][0]
        if len(seg) < 6:
            continue
        pts = np.asarray(seg, dtype=float).reshape(-1, 2) * scale
        pts = pts.astype(np.int32)
        cv2.polylines(preview, [pts], True, (255, 64, 0), max(1, int(2 * scale)))
        x, y, bw, bh = ann["bbox"]
        x1, y1 = int(x * scale), int(y * scale)
        x2, y2 = int((x + bw) * scale), int((y + bh) * scale)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), max(1, int(1 * scale)))

    cv2.imwrite(str(out_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def run_detection(
    ortho_path: Path,
    model_path: Path,
    output_dir: Path,
    score_thresh: float = 0.7,
    window_size: int = 2000,
    step_rate: float = 0.25,
    epsg: int = 25829,
    max_tiles: int | None = None,
    class_names: list[str] | None = None,
) -> dict:
    if not class_names:
        class_names = ["Boulder"]
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.INPUT.MAX_SIZE_TEST = 0
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.TEST.DETECTIONS_PER_IMAGE = 500
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    with rasterio.open(ortho_path) as ds:
        transform = ds.transform
        if ds.crs is not None and ds.crs.to_epsg() is not None:
            epsg = int(ds.crs.to_epsg())
        origin_x = float(transform.c)
        origin_y = float(transform.f)
        pixel_width = float(transform.a)
        pixel_height = float(transform.e)

    with TiffFile(ortho_path) as tif:
        mmap = tif.asarray(out="memmap")

    im_name = ortho_path.name
    stem = ortho_path.stem
    pix_im = mmap.shape
    pix_window = [window_size, window_size]
    pix_step = [max(1, int(round(window_size * step_rate)))] * 2

    row_starts = make_starts(pix_im[0], pix_window[0], pix_step[0])
    col_starts = make_starts(pix_im[1], pix_window[1], pix_step[1])
    tile_coords = [(r0, c0) for c0 in col_starts for r0 in row_starts]
    if max_tiles is not None:
        tile_coords = tile_coords[:max_tiles]

    image_grid = []
    for r0 in row_starts:
        for c0 in col_starts:
            image_grid.append(
                [r0, r0 + pix_window[0], c0, c0 + pix_window[1]]
            )

    coco = coco_initialize(class_names)
    coco["images"].append(
        {
            "id": 1,
            "width": int(pix_im[1]),
            "height": int(pix_im[0]),
            "file_name": im_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_capture": "",
            "window_size": pix_window,
            "window_step_rate": [step_rate, step_rate],
            "image_grid": image_grid,
        }
    )

    ann_id = 0
    tile_set = set(tile_coords)
    for c0 in tqdm(col_starts, desc="columns"):
        for r0 in row_starts:
            if max_tiles is not None and (r0, c0) not in tile_set:
                continue
            row_range = [r0, r0 + pix_window[0]]
            col_range = [c0, c0 + pix_window[1]]
            tile = np.array(mmap[row_range[0] : row_range[1], col_range[0] : col_range[1], :])
            tile = ensure_rgb_uint8(tile)
            tile_bgr = tile[:, :, ::-1]

            outputs = predictor(tile_bgr)
            instances = outputs["instances"].to("cpu")
            if len(instances) == 0:
                continue

            bboxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy()

            for idx in range(len(bboxes)):
                ann_id += 1
                boundary = mask_to_seg(masks[idx], (col_range[0], row_range[0]))
                bbox = [
                    float(bboxes[idx][0] + col_range[0]),
                    float(bboxes[idx][1] + row_range[0]),
                    float(bboxes[idx][2] - bboxes[idx][0] + 1),
                    float(bboxes[idx][3] - bboxes[idx][1] + 1),
                ]
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": 1,
                        "category_id": int(classes[idx]) + 1,
                        "segmentation": [boundary],
                        "area": poly_area(boundary),
                        "bbox": bbox,
                        "iscrowd": 0,
                        "attributes": {"occluded": 0},
                        "score": str(float(scores[idx])),
                    }
                )

    json_path = output_dir / f"{stem}_detect_object.json"
    geojson_path = output_dir / f"{stem}_detections.geojson"
    preview_path = output_dir / f"{stem}_detections_preview.jpg"

    json_path.write_text(json.dumps(coco))
    write_geojson(
        coco,
        geojson_path,
        origin_x,
        origin_y,
        pixel_width,
        pixel_height,
        epsg,
    )
    write_preview_png(ortho_path, coco, preview_path)

    summary = {
        "ortho": str(ortho_path),
        "json": str(json_path),
        "geojson": str(geojson_path),
        "preview": str(preview_path),
        "num_detections_raw": len(coco["annotations"]),
        "tiles_processed": len(tile_coords),
    }
    (output_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ortho", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-thresh", type=float, default=0.7)
    parser.add_argument("--window-size", type=int, default=2000)
    parser.add_argument("--step-rate", type=float, default=0.25)
    parser.add_argument("--epsg", type=int, default=25829)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--class-names",
        type=str,
        default="Boulder",
        help="Comma-separated class names matching the trained model (e.g. 'Boulder,BoulderDeposit').",
    )
    args = parser.parse_args()

    summary = run_detection(
        ortho_path=args.ortho,
        model_path=args.model,
        output_dir=args.output_dir,
        score_thresh=args.score_thresh,
        window_size=args.window_size,
        step_rate=args.step_rate,
        epsg=args.epsg,
        max_tiles=args.max_tiles,
        class_names=[c.strip() for c in args.class_names.split(",") if c.strip()],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
