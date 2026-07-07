#!/usr/bin/env python3
"""Offline COCO dataset augmentation (like the original BoulderCalculator paper).

The paper's training notebook notes "datasets have already been augmented" and
uses a NonAugmentationsTrainer -- i.e. augmentation was applied to the dataset
itself before training, not on the fly. This script reproduces that: it reads a
COCO dataset dir (as produced by gpkg_to_coco.py), applies deterministic
geometric variants (flips / right-angle rotations) plus optional random
brightness/contrast jitter to the TRAIN split, transforms the polygon
annotations exactly, and writes a new dataset dir. Valid/test splits are
copied through unchanged.

Example:
    python BoulderCalculator/scripts/augment_coco_dataset.py \
        --input-dir segmentation/coco_dataset_v2 \
        --output-dir segmentation/coco_dataset_v2_aug \
        --variants hflip,vflip,rot90,rot180,rot270 \
        --jitter 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import rasterio

VALID_VARIANTS = (
    "hflip",
    "vflip",
    "rot90",
    "rot180",
    "rot270",
    "transpose",
    "antitranspose",
)


def load_image(path: Path) -> np.ndarray:
    """Read tile as HxWx3 uint8 RGB using rasterio (handles GeoTIFF bands)."""
    with rasterio.open(path) as ds:
        arr = ds.read()
    if arr.shape[0] >= 3:
        img = np.transpose(arr[:3], (1, 2, 0))
    else:
        img = np.stack([arr[0]] * 3, axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def transform_image(img: np.ndarray, variant: str) -> np.ndarray:
    if variant == "hflip":
        return img[:, ::-1]
    if variant == "vflip":
        return img[::-1, :]
    if variant == "rot90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if variant == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180)
    if variant == "rot270":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if variant == "transpose":
        return np.ascontiguousarray(img.transpose(1, 0, 2))
    if variant == "antitranspose":
        return cv2.rotate(np.ascontiguousarray(img.transpose(1, 0, 2)), cv2.ROTATE_180)
    raise ValueError(variant)


def transform_points(xs: np.ndarray, ys: np.ndarray, variant: str, w: int, h: int):
    """Map pixel coordinates from the original image into the variant image."""
    if variant == "hflip":
        return w - xs, ys
    if variant == "vflip":
        return xs, h - ys
    if variant == "rot90":  # clockwise; new size (h, w) -> width=h, height=w
        return h - ys, xs
    if variant == "rot180":
        return w - xs, h - ys
    if variant == "rot270":  # counterclockwise
        return ys, w - xs
    if variant == "transpose":  # flip across main diagonal
        return ys, xs
    if variant == "antitranspose":  # flip across anti-diagonal
        return h - ys, w - xs
    raise ValueError(variant)


def variant_size(variant: str, w: int, h: int) -> tuple[int, int]:
    if variant in ("rot90", "rot270", "transpose", "antitranspose"):
        return h, w
    return w, h


def jitter_image(img: np.ndarray, rng: random.Random, amount: float) -> np.ndarray:
    alpha = 1.0 + rng.uniform(-amount, amount)  # contrast
    beta = rng.uniform(-amount, amount) * 128.0  # brightness
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def transform_annotation(ann: dict, variant: str, w: int, h: int) -> dict:
    new_segs = []
    for seg in ann["segmentation"]:
        xs = np.asarray(seg[0::2], dtype=float)
        ys = np.asarray(seg[1::2], dtype=float)
        nx, ny = transform_points(xs, ys, variant, w, h)
        out = np.empty(len(seg), dtype=float)
        out[0::2] = nx
        out[1::2] = ny
        new_segs.append(out.tolist())
    all_x = np.concatenate([np.asarray(s[0::2]) for s in new_segs])
    all_y = np.concatenate([np.asarray(s[1::2]) for s in new_segs])
    bbox = [
        float(all_x.min()),
        float(all_y.min()),
        float(all_x.max() - all_x.min()),
        float(all_y.max() - all_y.min()),
    ]
    new_ann = dict(ann)
    new_ann["segmentation"] = new_segs
    new_ann["bbox"] = bbox
    return new_ann


def augment_train(
    input_dir: Path,
    output_dir: Path,
    variants: list[str],
    jitter: float,
    seed: int,
) -> dict:
    coco = json.loads((input_dir / "train_annotations.json").read_text())
    out_img_dir = output_dir / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    new_images: list[dict] = []
    new_annotations: list[dict] = []
    image_id = 1
    ann_id = 1

    for image in coco["images"]:
        src_path = input_dir / "train" / image["file_name"]
        w, h = image["width"], image["height"]
        img = load_image(src_path)
        stem = Path(image["file_name"]).stem
        suffix = Path(image["file_name"]).suffix

        for variant in ["orig"] + variants:
            if variant == "orig":
                out_img = img
                file_name = image["file_name"]
                vw, vh = w, h
            else:
                out_img = transform_image(img, variant)
                file_name = f"{stem}_{variant}{suffix}"
                vw, vh = variant_size(variant, w, h)
            if jitter > 0 and variant != "orig":
                out_img = jitter_image(out_img, rng, jitter)

            cv2.imwrite(str(out_img_dir / file_name), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

            new_images.append(
                {
                    **{k: v for k, v in image.items() if k not in ("id", "file_name", "width", "height")},
                    "id": image_id,
                    "file_name": file_name,
                    "width": vw,
                    "height": vh,
                }
            )
            for ann in anns_by_image.get(image["id"], []):
                if variant == "orig":
                    new_ann = dict(ann)
                else:
                    new_ann = transform_annotation(ann, variant, w, h)
                new_ann["id"] = ann_id
                new_ann["image_id"] = image_id
                new_annotations.append(new_ann)
                ann_id += 1
            image_id += 1

    out_coco = dict(coco)
    out_coco["images"] = new_images
    out_coco["annotations"] = new_annotations
    (output_dir / "train_annotations.json").write_text(json.dumps(out_coco))

    return {
        "split": "train",
        "source_images": len(coco["images"]),
        "augmented_images": len(new_images),
        "source_annotations": len(coco["annotations"]),
        "augmented_annotations": len(new_annotations),
        "variants": ["orig"] + variants,
        "jitter": jitter,
    }


def copy_split(input_dir: Path, output_dir: Path, split: str, ann_name: str) -> dict:
    src_img_dir = input_dir / split
    dst_img_dir = output_dir / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src_img_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dst_img_dir / f.name)
            n += 1
    shutil.copy2(input_dir / ann_name, output_dir / ann_name)
    return {"split": split, "images": n, "copied": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        type=str,
        default="hflip,vflip,rot90,rot180,rot270,transpose,antitranspose",
        help=f"Comma-separated geometric variants ({','.join(VALID_VARIANTS)})",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Random brightness/contrast jitter fraction applied to augmented copies (e.g. 0.15). 0 = off.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VALID_VARIANTS:
            raise ValueError(f"Unknown variant: {v}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        augment_train(args.input_dir, args.output_dir, variants, args.jitter, args.seed),
        copy_split(args.input_dir, args.output_dir, "valid", "validation_annotations.json"),
        copy_split(args.input_dir, args.output_dir, "test", "testing_annotations.json"),
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
