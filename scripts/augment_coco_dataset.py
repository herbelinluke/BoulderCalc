#!/usr/bin/env python3
"""Offline COCO dataset augmentation (like the original BoulderCalculator paper).

The paper's training notebook notes "datasets have already been augmented" and
uses a NonAugmentationsTrainer -- i.e. augmentation was applied to the dataset
itself before training, not on the fly. This script reproduces that: it reads a
COCO dataset dir (as produced by gpkg_to_coco.py), applies deterministic
geometric variants (flips / right-angle rotations) plus optional coastal
photometric jitter, transforms the polygon annotations exactly, and writes a
new dataset dir.

By default only the train split is expanded (valid/test copied unchanged).
Pass ``--splits train,valid,test`` to offline-augment every split (used by the
geo-split weekend experiment).

Online training (``train_boulder_local.py``) additionally applies full-circle
rotation, scale jitter, blur, noise, and synthetic shadows via
``boulder_augmentations.py`` unless ``--no-rich-aug`` is set. Use this offline
script for exact dihedral expansion; use online augs for the lossy / continuous
transforms.

Example:
    python BoulderCalculator/scripts/augment_coco_dataset.py \
        --input-dir segmentation/coco_dataset_v2 \
        --output-dir segmentation/coco_dataset_v2_aug \
        --variants hflip,vflip,rot90,rot180,rot270 \
        --jitter 0.15

    # Expand all splits (geo-split experiment):
    python BoulderCalculator/scripts/augment_coco_dataset.py \
        --input-dir segmentation/coco_baseline_rgb_dsm \
        --output-dir segmentation/coco_baseline_rgb_dsm_aug \
        --splits train,valid,test --jitter 0.15
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

SPLIT_ANN = {
    "train": "train_annotations.json",
    "valid": "validation_annotations.json",
    "test": "testing_annotations.json",
}


def load_image(path: Path) -> np.ndarray:
    """Read tile as HxWxC uint8 (3-band RGB or 4-band RGB+DSM) via rasterio."""
    with rasterio.open(path) as ds:
        arr = ds.read()
    if arr.shape[0] >= 4:
        img = np.transpose(arr[:4], (1, 2, 0))
    elif arr.shape[0] >= 3:
        img = np.transpose(arr[:3], (1, 2, 0))
    else:
        img = np.stack([arr[0]] * 3, axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_image(path: Path, img: np.ndarray) -> None:
    """Write HxWxC uint8; use rasterio for 4-band GeoTIFFs, OpenCV for 3-band."""
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got {img.shape}")
    if img.shape[2] == 4:
        profile = {
            "driver": "GTiff",
            "height": img.shape[0],
            "width": img.shape[1],
            "count": 4,
            "dtype": "uint8",
            "compress": "deflate",
        }
        with rasterio.open(path, "w", **profile) as out:
            for i in range(4):
                out.write(img[:, :, i], i + 1)
        return
    if img.shape[2] != 3:
        raise ValueError(f"Expected 3 or 4 channels, got {img.shape[2]}")
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


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


def _clip_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _apply_rgb_only(img: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[:, :, :3] = rgb
    return out


def photometric_jitter(
    img: np.ndarray,
    rng: random.Random,
    amount: float,
    *,
    hue_delta: float = 0.04,
    saturation: float = 0.2,
    noise_std: float = 10.0,
    blur_prob: float = 0.35,
    shadow_prob: float = 0.4,
) -> np.ndarray:
    """Coastal photometric suite on RGB only; leave DSM (band 4) unchanged.

    ``amount`` scales brightness/contrast strength (e.g. 0.15–0.25). Hue /
    saturation / noise / blur / shadow use the dedicated kwargs (mild defaults).
    """
    if amount <= 0:
        return img

    out = img.copy()
    rgb = out[:, :, :3].astype(np.float32)

    # Brightness / contrast (moderate; cloud-cover variability).
    alpha = 1.0 + rng.uniform(-amount, amount)  # contrast
    beta = rng.uniform(-amount, amount) * 128.0  # brightness
    mean = rgb.mean(axis=(0, 1), keepdims=True)
    rgb = (rgb - mean) * alpha + mean + beta

    # Mild hue / saturation in HSV (OpenCV H in [0, 180]).
    bgr_u8 = _clip_u8(cv2.cvtColor(_clip_u8(rgb), cv2.COLOR_RGB2BGR))
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-hue_delta, hue_delta) * 180.0) % 180.0
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * (1.0 + rng.uniform(-saturation, saturation)), 0, 255
    )
    rgb = cv2.cvtColor(_clip_u8(hsv), cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Gaussian noise (sensor / session floors).
    if noise_std > 0:
        std = rng.uniform(0.0, noise_std)
        if std > 1e-3:
            rgb = rgb + np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(
                0.0, std, size=rgb.shape
            )

    rgb_u8 = _clip_u8(rgb)

    # Slight motion or defocus blur.
    if rng.random() < blur_prob:
        k = rng.choice([3, 5, 7])
        if rng.random() < 0.5:
            angle = rng.uniform(0.0, 180.0)
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0
            rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
            kernel = cv2.warpAffine(kernel, rot, (k, k))
            kernel /= max(kernel.sum(), 1e-6)
            rgb_u8 = cv2.filter2D(rgb_u8, -1, kernel)
        else:
            rgb_u8 = cv2.GaussianBlur(rgb_u8, (k, k), 0)

    # Synthetic soft shadow (low sun on exposed coastline).
    if rng.random() < shadow_prob:
        h, w = rgb_u8.shape[:2]
        strength = rng.uniform(0.25, 0.55)
        cx, cy = rng.uniform(0.15 * w, 0.85 * w), rng.uniform(0.15 * h, 0.85 * h)
        ax, ay = rng.uniform(0.25 * w, 0.7 * w), rng.uniform(0.12 * h, 0.45 * h)
        angle = rng.uniform(0.0, 180.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cos_a, sin_a = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        x, y = xx - cx, yy - cy
        xr, yr = cos_a * x + sin_a * y, -sin_a * x + cos_a * y
        r2 = (xr / max(ax, 1.0)) ** 2 + (yr / max(ay, 1.0)) ** 2
        mask = np.clip(1.0 - r2, 0.0, 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(ax, ay) * 0.08)
        shade = (1.0 - strength * mask).astype(np.float32)
        rgb_u8 = _clip_u8(rgb_u8.astype(np.float32) * shade[:, :, None])

    return _apply_rgb_only(out, rgb_u8)


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


def augment_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    variants: list[str],
    jitter: float,
    seed: int,
) -> dict:
    ann_name = SPLIT_ANN[split]
    coco = json.loads((input_dir / ann_name).read_text())
    out_img_dir = output_dir / split
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
        src_path = input_dir / split / image["file_name"]
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
                out_img = photometric_jitter(out_img, rng, jitter)

            write_image(out_img_dir / file_name, out_img)

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
    (output_dir / ann_name).write_text(json.dumps(out_coco))

    return {
        "split": split,
        "source_images": len(coco["images"]),
        "augmented_images": len(new_images),
        "source_annotations": len(coco["annotations"]),
        "augmented_annotations": len(new_annotations),
        "variants": ["orig"] + variants,
        "jitter": jitter,
        "photometric": [
            "brightness_contrast",
            "hue_saturation",
            "gaussian_noise",
            "motion_or_defocus_blur",
            "synthetic_shadow",
        ]
        if jitter > 0
        else [],
    }


def augment_train(
    input_dir: Path,
    output_dir: Path,
    variants: list[str],
    jitter: float,
    seed: int,
) -> dict:
    """Back-compat alias for augment_split(..., split='train')."""
    return augment_split(input_dir, output_dir, "train", variants, jitter, seed)


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
        help=(
            "Photometric strength for augmented copies (e.g. 0.15). Enables "
            "brightness/contrast plus mild hue/sat, noise, blur, and shadows. "
            "0 = off. DSM band is never modified."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        help=(
            "Comma-separated splits to offline-augment "
            "(train, valid, test). Default: train only; other splits are copied. "
            "Use train,valid,test for the geo-split weekend experiment."
        ),
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VALID_VARIANTS:
            raise ValueError(f"Unknown variant: {v}")

    augment_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in augment_splits:
        if s not in SPLIT_ANN:
            raise ValueError(f"Unknown split {s!r}; expected one of {list(SPLIT_ANN)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for split, ann_name in SPLIT_ANN.items():
        if split in augment_splits:
            # Distinct seed per split so jitter patterns differ.
            split_seed = args.seed + {"train": 0, "valid": 1, "test": 2}[split]
            summary.append(
                augment_split(
                    args.input_dir,
                    args.output_dir,
                    split,
                    variants,
                    args.jitter,
                    split_seed,
                )
            )
        else:
            summary.append(
                copy_split(args.input_dir, args.output_dir, split, ann_name)
            )
    print(json.dumps(summary, indent=2))

    from run_provenance import write_dataset_provenance

    write_dataset_provenance(
        args.output_dir,
        tool="augment_coco_dataset.py",
        flags={
            "jitter": args.jitter,
            "jitter_enabled": args.jitter > 0,
            "variants": variants,
            "seed": args.seed,
            "splits_augmented": augment_splits,
            "input_dir": str(args.input_dir),
        },
        splits_summary=summary,
        parents=[args.input_dir],
        notes="Offline geometric (+ optional photometric jitter) augmentation.",
    )


if __name__ == "__main__":
    main()
