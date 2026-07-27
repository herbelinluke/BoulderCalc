#!/usr/bin/env python3
"""Materialize a train/valid/test COCO dir from a shared offline-aug tile pool.

The geo-split weekend experiment builds and augments **all** tiles once
(``all_tiles.yaml`` → ``coco_geo_all_*_aug``). Each setup then filters
that pool into train/valid/test by geographic split config.

**Hold-out hygiene:** offline-aug variants (``*_hflip``, ``*_rot90``, …) are
included for **train** only. Valid/test drop those stems (or use
``--holdout-pool-dir`` pointing at the unaugmented pool). Pass
``--allow-aug-holdout`` only to restore the old contaminated behavior.

By default images are **hard-linked** into the output (same disk blocks, no
extra space; works on Windows guest / non-admin). Symlinks need admin or
Developer Mode on Windows. ``--link-mode copy`` duplicates files (avoid —
the pool is already ~20GB).

Example:
  python BoulderCalculator/scripts/materialize_geo_split_coco.py \\
    --pool-dir segmentation/coco_geo_all_rgb_dsm_aug \\
    --holdout-pool-dir segmentation/coco_geo_all_rgb_dsm \\
    --split-config BoulderCalculator/experiments/geo_splits/baseline.yaml \\
    --output-dir segmentation/coco_geo_baseline_from_pool
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from augment_coco_dataset import VALID_VARIANTS  # noqa: E402
from file_link import link_or_copy  # noqa: E402
from gpkg_to_coco import (  # noqa: E402
    expand_year_keys,
    load_split_config,
    resolve_tiles_by_year,
    year_key,
)

SPLIT_ANN = {
    "train": "train_annotations.json",
    "valid": "validation_annotations.json",
    "test": "testing_annotations.json",
}

# Year-prefixed ortho stems (parent 2000 tiles and optional 512 chips `_rR_cC`).
_STEM_24 = re.compile(
    r"^24_Sites1and2_2024_Orthomosaic_(\d+)_(\d+)(?:_r\d+_c\d+)?$"
)
_STEM_25 = re.compile(r"^25_25IniSouthOrt_(\d+)_(\d+)(?:_r\d+_c\d+)?$")


def strip_aug_variant(stem: str) -> str:
    for variant in VALID_VARIANTS:
        suffix = f"_{variant}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def is_aug_variant_name(file_name: str) -> bool:
    """True when ``file_name`` is an offline-aug stem (…_hflip, …_rot90, …)."""
    stem = Path(file_name).stem
    return strip_aug_variant(stem) != stem


def file_name_to_year_key(file_name: str) -> str | None:
    """Map parent or chip COCO names (optional aug suffix) → ``24_14_15``."""
    stem = strip_aug_variant(Path(file_name).stem)
    m = _STEM_24.match(stem)
    if m:
        return year_key(24, f"{int(m.group(1))}_{int(m.group(2))}")
    m = _STEM_25.match(stem)
    if m:
        return year_key(25, f"{int(m.group(1))}_{int(m.group(2))}")
    return None


def load_pool_images(pool_dir: Path) -> tuple[dict, list[dict], dict[int, list[dict]]]:
    """Load pool annotations; prefer train JSON (all-tiles pool), merge if needed."""
    images_by_id: dict[int, dict] = {}
    anns_by_image: dict[int, list[dict]] = {}
    categories = None
    licenses = None
    info = None

    for split, ann_name in SPLIT_ANN.items():
        path = pool_dir / ann_name
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if categories is None:
            categories = data.get("categories", [])
            licenses = data.get("licenses", [])
            info = data.get("info", {})
        for image in data["images"]:
            images_by_id[image["id"]] = {**image, "_pool_split": split}
        for ann in data["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    if not images_by_id:
        raise FileNotFoundError(f"No COCO images found under {pool_dir}")

    meta = {
        "categories": categories or [],
        "licenses": licenses or [{"name": "", "id": 0, "url": ""}],
        "info": info or {},
    }
    return meta, list(images_by_id.values()), anns_by_image


def resolve_pool_image(pool_dir: Path, pool_split: str, file_name: str) -> Path:
    """Find the on-disk image; pool usually keeps everything under train/."""
    candidates = [
        pool_dir / pool_split / file_name,
        pool_dir / "train" / file_name,
        pool_dir / "valid" / file_name,
        pool_dir / "test" / file_name,
        pool_dir / "images" / file_name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Pool image not found for {file_name!r} under {pool_dir}")


def materialize_split(
    *,
    pool_dir: Path,
    output_dir: Path,
    split_name: str,
    year_keys: set[str],
    pool_images: list[dict],
    anns_by_image: dict[int, list[dict]],
    meta: dict,
    link_mode: str,
    exclude_aug_variants: bool = False,
) -> dict:
    selected = []
    n_skipped_aug = 0
    for image in pool_images:
        yk = file_name_to_year_key(image["file_name"])
        if yk is None or yk not in year_keys:
            continue
        if exclude_aug_variants and is_aug_variant_name(image["file_name"]):
            n_skipped_aug += 1
            continue
        selected.append(image)

    out_img_dir = output_dir / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)

    new_images: list[dict] = []
    new_annotations: list[dict] = []
    image_id = 1
    ann_id = 1
    modes_used: dict[str, int] = {}

    for image in selected:
        src = resolve_pool_image(
            pool_dir, image.get("_pool_split", "train"), image["file_name"]
        )
        used = link_or_copy(src, out_img_dir / image["file_name"], link_mode)
        modes_used[used] = modes_used.get(used, 0) + 1
        new_images.append(
            {
                **{
                    k: v
                    for k, v in image.items()
                    if not k.startswith("_") and k != "id"
                },
                "id": image_id,
            }
        )
        for ann in anns_by_image.get(image["id"], []):
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            new_ann["image_id"] = image_id
            new_annotations.append(new_ann)
            ann_id += 1
        image_id += 1

    info = dict(meta.get("info") or {})
    info["description"] = (
        f"Materialized geo split '{split_name}' from pool {pool_dir.name} "
        f"({len(year_keys)} tile keys → {len(new_images)} images)"
    )
    coco = {
        "licenses": meta["licenses"],
        "info": info,
        "categories": meta["categories"],
        "images": new_images,
        "annotations": new_annotations,
    }
    ann_path = output_dir / SPLIT_ANN[split_name]
    ann_path.write_text(json.dumps(coco))
    return {
        "split": split_name,
        "tile_keys": len(year_keys),
        "images": len(new_images),
        "annotations": len(new_annotations),
        "skipped_aug_variants": n_skipped_aug,
        "exclude_aug_variants": exclude_aug_variants,
        "link_modes": modes_used,
        "json": str(ann_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("segmentation"),
        help="For tiles_used resolution when expanding the split config.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hard", "symlink", "copy"),
        default="auto",
        help=(
            "How to place pool images into the setup dir. "
            "auto (default): hard link → symlink → copy. "
            "Prefer hard on Windows guest (no admin, no extra disk). "
            "copy duplicates ~20GB×setups — avoid if possible."
        ),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Deprecated alias for --link-mode copy.",
    )
    parser.add_argument("--years", type=str, default="24,25")
    parser.add_argument(
        "--holdout-pool-dir",
        type=Path,
        default=None,
        help=(
            "Optional unaugmented COCO dir used only for valid/test images. "
            "When omitted, valid/test are taken from --pool-dir but offline-aug "
            "variants (*_hflip, *_rot90, …) are dropped so hold-outs stay clean."
        ),
    )
    parser.add_argument(
        "--allow-aug-holdout",
        action="store_true",
        help=(
            "Do NOT strip offline-aug variants from valid/test when using a "
            "single --pool-dir (legacy; not recommended)."
        ),
    )
    args = parser.parse_args()

    link_mode = "copy" if args.copy else args.link_mode
    years = sorted({int(p.strip()) for p in args.years.split(",") if p.strip()})
    split_config = load_split_config(args.split_config)
    tiles_by_year = resolve_tiles_by_year(args.segmentation_dir, None)
    train, valid, test = expand_year_keys(
        years, tiles_by_year, split_config=split_config
    )

    meta, pool_images, anns_by_image = load_pool_images(args.pool_dir)
    holdout_pool = args.holdout_pool_dir
    if holdout_pool is not None:
        hold_meta, hold_images, hold_anns = load_pool_images(holdout_pool)
    else:
        hold_meta, hold_images, hold_anns = meta, pool_images, anns_by_image

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train may include offline-aug variants from the (aug) pool.
    # Valid/test must be unaugmented originals.
    strip_holdout_aug = not args.allow_aug_holdout
    summary = []
    for split_name, keys, images, anns, pdir, meta_i, strip_aug in (
        ("train", train, pool_images, anns_by_image, args.pool_dir, meta, False),
        (
            "valid",
            valid,
            hold_images,
            hold_anns,
            holdout_pool or args.pool_dir,
            hold_meta,
            strip_holdout_aug,
        ),
        (
            "test",
            test,
            hold_images,
            hold_anns,
            holdout_pool or args.pool_dir,
            hold_meta,
            strip_holdout_aug,
        ),
    ):
        summary.append(
            materialize_split(
                pool_dir=pdir,
                output_dir=args.output_dir,
                split_name=split_name,
                year_keys=set(keys),
                pool_images=images,
                anns_by_image=anns,
                meta=meta_i,
                link_mode=link_mode,
                exclude_aug_variants=strip_aug,
            )
        )
    print(
        json.dumps(
            {
                "pool": str(args.pool_dir),
                "holdout_pool": str(holdout_pool) if holdout_pool else None,
                "setup": split_config.get("id"),
                "link_mode_requested": link_mode,
                "strip_aug_from_holdout": strip_holdout_aug,
                "splits": summary,
            },
            indent=2,
        )
    )

    from run_provenance import write_dataset_provenance

    write_dataset_provenance(
        args.output_dir,
        tool="materialize_geo_split_coco.py",
        flags={
            "split_config": str(args.split_config),
            "setup_id": split_config.get("id"),
            "pool_dir": str(args.pool_dir),
            "holdout_pool_dir": str(holdout_pool) if holdout_pool else None,
            "strip_aug_from_holdout": strip_holdout_aug,
            "link_mode": link_mode,
            "years": args.years,
        },
        splits_summary=summary,
        parents=[args.pool_dir]
        + ([holdout_pool] if holdout_pool is not None else []),
        notes=(
            "Geo-split view of a shared pool. Train may include offline-aug "
            "variants; valid/test exclude them unless --allow-aug-holdout."
        ),
    )


if __name__ == "__main__":
    main()
