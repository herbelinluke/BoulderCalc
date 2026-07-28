#!/usr/bin/env python3
"""Resample a COCO *train* split: oversample positives, thin easy empties.

Leaves valid/test JSON + image folders untouched (copy through).

Tile classes (non-crowd = trainable boulder)::

  positive       ≥1 trainable boulder → keep; optionally duplicate (oversample)
  deposit_heavy  crowd/deposit-dominated, ≤1 trainable → keep small fraction
  green_hilly    empty/near-empty + low row (map top / north) → keep small fraction
  empty_coast    empty + mid/high row (coastal belt) → keep moderate fraction
  empty_other    other empties → keep small fraction

Row convention (Inishmaan tiling): **larger row = further south / lower on map**;
green hills sit toward the **top** (smaller rows).

Example::

  python BoulderCalculator/scripts/resample_coco_train.py \\
    --input-dir segmentation/coco_geo_stratified_coastal_aug \\
    --output-dir segmentation/coco_geo_stratified_coastal_aug_balanced \\
    --tile-extents BoulderCalculator/experiments/geo_splits/tile_extents_stratified_coastal.geojson \\
    --seed 42

Writes ``resample_train_summary.json`` plus QGIS-oriented audits
(``resample_train_audit_parents.csv`` / ``.geojson``, ``…_images.csv``) describing
which tiles were oversampled, thinned, or removed and by how much.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from file_link import link_or_copy  # noqa: E402

SPLIT_ANN = {
    "train": "train_annotations.json",
    "valid": "validation_annotations.json",
    "test": "testing_annotations.json",
}

AUG_SUFFIXES = (
    "_hflip",
    "_vflip",
    "_rot90",
    "_rot180",
    "_rot270",
    "_transpose",
    "_antitranspose",
)

_ROW_RE = re.compile(
    r"(?:24_Sites1and2_2024_Orthomosaic_|25_25IniSouthOrt_|Sites1and2_2024_Orthomosaic_|25IniSouthOrt_)(\d+)_(\d+)",
    re.IGNORECASE,
)


def strip_aug_suffix(stem: str) -> tuple[str, str | None]:
    for suffix in AUG_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], suffix[1:]
    return stem, None


def parse_row_col(file_name: str) -> tuple[int | None, int | None]:
    stem, _ = strip_aug_suffix(Path(file_name).stem)
    m = _ROW_RE.search(stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.search(r"_(\d+)_(\d+)$", stem)
    if m2:
        return int(m2.group(1)), int(m2.group(2))
    return None, None


def infer_year(file_name: str) -> int | None:
    stem, _ = strip_aug_suffix(Path(file_name).stem)
    if stem.startswith("24_") or "2024_Orthomosaic" in stem:
        return 24
    if stem.startswith("25_") or "25IniSouthOrt" in stem or "IniSouthOrt" in stem:
        return 25
    return None


def year_key_from_file_name(file_name: str) -> str | None:
    year = infer_year(file_name)
    row, col = parse_row_col(file_name)
    if year is None or row is None or col is None:
        return None
    return f"{year}_{row:02d}_{col:02d}"


def classify_image(
    anns: list[dict[str, Any]],
    row: int | None,
    *,
    hilly_row_max: int,
) -> str:
    trainable = [a for a in anns if not a.get("iscrowd", 0)]
    crowd = [a for a in anns if a.get("iscrowd", 0)]
    n_dep = 0
    for a in crowd:
        reason = (a.get("attributes") or {}).get("ignore_reason")
        if reason == "deposit" or a.get("category_id") == 2:
            n_dep += 1

    if len(trainable) >= 1:
        return "positive"
    if n_dep >= 3 and len(trainable) <= 1:
        return "deposit_heavy"
    if len(trainable) == 0:
        if row is not None and row <= hilly_row_max:
            return "green_hilly"
        if row is not None and row > hilly_row_max:
            return "empty_coast"
        return "empty_other"
    return "empty_other"


def load_coco(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_split(
    images: list[dict],
    anns_by_id: dict[int, list[dict]],
    selected_old_ids: list[int],
) -> tuple[list[dict], list[dict]]:
    """Rebuild images/annotations with fresh contiguous ids; duplicates allowed."""
    new_images: list[dict] = []
    new_anns: list[dict] = []
    ann_id = 1
    for new_id, old_id in enumerate(selected_old_ids, start=1):
        src_im = next(im for im in images if int(im["id"]) == old_id)
        new_images.append({**{k: v for k, v in src_im.items() if k != "id"}, "id": new_id})
        for ann in anns_by_id.get(old_id, []):
            new_anns.append({**ann, "id": ann_id, "image_id": new_id})
            ann_id += 1
    return new_images, new_anns


def resample_train(
    coco: dict[str, Any],
    *,
    seed: int,
    positive_oversample: float,
    keep_deposit: float,
    keep_hilly: float,
    keep_empty_coast: float,
    keep_empty_other: float,
    hilly_row_max: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    images = list(coco["images"])
    anns_by: dict[int, list[dict]] = {}
    for a in coco.get("annotations", []):
        anns_by.setdefault(int(a["image_id"]), []).append(a)

    buckets: dict[str, list[int]] = {
        "positive": [],
        "deposit_heavy": [],
        "green_hilly": [],
        "empty_coast": [],
        "empty_other": [],
    }
    bucket_of: dict[int, str] = {}
    for im in images:
        iid = int(im["id"])
        row, _col = parse_row_col(im["file_name"])
        cls = classify_image(anns_by.get(iid, []), row, hilly_row_max=hilly_row_max)
        buckets[cls].append(iid)
        bucket_of[iid] = cls

    selected: list[int] = []
    kept_counts: dict[str, int] = {}

    # Positives: keep all + oversample (fractional → Bernoulli extras).
    pos = list(buckets["positive"])
    selected.extend(pos)
    extra = int(len(pos) * max(0.0, positive_oversample - 1.0))
    frac = (positive_oversample - 1.0) - int(positive_oversample - 1.0)
    if pos and frac > 0 and rng.random() < frac:
        extra += 1
    if extra > 0 and pos:
        selected.extend(rng.choices(pos, k=extra))
    kept_counts["positive_base"] = len(pos)
    kept_counts["positive_extra"] = extra

    def keep_frac(ids: list[int], frac: float, name: str) -> None:
        ids = list(ids)
        rng.shuffle(ids)
        n = int(round(len(ids) * max(0.0, min(1.0, frac))))
        chosen = ids[:n]
        selected.extend(chosen)
        kept_counts[name] = len(chosen)
        kept_counts[f"{name}_pool"] = len(ids)

    keep_frac(buckets["deposit_heavy"], keep_deposit, "deposit_heavy")
    keep_frac(buckets["green_hilly"], keep_hilly, "green_hilly")
    keep_frac(buckets["empty_coast"], keep_empty_coast, "empty_coast")
    keep_frac(buckets["empty_other"], keep_empty_other, "empty_other")

    copy_counts = Counter(selected)
    image_audit: list[dict[str, Any]] = []
    by_id = {int(im["id"]): im for im in images}
    for iid, im in by_id.items():
        n_out = int(copy_counts.get(iid, 0))
        stem = Path(im["file_name"]).stem
        parent_stem, aug_variant = strip_aug_suffix(stem)
        row, col = parse_row_col(im["file_name"])
        bucket = bucket_of[iid]
        if n_out == 0:
            action = "removed"
        elif n_out > 1:
            action = "oversampled"
        else:
            action = "kept"
        image_audit.append(
            {
                "image_id": iid,
                "file_name": im["file_name"],
                "parent_stem": parent_stem,
                "aug_variant": aug_variant or "orig",
                "year": infer_year(im["file_name"]),
                "row": row,
                "col": col,
                "year_key": year_key_from_file_name(im["file_name"]),
                "bucket": bucket,
                "action": action,
                "n_copies_out": n_out,
            }
        )

    rng.shuffle(selected)
    new_images, new_anns = rewrite_split(images, anns_by, selected)
    out = {
        **coco,
        "images": new_images,
        "annotations": new_anns,
    }
    stats = {
        "bucket_sizes_in": {k: len(v) for k, v in buckets.items()},
        "kept": kept_counts,
        "n_images_in": len(images),
        "n_images_out": len(new_images),
        "n_annotations_out": len(new_anns),
        "hilly_row_max": hilly_row_max,
        "positive_oversample": positive_oversample,
        "keep_deposit": keep_deposit,
        "keep_hilly": keep_hilly,
        "keep_empty_coast": keep_empty_coast,
        "keep_empty_other": keep_empty_other,
        "seed": seed,
        "n_removed_images": sum(1 for r in image_audit if r["action"] == "removed"),
        "n_oversampled_images": sum(1 for r in image_audit if r["action"] == "oversampled"),
        "n_kept_once_images": sum(1 for r in image_audit if r["action"] == "kept"),
    }
    return out, stats, image_audit


def aggregate_parent_audit(
    image_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Roll image-level decisions up to unaugmented parent tiles for QGIS."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_audit:
        key = row.get("year_key") or row["parent_stem"]
        groups[str(key)].append(row)

    parents: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        n_in = len(rows)
        n_out = sum(int(r["n_copies_out"]) for r in rows)
        n_removed = sum(1 for r in rows if r["action"] == "removed")
        bucket_counts = Counter(r["bucket"] for r in rows)
        primary_bucket = bucket_counts.most_common(1)[0][0]
        if n_out == 0:
            status = "removed"
        elif n_out > n_in:
            status = "oversampled"
        elif n_removed > 0 and n_out < n_in:
            status = "thinned"
        else:
            status = "kept"
        sample = rows[0]
        parents.append(
            {
                "year_key": sample.get("year_key"),
                "parent_stem": sample["parent_stem"],
                "year": sample.get("year"),
                "row": sample.get("row"),
                "col": sample.get("col"),
                "primary_bucket": primary_bucket,
                "bucket_counts": dict(bucket_counts),
                "status": status,
                "n_images_in": n_in,
                "n_images_out": n_out,
                "n_images_removed": n_removed,
                "oversample_factor": round(n_out / n_in, 4) if n_in else 0.0,
                "removed": int(n_out == 0),
                "oversampled": int(n_out > n_in),
            }
        )
    return parents


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            if "bucket_counts" in flat and isinstance(flat["bucket_counts"], dict):
                flat["bucket_counts"] = json.dumps(flat["bucket_counts"], sort_keys=True)
            writer.writerow(flat)


def write_parent_geojson(
    path: Path,
    parents: list[dict[str, Any]],
    extents_geojson: Path | None,
) -> int:
    """Write parent audit GeoJSON; join geometries from tile extents when available."""
    geom_by_key: dict[str, Any] = {}
    crs = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::25829"}}
    if extents_geojson is not None and extents_geojson.is_file():
        extents = json.loads(extents_geojson.read_text(encoding="utf-8"))
        crs = extents.get("crs", crs)
        for feat in extents.get("features", []):
            props = feat.get("properties") or {}
            yk = props.get("year_key")
            if yk:
                geom_by_key[str(yk)] = feat.get("geometry")

    features = []
    for row in parents:
        yk = row.get("year_key")
        geom = geom_by_key.get(str(yk)) if yk else None
        props = {
            k: v
            for k, v in row.items()
            if k != "bucket_counts"
        }
        props["bucket_counts"] = json.dumps(row.get("bucket_counts") or {}, sort_keys=True)
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": geom,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": path.stem,
                "crs": crs,
                "features": features,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return sum(1 for f in features if f.get("geometry") is not None)


def copy_holdout_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    link_mode: str,
) -> dict[str, Any]:
    ann_name = SPLIT_ANN[split]
    src_ann = input_dir / ann_name
    if not src_ann.is_file():
        return {"split": split, "status": "missing"}
    shutil.copy2(src_ann, output_dir / ann_name)
    src_img = input_dir / split
    dst_img = output_dir / split
    n = 0
    modes: dict[str, int] = {}
    if src_img.is_dir():
        dst_img.mkdir(parents=True, exist_ok=True)
        for path in src_img.iterdir():
            if path.is_file():
                used = link_or_copy(path, dst_img / path.name, link_mode)
                modes[used] = modes.get(used, 0) + 1
                n += 1
    return {"split": split, "status": "copied", "images": n, "link_modes": modes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--positive-oversample",
        type=float,
        default=1.5,
        help="Multiply positive tiles (1.0=no oversample, 1.5≈+50%% extras).",
    )
    parser.add_argument(
        "--keep-deposit",
        type=float,
        default=0.15,
        help="Fraction of deposit-heavy tiles to keep.",
    )
    parser.add_argument(
        "--keep-hilly",
        type=float,
        default=0.15,
        help="Fraction of green-hilly (low-row empty) tiles to keep.",
    )
    parser.add_argument(
        "--keep-empty-coast",
        type=float,
        default=0.30,
        help="Fraction of coastal empty tiles to keep (harder negatives).",
    )
    parser.add_argument(
        "--keep-empty-other",
        type=float,
        default=0.15,
        help="Fraction of other empty tiles to keep.",
    )
    parser.add_argument(
        "--hilly-row-max",
        type=int,
        default=6,
        help="Rows ≤ this count as map-top / green-hilly belt (default 6).",
    )
    parser.add_argument(
        "--link-mode",
        default="auto",
        choices=("auto", "hard", "symlink", "copy"),
    )
    parser.add_argument(
        "--tile-extents",
        type=Path,
        default=None,
        help=(
            "Optional tile extents GeoJSON (year_key property) to join into "
            "resample_train_audit_parents.geojson for QGIS before/after maps."
        ),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Where to write audit CSV/GeoJSON (default: --output-dir).",
    )
    args = parser.parse_args()

    train_ann = args.input_dir / SPLIT_ANN["train"]
    if not train_ann.is_file():
        raise SystemExit(f"Missing {train_ann}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = args.audit_dir or args.output_dir
    audit_dir.mkdir(parents=True, exist_ok=True)
    coco = load_coco(train_ann)
    new_coco, stats, image_audit = resample_train(
        coco,
        seed=args.seed,
        positive_oversample=args.positive_oversample,
        keep_deposit=args.keep_deposit,
        keep_hilly=args.keep_hilly,
        keep_empty_coast=args.keep_empty_coast,
        keep_empty_other=args.keep_empty_other,
        hilly_row_max=args.hilly_row_max,
    )
    (args.output_dir / SPLIT_ANN["train"]).write_text(
        json.dumps(new_coco), encoding="utf-8"
    )

    parent_audit = aggregate_parent_audit(image_audit)
    image_csv = audit_dir / "resample_train_audit_images.csv"
    parent_csv = audit_dir / "resample_train_audit_parents.csv"
    parent_geojson = audit_dir / "resample_train_audit_parents.geojson"
    write_csv(
        image_csv,
        image_audit,
        [
            "image_id",
            "file_name",
            "parent_stem",
            "aug_variant",
            "year",
            "row",
            "col",
            "year_key",
            "bucket",
            "action",
            "n_copies_out",
        ],
    )
    write_csv(
        parent_csv,
        parent_audit,
        [
            "year_key",
            "parent_stem",
            "year",
            "row",
            "col",
            "primary_bucket",
            "bucket_counts",
            "status",
            "n_images_in",
            "n_images_out",
            "n_images_removed",
            "oversample_factor",
            "removed",
            "oversampled",
        ],
    )
    n_joined = write_parent_geojson(parent_geojson, parent_audit, args.tile_extents)

    # Link train images referenced by the resampled JSON.
    train_img_out = args.output_dir / "train"
    train_img_out.mkdir(parents=True, exist_ok=True)
    modes: dict[str, int] = {}
    missing = []
    for im in new_coco["images"]:
        fn = im["file_name"]
        src = args.input_dir / "train" / fn
        if not src.is_file():
            for alt in ("valid", "test"):
                cand = args.input_dir / alt / fn
                if cand.is_file():
                    src = cand
                    break
        if not src.is_file():
            missing.append(fn)
            continue
        used = link_or_copy(src, train_img_out / fn, args.link_mode)
        modes[used] = modes.get(used, 0) + 1
    if missing:
        raise SystemExit(f"Missing {len(missing)} train images (e.g. {missing[0]})")

    holdouts = [
        copy_holdout_split(args.input_dir, args.output_dir, s, args.link_mode)
        for s in ("valid", "test")
    ]
    summary = {
        "input": str(args.input_dir),
        "output": str(args.output_dir),
        "train_resample": stats,
        "train_link_modes": modes,
        "holdouts": holdouts,
        "audit": {
            "images_csv": str(image_csv),
            "parents_csv": str(parent_csv),
            "parents_geojson": str(parent_geojson),
            "tile_extents": str(args.tile_extents) if args.tile_extents else None,
            "n_parents": len(parent_audit),
            "n_parents_with_geometry": n_joined,
            "n_parents_removed": sum(1 for p in parent_audit if p["removed"]),
            "n_parents_oversampled": sum(1 for p in parent_audit if p["oversampled"]),
            "policy": {
                "positive_oversample": args.positive_oversample,
                "keep_deposit": args.keep_deposit,
                "keep_hilly": args.keep_hilly,
                "keep_empty_coast": args.keep_empty_coast,
                "keep_empty_other": args.keep_empty_other,
                "hilly_row_max": args.hilly_row_max,
                "seed": args.seed,
            },
        },
    }
    (args.output_dir / "resample_train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    # Mirror summary into audit dir when separate (easy to find for QGIS).
    if audit_dir.resolve() != args.output_dir.resolve():
        (audit_dir / "resample_train_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))

    from run_provenance import write_dataset_provenance

    write_dataset_provenance(
        args.output_dir,
        tool="resample_coco_train.py",
        flags={
            "positive_oversample": args.positive_oversample,
            "keep_deposit": args.keep_deposit,
            "keep_hilly": args.keep_hilly,
            "keep_empty_coast": args.keep_empty_coast,
            "keep_empty_other": args.keep_empty_other,
            "hilly_row_max": args.hilly_row_max,
            "seed": args.seed,
            "link_mode": args.link_mode,
            "tile_extents": str(args.tile_extents) if args.tile_extents else None,
            "audit_dir": str(audit_dir),
        },
        splits_summary=summary,
        parents=[args.input_dir],
        notes=(
            "Train resampled (oversample positives / thin empties); valid/test copied. "
            "See resample_train_audit_parents.csv/.geojson for QGIS before/after."
        ),
    )


if __name__ == "__main__":
    main()
