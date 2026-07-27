#!/usr/bin/env python3
# RUNS ON: Windows (needs COCO jsons + optional GeoTIFF tiles for GSD; no model)
"""Dataset / config diagnostics for geo_split tiling-confound analysis.

For each discovered ``coco_geo512_{setup}_{modality}_from_pool`` (and optional
parent 2000 runs), writes one JSON with:

- effective network input GSD (m/px) from training ``image_size`` + tile size
- empty-chip fraction (trainable / any-annotation)
- boundary-truncation fraction
- instance counts in COCO S/M/L buckets — resized-pixel AND ground-m²
- anchor-scale vs boulder size histogram at network resolution

Also emits a CSV summary and a disk inventory of what was found (no hardcoded
run lists).

Example (project root)::

    python BoulderCalculator/scripts/geo_split_dataset_diagnostics.py \\
      --segmentation-dir segmentation \\
      --output-dir segmentation/geo_split_diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_split_run_index import (  # noqa: E402
    ANN_FILES,
    DEFAULT_GSD_M,
    SETUPS,
    analyze_coco_split,
    discover_geo512_coco_dirs,
    discover_geo512_runs,
    discover_parent_geo_runs,
    expected_run_dir,
    glob_report,
    load_json,
    load_provenance,
    read_gsd_m,
    resolve_dataset_dir,
)
from run_provenance import load_provenance as load_any_provenance  # noqa: E402


def _image_size_for_pair(
    setup: str,
    modality: str,
    coco_dir: Path,
    seg: Path,
    tiling: str,
    default_image_size: int,
) -> tuple[int, str, dict[str, Any] | None]:
    """Prefer training provenance image_size; else CLI default."""
    run_dir = expected_run_dir(seg, setup, modality, tiling=tiling)
    prov = load_provenance(run_dir) if run_dir.is_dir() else None
    if prov and (prov.get("flags") or {}).get("image_size") is not None:
        return int(prov["flags"]["image_size"]), "training_run_provenance", prov
    # Dataset provenance may not carry image_size; use tiling default.
    return default_image_size, "cli_default", prov


def _load_split(coco_dir: Path, split: str) -> dict[str, Any] | None:
    path = coco_dir / ANN_FILES[split]
    if not path.is_file():
        return None
    return load_json(path)


def diagnose_one(
    *,
    setup: str,
    modality: str,
    coco_dir: Path,
    seg: Path,
    tiling: str,
    splits: list[str],
    default_image_size: int,
    default_gsd: float,
) -> dict[str, Any]:
    image_size, image_size_source, train_prov = _image_size_for_pair(
        setup, modality, coco_dir, seg, tiling, default_image_size
    )
    ds_prov = load_any_provenance(coco_dir)
    row: dict[str, Any] = {
        "setup": setup,
        "modality": modality,
        "tiling_regime": tiling,
        "coco_dir": str(coco_dir),
        "image_size": image_size,
        "image_size_source": image_size_source,
        "training_run_dir": str(expected_run_dir(seg, setup, modality, tiling=tiling)),
        "training_flags": (train_prov or {}).get("flags"),
        "dataset_provenance_flags": (ds_prov or {}).get("flags") if ds_prov else None,
        "splits": {},
    }

    # Seed GSD from any available tile, else default.
    seed_gsd, seed_src = default_gsd, "cli_default"
    for split in splits:
        coco = _load_split(coco_dir, split)
        if not coco or not coco.get("images"):
            continue
        fn = coco["images"][0]["file_name"]
        for cand in (
            coco_dir / split / fn,
            coco_dir / "train" / fn,
            coco_dir / fn,
        ):
            if cand.is_file():
                seed_gsd, seed_src = read_gsd_m(cand, default=default_gsd)
                break
        break

    for split in splits:
        coco = _load_split(coco_dir, split)
        if coco is None:
            row["splits"][split] = {"error": f"missing {ANN_FILES[split]}"}
            continue
        stats = analyze_coco_split(
            coco,
            image_size=image_size,
            native_gsd_m=seed_gsd,
            gsd_source=seed_src,
            dataset_dir=coco_dir,
            split=split,
        )
        row["splits"][split] = stats

    # Headline fields from train split (fallback valid/test).
    primary = None
    for split in ("train", "valid", "test"):
        if split in row["splits"] and "n_images" in row["splits"][split]:
            primary = row["splits"][split]
            row["primary_split"] = split
            break
    if primary:
        row["empty_chip_fraction"] = primary["empty_chip_fraction_trainable"]
        row["boundary_truncation_fraction"] = primary["boundary_truncation_fraction"]
        row["effective_network_gsd_m"] = primary["effective_network_gsd_m"]
        row["native_gsd_m"] = primary["native_gsd_m"]
        row["size_buckets_pixel_resized"] = primary["size_buckets_pixel_resized"]
        row["size_buckets_ground_m2"] = primary["size_buckets_ground_m2"]
        row["bucket_flip_fraction_resized_vs_ground"] = primary[
            "bucket_flip_fraction_resized_vs_ground"
        ]
        row["n_instances_trainable"] = primary["n_instances_trainable"]
    return row


def _flatten_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "setup": row.get("setup"),
        "modality": row.get("modality"),
        "tiling_regime": row.get("tiling_regime"),
        "image_size": row.get("image_size"),
        "image_size_source": row.get("image_size_source"),
        "primary_split": row.get("primary_split"),
        "empty_chip_fraction": row.get("empty_chip_fraction"),
        "boundary_truncation_fraction": row.get("boundary_truncation_fraction"),
        "effective_network_gsd_m": row.get("effective_network_gsd_m"),
        "native_gsd_m": row.get("native_gsd_m"),
        "n_instances_trainable": row.get("n_instances_trainable"),
        "bucket_flip_fraction_resized_vs_ground": row.get(
            "bucket_flip_fraction_resized_vs_ground"
        ),
        "coco_dir": row.get("coco_dir"),
    }
    for kind, key in (
        ("pixel", "size_buckets_pixel_resized"),
        ("ground", "size_buckets_ground_m2"),
    ):
        buckets = row.get(key) or {}
        for b in ("small", "medium", "large"):
            flat[f"n_{kind}_{b}"] = buckets.get(b)
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("segmentation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmentation/geo_split_diagnostics"),
    )
    parser.add_argument(
        "--splits",
        default="train,valid,test",
        help="Comma-separated COCO splits to diagnose (default: train,valid,test).",
    )
    parser.add_argument(
        "--default-image-size-512",
        type=int,
        default=512,
        help="Fallback network resize when no training provenance (geo512).",
    )
    parser.add_argument(
        "--default-image-size-2000",
        type=int,
        default=2000,
        help="Fallback network resize for parent 2000 runs.",
    )
    parser.add_argument(
        "--default-gsd-m",
        type=float,
        default=DEFAULT_GSD_M,
        help=f"Fallback meters/pixel when GeoTIFF unreadable (default {DEFAULT_GSD_M}).",
    )
    parser.add_argument(
        "--include-parent-2000",
        action="store_true",
        help="Also diagnose coco dirs paired with training_run_geo_* (2000 tiles).",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only write disk inventory JSON (no COCO analysis).",
    )
    args = parser.parse_args()

    seg = args.segmentation_dir
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    inventory = glob_report(seg)
    # Also list expected matrix cells that are missing.
    found_keys = {
        (r["setup"], r["modality"]) for r in inventory["geo512_coco"]
    }
    missing = []
    for setup in SETUPS:
        for modality in ("rgb", "rgb_dsm", "rgb_local_relief"):
            if (setup, modality) not in found_keys:
                missing.append({"setup": setup, "modality": modality})
    inventory["missing_geo512_coco_cells"] = missing
    inv_path = out / "disk_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {inv_path}")
    print(
        f"Found geo512 COCO={inventory['n_geo512_coco']} "
        f"runs={inventory['n_geo512_runs']} "
        f"parent_runs={inventory['n_parent_runs']} "
        f"missing_cells={len(missing)}"
    )
    if args.inventory_only:
        return

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    rows: list[dict[str, Any]] = []

    for setup, modality, coco_dir in discover_geo512_coco_dirs(seg):
        print(f"Diagnosing {setup}/{modality} ({coco_dir.name})…")
        row = diagnose_one(
            setup=setup,
            modality=modality,
            coco_dir=coco_dir,
            seg=seg,
            tiling="512",
            splits=splits,
            default_image_size=args.default_image_size_512,
            default_gsd=args.default_gsd_m,
        )
        rows.append(row)
        path = out / f"diag_{setup}_{modality}_512.json"
        path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print(f"  Wrote {path}")

    if args.include_parent_2000:
        # Pair parent runs → their dataset dirs.
        seen: set[str] = set()
        for run in discover_parent_geo_runs(seg):
            dataset = resolve_dataset_dir(run, seg)
            if dataset is None:
                print(f"[skip] parent run {run.run_dir.name}: no dataset_dir")
                continue
            key = str(dataset.resolve())
            if key in seen:
                continue
            seen.add(key)
            print(f"Diagnosing parent {run.setup}/{run.modality}…")
            row = diagnose_one(
                setup=run.setup,
                modality=run.modality,
                coco_dir=dataset,
                seg=seg,
                tiling="2000",
                splits=splits,
                default_image_size=args.default_image_size_2000,
                default_gsd=args.default_gsd_m,
            )
            rows.append(row)
            path = out / f"diag_{run.setup}_{run.modality}_2000.json"
            path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            print(f"  Wrote {path}")

    if not rows:
        print(
            "WARNING: no COCO dirs found under "
            f"{seg}. Inventory written; re-run on Windows after datasets exist."
        )
        return

    csv_path = out / "diagnostics_summary.csv"
    flat_rows = [_flatten_csv_row(r) for r in rows]
    fieldnames = list(flat_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Wrote {csv_path} ({len(flat_rows)} rows)")

    # Convenience: also dump geo512 run list with checkpoints for Deliverable 2.
    runs = discover_geo512_runs(seg)
    runs_path = out / "discovered_runs.json"
    runs_path.write_text(
        json.dumps([r.to_dict() for r in runs], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {runs_path} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
