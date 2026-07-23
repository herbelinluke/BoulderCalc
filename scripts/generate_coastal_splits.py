#!/usr/bin/env python3
"""Generate alternate geographic train/valid/test split YAMLs + QGIS GeoJSONs.

Reads tile GeoTIFF footprints, bins centroids along the first coastal PCA axis
into 12 segments (matching the comment in gpkg_to_coco.py), and writes:

  - blocks_alt_a.yaml  valid={1,7}  test={4,10} + abutting buffer
  - blocks_alt_b.yaml  valid={0,6}  test={3,9}  + abutting buffer
  - north_south.yaml   contiguous coastal halves + buffer strip
  - sporadic_aligned.yaml  random locations, no buffer, year-aligned clusters

Also writes tile_extents_<id>.geojson (EPSG:25829) with a ``split`` property
for QGIS styling. Does not overwrite baseline.yaml (hand-curated).

Example:
  python BoulderCalculator/scripts/generate_coastal_splits.py \\
    --segmentation-dir segmentation \\
    --output-dir BoulderCalculator/experiments/geo_splits
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import yaml
from shapely.geometry import box, mapping

# Allow importing sibling scripts when run as a file.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from gpkg_to_coco import (  # noqa: E402
    TILES_24,
    TILES_25,
    canonical_key,
    footprint_overlaps,
    resolve_tile_path,
    resolve_tiles_by_year,
    year_key,
)

N_SEGMENTS = 12
TOUCH_TOL_M = 2.0  # treat near-touching footprints as abutting for buffers


def load_bounds(
    tile_dir: Path, years: list[int], tiles_by_year: dict[int, list[str]]
) -> dict[str, tuple[float, float, float, float]]:
    cache: dict[str, tuple[float, float, float, float]] = {}
    for year in years:
        for key in tiles_by_year[year]:
            yk = year_key(year, key)
            with rasterio.open(resolve_tile_path(tile_dir, yk)) as ds:
                b = ds.bounds
                cache[yk] = (b.left, b.bottom, b.right, b.top)
    return cache


def centroid(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def pca_axis_scores(bounds: dict[str, tuple[float, float, float, float]]) -> dict[str, float]:
    keys = sorted(bounds)
    pts = np.array([centroid(bounds[k]) for k in keys], dtype=float)
    mean = pts.mean(axis=0)
    centered = pts - mean
    # 2x2 covariance; first eigenvector = coastal axis
    cov = centered.T @ centered / max(len(keys) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    scores = centered @ axis
    return {k: float(s) for k, s in zip(keys, scores)}


def assign_segments(scores: dict[str, float], n: int = N_SEGMENTS) -> dict[str, int]:
    """Equal-count bins along PCA score (0 .. n-1)."""
    items = sorted(scores.items(), key=lambda kv: kv[1])
    if not items:
        return {}
    out: dict[str, int] = {}
    for i, (yk, _) in enumerate(items):
        # floor division into n nearly equal buckets
        seg = min(n - 1, (i * n) // len(items))
        out[yk] = seg
    return out


def expand_bounds(
    b: tuple[float, float, float, float], tol: float
) -> tuple[float, float, float, float]:
    return (b[0] - tol, b[1] - tol, b[2] + tol, b[3] + tol)


def abutting_buffer(
    bounds: dict[str, tuple[float, float, float, float]],
    holdout: set[str],
    eligible: set[str],
) -> set[str]:
    """Tiles that touch hold-outs but are not themselves hold-outs."""
    buf: set[str] = set()
    for yk in eligible:
        if yk in holdout:
            continue
        be = expand_bounds(bounds[yk], TOUCH_TOL_M)
        for h in holdout:
            if footprint_overlaps(be, bounds[h], min_area=0.01):
                buf.add(yk)
                break
    return buf


def year_maps_from_keys(keys: set[str]) -> dict[int, list[str]]:
    by: dict[int, list[str]] = defaultdict(list)
    for yk in sorted(keys):
        # year_key is f"{year}_{row:02d}_{col:02d}"
        parts = yk.split("_")
        year = int(parts[0])
        short = canonical_key(f"{int(parts[1])}_{int(parts[2])}")
        by[year].append(short)
    for year in by:
        by[year] = sorted(set(by[year]), key=lambda k: tuple(map(int, k.split("_"))))
    return {24: by.get(24, []), 25: by.get(25, [])}


def split_dict(
    setup_id: str,
    description: str,
    leakage_check: str,
    valid: set[str],
    test: set[str],
    excluded: set[str],
) -> dict:
    return {
        "id": setup_id,
        "description": description.strip(),
        "leakage_check": leakage_check,
        "valid": year_maps_from_keys(valid),
        "test": year_maps_from_keys(test),
        "excluded": year_maps_from_keys(excluded),
    }


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    )
    print(f"Wrote {path}")


def write_split_geojson(
    path: Path,
    bounds: dict[str, tuple[float, float, float, float]],
    assignment: dict[str, str],
) -> None:
    features = []
    for yk, split in sorted(assignment.items()):
        parts = yk.split("_")
        year = int(parts[0])
        short = f"{int(parts[1])}_{int(parts[2])}"
        poly = box(*bounds[yk])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "tile": short,
                    "year": year,
                    "year_key": yk,
                    "split": split,
                },
                "geometry": mapping(poly),
            }
        )
    geojson = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::25829"},
        },
        "features": features,
    }
    path.write_text(json.dumps(geojson))
    print(f"Wrote {len(features)} features to {path}")


def overlap_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def location_clusters(
    bounds: dict[str, tuple[float, float, float, float]],
) -> list[list[str]]:
    """Pair each tile with at most one opposite-year partner (greedy IoU/area).

    Avoids transitive coastal chains: we want year-aligned *locations*, not one
    giant connected component. Unmatched tiles become singletons.
    """
    keys_24 = [k for k in bounds if k.startswith("24_")]
    keys_25 = [k for k in bounds if k.startswith("25_")]
    pairs: list[tuple[float, str, str]] = []
    for a in keys_24:
        for b in keys_25:
            area = overlap_area(bounds[a], bounds[b])
            if area >= 1.0:
                pairs.append((area, a, b))
    pairs.sort(key=lambda t: t[0], reverse=True)

    used: set[str] = set()
    clusters: list[list[str]] = []
    for _area, a, b in pairs:
        if a in used or b in used:
            continue
        clusters.append([a, b])
        used.add(a)
        used.add(b)
    for k in bounds:
        if k not in used:
            clusters.append([k])
    return clusters


def build_block_setup(
    setup_id: str,
    description: str,
    segments: dict[str, int],
    bounds: dict[str, tuple[float, float, float, float]],
    valid_segs: set[int],
    test_segs: set[int],
) -> tuple[dict, dict[str, str]]:
    all_keys = set(segments)
    valid = {k for k, s in segments.items() if s in valid_segs}
    test = {k for k, s in segments.items() if s in test_segs}
    if valid & test:
        raise ValueError(f"{setup_id}: valid/test segment overlap in tile IDs")
    holdout = valid | test
    excluded = abutting_buffer(bounds, holdout, all_keys)
    # Drop any accidental hold-out from excluded
    excluded -= holdout
    assignment = {}
    for k in all_keys:
        if k in excluded:
            assignment[k] = "excluded"
        elif k in valid:
            assignment[k] = "valid"
        elif k in test:
            assignment[k] = "test"
        else:
            assignment[k] = "train"
    data = split_dict(
        setup_id,
        description,
        "geographic",
        valid,
        test,
        excluded,
    )
    return data, assignment


def build_north_south(
    scores: dict[str, float],
    bounds: dict[str, tuple[float, float, float, float]],
    valid_frac: float = 0.15,
    test_frac: float = 0.25,
) -> tuple[dict, dict[str, str]]:
    """Contiguous ends of the coast: low-score → valid, high-score → test."""
    items = sorted(scores.items(), key=lambda kv: kv[1])
    n = len(items)
    n_valid = max(1, int(round(n * valid_frac)))
    n_test = max(1, int(round(n * test_frac)))
    # Leave a gap in the middle for train; trim if needed
    while n_valid + n_test >= n and (n_valid > 1 or n_test > 1):
        if n_valid >= n_test and n_valid > 1:
            n_valid -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    valid = {k for k, _ in items[:n_valid]}
    test = {k for k, _ in items[-n_test:]}
    # If the coast is short enough that ends meet, shrink test
    overlap = valid & test
    if overlap:
        test -= overlap
    holdout = valid | test
    excluded = abutting_buffer(bounds, holdout, set(scores))
    excluded -= holdout
    assignment = {}
    for k in scores:
        if k in excluded:
            assignment[k] = "excluded"
        elif k in valid:
            assignment[k] = "valid"
        elif k in test:
            assignment[k] = "test"
        else:
            assignment[k] = "train"
    data = split_dict(
        "north_south",
        "Contiguous coastal ends along PCA axis (low→valid, high→test) + abutting buffer.",
        "geographic",
        valid,
        test,
        excluded,
    )
    return data, assignment


def build_sporadic(
    bounds: dict[str, tuple[float, float, float, float]],
    seed: int = 42,
    valid_frac: float = 0.15,
    test_frac: float = 0.25,
) -> tuple[dict, dict[str, str]]:
    clusters = location_clusters(bounds)
    rng = random.Random(seed)
    order = list(range(len(clusters)))
    rng.shuffle(order)
    n = len(clusters)
    n_valid = max(1, int(round(n * valid_frac)))
    n_test = max(1, int(round(n * test_frac)))
    while n_valid + n_test >= n and (n_valid > 1 or n_test > 1):
        if n_valid >= n_test and n_valid > 1:
            n_valid -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    valid_idx = set(order[:n_valid])
    test_idx = set(order[n_valid : n_valid + n_test])
    valid: set[str] = set()
    test: set[str] = set()
    for i, members in enumerate(clusters):
        if i in valid_idx:
            valid.update(members)
        elif i in test_idx:
            test.update(members)
    assignment = {}
    for k in bounds:
        if k in valid:
            assignment[k] = "valid"
        elif k in test:
            assignment[k] = "test"
        else:
            assignment[k] = "train"
    data = split_dict(
        "sporadic_aligned",
        (
            f"Random location clusters (seed={seed}), no buffer; "
            "footprint-overlapping tiles (incl. cross-year) stay in one split."
        ),
        "location_consistency",
        valid,
        test,
        set(),
    )
    return data, assignment


def assignment_from_config(
    bounds: dict[str, tuple[float, float, float, float]],
    data: dict,
) -> dict[str, str]:
    """Map year-prefixed keys → split using a loaded config dict."""
    valid = set()
    test = set()
    excluded = set()
    for year, keys in (data.get("valid") or {}).items():
        for k in keys:
            valid.add(year_key(int(year), k))
    for year, keys in (data.get("test") or {}).items():
        for k in keys:
            test.add(year_key(int(year), k))
    for year, keys in (data.get("excluded") or {}).items():
        for k in keys:
            excluded.add(year_key(int(year), k))
    assignment = {}
    for yk in bounds:
        if yk in excluded:
            assignment[yk] = "excluded"
        elif yk in valid:
            assignment[yk] = "valid"
        elif yk in test:
            assignment[yk] = "test"
        else:
            assignment[yk] = "train"
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=Path("segmentation"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("BoulderCalculator/experiments/geo_splits"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--also-baseline-geojson",
        action="store_true",
        help="Also write tile_extents_baseline.geojson from baseline.yaml",
    )
    args = parser.parse_args()

    seg_dir = args.segmentation_dir
    tile_dir = seg_dir / "tiling"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles_by_year = resolve_tiles_by_year(seg_dir, None)
    # Prefer on-disk tiles_used; fall back to baked lists if empty year
    if not tiles_by_year.get(24):
        tiles_by_year[24] = list(TILES_24)
    if not tiles_by_year.get(25):
        tiles_by_year[25] = list(TILES_25)

    years = [24, 25]
    bounds = load_bounds(tile_dir, years, tiles_by_year)
    scores = pca_axis_scores(bounds)
    segments = assign_segments(scores)

    # Segment occupancy summary
    hist = defaultdict(int)
    for s in segments.values():
        hist[s] += 1
    print("PCA segment tile counts:", dict(sorted(hist.items())))

    setups: list[tuple[dict, dict[str, str]]] = []

    setups.append(
        build_block_setup(
            "blocks_alt_a",
            "Coastal PCA bins; valid={1,7}, test={4,10} + abutting buffer.",
            segments,
            bounds,
            valid_segs={1, 7},
            test_segs={4, 10},
        )
    )
    setups.append(
        build_block_setup(
            "blocks_alt_b",
            "Coastal PCA bins; valid={0,6}, test={3,9} + abutting buffer.",
            segments,
            bounds,
            valid_segs={0, 6},
            test_segs={3, 9},
        )
    )
    setups.append(build_north_south(scores, bounds))
    setups.append(build_sporadic(bounds, seed=args.seed))

    for data, assignment in setups:
        write_yaml(out_dir / f"{data['id']}.yaml", data)
        write_split_geojson(out_dir / f"tile_extents_{data['id']}.geojson", bounds, assignment)
        n = {s: sum(1 for v in assignment.values() if v == s) for s in ("train", "valid", "test", "excluded")}
        print(f"  {data['id']}: {n}")

    if args.also_baseline_geojson:
        baseline_path = out_dir / "baseline.yaml"
        if baseline_path.exists():
            with baseline_path.open() as f:
                baseline = yaml.safe_load(f)
            assignment = assignment_from_config(bounds, baseline)
            write_split_geojson(
                out_dir / "tile_extents_baseline.geojson", bounds, assignment
            )
        else:
            print(f"Skip baseline geojson: {baseline_path} missing")


if __name__ == "__main__":
    main()
