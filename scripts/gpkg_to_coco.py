#!/usr/bin/env python3
"""Convert a QGIS GPKG annotation layer + GeoTIFF tiles into Detectron2 COCO JSON.

Supports year-based tile layouts under segmentation/tiling/{24,25}/, optional
ROI clipping, multi-year training in one dataset, and boulder-only mode.

Annotation GPKGs can be year-tagged so each year's polygons only label that
year's tiles (important if footprints ever overlap):

  --gpkg july14_24.gpkg:24,july14_25.gpkg:25

Tile lists default from ``segmentation/annotations/tiles_used.txt`` (or the
baked-in copy of that file). Override with ``--tiles-used``.

Class attribute handling:
  - numeric 0 / missing / string "Boulder"  -> Boulder (COCO category 1)
  - numeric 1 / string containing "deposit" -> BoulderDeposit

With ``--boulder-only`` (default), deposits and Boulder polygons smaller than
``--min-area-m2`` are kept as COCO ``iscrowd=1`` ignore regions (same Boulder
category). Train with ``train_boulder_local.py``, which treats those crowds as
neither positives nor negatives. Use ``--no-boulder-only`` for a trainable
two-class dataset (deposits as category 2; small boulders still become crowds
when ``--min-area-m2`` > 0).

Example (both years, per-year GPKGs, boulder-only):
    python BoulderCalculator/scripts/gpkg_to_coco.py \\
        --segmentation-dir segmentation \\
        --years 24,25 \\
        --gpkg segmentation/annotations/july14_24.gpkg:24,segmentation/annotations/july14_25.gpkg:25 \\
        --output-dir segmentation/coco_dataset_both \\
        --min-area-m2 1.0

ROI clipping is off by default. Pass ``--roi path.shp`` (or multiple
comma-separated paths) only if you want to re-enable it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import fiona
import rasterio
from rasterio.warp import transform as warp_transform
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shp_transform, unary_union

WORKING_EPSG = "EPSG:25829"

CATEGORIES_TWO = [
    {"id": 1, "name": "Boulder", "supercategory": "none"},
    {"id": 2, "name": "BoulderDeposit", "supercategory": "none"},
]
CATEGORIES_ONE = [
    {"id": 1, "name": "Boulder", "supercategory": "none"},
]

# Snapshot of segmentation/annotations/tiles_used.txt (2026-07-13).
# Keys are "R_C" (no zero-padding). Prefer --tiles-used when the file is present.
_TILES_USED_TEXT = """\
2025 tiles annotated:
3:35-38
4:33-38
5:30-35
6:27-31
7:24-29
8:20-27
9:18-25
10:5-9, 16-21
11:5-18
12:5-17
13:5-13

2024 tiles annotated:
4:44-46
5:42-46
6:40-45
7:38-43
8:35-40
9:32-36
10:5-7, 28-32
11:5-7, 24-29
12:7-9, 23-28
13:7-13, 21-23
14:7-22
15:7-20
16:7-17
"""


def parse_col_spec(spec: str) -> list[int]:
    """Parse '5-9, 16-21' or '36' into a list of column indices."""
    cols: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cols.extend(range(int(lo), int(hi) + 1))
        else:
            cols.append(int(part))
    return cols


def parse_tiles_used_text(text: str) -> dict[int, list[str]]:
    """Parse tiles_used.txt into {24: [...], 25: [...]}."""
    by_year: dict[int, list[str]] = {24: [], 25: []}
    year: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if "2025" in low and "tile" in low:
            year = 25
            continue
        if "2024" in low and "tile" in low:
            year = 24
            continue
        if year is None or ":" not in line:
            continue
        row_s, cols_s = line.split(":", 1)
        row = int(row_s.strip())
        for col in parse_col_spec(cols_s):
            by_year[year].append(f"{row}_{col}")
    return by_year


def load_tiles_used(path: Path) -> dict[int, list[str]]:
    data = parse_tiles_used_text(path.read_text())
    if not data.get(24) or not data.get(25):
        raise ValueError(f"{path} must define both 2024 and 2025 tile ranges")
    return data


_DEFAULT_TILES = parse_tiles_used_text(_TILES_USED_TEXT)
TILES_24 = list(_DEFAULT_TILES[24])
TILES_25 = list(_DEFAULT_TILES[25])

# Geographic leakage-safe hold-outs (EPSG:25829 footprint blocks along the coast).
# Built so no train/valid/test pair shares overlapping ground (incl. cross-year).
# EXCLUDED_* tiles abut hold-outs and are dropped from every split (spatial buffer).
# Blocks: 12 segments along the coast PCA axis; valid={2,8}, test={5,11}.
VALID_24 = [
    "7_38",
    "7_39",
    "8_37",
    "8_38",
    "8_39",
    "14_17",
    "14_18",
    "14_19",
    "14_20",
    "15_17",
    "15_18",
    "15_19",
    "15_20",
    "16_17",
]
TEST_24 = [
    "10_5",
    "10_6",
    "10_7",
    "10_28",
    "10_29",
    "11_5",
    "11_6",
    "11_7",
    "11_27",
    "11_28",
    "11_29",
    "12_7",
    "12_8",
    "12_27",
    "12_28",
    "13_7",
    "13_8",
    "13_9",
    "14_7",
    "14_8",
    "14_9",
    "15_7",
    "15_8",
    "15_9",
    "16_7",
    "16_8",
    "16_9",
]
EXCLUDED_24 = [
    "8_36",
    "9_36",
    "11_26",
    "12_9",
    "12_26",
    "13_10",
    "14_10",
    "14_16",
    "15_10",
    "15_16",
    "16_10",
    "16_16",
]
VALID_25 = [
    "4_33",
    "5_31",
    "5_32",
    "5_33",
    "6_31",
    "11_13",
    "11_14",
    "11_15",
    "12_13",
    "12_14",
    "12_15",
    "12_16",
]
TEST_25 = [
    "7_24",
    "8_22",
    "8_23",
    "8_24",
    "9_22",
    "9_23",
    "9_24",
    "9_25",
    "10_5",
    "11_5",
    "11_6",
    "12_5",
    "12_6",
    "13_5",
    "13_6",
]
EXCLUDED_25 = [
    "4_34",
    "5_34",
    "7_25",
    "8_25",
    "10_6",
    "11_16",
    "13_13",
]

# Back-compat aliases used by build_rgb_dsm_tiles.py
ALL_TILES = TILES_25
VALID_TILES = VALID_25
TEST_TILES = TEST_25
TRAIN_TILES = [
    k
    for k in ALL_TILES
    if k not in VALID_TILES + TEST_TILES + EXCLUDED_25
]


def normalize_key(key: str) -> tuple[int, int]:
    """Parse '11_7' / '11_07' / '24_11_07' into (row, col). Year prefix optional."""
    parts = key.strip().split("_")
    if len(parts) == 3:
        parts = parts[1:]
    if len(parts) != 2:
        raise ValueError(f"Bad tile key: {key}")
    return int(parts[0]), int(parts[1])


def tile_key(row: int, col: int) -> str:
    return f"{row}_{col}"


def canonical_key(key: str) -> str:
    row, col = normalize_key(key)
    return tile_key(row, col)


def tile_filename(key: str, year: int = 25) -> str:
    row, col = normalize_key(key)
    if year == 24:
        return f"Sites1and2_2024_Orthomosaic_{row:02d}_{col:02d}.tif"
    return f"25IniSouthOrt_{row:02d}_{col:02d}.tif"


def parse_year_key(key: str, default_year: int | None = None) -> tuple[int, str]:
    """Parse '14_15' or '24_14_15' into (year, '14_15')."""
    parts = key.strip().split("_")
    if len(parts) == 3 and parts[0] in ("24", "25"):
        return int(parts[0]), tile_key(int(parts[1]), int(parts[2]))
    if len(parts) == 2:
        if default_year is None:
            raise ValueError(f"Tile key {key!r} needs a year prefix (e.g. 24_14_15) or --year")
        return default_year, tile_key(int(parts[0]), int(parts[1]))
    raise ValueError(f"Bad tile key: {key}")


def year_key(year: int, key: str) -> str:
    row, col = normalize_key(key)
    return f"{year}_{row:02d}_{col:02d}"


def tiles_for_year(year: int, tiles_by_year: dict[int, list[str]] | None = None) -> list[str]:
    source = tiles_by_year or {24: TILES_24, 25: TILES_25}
    return [canonical_key(k) for k in source[year]]


def expand_year_keys(
    years: list[int],
    tiles_by_year: dict[int, list[str]] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (train, valid, test) year-prefixed keys for the given years.

    Hold-outs are geographic blocks shared across years (no footprint overlap
    between train/valid/test). EXCLUDED_* tiles are omitted from every split.
    """
    train, valid, test = [], [], []
    for year in years:
        tiles = tiles_for_year(year, tiles_by_year)
        valid_set = {canonical_key(k) for k in (VALID_24 if year == 24 else VALID_25)}
        test_set = {canonical_key(k) for k in (TEST_24 if year == 24 else TEST_25)}
        excluded_set = {
            canonical_key(k) for k in (EXCLUDED_24 if year == 24 else EXCLUDED_25)
        }
        missing_holdouts = sorted((valid_set | test_set) - set(tiles))
        if missing_holdouts:
            raise ValueError(
                f"Hold-out tiles not in year {year} tile list: {missing_holdouts}. "
                "Update VALID_*/TEST_* or tiles_used.txt."
            )
        overlap_holdouts = sorted(valid_set & test_set)
        if overlap_holdouts:
            raise ValueError(f"Tile(s) listed in both valid and test for {year}: {overlap_holdouts}")
        for key in tiles:
            if key in excluded_set:
                continue
            yk = year_key(year, key)
            if key in valid_set:
                valid.append(yk)
            elif key in test_set:
                test.append(yk)
            else:
                train.append(yk)
    # ID-level leakage guard
    s_train, s_valid, s_test = set(train), set(valid), set(test)
    if s_train & s_valid or s_train & s_test or s_valid & s_test:
        raise ValueError(
            "Split leakage: duplicate year-prefixed tile IDs across train/valid/test"
        )
    return train, valid, test


def footprint_overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float], min_area: float = 1.0) -> bool:
    """Axis-aligned bounds overlap with area >= min_area (map units^2)."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0) >= min_area


def assert_no_geographic_leakage(
    tile_dir: Path,
    train: list[str],
    valid: list[str],
    test: list[str],
) -> None:
    """Raise if any train/valid/test pair shares overlapping GeoTIFF footprints."""
    import rasterio

    def bounds(yk: str) -> tuple[float, float, float, float]:
        with rasterio.open(resolve_tile_path(tile_dir, yk)) as ds:
            b = ds.bounds
            return (b.left, b.bottom, b.right, b.top)

    cache = {yk: bounds(yk) for yk in train + valid + test}

    def check(name_a: str, keys_a: list[str], name_b: str, keys_b: list[str]) -> None:
        hits = []
        for a in keys_a:
            ba = cache[a]
            for b in keys_b:
                if footprint_overlaps(ba, cache[b]):
                    hits.append((a, b))
        if hits:
            sample = ", ".join(f"{a}∩{b}" for a, b in hits[:8])
            more = f" (+{len(hits) - 8} more)" if len(hits) > 8 else ""
            raise ValueError(
                f"Geographic leakage between {name_a} and {name_b}: "
                f"{len(hits)} overlapping tile pair(s). e.g. {sample}{more}"
            )

    check("train", train, "valid", valid)
    check("train", train, "test", test)
    check("valid", valid, "test", test)


def resolve_tile_path(tile_dir: Path, key: str, year: int | None = None) -> Path:
    """Find a tile under tiling/{year}/ or flat tiling/ (legacy).

    ``key`` may be ``14_15`` (with year=) or year-prefixed ``24_14_15``.
    """
    year, short = parse_year_key(key, year)
    name = tile_filename(short, year)
    candidates = [
        tile_dir / str(year) / name,
        tile_dir / name,
    ]
    if year == 25:
        row, col = normalize_key(short)
        candidates.append(tile_dir / f"25IniSouthOrt_{row}_{col}.tif")
        candidates.append(tile_dir / "25" / f"25IniSouthOrt_{row}_{col}.tif")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Tile not found for key={key} year={year}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def make_reprojector(src_crs, dst_crs):
    def func(xs, ys):
        out_x, out_y = warp_transform(src_crs, dst_crs, list(xs), list(ys))
        return out_x, out_y

    return func


def parse_class_value(raw) -> int:
    """Map Class attribute to 0=Boulder, 1=BoulderDeposit."""
    if raw is None:
        return 0
    if isinstance(raw, (int, float)):
        return 1 if int(raw) == 1 else 0
    text = str(raw).strip().lower()
    if not text or text in ("0", "boulder", "boulders"):
        return 0
    if "deposit" in text or text == "1":
        return 1
    return 0


def load_annotations(
    gpkg_path: Path,
    layer: str | None,
    class_field: str,
    boulder_only: bool,
    year: int | None = None,
) -> list[tuple]:
    """Return [(geom in WORKING_EPSG, class 0|1, year|None)].

    ``year`` tags features so they only label tiles from that ortho year.
    ``None`` means the feature may apply to any selected year (merged GPKG).

    ``boulder_only`` no longer drops deposits; they are kept and emitted as
    ``iscrowd=1`` later in ``convert_tile``. The flag is retained for call-site
    compatibility / logging.
    """
    del boulder_only  # deposits are loaded; crowding happens at convert time
    kwargs = {"layer": layer} if layer else {}
    feats: list[tuple] = []
    unknown_class = 0
    n_deposit = 0
    skipped_null_geom = 0
    skipped_bad_geom = 0
    with fiona.open(gpkg_path, **kwargs) as src:
        reproject = make_reprojector(src.crs, WORKING_EPSG)
        for feat in src:
            raw_geom = feat.get("geometry")
            # QGIS exports sometimes leave null geometry rows; skip them.
            # (Do not require dict — Fiona may return Mapping / Geometry objects.)
            if raw_geom is None:
                skipped_null_geom += 1
                continue
            try:
                geom = shape(raw_geom)
            except (AttributeError, ValueError, TypeError):
                skipped_bad_geom += 1
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                skipped_null_geom += 1
                continue
            geom = shp_transform(reproject, geom)
            raw = feat["properties"].get(class_field)
            if raw is None:
                unknown_class += 1
            cls = parse_class_value(raw)
            if cls == 1:
                n_deposit += 1
            feats.append((geom, cls, year))
    tag = f" year={year}" if year is not None else " year=any"
    if skipped_null_geom:
        print(
            f"WARNING: skipped {skipped_null_geom} feature(s) with null/empty "
            f"geometry in {gpkg_path.name} [{tag.strip()}]"
        )
    if skipped_bad_geom:
        print(
            f"WARNING: skipped {skipped_bad_geom} feature(s) with unreadable "
            f"geometry in {gpkg_path.name} [{tag.strip()}]"
        )
    if unknown_class:
        print(
            f"WARNING: {unknown_class} feature(s) in {gpkg_path.name} missing "
            f"'{class_field}', treated as Boulder (0) [{tag.strip()}]"
        )
    if n_deposit:
        print(
            f"Loaded {n_deposit} BoulderDeposit polygon(s) from "
            f"{gpkg_path.name} (crowd-ignore when --boulder-only) [{tag.strip()}]"
        )
    print(f"Loaded {len(feats)} features from {gpkg_path} [{tag.strip()}]")
    return feats


def infer_year_from_path(path: Path) -> int | None:
    name = path.name.lower()
    if "24" in name and "25" not in name:
        return 24
    if "25" in name and "24" not in name:
        return 25
    # Common patterns: july13_24, july9_24input, roi_24, ...
    for token in name.replace(".", "_").replace("-", "_").split("_"):
        if token == "24":
            return 24
        if token == "25":
            return 25
    return None


def parse_gpkg_specs(
    value: str | None,
    years: list[int],
    defaults: list[tuple[Path, int | None]],
) -> list[tuple[Path, int | None]]:
    """Parse --gpkg into [(path, year|None), ...].

    Accepts:
      path.gpkg
      path24.gpkg,path25.gpkg
      path24.gpkg:24,path25.gpkg:25
    """
    if value is None:
        return defaults

    specs: list[tuple[Path, int | None]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            path_s, year_s = part.rsplit(":", 1)
            year_s = year_s.strip()
            if year_s in ("24", "25"):
                specs.append((Path(path_s.strip()), int(year_s)))
                continue
            # Colon was part of a Windows drive letter? (C:\...) — treat whole as path
            if len(path_s) == 1 and path_s.isalpha():
                specs.append((Path(part), None))
                continue
        specs.append((Path(part), None))

    # Fill missing years: filename heuristic, then zip with --years order.
    unresolved = [i for i, (_, y) in enumerate(specs) if y is None]
    if unresolved:
        for i in unresolved:
            inferred = infer_year_from_path(specs[i][0])
            if inferred is not None:
                specs[i] = (specs[i][0], inferred)
        unresolved = [i for i, (_, y) in enumerate(specs) if y is None]

    if len(specs) == 1 and specs[0][1] is None and len(years) == 1:
        specs[0] = (specs[0][0], years[0])
    elif len(specs) == len(years) and all(y is None for _, y in specs):
        specs = [(p, y) for (p, _), y in zip(specs, years)]
    elif len(specs) == 1 and specs[0][1] is None and len(years) > 1:
        # Single merged GPKG spanning multiple years.
        specs = [(specs[0][0], None)]
    elif unresolved:
        raise ValueError(
            "Could not assign a year to every --gpkg entry. Use "
            "path.gpkg:24,path.gpkg:25 or names containing 24/25."
        )

    return specs


def load_roi(roi_path: Path):
    """Union of all ROI polygons from a .shp or .gpkg, reprojected to WORKING_EPSG."""
    layers = fiona.listlayers(roi_path) if roi_path.suffix.lower() == ".gpkg" else [None]
    geoms = []
    for layer in layers:
        kwargs = {"layer": layer} if layer is not None else {}
        with fiona.open(roi_path, **kwargs) as src:
            reproject = make_reprojector(src.crs, WORKING_EPSG)
            for feat in src:
                raw_geom = feat.get("geometry")
                if raw_geom is None:
                    continue
                try:
                    geom = shape(raw_geom)
                except (AttributeError, ValueError, TypeError):
                    continue
                if not geom.is_valid:
                    geom = geom.buffer(0)
                if geom.is_empty:
                    continue
                geoms.append(shp_transform(reproject, geom))
    if not geoms:
        raise ValueError(f"No polygons found in ROI: {roi_path}")
    return unary_union(geoms)


def polygons_of(geom) -> list:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        result = []
        for part in geom.geoms:
            result.extend(polygons_of(part))
        return result
    return []


def ring_to_seg(ring_coords, inv_transform) -> list[float]:
    coords = list(ring_coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    seg: list[float] = []
    for x, y in coords:
        col, row = inv_transform * (x, y)
        seg.extend([float(col), float(row)])
    return seg


def bbox_from_seg(seg: list[float]) -> list[float]:
    xs = seg[0::2]
    ys = seg[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def poly_area(seg: list[float]) -> float:
    xs = seg[0::2]
    ys = seg[1::2]
    area = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def convert_tile(
    image_path: Path,
    feats: list[tuple],
    roi,
    image_id: int,
    ann_start_id: int,
    min_area_m2: float,
    boulder_only: bool,
    tile_year: int | None = None,
) -> tuple[dict, list[dict], int, dict]:
    with rasterio.open(image_path) as ds:
        transform = ds.transform
        width = ds.width
        height = ds.height
        tile_poly = box(*ds.bounds)

    inv_transform = ~transform
    image_info = {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": image_path.name,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": "",
    }

    annotations: list[dict] = []
    ann_id = ann_start_id
    # keys: trainable boulder, crowd-ignored (deposit/small), trainable deposit
    per_class = {"boulder": 0, "crowd": 0, "deposit": 0}
    for item in feats:
        geom, cls = item[0], item[1]
        feat_year = item[2] if len(item) > 2 else None
        # Year-tagged features only label tiles from that ortho year.
        if feat_year is not None and tile_year is not None and feat_year != tile_year:
            continue
        if not geom.intersects(tile_poly):
            continue
        clipped = geom
        if roi is not None:
            clipped = clipped.intersection(roi)
            if clipped.is_empty:
                continue

        # Decide crowd vs trainable before tile clip (area filter uses map units).
        is_crowd = 0
        ignore_reason = None
        if cls == 1 and boulder_only:
            # Single-class training: deposits become ignore regions.
            is_crowd = 1
            ignore_reason = "deposit"
            category_id = 1
        elif cls == 1:
            category_id = 2
        else:
            category_id = 1
            if min_area_m2 > 0 and clipped.area < min_area_m2:
                is_crowd = 1
                ignore_reason = "small"

        clipped = clipped.intersection(tile_poly)
        for poly in polygons_of(clipped):
            seg = ring_to_seg(poly.exterior.coords, inv_transform)
            if len(seg) < 6:
                continue
            area = poly_area(seg)
            if area <= 0:
                continue
            attrs = {"occluded": False}
            if ignore_reason is not None:
                attrs["ignore_reason"] = ignore_reason
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [seg],
                    "area": area,
                    "bbox": bbox_from_seg(seg),
                    "iscrowd": is_crowd,
                    "attributes": attrs,
                }
            )
            if is_crowd:
                per_class["crowd"] += 1
            elif category_id == 2:
                per_class["deposit"] += 1
            else:
                per_class["boulder"] += 1
            ann_id += 1

    return image_info, annotations, ann_id, per_class


def build_split(
    split_name: str,
    tile_keys: list[str],
    tile_dir: Path,
    output_dir: Path,
    feats: list[tuple],
    roi,
    min_area_m2: float,
    boulder_only: bool,
) -> dict:
    split_image_dir = output_dir / split_name
    split_image_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 1
    ann_id = 1
    per_class_total = {"boulder": 0, "crowd": 0, "deposit": 0}
    categories = CATEGORIES_ONE if boulder_only else CATEGORIES_TWO
    years_used: set[int] = set()

    for key in tile_keys:
        year, _ = parse_year_key(key)
        years_used.add(year)
        src_image = resolve_tile_path(tile_dir, key)
        # Year-prefix copied filenames so 24/25 tiles never collide in one split dir.
        dst_name = f"{year}_{src_image.name}"
        shutil.copy2(src_image, split_image_dir / dst_name)

        image_info, anns, ann_id, per_class = convert_tile(
            src_image,
            feats,
            roi,
            image_id,
            ann_id,
            min_area_m2,
            boulder_only,
            tile_year=year,
        )
        image_info["file_name"] = dst_name
        images.append(image_info)
        annotations.extend(anns)
        for k, v in per_class.items():
            per_class_total[k] += v
        image_id += 1

    coco = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": (
                f"Boulder training split: {split_name} "
                f"(years={sorted(years_used)}; "
                f"crowd-ignore deposits/small when boulder-only)"
            ),
            "url": "",
            "version": "3.1",
            "year": "2026",
        },
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    ann_name = {
        "train": "train_annotations.json",
        "valid": "validation_annotations.json",
        "test": "testing_annotations.json",
    }[split_name]
    out_json = output_dir / ann_name
    out_json.write_text(json.dumps(coco))
    n_pos = per_class_total["boulder"] + per_class_total["deposit"]
    return {
        "split": split_name,
        "years": sorted(years_used),
        "tiles": tile_keys,
        "images": len(images),
        "annotations": len(annotations),
        "boulders": per_class_total["boulder"],
        "deposits": per_class_total["deposit"],
        "crowd_ignore": per_class_total["crowd"],
        "trainable": n_pos,
        "json": str(out_json),
        "image_dir": str(split_image_dir),
    }


def write_tile_extents(
    tile_keys: list[str], tile_dir: Path, out_path: Path
) -> None:
    features = []
    for key in tile_keys:
        try:
            path = resolve_tile_path(tile_dir, key)
            year, short = parse_year_key(key)
        except (FileNotFoundError, ValueError):
            continue
        with rasterio.open(path) as ds:
            poly = box(*ds.bounds)
        features.append(
            {
                "type": "Feature",
                "properties": {"tile": short, "year": year, "file": path.name},
                "geometry": mapping(poly),
            }
        )
    geojson = {
        "type": "FeatureCollection",
        "name": out_path.stem,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::25829"}},
        "features": features,
    }
    out_path.write_text(json.dumps(geojson))
    print(f"Wrote {len(features)} tile extents to {out_path}")


def parse_tile_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_years(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    years = sorted({int(p.strip()) for p in value.split(",") if p.strip()})
    for y in years:
        if y not in (24, 25):
            raise ValueError(f"Unsupported year: {y}")
    return years


def parse_path_list(value: str | None) -> list[Path] | None:
    if value is None:
        return None
    if str(value).lower() == "none":
        return []
    return [Path(p.strip()) for p in str(value).split(",") if p.strip()]


def default_gpkg_specs(years: list[int], seg_dir: Path) -> list[tuple[Path, int | None]]:
    """Prefer newest per-year GPKGs (july14 → july13 → july9)."""
    ann = seg_dir / "annotations"

    def pick_year(year: int) -> Path:
        for name in (f"july14_{year}.gpkg", f"july13_{year}.gpkg", f"july9_{year}input.gpkg"):
            path = ann / name
            if path.exists():
                return path
        return ann / f"july14_{year}.gpkg"

    if years == [24]:
        return [(pick_year(24), 24)]
    if years == [25]:
        return [(pick_year(25), 25)]

    july14 = [(ann / "july14_24.gpkg", 24), (ann / "july14_25.gpkg", 25)]
    if all(p.exists() for p, _ in july14):
        return july14
    july13 = [(ann / "july13_24.gpkg", 24), (ann / "july13_25.gpkg", 25)]
    if all(p.exists() for p, _ in july13):
        return july13
    per_year = [
        (ann / "july9_24input.gpkg", 24),
        (ann / "july9_25input.gpkg", 25),
    ]
    if all(p.exists() for p, _ in per_year):
        return per_year
    return [(ann / "july9_input.gpkg", None)]


def default_tiles_used_path(seg_dir: Path) -> Path:
    return seg_dir / "annotations" / "tiles_used.txt"


def resolve_tiles_by_year(seg_dir: Path, tiles_used: Path | None) -> dict[int, list[str]]:
    """Load tile lists from --tiles-used, else annotations/tiles_used.txt, else baked-in."""
    candidates = []
    if tiles_used is not None:
        candidates.append(tiles_used)
    candidates.append(default_tiles_used_path(seg_dir))
    for path in candidates:
        if path.exists():
            data = load_tiles_used(path)
            print(f"Tile list: {path} (24={len(data[24])}, 25={len(data[25])})")
            return data
    print(
        f"Tile list: baked-in tiles_used snapshot "
        f"(24={len(TILES_24)}, 25={len(TILES_25)})"
    )
    return {24: list(TILES_24), 25: list(TILES_25)}


def default_roi_paths(years: list[int], seg_dir: Path) -> list[Path]:
    te = seg_dir / "tile_extents"
    if years == [24]:
        return [te / "roi_24_0709.gpkg"]
    if years == [25]:
        return [te / "roi.shp"]
    return [te / "roi_24_0709.gpkg", te / "roi.shp"]


def default_output_dir(years: list[int], seg_dir: Path) -> Path:
    if years == [24]:
        return seg_dir / "coco_dataset_24"
    if years == [25]:
        return seg_dir / "coco_dataset_25"
    return seg_dir / "coco_dataset_both"


def load_rois(roi_paths: list[Path]):
    """Union ROI polygons from one or more .shp/.gpkg files."""
    geoms = []
    for path in roi_paths:
        geoms.append(load_roi(path))
        print(f"ROI: {path}")
    if not geoms:
        return None
    return unary_union(geoms) if len(geoms) > 1 else geoms[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=Path("segmentation"))
    parser.add_argument(
        "--years",
        type=str,
        default="24,25",
        help="Comma-separated ortho years to include (default: 24,25). Use 24 or 25 alone for a single year.",
    )
    parser.add_argument(
        "--year",
        type=int,
        choices=[24, 25],
        default=None,
        help="Deprecated alias for a single year; prefer --years.",
    )
    parser.add_argument(
        "--gpkg",
        type=str,
        default=None,
        help=(
            "Annotation GPKG(s). Comma-separated; optional :24/:25 year tags. "
            "Examples: july14_24.gpkg:24,july14_25.gpkg:25. "
            "Default: july14_24/25.gpkg when present, else july13, else july9."
        ),
    )
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--class-field", type=str, default="Class")
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help=(
            "Optional comma-separated ROI .shp/.gpkg paths to clip annotations. "
            "ROI clipping is off by default; pass paths here to enable it, or "
            "'none' / --no-roi to keep it disabled."
        ),
    )
    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Disable ROI clipping (default behavior; kept for backward compatibility).",
    )
    parser.add_argument(
        "--tiles-used",
        type=Path,
        default=None,
        help=(
            "Path to tiles_used.txt (row:col-range format). "
            "Default: segmentation/annotations/tiles_used.txt when present."
        ),
    )
    parser.add_argument("--tile-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=0.0,
        help=(
            "Boulder polygons smaller than this (m2, ROI-clipped geom) become "
            "COCO iscrowd=1 ignore regions instead of trainable GT. 0 = off."
        ),
    )
    parser.add_argument(
        "--boulder-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Emit a 1-class COCO dataset; BoulderDeposit polygons become "
            "iscrowd=1 ignore regions (default: on). Use --no-boulder-only for "
            "trainable deposits as category 2."
        ),
    )
    parser.add_argument("--train-tiles", type=str, default=None)
    parser.add_argument("--valid-tiles", type=str, default=None)
    parser.add_argument("--test-tiles", type=str, default=None)
    parser.add_argument(
        "--write-extents",
        action="store_true",
        help="Write tile footprints GeoJSON into segmentation-dir/tile_extents/",
    )
    args = parser.parse_args()

    years = [args.year] if args.year is not None else parse_years(args.years, [24, 25])
    seg_dir = args.segmentation_dir
    tile_dir = args.tile_dir or (seg_dir / "tiling")
    output_dir = args.output_dir or default_output_dir(years, seg_dir)

    gpkg_specs = parse_gpkg_specs(
        args.gpkg, years, default_gpkg_specs(years, seg_dir)
    )
    for path, _ in gpkg_specs:
        if not path.exists():
            raise FileNotFoundError(f"Annotation GPKG not found: {path}")

    tiles_by_year = resolve_tiles_by_year(seg_dir, args.tiles_used)
    train_default, valid_default, test_default = expand_year_keys(years, tiles_by_year)
    n_excluded = sum(
        len(EXCLUDED_24 if y == 24 else EXCLUDED_25)
        for y in years
    )
    print(
        f"Splits: train={len(train_default)} valid={len(valid_default)} "
        f"test={len(test_default)} excluded_buffer={n_excluded}"
    )
    print(
        "Hold-outs: valid="
        + ",".join(valid_default)
        + " test="
        + ",".join(test_default)
    )
    # Verify no cross-year (or within-year) footprint overlap between splits.
    try:
        assert_no_geographic_leakage(
            tile_dir, train_default, valid_default, test_default
        )
        print("Geographic leakage check: OK (no overlapping footprints across splits)")
    except FileNotFoundError as exc:
        print(f"Geographic leakage check skipped (missing tile): {exc}")

    if (
        args.no_roi
        or args.roi is None
        or str(args.roi).lower() == "none"
    ):
        roi_paths: list[Path] = []
    else:
        roi_paths = parse_path_list(args.roi) or []
    if roi_paths:
        roi = load_rois(roi_paths)
    else:
        roi = None
        print("ROI: none (clipping disabled)")

    feats: list[tuple] = []
    for path, year in gpkg_specs:
        feats.extend(
            load_annotations(path, args.layer, args.class_field, args.boulder_only, year)
        )
    n_boulder = sum(1 for item in feats if item[1] == 0)
    n_deposit = sum(1 for item in feats if item[1] == 1)
    mode = (
        "boulder-only + crowd-ignore deposits/small"
        if args.boulder_only
        else "two-class (small boulders still crowd-ignored if --min-area-m2 > 0)"
    )
    src_desc = ", ".join(
        f"{p.name}:{y if y is not None else 'any'}" for p, y in gpkg_specs
    )
    print(
        f"Total {len(feats)} features from [{src_desc}] "
        f"({n_boulder} Boulder, {n_deposit} BoulderDeposit) [{mode}]"
    )
    print(f"Years: {years}")

    train_tiles = parse_tile_list(args.train_tiles, train_default)
    valid_tiles = parse_tile_list(args.valid_tiles, valid_default)
    test_tiles = parse_tile_list(args.test_tiles, test_default)
    # Allow unprefixed overrides by attaching the sole year when only one year is selected.
    if len(years) == 1:
        y = years[0]

        def ensure_year(keys: list[str]) -> list[str]:
            out = []
            for k in keys:
                try:
                    parse_year_key(k)
                    out.append(k)
                except ValueError:
                    out.append(year_key(y, k))
            return out

        train_tiles = ensure_year(train_tiles)
        valid_tiles = ensure_year(valid_tiles)
        test_tiles = ensure_year(test_tiles)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_extents:
        tag = "_".join(str(y) for y in years)
        extents_path = seg_dir / "tile_extents" / f"tile_extents_{tag}_auto.geojson"
        write_tile_extents(train_tiles + valid_tiles + test_tiles, tile_dir, extents_path)

    summary = [
        build_split(
            "train",
            train_tiles,
            tile_dir,
            output_dir,
            feats,
            roi,
            args.min_area_m2,
            args.boulder_only,
        ),
        build_split(
            "valid",
            valid_tiles,
            tile_dir,
            output_dir,
            feats,
            roi,
            args.min_area_m2,
            args.boulder_only,
        ),
        build_split(
            "test",
            test_tiles,
            tile_dir,
            output_dir,
            feats,
            roi,
            args.min_area_m2,
            args.boulder_only,
        ),
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
