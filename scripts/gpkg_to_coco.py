#!/usr/bin/env python3
"""Convert a QGIS GPKG annotation layer + GeoTIFF tiles into Detectron2 COCO JSON.

Supports year-based tile layouts under segmentation/tiling/{24,25}/, ROI as
shapefile or GeoPackage, and boulder-only training (deposit polygons dropped).

Class attribute handling:
  - numeric 0 / missing / string "Boulder"  -> Boulder (COCO category 1)
  - numeric 1 / string containing "deposit" -> BoulderDeposit (dropped when
    --boulder-only is set, which is the default for the 2024 workflow)

Example (2024 boulder-only):
    python BoulderCalculator/scripts/gpkg_to_coco.py \\
        --segmentation-dir segmentation \\
        --year 24 \\
        --gpkg segmentation/annotations/july8_24annot.gpkg \\
        --roi segmentation/tile_extents/roi_24_0709.gpkg \\
        --output-dir segmentation/coco_dataset_24 \\
        --min-area-m2 1.0
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

# 2025 tiles (legacy v3 list). Keys are "RR_CC" without year prefix.
TILES_25 = (
    [f"11_{c}" for c in range(17, 19)]
    + [f"10_{c}" for c in range(17, 22)]
    + [f"09_{c}" for c in range(18, 26)]
    + [f"08_{c}" for c in range(20, 28)]
    + [f"07_{c}" for c in range(24, 30)]
    + [f"06_{c}" for c in range(27, 32)]
    + [f"05_{c}" for c in range(30, 36)]
    + [f"04_{c}" for c in range(33, 39)]
    + [f"03_{c}" for c in range(36, 39)]
)
VALID_25 = ["05_33", "08_24"]
TEST_25 = ["04_35", "05_34", "06_29"]

# 2024 tiles: 11:7, 12:7-9, 13:7-13, 14:7-20, 15:7-20, 16:7-17
TILES_24 = (
    [f"11_{c}" for c in [7]]
    + [f"12_{c}" for c in range(7, 10)]
    + [f"13_{c}" for c in range(7, 14)]
    + [f"14_{c}" for c in range(7, 21)]
    + [f"15_{c}" for c in range(7, 21)]
    + [f"16_{c}" for c in range(7, 18)]
)
# Hold-out tiles with annotations, spread across the 2024 strip.
VALID_24 = ["13_9", "15_15"]
TEST_24 = ["12_8", "14_15", "16_14"]

# Back-compat aliases used by build_hillshade_tiles.py
ALL_TILES = TILES_25
VALID_TILES = VALID_25
TEST_TILES = TEST_25
TRAIN_TILES = [k for k in ALL_TILES if k not in VALID_TILES + TEST_TILES]


def normalize_key(key: str) -> tuple[int, int]:
    """Parse '11_7' / '11_07' / '24_11_07' into (row, col). Year prefix optional."""
    parts = key.strip().split("_")
    if len(parts) == 3:
        parts = parts[1:]
    if len(parts) != 2:
        raise ValueError(f"Bad tile key: {key}")
    return int(parts[0]), int(parts[1])


def tile_filename(key: str, year: int = 25) -> str:
    row, col = normalize_key(key)
    if year == 24:
        return f"Sites1and2_2024_Orthomosaic_{row:02d}_{col:02d}.tif"
    return f"25IniSouthOrt_{row:02d}_{col:02d}.tif"


def resolve_tile_path(tile_dir: Path, key: str, year: int) -> Path:
    """Find a tile under tiling/{year}/ or flat tiling/ (legacy)."""
    name = tile_filename(key, year)
    candidates = [
        tile_dir / str(year) / name,
        tile_dir / name,
    ]
    # Also try unpadded 25-style names for older flat layouts.
    if year == 25:
        row, col = normalize_key(key)
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
) -> list[tuple]:
    """Return [(shapely geometry in WORKING_EPSG, class value 0|1)]."""
    kwargs = {"layer": layer} if layer else {}
    feats: list[tuple] = []
    unknown_class = 0
    dropped_deposit = 0
    with fiona.open(gpkg_path, **kwargs) as src:
        reproject = make_reprojector(src.crs, WORKING_EPSG)
        for feat in src:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            geom = shp_transform(reproject, geom)
            raw = feat["properties"].get(class_field)
            if raw is None:
                unknown_class += 1
            cls = parse_class_value(raw)
            if boulder_only and cls == 1:
                dropped_deposit += 1
                continue
            feats.append((geom, cls))
    if unknown_class:
        print(
            f"WARNING: {unknown_class} feature(s) missing '{class_field}', "
            "treated as Boulder (0)"
        )
    if dropped_deposit:
        print(f"Dropped {dropped_deposit} BoulderDeposit polygon(s) (--boulder-only)")
    return feats


def load_roi(roi_path: Path):
    """Union of all ROI polygons from a .shp or .gpkg, reprojected to WORKING_EPSG."""
    layers = fiona.listlayers(roi_path) if roi_path.suffix.lower() == ".gpkg" else [None]
    geoms = []
    for layer in layers:
        kwargs = {"layer": layer} if layer is not None else {}
        with fiona.open(roi_path, **kwargs) as src:
            reproject = make_reprojector(src.crs, WORKING_EPSG)
            for feat in src:
                geom = shape(feat["geometry"])
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
    per_class = {1: 0, 2: 0}
    for geom, cls in feats:
        if not geom.intersects(tile_poly):
            continue
        clipped = geom
        if roi is not None:
            clipped = clipped.intersection(roi)
            if clipped.is_empty:
                continue
        # Size filter on Boulders only; deposits kept when not boulder-only.
        if min_area_m2 > 0 and cls == 0 and clipped.area < min_area_m2:
            continue
        clipped = clipped.intersection(tile_poly)
        for poly in polygons_of(clipped):
            seg = ring_to_seg(poly.exterior.coords, inv_transform)
            if len(seg) < 6:
                continue
            area = poly_area(seg)
            if area <= 0:
                continue
            if boulder_only:
                category_id = 1
            else:
                category_id = min(max(cls, 0), 1) + 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [seg],
                    "area": area,
                    "bbox": bbox_from_seg(seg),
                    "iscrowd": 0,
                    "attributes": {"occluded": False},
                }
            )
            per_class[category_id] += 1
            ann_id += 1

    return image_info, annotations, ann_id, per_class


def build_split(
    split_name: str,
    tile_keys: list[str],
    tile_dir: Path,
    year: int,
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
    per_class_total = {1: 0, 2: 0}
    categories = CATEGORIES_ONE if boulder_only else CATEGORIES_TWO

    for key in tile_keys:
        src_image = resolve_tile_path(tile_dir, key, year)
        shutil.copy2(src_image, split_image_dir / src_image.name)

        image_info, anns, ann_id, per_class = convert_tile(
            src_image, feats, roi, image_id, ann_id, min_area_m2, boulder_only
        )
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
            "description": f"Boulder training split: {split_name} (year={year})",
            "url": "",
            "version": "3.0",
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
    return {
        "split": split_name,
        "year": year,
        "tiles": tile_keys,
        "images": len(images),
        "annotations": len(annotations),
        "boulders": per_class_total[1],
        "deposits": per_class_total[2],
        "json": str(out_json),
        "image_dir": str(split_image_dir),
    }


def write_tile_extents(
    tile_keys: list[str], tile_dir: Path, year: int, out_path: Path
) -> None:
    features = []
    for key in tile_keys:
        try:
            path = resolve_tile_path(tile_dir, key, year)
        except FileNotFoundError:
            continue
        with rasterio.open(path) as ds:
            poly = box(*ds.bounds)
        features.append(
            {
                "type": "Feature",
                "properties": {"tile": key, "year": year, "file": path.name},
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


def default_paths(year: int, seg_dir: Path) -> tuple[Path, Path, Path]:
    """Return (gpkg, roi, output_dir) defaults for a year."""
    if year == 24:
        return (
            seg_dir / "annotations" / "july8_24annot.gpkg",
            seg_dir / "tile_extents" / "roi_24_0709.gpkg",
            seg_dir / "coco_dataset_24",
        )
    return (
        seg_dir / "annotations" / "july7_training_input.gpkg",
        seg_dir / "tile_extents" / "roi.shp",
        seg_dir / "coco_dataset_v3",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=Path("segmentation"))
    parser.add_argument(
        "--year",
        type=int,
        choices=[24, 25],
        default=24,
        help="Ortho year: 24 -> tiling/24/Sites1and2_2024_..., 25 -> tiling/25/25IniSouthOrt_...",
    )
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=None,
        help="Annotation GPKG (default depends on --year)",
    )
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--class-field", type=str, default="Class")
    parser.add_argument(
        "--roi",
        type=Path,
        default=None,
        help="ROI .shp or .gpkg (default depends on --year; pass 'none' to disable)",
    )
    parser.add_argument("--tile-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=0.0,
        help="Drop Boulder polygons smaller than this in m2 (whole ROI-clipped geometry). 0 = no filter.",
    )
    parser.add_argument(
        "--boulder-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop BoulderDeposit polygons and emit a 1-class COCO dataset (default: on).",
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

    seg_dir = args.segmentation_dir
    default_gpkg, default_roi, default_out = default_paths(args.year, seg_dir)
    gpkg = args.gpkg or default_gpkg
    tile_dir = args.tile_dir or (seg_dir / "tiling")
    output_dir = args.output_dir or default_out

    all_tiles = TILES_24 if args.year == 24 else TILES_25
    valid_default = VALID_24 if args.year == 24 else VALID_25
    test_default = TEST_24 if args.year == 24 else TEST_25
    train_default = [k for k in all_tiles if k not in valid_default + test_default]

    roi = None
    if args.roi is None:
        if default_roi.exists():
            roi = load_roi(default_roi)
            print(f"ROI: {default_roi}")
        else:
            print(f"ROI default missing ({default_roi}); continuing without ROI clip")
    elif str(args.roi).lower() != "none":
        roi = load_roi(args.roi)
        print(f"ROI: {args.roi}")
    else:
        print("ROI: none")

    feats = load_annotations(gpkg, args.layer, args.class_field, args.boulder_only)
    n_boulder = sum(1 for _, c in feats if c == 0)
    n_deposit = sum(1 for _, c in feats if c == 1)
    mode = "boulder-only" if args.boulder_only else "two-class"
    print(
        f"Loaded {len(feats)} features from {gpkg} "
        f"({n_boulder} Boulder, {n_deposit} BoulderDeposit) [{mode}]"
    )

    train_tiles = parse_tile_list(args.train_tiles, train_default)
    valid_tiles = parse_tile_list(args.valid_tiles, valid_default)
    test_tiles = parse_tile_list(args.test_tiles, test_default)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_extents:
        extents_path = seg_dir / "tile_extents" / f"tile_extents_{args.year}_auto.geojson"
        write_tile_extents(
            train_tiles + valid_tiles + test_tiles, tile_dir, args.year, extents_path
        )

    summary = [
        build_split(
            "train", train_tiles, tile_dir, args.year, output_dir,
            feats, roi, args.min_area_m2, args.boulder_only,
        ),
        build_split(
            "valid", valid_tiles, tile_dir, args.year, output_dir,
            feats, roi, args.min_area_m2, args.boulder_only,
        ),
        build_split(
            "test", test_tiles, tile_dir, args.year, output_dir,
            feats, roi, args.min_area_m2, args.boulder_only,
        ),
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
