#!/usr/bin/env python3
"""Convert a QGIS GPKG annotation layer + GeoTIFF tiles directly into Detectron2 COCO JSON.

Reads polygons (with a Class attribute: 0 = Boulder, 1 = BoulderDeposit) from a
GeoPackage, clips them to an optional ROI polygon layer and to each tile's
extent (computed from the GeoTIFF itself -- no tile_extents files needed), and
writes train/valid/test COCO datasets.

Example:
    python BoulderCalculator/scripts/gpkg_to_coco.py \
        --segmentation-dir segmentation \
        --gpkg "segmentation/annotations/july5_deposits&more.gpkg" \
        --roi segmentation/tile_extents/roi.shp \
        --output-dir segmentation/coco_dataset_v2
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

# Horizontal CRS of the ortho tiles (ETRS89 / UTM zone 29N). Annotation and ROI
# layers are reprojected into this CRS so areas are in square meters.
WORKING_EPSG = "EPSG:25829"

CATEGORIES = [
    {"id": 1, "name": "Boulder", "supercategory": "none"},
    {"id": 2, "name": "BoulderDeposit", "supercategory": "none"},
]

# Tile keys (row_col of 25IniSouthOrt_ROW_COL.tif).
# Rows 11: 17-18, 10: 17-21, 9: 18-25, 8: 20-27, 7: 24-29, 6: 27-31,
#      5: 30-35, 4: 33-38, 3: 36-38.
ALL_TILES = (
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
VALID_TILES = ["05_33", "08_24"]
TEST_TILES = ["04_35", "05_34", "06_29"]
TRAIN_TILES = [k for k in ALL_TILES if k not in VALID_TILES + TEST_TILES]


def tile_filename(key: str) -> str:
    return f"25IniSouthOrt_{key}.tif"


def make_reprojector(src_crs, dst_crs):
    def func(xs, ys):
        out_x, out_y = warp_transform(src_crs, dst_crs, list(xs), list(ys))
        return out_x, out_y

    return func


def load_annotations(
    gpkg_path: Path, layer: str | None, class_field: str
) -> list[tuple]:
    """Return [(shapely geometry in WORKING_EPSG, class value)]."""
    kwargs = {"layer": layer} if layer else {}
    feats: list[tuple] = []
    unknown_class = 0
    with fiona.open(gpkg_path, **kwargs) as src:
        reproject = make_reprojector(src.crs, WORKING_EPSG)
        for feat in src:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            geom = shp_transform(reproject, geom)
            cls = feat["properties"].get(class_field)
            if cls is None:
                unknown_class += 1
                cls = 0
            feats.append((geom, int(cls)))
    if unknown_class:
        print(f"WARNING: {unknown_class} feature(s) missing '{class_field}', treated as Boulder (0)")
    return feats


def load_roi(roi_path: Path):
    """Union of all ROI polygons, reprojected to WORKING_EPSG."""
    with fiona.open(roi_path) as src:
        reproject = make_reprojector(src.crs, WORKING_EPSG)
        geoms = []
        for feat in src:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            geoms.append(shp_transform(reproject, geom))
    return unary_union(geoms)


def polygons_of(geom) -> list:
    """Extract Polygon components from any clip result."""
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
        if min_area_m2 > 0 and clipped.area < min_area_m2:
            continue
        clipped = clipped.intersection(tile_poly)
        for poly in polygons_of(clipped):
            seg = ring_to_seg(poly.exterior.coords, inv_transform)
            if len(seg) < 6:
                continue
            area = poly_area(seg)
            if area <= 0:
                continue
            category_id = min(max(cls, 0), 1) + 1  # 0 -> 1 Boulder, 1 -> 2 BoulderDeposit
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
    output_dir: Path,
    feats: list[tuple],
    roi,
    min_area_m2: float,
) -> dict:
    split_image_dir = output_dir / split_name
    split_image_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 1
    ann_id = 1
    per_class_total = {1: 0, 2: 0}

    for key in tile_keys:
        src_image = tile_dir / tile_filename(key)
        if not src_image.exists():
            raise FileNotFoundError(src_image)
        shutil.copy2(src_image, split_image_dir / src_image.name)

        image_info, anns, ann_id, per_class = convert_tile(
            src_image, feats, roi, image_id, ann_id, min_area_m2
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
            "description": f"Boulder training split: {split_name}",
            "url": "",
            "version": "2.0",
            "year": "2026",
        },
        "categories": CATEGORIES,
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
        "tiles": tile_keys,
        "images": len(images),
        "annotations": len(annotations),
        "boulders": per_class_total[1],
        "deposits": per_class_total[2],
        "json": str(out_json),
        "image_dir": str(split_image_dir),
    }


def write_tile_extents(tile_keys: list[str], tile_dir: Path, out_path: Path) -> None:
    """Write tile footprints (in WORKING_EPSG) as GeoJSON for QGIS reference."""
    features = []
    for key in tile_keys:
        path = tile_dir / tile_filename(key)
        if not path.exists():
            continue
        with rasterio.open(path) as ds:
            poly = box(*ds.bounds)
        features.append(
            {
                "type": "Feature",
                "properties": {"tile": key, "file": path.name},
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=Path("segmentation"))
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=None,
        help="Annotation GPKG (default: segmentation-dir/annotations/july7_training_input.gpkg)",
    )
    parser.add_argument("--layer", type=str, default=None, help="GPKG layer name (default: first layer)")
    parser.add_argument("--class-field", type=str, default="Class")
    parser.add_argument(
        "--roi",
        type=Path,
        default=None,
        help="ROI polygon file to clip annotations to (default: segmentation-dir/tile_extents/roi.shp; pass 'none' to disable)",
    )
    parser.add_argument("--tile-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-area-m2", type=float, default=0.0, help="Drop features smaller than this (0 = no filter)")
    parser.add_argument("--train-tiles", type=str, default=None)
    parser.add_argument("--valid-tiles", type=str, default=None)
    parser.add_argument("--test-tiles", type=str, default=None)
    parser.add_argument(
        "--write-extents",
        action="store_true",
        help="Also write tile footprints GeoJSON into segmentation-dir/tile_extents/",
    )
    args = parser.parse_args()

    seg_dir = args.segmentation_dir
    gpkg = args.gpkg or (seg_dir / "annotations" / "july7_training_input.gpkg")
    tile_dir = args.tile_dir or (seg_dir / "tiling")
    output_dir = args.output_dir or (seg_dir / "coco_dataset_v2")

    roi = None
    if args.roi is None:
        default_roi = seg_dir / "tile_extents" / "roi.shp"
        if default_roi.exists():
            roi = load_roi(default_roi)
            print(f"ROI: {default_roi}")
    elif str(args.roi).lower() != "none":
        roi = load_roi(args.roi)
        print(f"ROI: {args.roi}")
    if roi is None:
        print("ROI: none (annotations only clipped to tile extents)")

    feats = load_annotations(gpkg, args.layer, args.class_field)
    n_boulder = sum(1 for _, c in feats if c == 0)
    n_deposit = sum(1 for _, c in feats if c == 1)
    print(f"Loaded {len(feats)} features from {gpkg} ({n_boulder} Boulder, {n_deposit} BoulderDeposit)")

    train_tiles = parse_tile_list(args.train_tiles, TRAIN_TILES)
    valid_tiles = parse_tile_list(args.valid_tiles, VALID_TILES)
    test_tiles = parse_tile_list(args.test_tiles, TEST_TILES)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_extents:
        extents_path = seg_dir / "tile_extents" / "tile_extents_auto.geojson"
        write_tile_extents(train_tiles + valid_tiles + test_tiles, tile_dir, extents_path)

    summary = [
        build_split("train", train_tiles, tile_dir, output_dir, feats, roi, args.min_area_m2),
        build_split("valid", valid_tiles, tile_dir, output_dir, feats, roi, args.min_area_m2),
        build_split("test", test_tiles, tile_dir, output_dir, feats, roi, args.min_area_m2),
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
