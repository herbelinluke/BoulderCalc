#!/usr/bin/env python3
"""Convert per-tile GeoJSON annotations + GeoTIFF tiles into Detectron2 COCO JSON."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import rasterio
from rasterio.warp import transform as warp_transform
from shapely.geometry import shape

# Tile key -> ortho filename in segmentation/tiling/
TILE_MAP: dict[str, str] = {
    "25_04_34": "25IniSouthOrt_04_34.tif",
    "25_04_35": "25IniSouthOrt_04_35.tif",
    "25_04_36": "25IniSouthOrt_04_36.tif",
    "25_04_37": "25IniSouthOrt_04_37.tif",
    "25_05_31": "25IniSouthOrt_05_31.tif",
    "25_05_32": "25IniSouthOrt_05_32.tif",
    "25_05_33": "25IniSouthOrt_05_33.tif",
    "25_05_34": "25IniSouthOrt_05_34.tif",
    "25_05_35": "25IniSouthOrt_05_35.tif",
    "25_06_27": "25IniSouthOrt_06_27.tif",
    "25_06_28": "25IniSouthOrt_06_28.tif",
    "25_06_29": "25IniSouthOrt_06_29.tif",
    "25_06_30": "25IniSouthOrt_06_30.tif",
    "25_07_24": "25IniSouthOrt_07_24.tif",
    "25_07_25": "25IniSouthOrt_07_25.tif",
    "25_07_26": "25IniSouthOrt_07_26.tif",
    "25_07_28": "25IniSouthOrt_07_28.tif",
    "25_08_23": "25IniSouthOrt_08_23.tif",
    "25_08_24": "25IniSouthOrt_08_24.tif",
    "25_08_25": "25IniSouthOrt_08_25.tif",
}

# Tile key -> GeoJSON filename in segmentation/tile_geojsons/
GEOJSON_MAP: dict[str, str] = {
    "25_04_34": "25_04_34_2nd.geojson",
    "25_04_35": "25_04_35_2nd.geojson",
    "25_04_36": "25_04_36_2nd.geojson",
    "25_04_37": "25_04_37_2nd.geojson",
    "25_05_31": "25_05_31_1st.geojson",
    "25_05_32": "25_05_32_1st.geojson",
    "25_05_33": "25_05_33_1st.geojson",
    "25_05_34": "25_05_34_1st.geojson",
    "25_05_35": "25_05_35_2nd.geojson",
    "25_06_27": "25_06_27_2nd.geojson",
    "25_06_28": "25_06_28_2nd.geojson",
    "25_06_29": "25_06_29_2nd.geojson",
    "25_06_30": "25_06_30_2nd.geojson",
    "25_07_24": "25_07_24_2nd.geojson",
    "25_07_25": "25_07_25_2nd.geojson",
    "25_07_26": "25_07_26_2nd.geojson",
    "25_07_28": "25_07_28_2nd.geojson",
    "25_08_23": "25_08_23_2nd.geojson",
    "25_08_24": "25_08_24_2nd.geojson",
    "25_08_25": "25_08_25_2nd.geojson",
}

# ~62% train / 9% valid / 29% test by annotation count; test tiles span rows 04–06
TRAIN_TILES = [
    "25_04_34",
    "25_04_36",
    "25_04_37",
    "25_05_31",
    "25_05_32",
    "25_05_35",
    "25_06_27",
    "25_06_28",
    "25_06_30",
    "25_07_24",
    "25_07_25",
    "25_07_26",
    "25_07_28",
    "25_08_23",
    "25_08_25",
]
VALID_TILES = ["25_05_33", "25_08_24"]
TEST_TILES = ["25_04_35", "25_05_34", "25_06_29"]


def polygon_coords_to_pixels(coords, transform, src_crs, dst_crs) -> list[float]:
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    xs, ys = warp_transform(src_crs, dst_crs, lons, lats)
    seg: list[float] = []
    for x, y in zip(xs, ys):
        row, col = rasterio.transform.rowcol(transform, x, y)
        seg.extend([float(col), float(row)])
    return seg


def flatten_polygons(geom) -> list[list[tuple[float, float]]]:
    g = shape(geom)
    if g.geom_type == "Polygon":
        return [list(g.exterior.coords)]
    if g.geom_type == "MultiPolygon":
        return [list(p.exterior.coords) for p in g.geoms]
    return []


def bbox_from_seg(seg: list[float]) -> list[float]:
    xs = seg[0::2]
    ys = seg[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def poly_area(seg: list[float]) -> float:
    xs = seg[0::2]
    ys = seg[1::2]
    area = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def convert_tile(
    geojson_path: Path,
    image_path: Path,
    image_id: int,
    ann_start_id: int,
) -> tuple[dict, list[dict], int]:
    data = json.loads(geojson_path.read_text())
    src_crs = data.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if "CRS84" in src_crs:
        src_crs = "EPSG:4326"

    with rasterio.open(image_path) as ds:
        transform = ds.transform
        dst_crs = ds.crs
        width = ds.width
        height = ds.height

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
    for feature in data["features"]:
        for ring in flatten_polygons(feature["geometry"]):
            if len(ring) < 4:
                continue
            seg = polygon_coords_to_pixels(ring, transform, src_crs, dst_crs)
            if len(seg) < 6:
                continue
            area = poly_area(seg)
            if area <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [seg],
                    "area": area,
                    "bbox": bbox_from_seg(seg),
                    "iscrowd": 0,
                    "attributes": {"occluded": False},
                }
            )
            ann_id += 1

    return image_info, annotations, ann_id


def build_split(
    split_name: str,
    tile_keys: list[str],
    geojson_dir: Path,
    tile_dir: Path,
    output_dir: Path,
) -> dict:
    split_image_dir = output_dir / split_name
    split_image_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 1
    ann_id = 1

    for key in tile_keys:
        if key not in TILE_MAP:
            raise KeyError(f"Unknown tile key: {key}")
        geojson_name = GEOJSON_MAP.get(key, f"{key}_1st.geojson")
        geojson_path = geojson_dir / geojson_name
        image_name = TILE_MAP[key]
        src_image = tile_dir / image_name
        dst_image = split_image_dir / image_name
        if not src_image.exists():
            raise FileNotFoundError(src_image)
        if not geojson_path.exists():
            raise FileNotFoundError(geojson_path)
        shutil.copy2(src_image, dst_image)

        image_info, anns, ann_id = convert_tile(geojson_path, src_image, image_id, ann_id)
        images.append(image_info)
        annotations.extend(anns)
        image_id += 1

    coco = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": f"Boulder training split: {split_name}",
            "url": "",
            "version": "1.0",
            "year": "2026",
        },
        "categories": [{"id": 1, "name": "Boulder", "supercategory": "none"}],
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
        "json": str(out_json),
        "image_dir": str(split_image_dir),
    }


def parse_tile_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("/home/herbs/Documents/tamucc/segmentation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/herbs/Documents/tamucc/segmentation/coco_dataset"),
    )
    parser.add_argument(
        "--tile-dir",
        type=Path,
        default=None,
        help="Image source directory (default: segmentation-dir/tiling). Use tiling_hillshade for DSM hillshade.",
    )
    parser.add_argument(
        "--train-tiles",
        type=str,
        default=None,
        help="Comma-separated tile keys overriding TRAIN_TILES.",
    )
    parser.add_argument(
        "--valid-tiles",
        type=str,
        default=None,
        help="Comma-separated tile keys overriding VALID_TILES.",
    )
    parser.add_argument(
        "--test-tiles",
        type=str,
        default=None,
        help="Comma-separated tile keys overriding TEST_TILES.",
    )
    args = parser.parse_args()

    geojson_dir = args.segmentation_dir / "tile_geojsons"
    tile_dir = args.tile_dir or (args.segmentation_dir / "tiling")
    train_tiles = parse_tile_list(args.train_tiles, TRAIN_TILES)
    valid_tiles = parse_tile_list(args.valid_tiles, VALID_TILES)
    test_tiles = parse_tile_list(args.test_tiles, TEST_TILES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = [
        build_split("train", train_tiles, geojson_dir, tile_dir, args.output_dir),
        build_split("valid", valid_tiles, geojson_dir, tile_dir, args.output_dir),
        build_split("test", test_tiles, geojson_dir, tile_dir, args.output_dir),
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
