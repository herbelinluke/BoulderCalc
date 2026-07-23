"""Build paired 2024/2025 tiles, run 4-band inference, match, and visualize.

Designed for ``training_run_rgb_dsm_4000`` test tiles: for each 2025 test
GeoTIFF, crop a same-extent 2024 RGB+DSM window, predict boulder masks with
the trained model, convert masks to georeferenced polygons, then run the
boulder matcher.

Example:
  python -m matching.run_inference_match \\
    --model ../../segmentation/training_run_rgb_dsm_4000/model_final.pth \\
    --outdir ../../segmentation/training_run_rgb_dsm_4000/matching
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine, xy as transform_xy
from rasterio.warp import Resampling, reproject
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Matching package
from .candidates import run_matcher_with_candidates, write_missed_candidates
from .dedupe import dedupe_polygons
from .matcher import BoulderMatcher
from .qc import run_dod_qc, write_dod_qc
from .survey import BoulderSurvey
from .visualize import export_screenshots, load_results, run_gui

# BoulderCalculator script helpers (4-band IO)
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from multiband_io import (  # noqa: E402
    FOUR_BAND_PIXEL_MEAN,
    FOUR_BAND_PIXEL_STD,
    load_bgrd_uint8,
)


def _project_root() -> Path:
    # Matching/matching/thisfile -> Matching -> BoulderCalculator -> tamucc
    return Path(__file__).resolve().parents[3]


def elevation_to_uint8(dem: np.ndarray) -> np.ndarray:
    dem = dem.astype(np.float32)
    finite = dem[np.isfinite(dem)]
    if finite.size == 0:
        return np.zeros(dem.shape, dtype=np.uint8)
    fill = float(np.nanmedian(finite))
    dem = np.where(np.isfinite(dem), dem, fill)
    lo, hi = np.percentile(dem, [2, 98])
    scaled = (dem - lo) / max(hi - lo, 1e-6)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def build_aligned_4band_window(
    ortho_path: Path,
    dsm_path: Path,
    bounds: tuple[float, float, float, float],
    out_path: Path,
    dst_crs,
    width: int,
    height: int,
    dst_transform: Affine,
) -> Path:
    """Write a 4-band RGB+DSM GeoTIFF on the destination grid of a 2025 tile."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.zeros((3, height, width), dtype=np.uint8)
    dem = np.zeros((height, width), dtype=np.float32)

    with rasterio.open(ortho_path) as ortho:
        for i in range(1, 4):
            reproject(
                source=rasterio.band(ortho, i),
                destination=rgb[i - 1],
                src_transform=ortho.transform,
                src_crs=ortho.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

    with rasterio.open(dsm_path) as dsm:
        reproject(
            source=rasterio.band(dsm, 1),
            destination=dem,
            src_transform=dsm.transform,
            src_crs=dsm.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    dsm_u8 = elevation_to_uint8(dem)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "uint8",
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(rgb[0], 1)
        dst.write(rgb[1], 2)
        dst.write(rgb[2], 3)
        dst.write(dsm_u8, 4)
    return out_path


def mask_to_polygon(mask: np.ndarray, transform: Affine) -> Polygon | None:
    mask_u8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    polys = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # OpenCV contours are (col, row) = (x, y). rasterio xy() expects (row, col).
        pts = contour.reshape(-1, 2).astype(float)
        cols, rows = pts[:, 0], pts[:, 1]
        xs, ys = transform_xy(transform, rows, cols, offset="center")
        ring = list(zip(xs, ys))
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0:
            polys.append(poly)
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.geom_type == "MultiPolygon":
        # Keep largest part for matching
        merged = max(merged.geoms, key=lambda g: g.area)
    return merged if not merged.is_empty else None


def build_predictor(model_path: Path, score_thresh: float, device: str, image_size: int = 2000):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device
    cfg.INPUT.MAX_SIZE_TEST = image_size
    cfg.INPUT.MIN_SIZE_TEST = image_size
    cfg.INPUT.FORMAT = "BGR"
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.MODEL.PIXEL_MEAN = FOUR_BAND_PIXEL_MEAN
    cfg.MODEL.PIXEL_STD = FOUR_BAND_PIXEL_STD
    return DefaultPredictor(cfg)


def predict_tile_geojson(
    predictor,
    tile_path: Path,
    score_thresh: float,
    year_tag: str,
    tile_key: str,
) -> gpd.GeoDataFrame:
    with rasterio.open(tile_path) as src:
        transform = src.transform
        crs = src.crs

    model_image = load_bgrd_uint8(tile_path)
    outputs = predictor(model_image)
    instances = outputs["instances"].to("cpu")

    records = []
    if len(instances) == 0:
        return gpd.GeoDataFrame(records, geometry=[], crs=crs)

    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score < score_thresh:
            continue
        poly = mask_to_polygon(mask, transform)
        if poly is None or poly.area < 0.05:
            continue
        records.append(
            {
                "pred_id": i,
                "score": float(score),
                "year": year_tag,
                "tile_key": tile_key,
                "source_tile": tile_path.name,
                "area": float(poly.area),
                "geometry": poly,
            }
        )

    gdf = gpd.GeoDataFrame(records, crs=crs)
    if not gdf.empty and gdf.crs is not None:
        # Normalize compound CRS → EPSG:25829 for matching/volumes
        try:
            gdf = gdf.to_crs("EPSG:25829")
        except Exception:
            gdf = gdf.set_crs("EPSG:25829", allow_override=True)
    return gdf


def pad_key(key: str) -> str:
    row, col = key.split("_")[:2]
    return f"{int(row):02d}_{int(col):02d}"


def tile_key_from_name(name: str) -> str:
    # 25IniSouthOrt_04_35.tif → 04_35
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return stem


def load_gpkg_test_keys() -> tuple[list[str], list[str]]:
    """TEST_24 + TEST_25 from gpkg_to_coco (42 tiles total)."""
    sys.path.insert(0, str(_SCRIPTS))
    from gpkg_to_coco import TEST_24, TEST_25  # noqa: WPS433

    return list(TEST_24), list(TEST_25)


def find_rgb_tile(segmentation_dir: Path, year: int, key: str) -> Path:
    """Locate a 3-band (or already 4-band) ortho tile under segmentation/tiling/{year}."""
    pk = pad_key(key)
    year_dir = segmentation_dir / "tiling" / str(year)
    patterns = [
        f"*_{pk}.tif",
        f"*_{key}.tif",
    ]
    for pat in patterns:
        hits = sorted(year_dir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No RGB tile for year={year} key={key} under {year_dir}")


def find_existing_4band(segmentation_dir: Path, year: int, key: str) -> Path | None:
    pk = pad_key(key)
    for base in (
        segmentation_dir / f"tiling_rgb_dsm_{year}",
        segmentation_dir / "smoke_4band_july14" / f"tiling_rgb_dsm_{year}",
    ):
        if not base.exists():
            continue
        for pat in (f"*_{pk}.tif", f"*_{key}.tif"):
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    return None


def build_native_4band(
    rgb_tile: Path,
    dsm_path: Path,
    out_path: Path,
) -> Path:
    """Warp DSM onto an existing ortho tile grid and write RGB+DSM uint8."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(rgb_tile) as src:
        if src.count >= 4:
            # Already 4-band — copy through
            shutil.copy2(rgb_tile, out_path)
            return out_path
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        rgb = src.read([1, 2, 3])

    dem = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(dsm_path) as dsm:
        reproject(
            source=rasterio.band(dsm, 1),
            destination=dem,
            src_transform=dsm.transform,
            src_crs=dsm.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
        )
    dsm_u8 = elevation_to_uint8(dem)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        dst.write(rgb_u8[0], 1)
        dst.write(rgb_u8[1], 2)
        dst.write(rgb_u8[2], 3)
        dst.write(dsm_u8, 4)
    return out_path


def discover_split_test_tiles(
    root: Path,
) -> list[dict]:
    """Return [{year, key, rgb_path}, …] for all 42 hold-out test tiles."""
    seg = root / "segmentation"
    test_24, test_25 = load_gpkg_test_keys()
    entries = []
    for year, keys in ((24, test_24), (25, test_25)):
        for key in keys:
            rgb = find_rgb_tile(seg, year, key)
            entries.append({"year": year, "key": pad_key(key), "rgb_path": rgb})
    return entries


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        crs = gdf.crs or "EPSG:25829"
        empty = gpd.GeoDataFrame({"geometry": []}, crs=crs)
        empty.to_file(path, driver="GeoJSON")
        return
    gdf.to_file(path, driver="GeoJSON")


def run_pipeline(args: argparse.Namespace) -> dict:
    root = Path(args.project_root) if args.project_root else _project_root()
    outdir = Path(args.outdir)
    tiles_dir = outdir / "inference_tiles"
    pred_dir = outdir / "predictions"
    results_dir = outdir / "results"
    shots_dir = outdir / "screenshots"
    for d in (tiles_dir / "24", tiles_dir / "25", pred_dir, results_dir, shots_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = Path(args.model)
    ortho24 = Path(args.ortho_24) if args.ortho_24 else (
        root / "2024" / "Sites1and2_2024_Orthomosaic.tif"
    )
    ortho25 = Path(args.ortho_25) if args.ortho_25 else (
        root / "2025" / "25IniSouthOrt.tif"
    )
    dsm24 = Path(args.dsm_24) if args.dsm_24 else (
        root / "2024" / "Sites1and2_2024_DSM_30mm.tif"
    )
    dsm25 = Path(args.dsm_25) if args.dsm_25 else (
        root / "2025" / "25IniSouthDSM.tif"
    )

    if args.test_dir:
        # Legacy: only 2025 tiles in a folder (paired with aligned 2024 windows)
        test_entries = [
            {
                "year": 25,
                "key": pad_key(tile_key_from_name(p.name)),
                "rgb_path": p,
            }
            for p in sorted(Path(args.test_dir).glob("*.tif"))
        ]
    else:
        test_entries = discover_split_test_tiles(root)

    if not test_entries:
        raise SystemExit("No test tiles discovered")

    n24 = sum(1 for e in test_entries if e["year"] == 24)
    n25 = sum(1 for e in test_entries if e["year"] == 25)
    print(f"Found {len(test_entries)} test tiles (24:{n24} + 25:{n25})")
    print(f"Building predictor ({args.device}, score>={args.score_thresh}) ...")
    predictor = build_predictor(model, args.score_thresh, args.device, args.image_size)

    before_parts: list[gpd.GeoDataFrame] = []
    after_parts: list[gpd.GeoDataFrame] = []
    pair_meta = []
    seg = root / "segmentation"

    for idx, entry in enumerate(test_entries, start=1):
        year = entry["year"]
        key = entry["key"]
        rgb_path: Path = entry["rgb_path"]
        print(f"\n=== [{idx}/{len(test_entries)}] year={year} tile={key} ({rgb_path.name}) ===")

        # Native-year 4-band on the test tile's own grid
        native_out = tiles_dir / str(year) / f"native_{year}_{key}.tif"
        existing4 = find_existing_4band(seg, year, key)
        if args.rebuild_tiles or not native_out.exists():
            if existing4 is not None and not args.rebuild_tiles:
                print(f"  Using existing 4-band {existing4.name}")
                shutil.copy2(existing4, native_out)
            else:
                print(f"  Building native 4-band → {native_out.name}")
                build_native_4band(
                    rgb_path,
                    dsm24 if year == 24 else dsm25,
                    native_out,
                )
        else:
            print(f"  Reusing {native_out.name}")

        with rasterio.open(native_out) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            bounds = src.bounds

        # Opposite-year window warped onto the same grid
        opp_year = 25 if year == 24 else 24
        opp_out = tiles_dir / str(opp_year) / f"aligned_{opp_year}_on_{year}_{key}.tif"
        if args.rebuild_tiles or not opp_out.exists():
            print(f"  Building aligned {opp_year} window → {opp_out.name}")
            build_aligned_4band_window(
                ortho_path=ortho24 if opp_year == 24 else ortho25,
                dsm_path=dsm24 if opp_year == 24 else dsm25,
                bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
                out_path=opp_out,
                dst_crs=crs,
                width=width,
                height=height,
                dst_transform=transform,
            )
        else:
            print(f"  Reusing {opp_out.name}")

        tile24 = native_out if year == 24 else opp_out
        tile25 = native_out if year == 25 else opp_out

        print("  Inferring 2024 …")
        gdf24 = predict_tile_geojson(
            predictor, tile24, args.score_thresh, year_tag="24", tile_key=f"{year}_{key}"
        )
        print(f"  2024 detections: {len(gdf24)}")

        print("  Inferring 2025 …")
        gdf25 = predict_tile_geojson(
            predictor, tile25, args.score_thresh, year_tag="25", tile_key=f"{year}_{key}"
        )
        print(f"  2025 detections: {len(gdf25)}")

        write_geojson(gdf24, pred_dir / f"pred_24_{year}_{key}.geojson")
        write_geojson(gdf25, pred_dir / f"pred_25_{year}_{key}.geojson")

        before_parts.append(gdf24)
        after_parts.append(gdf25)
        pair_meta.append(
            {
                "source_year": year,
                "tile_key": key,
                "tile_24": str(tile24),
                "tile_25": str(tile25),
                "n_24": len(gdf24),
                "n_25": len(gdf25),
            }
        )

    def _concat(parts: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        nonempty = [p for p in parts if p is not None and not p.empty]
        if not nonempty:
            return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:25829")
        return gpd.GeoDataFrame(pd.concat(nonempty, ignore_index=True), crs=nonempty[0].crs)

    before = _concat(before_parts)
    after = _concat(after_parts)

    before_raw_path = pred_dir / "before_inferred_boulders_raw.geojson"
    after_raw_path = pred_dir / "after_inferred_boulders_raw.geojson"
    write_geojson(before, before_raw_path)
    write_geojson(after, after_raw_path)
    print(f"\nCombined predictions (raw): before={len(before)} after={len(after)}")

    n_before_raw, n_after_raw = len(before), len(after)
    if getattr(args, "dedupe", True) and (not before.empty or not after.empty):
        before = dedupe_polygons(
            before,
            iou_thresh=args.dedupe_iou,
            centroid_dist_m=args.dedupe_centroid_m,
        )
        after = dedupe_polygons(
            after,
            iou_thresh=args.dedupe_iou,
            centroid_dist_m=args.dedupe_centroid_m,
        )
        print(
            f"Dedupe (IoU>={args.dedupe_iou} or centroid≤{args.dedupe_centroid_m}m): "
            f"before {n_before_raw} → {len(before)}, after {n_after_raw} → {len(after)}"
        )

    before_path = pred_dir / "before_inferred_boulders.geojson"
    after_path = pred_dir / "after_inferred_boulders.geojson"
    write_geojson(before, before_path)
    write_geojson(after, after_path)
    print(f"Predictions used for matching: before={len(before)} after={len(after)}")

    if before.empty and after.empty:
        summary = {
            "model": str(model),
            "score_thresh": args.score_thresh,
            "tiles": pair_meta,
            "n_before_raw": n_before_raw,
            "n_after_raw": n_after_raw,
            "n_before": 0,
            "n_after": 0,
            "matches": 0,
            "appeared": 0,
            "disappeared": 0,
            "note": "No detections above score threshold on any test tile.",
        }
        (outdir / "match_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return summary

    before_survey = BoulderSurvey("before", str(before_path), dsm_path=str(dsm24)).compute_attributes()
    after_survey = BoulderSurvey("after", str(after_path), dsm_path=str(dsm25)).compute_attributes()
    if args.compute_volume and len(before) and len(after):
        print("Computing DSM volumes …")
        before_survey.compute_volume()
        after_survey.compute_volume()

    results = run_matcher_with_candidates(
        before_survey,
        after_survey,
        search_radius=args.search_radius,
        min_score=args.min_score,
        candidate_radius=getattr(args, "candidate_radius", None),
        candidate_min_score=getattr(args, "candidate_min_score", 0.35),
    )
    write_geojson(results["matches"], results_dir / "matched_boulders.geojson")
    write_geojson(results["appeared"], results_dir / "appeared_boulders.geojson")
    write_geojson(results["disappeared"], results_dir / "disappeared_boulders.geojson")
    write_geojson(results["vectors"], results_dir / "movement_vectors.geojson")
    write_missed_candidates(
        results["missed_candidates"], results_dir / "missed_candidates.geojson"
    )
    print(f"Missed-match candidates for review: {len(results['missed_candidates'])}")

    dod_summary = None
    if getattr(args, "dod_qc", True) and dsm24.exists() and dsm25.exists():
        print("Running DoD QC …")
        try:
            qc = run_dod_qc(
                results,
                before_polygons=before_survey.polygons,
                after_polygons=after_survey.polygons,
                before_dsm=dsm24,
                after_dsm=dsm25,
                lod_m=args.dod_lod_m,
                min_change_m3=args.dod_min_change_m3,
            )
            qc_dir = results_dir / "dod_qc"
            write_dod_qc(qc, qc_dir)
            dod_summary = qc["summary"]
            print(json.dumps(dod_summary, indent=2))
            print(f"DoD QC written to {qc_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"DoD QC failed (continuing without it): {exc}")

    summary = {
        "model": str(model),
        "score_thresh": args.score_thresh,
        "search_radius": args.search_radius,
        "min_score": args.min_score,
        "tiles": pair_meta,
        "n_before_raw": n_before_raw,
        "n_after_raw": n_after_raw,
        "n_before": len(before),
        "n_after": len(after),
        "matches": len(results["matches"]),
        "appeared": len(results["appeared"]),
        "disappeared": len(results["disappeared"]),
        "missed_candidates": len(results["missed_candidates"]),
        "candidate_radius": results.get("candidate_radius"),
        "dod_qc": dod_summary,
    }
    (outdir / "match_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "tiles"}, indent=2))

    if not args.no_screenshots:
        export_screenshots(
            load_results(results_dir),
            outdir=shots_dir,
            before=before,
            after=after,
            before_raster=Path(pair_meta[0]["tile_24"]) if pair_meta else None,
            after_raster=Path(pair_meta[0]["tile_25"]) if pair_meta else None,
            max_matches=args.max_matches,
            pad_m=args.pad_m,
            side_by_side=True,
            pair_tiles=[(m["tile_24"], m["tile_25"]) for m in pair_meta],
        )

    if args.gui:
        run_gui(
            load_results(results_dir),
            before=before,
            after=after,
            before_raster=Path(pair_meta[0]["tile_24"]) if pair_meta else None,
            after_raster=Path(pair_meta[0]["tile_25"]) if pair_meta else None,
            pad_m=args.pad_m,
            side_by_side=True,
            pair_tiles=[(m["tile_24"], m["tile_25"]) for m in pair_meta],
        )

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None,
                        help="Optional legacy folder of *.tif (defaults to gpkg TEST_24+TEST_25 = 42 tiles)")
    parser.add_argument("--ortho-24", type=Path, default=None)
    parser.add_argument("--ortho-25", type=Path, default=None)
    parser.add_argument("--dsm-24", type=Path, default=None)
    parser.add_argument("--dsm-25", type=Path, default=None)
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument(
        "--search-radius",
        type=float,
        default=BoulderMatcher.DEFAULT_SEARCH_RADIUS,
    )
    parser.add_argument("--min-score", type=float, default=BoulderMatcher.DEFAULT_MIN_SCORE)
    parser.add_argument(
        "--candidate-radius",
        type=float,
        default=None,
        help="Missed appeared↔disappeared review radius (default 1.5× search-radius)",
    )
    parser.add_argument("--candidate-min-score", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=2000)
    parser.add_argument("--compute-volume", action="store_true", default=True)
    parser.add_argument("--no-volume", action="store_true")
    parser.add_argument("--rebuild-tiles", action="store_true")
    parser.add_argument("--max-matches", type=int, default=40)
    parser.add_argument("--pad-m", type=float, default=8.0)
    parser.add_argument("--no-screenshots", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--dedupe", action="store_true", default=True)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--dedupe-iou", type=float, default=0.4)
    parser.add_argument("--dedupe-centroid-m", type=float, default=0.75)
    parser.add_argument("--dod-qc", action="store_true", default=True)
    parser.add_argument("--no-dod-qc", action="store_true")
    parser.add_argument("--dod-lod-m", type=float, default=0.08)
    parser.add_argument("--dod-min-change-m3", type=float, default=0.05)
    args = parser.parse_args()
    if args.no_volume:
        args.compute_volume = False
    if args.no_dedupe:
        args.dedupe = False
    if args.no_dod_qc:
        args.dod_qc = False
    run_pipeline(args)


if __name__ == "__main__":
    main()
