#!/usr/bin/env python3
"""Discover and parse geo512 / geo (2000) training runs and COCO datasets.

Naming (established by ``experiments/geo_splits_512/smoke_geo_splits_512.py``)::

    coco_geo512_{setup}_{modality}_from_pool/
    training_run_geo512_{setup}_{modality}/

Parent 2000×2000 experiment (for 512-vs-2000 comparison)::

    coco_geo_{setup}_from_pool/          # 4-band weekend
    training_run_geo_{setup}/

Modalities in repo: ``rgb``, ``rgb_dsm``, ``rgb_local_relief``
(not ``rgb_dsm_local_relief`` — that name does not appear in runners).

Checkpoints typically present: ``model_best.pth`` (early-stop), ``model_final.pth``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SETUPS: tuple[str, ...] = (
    "baseline",
    "blocks_alt_a",
    "blocks_alt_b",
    "north_south",
    "sporadic_aligned",
)

MODALITIES: tuple[str, ...] = ("rgb", "rgb_dsm", "rgb_local_relief")

# COCO area buckets (pixel² at the resolution used for eval / network input).
COCO_AREA_S = 32**2
COCO_AREA_M = 96**2

# Mask R-CNN R50-FPN zoo defaults (Detectron2 COCO-InstanceSegmentation).
DEFAULT_ANCHOR_SIZES: tuple[tuple[int, ...], ...] = (
    (32,),
    (64,),
    (128,),
    (256,),
    (512,),
)
DEFAULT_ANCHOR_ASPECT_RATIOS: tuple[float, ...] = (0.5, 1.0, 2.0)

# Inishmaan ortho GSD fallback when no GeoTIFF transform is readable (~3 cm).
DEFAULT_GSD_M = 0.03

ANN_FILES = {
    "train": "train_annotations.json",
    "valid": "validation_annotations.json",
    "test": "testing_annotations.json",
}

_GEO512_RUN_RE = re.compile(
    r"^training_run_geo512_(?P<setup>.+)_(?P<modality>rgb(?:_dsm|_local_relief)?)$"
)
_GEO512_COCO_RE = re.compile(
    r"^coco_geo512_(?P<setup>.+)_(?P<modality>rgb(?:_dsm|_local_relief)?)_from_pool$"
)
_GEO_RUN_RE = re.compile(r"^training_run_geo_(?P<setup>.+)$")
_GEO_COCO_RE = re.compile(r"^coco_geo_(?P<setup>.+)_from_pool$")


BucketDef = Literal["pixel", "ground_m2"]
SizeBucket = Literal["small", "medium", "large"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GeoRun:
    """One discovered training run."""

    setup: str
    modality: str
    run_dir: Path
    dataset_dir: Path | None
    tiling_regime: str  # "512" | "2000" | "unknown"
    checkpoint: Path | None = None
    provenance: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        return f"{self.setup}__{self.modality}__{self.tiling_regime}"

    @property
    def image_size(self) -> int | None:
        if not self.provenance:
            return None
        flags = self.provenance.get("flags") or {}
        val = flags.get("image_size")
        return int(val) if val is not None else None

    @property
    def four_band(self) -> bool:
        if self.modality in ("rgb_dsm", "rgb_local_relief"):
            return True
        if self.provenance:
            return bool((self.provenance.get("flags") or {}).get("four_band"))
        return False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["run_id"] = self.run_id
        d["run_dir"] = str(self.run_dir)
        d["dataset_dir"] = str(self.dataset_dir) if self.dataset_dir else None
        d["checkpoint"] = str(self.checkpoint) if self.checkpoint else None
        d["image_size"] = self.image_size
        d["four_band"] = self.four_band
        return d


@dataclass
class SizeBucketCounts:
    small: int = 0
    medium: int = 0
    large: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"small": self.small, "medium": self.medium, "large": self.large}

    def total(self) -> int:
        return self.small + self.medium + self.large


@dataclass
class InstanceSizeRecord:
    ann_id: int
    image_id: int
    area_px: float
    area_px_resized: float
    area_m2: float
    bucket_pixel: SizeBucket
    bucket_ground: SizeBucket
    bbox: list[float]
    touches_boundary: bool


# ---------------------------------------------------------------------------
# Discovery / naming
# ---------------------------------------------------------------------------


def parse_geo512_run_name(name: str) -> tuple[str, str] | None:
    m = _GEO512_RUN_RE.match(name)
    if not m:
        return None
    return m.group("setup"), m.group("modality")


def parse_geo512_coco_name(name: str) -> tuple[str, str] | None:
    m = _GEO512_COCO_RE.match(name)
    if not m:
        return None
    return m.group("setup"), m.group("modality")


def expected_coco_dir(seg: Path, setup: str, modality: str, *, tiling: str = "512") -> Path:
    if tiling == "512":
        return seg / f"coco_geo512_{setup}_{modality}_from_pool"
    # Parent 2000 weekend is RGB+DSM only under coco_geo_{setup}_from_pool
    if modality == "rgb_dsm":
        return seg / f"coco_geo_{setup}_from_pool"
    return seg / f"coco_geo_{setup}_{modality}_from_pool"


def resolve_parent_coco_dir(seg: Path, setup: str, modality: str) -> Path | None:
    """Try several historical parent COCO naming variants."""
    candidates = [
        expected_coco_dir(seg, setup, modality, tiling="2000"),
        seg / f"coco_geo_{setup}",
        seg / f"coco_geo_{setup}_rgb_dsm",
        seg / f"coco_geo_{setup}_{modality}",
        seg / f"coco_geo_{setup}_{modality}_from_pool",
    ]
    for path in candidates:
        if (path / "testing_annotations.json").is_file() or (
            path / "train_annotations.json"
        ).is_file():
            return path
    return None


def expected_run_dir(seg: Path, setup: str, modality: str, *, tiling: str = "512") -> Path:
    if tiling == "512":
        return seg / f"training_run_geo512_{setup}_{modality}"
    if modality == "rgb_dsm":
        return seg / f"training_run_geo_{setup}"
    return seg / f"training_run_geo_{setup}_{modality}"


def prefer_checkpoint(run_dir: Path) -> Path | None:
    """Prefer model_best.pth, then model_final.pth, then latest model_*.pth."""
    run_dir = Path(run_dir)
    for name in ("model_best.pth", "model_final.pth"):
        path = run_dir / name
        if path.is_file():
            return path
    numbered = sorted(run_dir.glob("model_*.pth"))
    numbered = [p for p in numbered if p.name not in ("model_best.pth", "model_final.pth")]
    if numbered:
        return numbered[-1]
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_provenance(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "training_run_provenance.json"
    if not path.is_file():
        return None
    return load_json(path)


def resolve_dataset_dir(run: GeoRun, seg: Path) -> Path | None:
    if run.dataset_dir and Path(run.dataset_dir).is_dir():
        return Path(run.dataset_dir)
    if run.provenance and run.provenance.get("dataset_dir"):
        cand = Path(run.provenance["dataset_dir"])
        if cand.is_dir():
            return cand
    if run.tiling_regime == "2000":
        found = resolve_parent_coco_dir(seg, run.setup, run.modality)
        if found is not None:
            return found
    expected = expected_coco_dir(seg, run.setup, run.modality, tiling=run.tiling_regime)
    if expected.is_dir():
        return expected
    return None


def discover_geo512_runs(
    segmentation_dir: Path | str,
    *,
    prefer_ckpt: bool = True,
) -> list[GeoRun]:
    """Glob ``training_run_geo512_*`` and parse (setup, modality)."""
    seg = Path(segmentation_dir)
    runs: list[GeoRun] = []
    if not seg.is_dir():
        return runs
    for path in sorted(seg.glob("training_run_geo512_*")):
        if not path.is_dir():
            continue
        parsed = parse_geo512_run_name(path.name)
        if parsed is None:
            continue
        setup, modality = parsed
        prov = load_provenance(path)
        dataset = None
        if prov and prov.get("dataset_dir"):
            dataset = Path(prov["dataset_dir"])
        else:
            dataset = expected_coco_dir(seg, setup, modality, tiling="512")
            if not dataset.is_dir():
                dataset = None
        ckpt = prefer_checkpoint(path) if prefer_ckpt else None
        runs.append(
            GeoRun(
                setup=setup,
                modality=modality,
                run_dir=path,
                dataset_dir=dataset,
                tiling_regime="512",
                checkpoint=ckpt,
                provenance=prov,
            )
        )
    return runs


def discover_geo512_coco_dirs(segmentation_dir: Path | str) -> list[tuple[str, str, Path]]:
    """Return (setup, modality, path) for materialized geo512 COCO dirs."""
    seg = Path(segmentation_dir)
    out: list[tuple[str, str, Path]] = []
    if not seg.is_dir():
        return out
    for path in sorted(seg.glob("coco_geo512_*_from_pool")):
        parsed = parse_geo512_coco_name(path.name)
        if parsed is None:
            continue
        setup, modality = parsed
        out.append((setup, modality, path))
    return out


def discover_parent_geo_runs(segmentation_dir: Path | str) -> list[GeoRun]:
    """Discover 2000-tile ``training_run_geo_{setup}`` (RGB+DSM weekend)."""
    seg = Path(segmentation_dir)
    runs: list[GeoRun] = []
    if not seg.is_dir():
        return runs
    for path in sorted(seg.glob("training_run_geo_*")):
        if path.name.startswith("training_run_geo512_"):
            continue
        if path.name.endswith("_smoke"):
            continue
        m = _GEO_RUN_RE.match(path.name)
        if not m:
            continue
        setup = m.group("setup")
        # Skip modality-suffixed names that aren't plain setups.
        if setup not in SETUPS:
            # Allow training_run_geo_{setup}_{modality} if present.
            for mod in MODALITIES:
                suffix = f"_{mod}"
                if setup.endswith(suffix):
                    real_setup = setup[: -len(suffix)]
                    if real_setup in SETUPS:
                        setup, modality = real_setup, mod
                        break
            else:
                continue
        else:
            modality = "rgb_dsm"  # parent weekend default
        prov = load_provenance(path)
        dataset = None
        if prov and prov.get("dataset_dir"):
            dataset = Path(prov["dataset_dir"])
        else:
            dataset = expected_coco_dir(seg, setup, modality, tiling="2000")
            if not dataset.is_dir():
                dataset = resolve_parent_coco_dir(seg, setup, modality)
        runs.append(
            GeoRun(
                setup=setup,
                modality=modality,
                run_dir=path,
                dataset_dir=dataset,
                tiling_regime="2000",
                checkpoint=prefer_checkpoint(path),
                provenance=prov,
            )
        )
    return runs


def glob_report(segmentation_dir: Path | str) -> dict[str, Any]:
    """Enumerate what exists on disk (no assumptions)."""
    seg = Path(segmentation_dir)
    geo512_runs = discover_geo512_runs(seg)
    geo512_coco = discover_geo512_coco_dirs(seg)
    parent_runs = discover_parent_geo_runs(seg)
    return {
        "segmentation_dir": str(seg),
        "exists": seg.is_dir(),
        "geo512_runs": [r.to_dict() for r in geo512_runs],
        "geo512_coco": [
            {"setup": s, "modality": m, "path": str(p)} for s, m, p in geo512_coco
        ],
        "parent_2000_runs": [r.to_dict() for r in parent_runs],
        "n_geo512_runs": len(geo512_runs),
        "n_geo512_coco": len(geo512_coco),
        "n_parent_runs": len(parent_runs),
    }


# ---------------------------------------------------------------------------
# GSD / size buckets / anchors
# ---------------------------------------------------------------------------


def read_gsd_m(image_path: Path | str | None, default: float = DEFAULT_GSD_M) -> tuple[float, str]:
    """Return (gsd_meters, source). Reads GeoTIFF transform when possible."""
    if image_path is None:
        return default, "default_fallback"
    path = Path(image_path)
    if not path.is_file():
        return default, "default_fallback"
    try:
        import rasterio

        with rasterio.open(path) as ds:
            gsd = abs(float(ds.transform.a))
            if gsd > 0:
                return gsd, "geotiff_transform"
    except Exception:
        pass
    return default, "default_fallback"


def coco_size_bucket(area: float) -> SizeBucket:
    if area < COCO_AREA_S:
        return "small"
    if area < COCO_AREA_M:
        return "medium"
    return "large"


def ground_area_thresholds_m2(gsd_m: float) -> tuple[float, float]:
    """COCO pixel thresholds expressed as ground m² at ``gsd_m``."""
    return COCO_AREA_S * gsd_m * gsd_m, COCO_AREA_M * gsd_m * gsd_m


def ground_size_bucket(area_m2: float, gsd_m: float) -> SizeBucket:
    t_s, t_m = ground_area_thresholds_m2(gsd_m)
    if area_m2 < t_s:
        return "small"
    if area_m2 < t_m:
        return "medium"
    return "large"


def bbox_touches_boundary(
    bbox_xywh: list[float] | tuple[float, ...],
    width: int,
    height: int,
    *,
    eps: float = 1.0,
) -> bool:
    x, y, w, h = map(float, bbox_xywh)
    return (
        x <= eps
        or y <= eps
        or (x + w) >= (width - eps)
        or (y + h) >= (height - eps)
    )


def resolve_split_image(
    dataset_dir: Path,
    split: str,
    file_name: str,
) -> Path | None:
    candidates = [
        dataset_dir / split / file_name,
        dataset_dir / "train" / file_name,
        dataset_dir / "valid" / file_name,
        dataset_dir / "test" / file_name,
        dataset_dir / file_name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def effective_network_gsd(
    native_gsd_m: float,
    tile_w: int,
    tile_h: int,
    image_size: int,
) -> float:
    """Meters/pixel after Detectron2 square resize to ``image_size``."""
    # Detectron2 ResizeShortestEdge with equal min/max → scale so short side = image_size.
    short = min(tile_w, tile_h)
    if short <= 0:
        return native_gsd_m
    scale = image_size / float(short)
    # Larger network pixels → coarser effective GSD when upscaling? No:
    # if we upscale, each network pixel covers less ground. scale>1 → finer.
    return native_gsd_m / scale


def flat_anchor_sizes(anchor_sizes: Iterable[Iterable[int]] | None = None) -> list[int]:
    sizes = anchor_sizes or DEFAULT_ANCHOR_SIZES
    out: list[int] = []
    for level in sizes:
        out.extend(int(s) for s in level)
    return out


def analyze_coco_split(
    coco: dict[str, Any],
    *,
    image_size: int,
    native_gsd_m: float,
    gsd_source: str,
    dataset_dir: Path | None = None,
    split: str = "train",
    skip_crowd: bool = True,
    boundary_eps: float = 1.0,
    sample_gsd_from_images: bool = True,
    max_gsd_samples: int = 8,
) -> dict[str, Any]:
    """Compute empty-chip / truncation / size-bucket / anchor-vs-size diagnostics."""
    images = {int(im["id"]): im for im in coco.get("images", [])}
    anns_by_image: dict[int, list[dict]] = {i: [] for i in images}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    # Optionally refine GSD from a few GeoTIFFs.
    gsd = native_gsd_m
    gsd_src = gsd_source
    if sample_gsd_from_images and dataset_dir is not None:
        sampled = 0
        for im in coco.get("images", []):
            if sampled >= max_gsd_samples:
                break
            path = resolve_split_image(dataset_dir, split, im["file_name"])
            if path is None:
                continue
            gsd_i, src_i = read_gsd_m(path, default=native_gsd_m)
            if src_i == "geotiff_transform":
                gsd = gsd_i
                gsd_src = src_i
                sampled += 1
                break  # one reliable transform is enough (constant GSD)

    n_images = len(images)
    empty = 0
    empty_any = 0
    n_inst = 0
    n_trunc = 0
    pixel_counts = SizeBucketCounts()
    ground_counts = SizeBucketCounts()
    widths: list[float] = []
    heights: list[float] = []
    areas_px_resized: list[float] = []
    areas_m2: list[float] = []
    sqrt_areas_net: list[float] = []

    tile_w = tile_h = image_size
    if images:
        sample_im = next(iter(images.values()))
        tile_w = int(sample_im.get("width") or image_size)
        tile_h = int(sample_im.get("height") or image_size)

    eff_gsd = effective_network_gsd(gsd, tile_w, tile_h, image_size)
    scale_x = image_size / float(tile_w) if tile_w else 1.0
    scale_y = image_size / float(tile_h) if tile_h else 1.0
    # Area scale under isotropic-ish resize (short-side resize approximates this).
    area_scale = scale_x * scale_y

    for image_id, im in images.items():
        w = int(im["width"])
        h = int(im["height"])
        anns = anns_by_image.get(image_id, [])
        if not anns:
            empty_any += 1
        trainable = [a for a in anns if not (skip_crowd and a.get("iscrowd", 0))]
        if not trainable:
            empty += 1
        for ann in trainable:
            n_inst += 1
            bbox = list(map(float, ann["bbox"]))
            area_px = float(ann.get("area") or max(0.0, bbox[2] * bbox[3]))
            area_resized = area_px * area_scale
            area_m2 = area_px * gsd * gsd
            bp = coco_size_bucket(area_resized)
            bg = ground_size_bucket(area_m2, gsd)
            if bp == "small":
                pixel_counts.small += 1
            elif bp == "medium":
                pixel_counts.medium += 1
            else:
                pixel_counts.large += 1
            if bg == "small":
                ground_counts.small += 1
            elif bg == "medium":
                ground_counts.medium += 1
            else:
                ground_counts.large += 1
            touches = bbox_touches_boundary(bbox, w, h, eps=boundary_eps)
            if touches:
                n_trunc += 1
            # Network-pixel bbox size (after resize).
            net_w = bbox[2] * scale_x
            net_h = bbox[3] * scale_y
            widths.append(net_w)
            heights.append(net_h)
            areas_px_resized.append(area_resized)
            areas_m2.append(area_m2)
            sqrt_areas_net.append(math.sqrt(max(area_resized, 0.0)))

    anchors = flat_anchor_sizes()
    # Histogram of sqrt(area) vs nearest anchor.
    hist_bins = {
        "lt_half_nearest": 0,
        "within_2x_nearest": 0,
        "gt_2x_nearest": 0,
        "nearest_anchor_counts": {str(a): 0 for a in anchors},
    }
    for s in sqrt_areas_net:
        if not anchors:
            break
        nearest = min(anchors, key=lambda a: abs(a - s))
        hist_bins["nearest_anchor_counts"][str(nearest)] += 1
        ratio = s / float(nearest) if nearest else 0.0
        if ratio < 0.5:
            hist_bins["lt_half_nearest"] += 1
        elif ratio <= 2.0:
            hist_bins["within_2x_nearest"] += 1
        else:
            hist_bins["gt_2x_nearest"] += 1

    # Native-pixel vs ground-area flip using same GSD-mapped thresholds.
    flip_native_vs_ground = 0
    for area_px, area_m2 in zip(
        # rebuild from lists: area_px = area_resized / area_scale
        [a / area_scale if area_scale else a for a in areas_px_resized],
        areas_m2,
    ):
        if coco_size_bucket(area_px) != ground_size_bucket(area_m2, gsd):
            flip_native_vs_ground += 1

    # Resized-pixel vs ground (using thresholds at native gsd — intentional confound probe).
    flip_resized_vs_ground = 0
    for area_r, area_m2 in zip(areas_px_resized, areas_m2):
        if coco_size_bucket(area_r) != ground_size_bucket(area_m2, gsd):
            flip_resized_vs_ground += 1

    return {
        "n_images": n_images,
        "n_instances_trainable": n_inst,
        "empty_chip_fraction_trainable": (empty / n_images) if n_images else float("nan"),
        "empty_chip_fraction_any_ann": (empty_any / n_images) if n_images else float("nan"),
        "n_empty_trainable": empty,
        "n_empty_any_ann": empty_any,
        "boundary_truncation_fraction": (n_trunc / n_inst) if n_inst else float("nan"),
        "n_truncated": n_trunc,
        "tile_width": tile_w,
        "tile_height": tile_h,
        "image_size_network": image_size,
        "native_gsd_m": gsd,
        "gsd_source": gsd_src,
        "effective_network_gsd_m": eff_gsd,
        "resize_scale_xy": [scale_x, scale_y],
        "size_buckets_pixel_resized": pixel_counts.as_dict(),
        "size_buckets_ground_m2": ground_counts.as_dict(),
        "ground_area_thresholds_m2": {
            "small_lt": ground_area_thresholds_m2(gsd)[0],
            "medium_lt": ground_area_thresholds_m2(gsd)[1],
        },
        "bucket_flip_native_px_vs_ground": flip_native_vs_ground,
        "bucket_flip_resized_px_vs_ground": flip_resized_vs_ground,
        "bucket_flip_fraction_resized_vs_ground": (
            flip_resized_vs_ground / n_inst if n_inst else float("nan")
        ),
        "anchor_sizes_px": anchors,
        "anchor_aspect_ratios": list(DEFAULT_ANCHOR_ASPECT_RATIOS),
        "anchor_vs_sqrt_area_hist": hist_bins,
        "sqrt_area_net_px_summary": _summary(sqrt_areas_net),
        "bbox_w_net_px_summary": _summary(widths),
        "bbox_h_net_px_summary": _summary(heights),
        "area_m2_summary": _summary(areas_m2),
    }


def _summary(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = sorted(vals)
    n = len(arr)

    def pct(p: float) -> float:
        if n == 1:
            return arr[0]
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return arr[idx]

    return {
        "n": n,
        "mean": sum(arr) / n,
        "p50": pct(50),
        "p90": pct(90),
        "min": arr[0],
        "max": arr[-1],
    }


def rewrite_areas_for_ground_eval(
    coco: dict[str, Any],
    gsd_m: float,
) -> dict[str, Any]:
    """Copy COCO with ``area`` set to ground m² so COCOeval areaRng can use m²."""
    out = json.loads(json.dumps(coco))  # deep copy via JSON
    for ann in out.get("annotations", []):
        bbox = ann.get("bbox") or [0, 0, 0, 0]
        area_px = float(ann.get("area") or max(0.0, float(bbox[2]) * float(bbox[3])))
        ann["area"] = area_px * gsd_m * gsd_m
        ann["_area_px"] = area_px
    return out


def scale_annotation_areas_for_resize(
    coco: dict[str, Any],
    image_size: int,
) -> dict[str, Any]:
    """Scale annotation areas/bboxes to network input resolution for pixel-bucket eval."""
    out = json.loads(json.dumps(coco))
    for im in out.get("images", []):
        w = int(im["width"])
        h = int(im["height"])
        short = min(w, h) or 1
        scale = image_size / float(short)
        im["_scale"] = scale
        im["width"] = int(round(w * scale))
        im["height"] = int(round(h * scale))
    scales = {int(im["id"]): float(im.get("_scale", 1.0)) for im in out["images"]}
    for ann in out.get("annotations", []):
        s = scales.get(int(ann["image_id"]), 1.0)
        x, y, bw, bh = map(float, ann["bbox"])
        ann["bbox"] = [x * s, y * s, bw * s, bh * s]
        ann["area"] = float(ann.get("area", bw * bh)) * s * s
    return out
