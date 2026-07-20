"""Shared helpers for comparing training runs and per-tile AP/AR evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Metrics JSON (whole-split / learning curves)
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)


def load_metrics_valid(path: Path | str) -> dict[str, dict[str, float]]:
    """Load a Detectron2-style ``metrics_valid.json`` (nested bbox/segm dicts)."""
    data = json.loads(Path(path).read_text())
    out: dict[str, dict[str, float]] = {}
    for task in ("bbox", "segm"):
        if task in data and isinstance(data[task], dict):
            out[task] = {k: float(v) for k, v in data[task].items() if _is_number(v)}
    return out


def load_metrics_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load Detectron2 ``metrics.json`` (JSONL event stream)."""
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def extract_eval_curve(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Periodic validation rows from ``metrics.json`` (have bbox/AP* or segm/AP*)."""
    curve = []
    for r in rows:
        if not any(k.startswith(("bbox/", "segm/")) for k in r):
            continue
        point = {"iteration": int(r.get("iteration", -1))}
        for task in ("bbox", "segm"):
            for key in _METRIC_KEYS:
                flat = f"{task}/{key}"
                if flat in r and _is_number(r[flat]):
                    point[flat] = float(r[flat])
        curve.append(point)
    return curve


def compare_metrics_valid(
    runs: dict[str, Path | str],
    tasks: tuple[str, ...] = ("bbox", "segm"),
    keys: tuple[str, ...] = ("AP", "AP50", "AP75", "AR100", "ARs", "ARm", "ARl"),
) -> "Any":
    """Build a pandas DataFrame comparing several ``metrics_valid.json`` files.

    ``runs`` maps display name → path. Missing keys become NaN.
    """
    import pandas as pd

    records = []
    for name, path in runs.items():
        metrics = load_metrics_valid(path)
        row: dict[str, Any] = {"run": name, "path": str(path)}
        for task in tasks:
            block = metrics.get(task, {})
            for key in keys:
                row[f"{task}/{key}"] = block.get(key, float("nan"))
        records.append(row)
    return pd.DataFrame(records).set_index("run")


def discover_geo_runs(
    segmentation_dir: Path | str,
    prefix: str = "training_run_geo_",
) -> dict[str, Path]:
    """Find ``segmentation/training_run_geo_*/metrics_valid.json``."""
    root = Path(segmentation_dir)
    found: dict[str, Path] = {}
    for path in sorted(root.glob(f"{prefix}*/metrics_valid.json")):
        name = path.parent.name
        if name.startswith(prefix):
            name = name[len(prefix) :]
        found[name] = path
    return found


# ---------------------------------------------------------------------------
# Tile naming / geometry
# ---------------------------------------------------------------------------

_STEM_RE = re.compile(
    r"(?:(?P<year>24|25)_)?(?:Sites1and2_2024_Orthomosaic_|25IniSouthOrt_)?(?P<row>\d+)_(?P<col>\d+)",
    re.IGNORECASE,
)


def parse_tile_stem(stem_or_name: str) -> tuple[int | None, int, int] | None:
    """Parse ``24_..._07_38`` / ``25IniSouthOrt_05_31`` → (year|None, row, col)."""
    stem = Path(stem_or_name).stem
    # Prefer year-prefixed COCO filenames.
    m = re.match(r"^(24|25)_.+_(\d+)_(\d+)$", stem)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.match(r"^Sites1and2_2024_Orthomosaic_(\d+)_(\d+)$", stem)
    if m:
        return 24, int(m.group(1)), int(m.group(2))
    m = re.match(r"^25IniSouthOrt_(\d+)_(\d+)$", stem)
    if m:
        return 25, int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d+)_(\d+)$", stem)
    if m:
        return None, int(m.group(1)), int(m.group(2))
    return None


def year_key(year: int | None, row: int, col: int) -> str | None:
    if year is None:
        return None
    return f"{year}_{row:02d}_{col:02d}"


def load_split_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML required to load split YAML configs") from exc
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def tiles_in_split(split_cfg: dict[str, Any], split: str) -> set[str]:
    """Return year_keys like ``24_07_38`` for a split name (train/valid/test/excluded)."""
    block = split_cfg.get(split, {}) or {}
    keys: set[str] = set()
    for year, tiles in block.items():
        y = int(year)
        for t in tiles or []:
            parts = str(t).strip().split("_")
            if len(parts) != 2:
                continue
            row, col = int(parts[0]), int(parts[1])
            keys.add(f"{y}_{row:02d}_{col:02d}")
    return keys


# ---------------------------------------------------------------------------
# Detection matching / NMS (sliding-window merge)
# ---------------------------------------------------------------------------


def _bbox_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def detection_to_xyxy(det: dict[str, Any]) -> list[float]:
    """Normalize a detection dict to ``[x1, y1, x2, y2]``."""
    if "bbox_xyxy" in det:
        return list(map(float, det["bbox_xyxy"]))
    bbox = list(map(float, det["bbox"]))
    fmt = det.get("bbox_format")
    if fmt == "xyxy" or det.get("source") == "inference_summary":
        return bbox
    if fmt == "xywh" or det.get("source") == "coco" or "segmentation" in det:
        return _bbox_xywh_to_xyxy(bbox)
    # Default: COCO xywh.
    return _bbox_xywh_to_xyxy(bbox)


def nms_detections(
    detections: list[dict[str, Any]],
    iou_thresh: float = 0.4,
    score_key: str = "score",
) -> list[dict[str, Any]]:
    """Greedy NMS on detections with COCO ``bbox`` [x,y,w,h] or xyxy.

    Use this after sliding-window inference to merge overlapping detections
    before per-tile scoring (``merge``), or skip it to score raw overlaps
    (``no merge``).
    """
    if not detections:
        return []
    work = []
    for d in detections:
        item = dict(d)
        try:
            xyxy = detection_to_xyxy(item)
        except (KeyError, TypeError, ValueError):
            continue
        item["_xyxy"] = xyxy
        item[score_key] = float(item.get(score_key, 0.0))
        work.append(item)

    order = sorted(range(len(work)), key=lambda i: work[i][score_key], reverse=True)
    keep: list[dict[str, Any]] = []
    suppressed = [False] * len(work)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(work[i])
        for j in order:
            if j == i or suppressed[j]:
                continue
            if _iou_xyxy(work[i]["_xyxy"], work[j]["_xyxy"]) >= iou_thresh:
                suppressed[j] = True
    for d in keep:
        d.pop("_xyxy", None)
    return keep


def match_precision_recall(
    gt_boxes_xyxy: list[list[float]],
    pred_boxes_xyxy: list[list[float]],
    pred_scores: list[float] | None = None,
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Greedy one-to-one matching at a fixed IoU → precision / recall / F1."""
    n_gt = len(gt_boxes_xyxy)
    n_pred = len(pred_boxes_xyxy)
    if n_gt == 0 and n_pred == 0:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "gt_count": 0,
            "pred_count": 0,
        }
    order = list(range(n_pred))
    if pred_scores is not None:
        order = sorted(order, key=lambda i: pred_scores[i], reverse=True)
    matched_gt = set()
    tp = 0
    for pi in order:
        best_iou, best_g = 0.0, -1
        for gi, gb in enumerate(gt_boxes_xyxy):
            if gi in matched_gt:
                continue
            iou = _iou_xyxy(pred_boxes_xyxy[pi], gb)
            if iou > best_iou:
                best_iou, best_g = iou, gi
        if best_iou >= iou_thresh and best_g >= 0:
            matched_gt.add(best_g)
            tp += 1
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_count": n_gt,
        "pred_count": n_pred,
    }


# ---------------------------------------------------------------------------
# Per-image COCO AP / AR
# ---------------------------------------------------------------------------


def _is_number(v: Any) -> bool:
    try:
        return v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def coco_ap_ar_for_image(
    gt_anns: list[dict[str, Any]],
    pred_anns: list[dict[str, Any]],
    width: int,
    height: int,
    iou_type: str = "bbox",
) -> dict[str, float]:
    """Run pycocotools COCOeval on a single image; return AP/AR keys (0–100).

    Per-tile AP/AR is noisier than whole-split metrics (few objects), but is
    useful for heatmaps and comparing whether certain hold-out tiles are easier.
    Crowd GT (``iscrowd=1``) is kept so COCO can ignore regions correctly.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    nan = float("nan")
    empty = {k: nan for k in _METRIC_KEYS}
    trainable = [a for a in gt_anns if not a.get("iscrowd", 0)]
    if not trainable:
        # No positive GT on this tile — COCO AP/AR are undefined; leave NaN.
        # (Crowd-only tiles are common with boulder-only + min-area ignore.)
        return empty

    if not pred_anns:
        return {k: 0.0 for k in _METRIC_KEYS}

    gt = {
        "images": [{"id": 1, "width": width, "height": height, "file_name": "tile.tif"}],
        "categories": [{"id": 1, "name": "Boulder"}],
        "annotations": [],
    }
    for i, ann in enumerate(gt_anns, start=1):
        item = {
            "id": i,
            "image_id": 1,
            "category_id": int(ann.get("category_id", 1)),
            "iscrowd": int(ann.get("iscrowd", 0)),
            "area": float(ann.get("area", 0.0)),
            "bbox": list(map(float, ann["bbox"])),
        }
        if "segmentation" in ann:
            item["segmentation"] = ann["segmentation"]
        if item["area"] <= 0 and item["bbox"]:
            item["area"] = max(0.0, float(item["bbox"][2]) * float(item["bbox"][3]))
        if iou_type == "segm" and "segmentation" not in item:
            x, y, w, h = item["bbox"]
            item["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        gt["annotations"].append(item)

    dt = []
    for i, ann in enumerate(pred_anns, start=1):
        bbox = list(map(float, ann["bbox"]))
        if ann.get("bbox_format") == "xyxy" or ann.get("source") == "inference_summary":
            x1, y1, x2, y2 = bbox
            bbox = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
        elif ann.get("bbox_format") != "xywh" and ann.get("source") != "coco":
            # Already xywh from model export path.
            pass
        item = {
            "id": i,
            "image_id": 1,
            "category_id": int(ann.get("category_id", 1)),
            "bbox": bbox,
            "score": float(ann.get("score", 1.0)),
            "area": float(ann.get("area", bbox[2] * bbox[3])),
        }
        if iou_type == "segm":
            if "segmentation" in ann:
                item["segmentation"] = ann["segmentation"]
            else:
                x, y, w, h = bbox
                item["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        dt.append(item)

    import io
    from contextlib import redirect_stdout

    # Silence pycocotools console spam (createIndex / loadRes / summarize).
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt)
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.params.imgIds = [1]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    names = list(_METRIC_KEYS)
    results = {}
    for idx, name in enumerate(names):
        val = float(ev.stats[idx]) if idx < len(ev.stats) else nan
        results[name] = val * 100.0 if np.isfinite(val) and val >= 0 else nan
    return results


# ---------------------------------------------------------------------------
# Load GT / predictions for a COCO split
# ---------------------------------------------------------------------------


def load_coco(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def coco_split_paths(dataset_dir: Path | str, split: str) -> tuple[Path, Path]:
    dataset_dir = Path(dataset_dir)
    ann_name = {
        "train": "train_annotations.json",
        "valid": "validation_annotations.json",
        "test": "testing_annotations.json",
    }[split]
    return dataset_dir / ann_name, dataset_dir / split


def resolve_project_root(start: Path | str | None = None) -> Path:
    """Walk up from ``start`` (default cwd) until ``BoulderCalculator/scripts/eval_utils.py`` exists."""
    cur = Path(start or Path.cwd()).resolve()
    for cand in [cur, *cur.parents]:
        marker = cand / "BoulderCalculator" / "scripts" / "eval_utils.py"
        if marker.is_file():
            return cand
    raise FileNotFoundError(
        f"Could not find project root (BoulderCalculator/scripts/eval_utils.py) from {cur}"
    )


def resolve_image_file(file_name: str, image_dirs: Iterable[Path | str]) -> Path | None:
    """Locate a tile under one or more dirs; try stripping a leading ``24_`` / ``25_`` prefix."""
    names = [file_name, Path(file_name).name]
    for prefix in ("24_", "25_"):
        base = Path(file_name).name
        if base.startswith(prefix):
            names.append(base[len(prefix) :])
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        for d in image_dirs:
            path = Path(d) / name
            if path.exists():
                return path
    return None


def filter_coco_to_existing_images(
    coco: dict[str, Any],
    image_dirs: Iterable[Path | str],
) -> tuple[dict[str, Any], list[str]]:
    """Keep only COCO images whose files resolve under ``image_dirs``. Returns (coco, missing stems)."""
    keep_ids: set[int] = set()
    missing: list[str] = []
    kept_images = []
    for img in coco.get("images", []):
        path = resolve_image_file(img["file_name"], image_dirs)
        if path is None:
            missing.append(Path(img["file_name"]).stem)
            continue
        keep_ids.add(int(img["id"]))
        kept_images.append(img)
    kept_anns = [a for a in coco.get("annotations", []) if int(a["image_id"]) in keep_ids]
    out = dict(coco)
    out["images"] = kept_images
    out["annotations"] = kept_anns
    return out, missing


def suggest_coco_dirs(segmentation_dir: Path | str, hint: str | None = None) -> list[Path]:
    """List COCO-looking dirs under segmentation that have testing_annotations.json."""
    root = Path(segmentation_dir)
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.glob("coco*")):
        if (path / "testing_annotations.json").is_file():
            found.append(path)
    if hint:
        # Prefer names similar to the failed hint.
        found.sort(key=lambda p: (hint not in p.name, p.name))
    return found


def resolve_gt_annotations(
    dataset_dir: Path | str | None,
    split: str,
    gt_json: Path | str | None = None,
) -> Path:
    """Resolve annotation JSON path; try RGB sibling if ``*_rgb_dsm`` is a stub."""
    if gt_json is not None:
        path = Path(gt_json)
        if path.is_file():
            return path
        raise FileNotFoundError(f"GT JSON not found: {path}")

    if dataset_dir is None:
        raise FileNotFoundError("Pass --gt-json and/or --dataset-dir.")

    dataset_dir = Path(dataset_dir)
    ann_name = {
        "train": "train_annotations.json",
        "valid": "validation_annotations.json",
        "test": "testing_annotations.json",
    }[split]
    path = dataset_dir / ann_name
    if path.is_file():
        return path

    # Common laptop case: coco_geo_*_rgb_dsm exists as an empty stub; RGB sibling has JSONs.
    sibling = None
    if dataset_dir.name.endswith("_rgb_dsm"):
        sibling = dataset_dir.with_name(dataset_dir.name[: -len("_rgb_dsm")])
        sib_path = sibling / ann_name
        if sib_path.is_file():
            print(
                f"Note: {path} missing; using RGB annotations from {sib_path}\n"
                f"      Pair with --image-dir tiling_rgb_dsm_* (and --four-band) for 4-band models,\n"
                f"      or rebuild via build_coco_rgb_dsm.py."
            )
            return sib_path

    suggestions = suggest_coco_dirs(dataset_dir.parent, hint=dataset_dir.name)
    msg = [f"Annotation file not found: {path}"]
    if sibling is not None and not (sibling / ann_name).is_file():
        msg.append(f"RGB sibling also missing: {sibling / ann_name}")
    if suggestions:
        msg.append("Datasets on disk with testing_annotations.json:")
        for s in suggestions[:12]:
            msg.append(f"  - {s}")
    raise FileNotFoundError("\n".join(msg))


def load_prediction_file(path: Path, image_name: str, width: int, height: int) -> list[dict]:
    """Load ``*_detections.coco.json`` or ``*_inference_summary.json`` → pred anns."""
    data = json.loads(Path(path).read_text())
    if path.name.endswith("_inference_summary.json"):
        preds = []
        for det in data.get("detections", []):
            preds.append(
                {
                    "bbox": list(map(float, det["bbox"])),
                    "bbox_format": "xyxy",
                    "score": float(det.get("score", 1.0)),
                    "category_id": int(det.get("class_id", 0)) + 1,
                    "source": "inference_summary",
                }
            )
        return preds
    # COCO-style predictions file (either full doc or results list)
    if isinstance(data, list):
        anns = data
    else:
        anns = data.get("annotations", data.get("annotations", []))
        if not anns and "images" in data:
            anns = data.get("annotations", [])
    out = []
    for ann in anns:
        item = dict(ann)
        item["source"] = "coco"
        item["bbox_format"] = "xywh"
        out.append(item)
    return out


def collect_prediction_map(predictions_dir: Path | str) -> dict[str, Path]:
    """Map image stem → prediction JSON path."""
    predictions_dir = Path(predictions_dir)
    mapping: dict[str, Path] = {}
    for pattern in ("*_detections.coco.json", "*_inference_summary.json"):
        for path in predictions_dir.glob(pattern):
            name = path.name
            for suffix in ("_detections.coco.json", "_inference_summary.json"):
                if name.endswith(suffix):
                    stem = name[: -len(suffix)]
                    break
            else:
                stem = path.stem
            # Prefer coco detections over summaries when both exist.
            if stem not in mapping or path.name.endswith("_detections.coco.json"):
                mapping[stem] = path
    return mapping


def evaluate_coco_split_per_tile(
    gt_coco: dict[str, Any],
    predictions_by_stem: dict[str, list[dict[str, Any]]],
    *,
    iou_type: str = "bbox",
    match_iou: float = 0.5,
    merge_iou: float | None = None,
    skip_crowd_as_gt: bool = True,
) -> list[dict[str, Any]]:
    """Score every image in a COCO dict; return one row per tile.

    Parameters
    ----------
    merge_iou
        If set (e.g. 0.4), run NMS on that tile's predictions first — the
        sliding-window ``with merging`` mode. ``None`` keeps all detections.
    """
    images = {img["id"]: img for img in gt_coco["images"]}
    anns_by_image: dict[int, list] = {i: [] for i in images}
    for ann in gt_coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    rows = []
    for image_id, img in images.items():
        file_name = img["file_name"]
        stem = Path(file_name).stem
        width, height = int(img["width"]), int(img["height"])
        gt_anns = anns_by_image.get(image_id, [])
        preds = list(predictions_by_stem.get(stem, []))
        if merge_iou is not None and preds:
            preds = nms_detections(preds, iou_thresh=merge_iou)

        # Fixed-IoU P/R on non-crowd GT.
        gt_boxes = []
        for ann in gt_anns:
            if skip_crowd_as_gt and ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
        pred_boxes = []
        pred_scores = []
        for ann in preds:
            pred_boxes.append(detection_to_xyxy(ann))
            pred_scores.append(float(ann.get("score", 1.0)))
        pr = match_precision_recall(gt_boxes, pred_boxes, pred_scores, iou_thresh=match_iou)
        coco_metrics = coco_ap_ar_for_image(gt_anns, preds, width, height, iou_type=iou_type)

        parsed = parse_tile_stem(file_name)
        year = parsed[0] if parsed else None
        row_i = parsed[1] if parsed else None
        col_i = parsed[2] if parsed else None
        rows.append(
            {
                "file_name": file_name,
                "stem": stem,
                "year": year,
                "row": row_i,
                "col": col_i,
                "year_key": year_key(year, row_i, col_i) if parsed and year else None,
                "width": width,
                "height": height,
                "merge_iou": merge_iou,
                **{f"coco_{k}": v for k, v in coco_metrics.items()},
                **pr,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Heatmaps / GeoJSON
# ---------------------------------------------------------------------------


def tile_metric_grid(
    rows: Iterable[dict[str, Any]],
    metric: str,
    year: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Build a row×col grid of ``metric`` for one ortho year."""
    subset = [r for r in rows if r.get("year") == year and r.get("row") is not None]
    if not subset:
        return np.full((0, 0), np.nan), [], []
    rows_idx = sorted({int(r["row"]) for r in subset})
    cols_idx = sorted({int(r["col"]) for r in subset})
    grid = np.full((len(rows_idx), len(cols_idx)), np.nan, dtype=float)
    rmap = {r: i for i, r in enumerate(rows_idx)}
    cmap = {c: i for i, c in enumerate(cols_idx)}
    for r in subset:
        val = r.get(metric)
        if val is None or not _is_number(val):
            continue
        grid[rmap[int(r["row"])], cmap[int(r["col"])]] = float(val)
    return grid, rows_idx, cols_idx


def plot_tile_heatmaps(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...] = ("coco_AP50", "coco_AR100", "recall", "precision"),
    years: tuple[int, ...] = (24, 25),
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 100.0,
    title_prefix: str = "",
):
    """Matplotlib heatmaps: one figure per metric, panels per year."""
    import matplotlib.pyplot as plt

    figs = []
    for metric in metrics:
        # Precision/recall are 0–1; COCO metrics are 0–100.
        use_vmax = 1.0 if metric in {"precision", "recall", "f1"} else vmax
        use_vmin = 0.0 if metric in {"precision", "recall", "f1"} else vmin
        present_years = [y for y in years if any(r.get("year") == y for r in rows)]
        if not present_years:
            continue
        fig, axes = plt.subplots(
            1,
            len(present_years),
            figsize=(6 * len(present_years), 5),
            squeeze=False,
        )
        for ax, year in zip(axes[0], present_years):
            grid, row_ids, col_ids = tile_metric_grid(rows, metric, year)
            if grid.size == 0:
                ax.set_visible(False)
                continue
            im = ax.imshow(grid, cmap=cmap, vmin=use_vmin, vmax=use_vmax, aspect="equal")
            ax.set_title(f"{title_prefix}{metric} — year {year}")
            ax.set_xlabel("col")
            ax.set_ylabel("row")
            ax.set_xticks(range(len(col_ids)))
            ax.set_xticklabels(col_ids, rotation=90, fontsize=7)
            ax.set_yticks(range(len(row_ids)))
            ax.set_yticklabels(row_ids, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        figs.append((metric, fig))
    return figs


def rows_to_geojson(
    rows: list[dict[str, Any]],
    extents_geojson: Path | str,
    metrics: tuple[str, ...] = ("coco_AP50", "coco_AR100", "precision", "recall"),
) -> dict[str, Any]:
    """Join per-tile metrics onto ``tile_extents_*.geojson`` features by ``year_key``."""
    extents = json.loads(Path(extents_geojson).read_text())
    by_key = {r["year_key"]: r for r in rows if r.get("year_key")}
    features = []
    for feat in extents.get("features", []):
        props = dict(feat.get("properties") or {})
        key = props.get("year_key")
        row = by_key.get(key)
        if row:
            for m in metrics:
                if m in row and _is_number(row[m]):
                    props[m] = float(row[m])
            props["gt_count"] = row.get("gt_count")
            props["pred_count"] = row.get("pred_count")
        features.append({"type": "Feature", "properties": props, "geometry": feat.get("geometry")})
    return {
        "type": "FeatureCollection",
        "name": Path(extents_geojson).stem + "_metrics",
        "crs": extents.get("crs"),
        "features": features,
    }


def summarize_by_split_membership(
    rows: list[dict[str, Any]],
    split_cfg: dict[str, Any],
    metric: str = "coco_AP50",
    membership_split: str = "test",
) -> dict[str, float]:
    """Mean ``metric`` over tiles that belong to ``membership_split`` in a setup.

    Use the *same* per-tile scores (ideally from one model on a common tile pool)
    with different setup YAMLs to test whether a hold-out set is intrinsically
    easier (higher mean score) than another.
    """
    keys = tiles_in_split(split_cfg, membership_split)
    vals = [
        float(r[metric])
        for r in rows
        if r.get("year_key") in keys and metric in r and _is_number(r[metric])
    ]
    if not vals:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def plot_learning_curves(
    curves: dict[str, list[dict[str, Any]]],
    metric: str = "bbox/AP50",
):
    """Plot validation ``metric`` vs iteration for several runs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in curves.items():
        xs = [p["iteration"] for p in curve if metric in p]
        ys = [p[metric] for p in curve if metric in p]
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel(metric)
    ax.set_title(f"Validation {metric} vs iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig
