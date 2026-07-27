#!/usr/bin/env python3
# RUNS ON: laptop (analysis only)
"""Unzip a geo_split transfer package and build the master analysis table.

Surfaces evidence for the 512-vs-2000 confound question:

- Does the large-object recall spike survive once per-bucket *n* is shown?
- Does empty-chip fraction correlate with the 512-vs-2000 performance gap?
- Do ground-area vs pixel-area size buckets diverge for the same instances?

Outputs CSV + markdown tables (no dashboard).

Example::

    python BoulderCalculator/scripts/analyze_geo_split_transfer.py \\
      --package segmentation/transfer_packages/geo_split_transfer_YYYYMMDD_HHMMSS.zip \\
      --output-dir segmentation/geo_split_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Any, *keys, default=float("nan")):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    if cur is None:
        return default
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _extract_package(package: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    if package.is_dir():
        return package
    if not package.is_file():
        raise SystemExit(f"Package not found: {package}")
    extract_root = work_dir / "extracted"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)
    with zipfile.ZipFile(package, "r") as zf:
        zf.extractall(extract_root)
    # Prefer the dated folder inside the zip.
    children = [p for p in extract_root.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return extract_root


def _index_diagnostics(diag_dir: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    if not diag_dir.is_dir():
        return out
    for path in diag_dir.glob("diag_*.json"):
        data = _load(path)
        key = (
            str(data.get("setup")),
            str(data.get("modality")),
            str(data.get("tiling_regime")),
        )
        out[key] = data
    return out


def _index_eval(eval_dir: Path) -> dict[tuple[str, str, str], dict]:
    """Map (setup, modality, tiling) → best available run_eval_summary.json."""
    out: dict[tuple[str, str, str], dict] = {}
    if not eval_dir.is_dir():
        return out
    for summary in eval_dir.rglob("run_eval_summary.json"):
        data = _load(summary)
        key = (
            str(data.get("setup")),
            str(data.get("modality")),
            str(data.get("tiling_regime")),
        )
        # Prefer model_best over model_final when both exist.
        stem = str(data.get("checkpoint_stem") or "")
        prev = out.get(key)
        if prev is None:
            out[key] = data
        elif stem == "model_best":
            out[key] = data
        elif prev.get("checkpoint_stem") != "model_best" and stem == "model_final":
            out[key] = data
    return out


def _index_training_metrics(runs_dir: Path) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    if not runs_dir.is_dir():
        return out
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        parts = run_dir.name.split("__")
        if len(parts) != 3:
            continue
        setup, modality, tiling = parts
        entry: dict[str, Any] = {
            "setup": setup,
            "modality": modality,
            "tiling_regime": tiling,
            "run_dir": str(run_dir),
        }
        for name, key in (
            ("metrics_valid.json", "metrics_valid"),
            ("metrics_test.json", "metrics_test"),
            ("metrics_valid_at_best.json", "metrics_valid_at_best"),
            ("training_run_provenance.json", "provenance"),
        ):
            path = run_dir / name
            if path.is_file():
                entry[key] = _load(path)
        out[(setup, modality, tiling)] = entry
    return out


def _metric_block(payload: Any) -> dict:
    """Normalize metrics_valid / metrics_test shapes to flat bbox/segm dicts."""
    if not isinstance(payload, dict):
        return {}
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return payload["metrics"]
    return payload


def _build_rows(
    diagnostics: dict,
    evals: dict,
    trains: dict,
) -> list[dict[str, Any]]:
    keys = set(diagnostics) | set(evals) | set(trains)
    rows: list[dict[str, Any]] = []
    for key in sorted(keys):
        setup, modality, tiling = key
        diag = diagnostics.get(key) or {}
        ev = evals.get(key) or {}
        tr = trains.get(key) or {}
        overall = (ev.get("overall") or {})
        ground = overall.get("overall_ground_m2") or {}
        pixel = overall.get("overall_pixel_native") or {}
        test_m = _metric_block(tr.get("metrics_test"))
        valid_m = _metric_block(tr.get("metrics_valid_at_best")) or _metric_block(
            tr.get("metrics_valid")
        )
        test_bbox = test_m.get("bbox") or {}
        valid_bbox = valid_m.get("bbox") or {}
        buckets_g = diag.get("size_buckets_ground_m2") or {}
        buckets_p = diag.get("size_buckets_pixel_resized") or {}

        row = {
            "setup": setup,
            "modality": modality,
            "tiling_regime": tiling,
            "empty_chip_fraction": diag.get("empty_chip_fraction"),
            "boundary_truncation_fraction": diag.get("boundary_truncation_fraction"),
            "effective_network_gsd_m": diag.get("effective_network_gsd_m"),
            "native_gsd_m": diag.get("native_gsd_m"),
            "n_instances_trainable": diag.get("n_instances_trainable"),
            "n_ground_small": buckets_g.get("small"),
            "n_ground_medium": buckets_g.get("medium"),
            "n_ground_large": buckets_g.get("large"),
            "n_pixel_small": buckets_p.get("small"),
            "n_pixel_medium": buckets_p.get("medium"),
            "n_pixel_large": buckets_p.get("large"),
            "bucket_flip_fraction_resized_vs_ground": diag.get(
                "bucket_flip_fraction_resized_vs_ground"
            ),
            # Eval-per-tile / overall (ground primary)
            "eval_AP": ground.get("AP"),
            "eval_AP50": ground.get("AP50"),
            "eval_AR100": ground.get("AR100"),
            "eval_ARs": ground.get("ARs"),
            "eval_ARm": ground.get("ARm"),
            "eval_ARl": ground.get("ARl"),
            "eval_pixel_AP": pixel.get("AP"),
            "eval_pixel_AP50": pixel.get("AP50"),
            "eval_pixel_AR100": pixel.get("AR100"),
            "eval_pixel_ARs": pixel.get("ARs"),
            "eval_pixel_ARm": pixel.get("ARm"),
            "eval_pixel_ARl": pixel.get("ARl"),
            # Train-time holdout files
            "test_bbox_AP": test_bbox.get("AP"),
            "test_bbox_AP50": test_bbox.get("AP50"),
            "test_bbox_AR100": test_bbox.get("AR100"),
            "test_bbox_ARl": test_bbox.get("ARl"),
            "valid_bbox_AP": valid_bbox.get("AP"),
            "valid_bbox_AP50": valid_bbox.get("AP50"),
            "valid_bbox_AR100": valid_bbox.get("AR100"),
            "valid_bbox_ARl": valid_bbox.get("ARl"),
        }
        # Val − test gap (positive ⇒ val looks better than test).
        for metric in ("AP", "AP50", "AR100", "ARl"):
            v = row.get(f"valid_bbox_{metric}")
            t = row.get(f"test_bbox_{metric}")
            try:
                row[f"val_minus_test_{metric}"] = float(v) - float(t)
            except (TypeError, ValueError):
                row[f"val_minus_test_{metric}"] = float("nan")
        rows.append(row)
    return rows


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n < 3:
        return float("nan")
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    denx = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    deny = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def _flag_large_recall(rows: list[dict]) -> list[str]:
    lines = [
        "## Large-object recall vs bucket *n*",
        "",
        "Flag ARl as likely noise when ground-large *n* is tiny (<30).",
        "",
        "| setup | modality | tiling | ARl (ground eval) | n_ground_large | flag |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in rows:
        n = r.get("n_ground_large")
        arl = r.get("eval_ARl")
        try:
            n_i = int(n) if n is not None and str(n) != "nan" else -1
        except (TypeError, ValueError):
            n_i = -1
        flag = ""
        if n_i >= 0 and n_i < 30 and arl is not None:
            flag = "LIKELY_NOISE (small n)"
        arl_s = f"{float(arl):.1f}" if arl is not None and math.isfinite(float(arl)) else "nan"
        lines.append(
            f"| {r['setup']} | {r['modality']} | {r['tiling_regime']} | "
            f"{arl_s} | {n_i if n_i >= 0 else 'nan'} | {flag} |"
        )
    return lines


def _flag_empty_chip_vs_gap(rows: list[dict]) -> list[str]:
    lines = [
        "",
        "## Empty-chip fraction vs 512–2000 performance gap",
        "",
    ]
    # Pair same setup+modality across tilings.
    by_sm: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        by_sm.setdefault((r["setup"], r["modality"]), {})[r["tiling_regime"]] = r

    gap_rows = []
    for (setup, modality), tilings in sorted(by_sm.items()):
        if "512" not in tilings or "2000" not in tilings:
            continue
        a = tilings["512"]
        b = tilings["2000"]
        try:
            gap_ap50 = float(a.get("eval_AP50")) - float(b.get("eval_AP50"))
        except (TypeError, ValueError):
            gap_ap50 = float("nan")
        try:
            empty_512 = float(a.get("empty_chip_fraction"))
        except (TypeError, ValueError):
            empty_512 = float("nan")
        try:
            empty_2000 = float(b.get("empty_chip_fraction"))
        except (TypeError, ValueError):
            empty_2000 = float("nan")
        gap_rows.append(
            {
                "setup": setup,
                "modality": modality,
                "empty_512": empty_512,
                "empty_2000": empty_2000,
                "empty_delta": empty_512 - empty_2000
                if math.isfinite(empty_512) and math.isfinite(empty_2000)
                else float("nan"),
                "ap50_512_minus_2000": gap_ap50,
            }
        )

    if not gap_rows:
        lines.append(
            "_No paired 512+2000 rows found. Re-run diagnostics/eval with "
            "`--include-parent-2000` on Windows, then re-package._"
        )
        return lines

    lines.extend(
        [
            "| setup | modality | empty_512 | empty_2000 | Δempty | AP50_512−2000 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for g in gap_rows:
        lines.append(
            f"| {g['setup']} | {g['modality']} | "
            f"{g['empty_512']:.3f} | {g['empty_2000']:.3f} | "
            f"{g['empty_delta']:.3f} | {g['ap50_512_minus_2000']:.2f} |"
        )
    corr = _pearson(
        [g["empty_delta"] for g in gap_rows],
        [g["ap50_512_minus_2000"] for g in gap_rows],
    )
    lines.append("")
    lines.append(
        f"Pearson(Δempty_chip, AP50_512−2000) = **{corr:.3f}** "
        f"(n={len(gap_rows)}; need ≥3 pairs)."
    )
    return lines


def _flag_bucket_flip(rows: list[dict]) -> list[str]:
    lines = [
        "",
        "## Ground-area vs pixel-area bucket assignment divergence",
        "",
        "| setup | modality | tiling | flip_fraction | n_inst |",
        "|---|---|---|---:|---:|",
    ]
    high = 0
    for r in rows:
        flip = r.get("bucket_flip_fraction_resized_vs_ground")
        n = r.get("n_instances_trainable")
        try:
            flip_f = float(flip)
        except (TypeError, ValueError):
            flip_f = float("nan")
        if math.isfinite(flip_f) and flip_f > 0.1:
            high += 1
        flip_s = f"{flip_f:.3f}" if math.isfinite(flip_f) else "nan"
        lines.append(
            f"| {r['setup']} | {r['modality']} | {r['tiling_regime']} | "
            f"{flip_s} | {n} |"
        )
    lines.append("")
    lines.append(
        f"{high}/{len(rows)} runs have >10% instances changing S/M/L bucket "
        "between resized-pixel and ground-m² definitions — a plausible "
        "explanation for inconsistent size-bucket results across differently "
        "resized pipelines."
    )
    return lines


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        required=True,
        help="Path to .zip from package_geo_split_transfer.py, or an unzipped folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmentation/geo_split_analysis"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Scratch dir for unzip (default: <output-dir>/_work).",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    work = args.work_dir or (out / "_work")
    root = _extract_package(args.package, work)
    print(f"Package root: {root}")

    diagnostics = _index_diagnostics(root / "diagnostics")
    evals = _index_eval(root / "eval_geo512_all")
    trains = _index_training_metrics(root / "training_runs")
    rows = _build_rows(diagnostics, evals, trains)

    master_csv = out / "master_table.csv"
    _write_csv(master_csv, rows)
    print(f"Wrote {master_csv} ({len(rows)} rows)")

    md_lines = [
        "# Geo-split tiling confound analysis",
        "",
        f"Package: `{args.package}`",
        f"Rows: {len(rows)}",
        "",
        "## Master table (headline columns)",
        "",
        "| setup | modality | tiling | empty | trunc | eff_GSD | AP50_g | AR100_g | ARl_g | n_large_g | val−test AP50 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(k, nd=3):
            v = r.get(k)
            try:
                return f"{float(v):.{nd}f}"
            except (TypeError, ValueError):
                return "nan"

        md_lines.append(
            f"| {r['setup']} | {r['modality']} | {r['tiling_regime']} | "
            f"{fmt('empty_chip_fraction')} | {fmt('boundary_truncation_fraction')} | "
            f"{fmt('effective_network_gsd_m', 5)} | {fmt('eval_AP50', 2)} | "
            f"{fmt('eval_AR100', 2)} | {fmt('eval_ARl', 2)} | "
            f"{r.get('n_ground_large')} | {fmt('val_minus_test_AP50', 2)} |"
        )

    md_lines.extend(_flag_large_recall(rows))
    md_lines.extend(_flag_empty_chip_vs_gap(rows))
    md_lines.extend(_flag_bucket_flip(rows))
    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Ground-m² size buckets are primary; pixel-area columns are for comparison.",
            "- `metrics_valid.json` from training is final-weights eval; prefer "
            "`metrics_valid_at_best.json` for the val−test gap when present.",
            "- Periodic validation curves in `metrics.json` are unchanged and must "
            "not be mixed with `metrics_test.json`.",
            "",
        ]
    )
    md_path = out / "analysis_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")

    index = {
        "n_rows": len(rows),
        "n_diagnostics": len(diagnostics),
        "n_evals": len(evals),
        "n_training_metric_dirs": len(trains),
        "package": str(args.package),
        "package_root": str(root),
    }
    (out / "analysis_index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
