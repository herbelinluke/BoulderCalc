#!/usr/bin/env python3
# RUNS ON: Windows (after Deliverables 1–3 have produced outputs)
"""Package geo_split diagnostic / eval / test-metric outputs for laptop transfer.

Collects lightweight artifacts only (JSON/CSV/PNG/config sidecars) — NOT
checkpoints or image tiles — into a dated folder and zips it.

Included when present
---------------------
- ``segmentation/geo_split_diagnostics/``
- ``segmentation/eval_geo512_all/`` (or ``--eval-dir``)
- Per-run: ``metrics_valid.json``, ``metrics_test.json``,
  ``metrics_valid_at_best.json``, ``metrics.json``,
  ``training_run_provenance.json``, ``config.yaml`` (if Detectron2 wrote it)
- Dataset ``dataset_provenance.json`` for matched COCO dirs

Example::

    python BoulderCalculator/scripts/package_geo_split_transfer.py \\
      --segmentation-dir segmentation \\
      --output-dir segmentation/transfer_packages
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_split_run_index import (  # noqa: E402
    discover_geo512_runs,
    discover_parent_geo_runs,
    resolve_dataset_dir,
)

LIGHTWEIGHT_RUN_FILES = (
    "metrics_valid.json",
    "metrics_test.json",
    "metrics_valid_at_best.json",
    "metrics.json",
    "training_run_provenance.json",
    "config.yaml",
    "last_checkpoint",
)


def _copy_file(src: Path, dst: Path, manifest: list[dict]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "src": str(src),
            "dst": str(dst),
            "bytes": src.stat().st_size,
        }
    )


def _copy_tree_filtered(
    src: Path,
    dst: Path,
    manifest: list[dict],
    *,
    suffixes: tuple[str, ...] = (".json", ".csv", ".md", ".png", ".txt", ".yaml", ".yml"),
    max_file_mb: float = 50.0,
) -> None:
    if not src.is_dir():
        return
    max_bytes = int(max_file_mb * 1024 * 1024)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if path.stat().st_size > max_bytes:
            continue
        rel = path.relative_to(src)
        _copy_file(path, dst / rel, manifest)


def _package_run(
    run,
    *,
    seg: Path,
    dest_runs: Path,
    manifest: list[dict],
    include_parent: bool,
) -> None:
    if run.tiling_regime == "2000" and not include_parent:
        return
    key = f"{run.setup}__{run.modality}__{run.tiling_regime}"
    out = dest_runs / key
    for name in LIGHTWEIGHT_RUN_FILES:
        src = run.run_dir / name
        if src.is_file():
            _copy_file(src, out / name, manifest)
    # Small eval sidecars only (skip prediction dumps).
    for sub in ("eval", "eval_test", "eval_valid_at_best"):
        src = run.run_dir / sub
        if src.is_dir():
            for path in src.rglob("*.json"):
                if path.stat().st_size > 5_000_000:
                    continue
                _copy_file(path, out / sub / path.relative_to(src), manifest)

    dataset = resolve_dataset_dir(run, seg)
    if dataset is not None:
        prov = dataset / "dataset_provenance.json"
        if prov.is_file():
            _copy_file(prov, out / "dataset_provenance.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-dir", type=Path, default=Path("segmentation"))
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help="Default: <seg>/geo_split_diagnostics",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Default: <seg>/eval_geo512_all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("segmentation/transfer_packages"),
    )
    parser.add_argument(
        "--include-parent-2000",
        action="store_true",
        help="Include parent training_run_geo_* lightweight artifacts.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Leave the dated folder unzipped.",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="Folder stamp (default: UTC YYYYMMDD_HHMMSS).",
    )
    args = parser.parse_args()

    seg = args.segmentation_dir
    diagnostics = args.diagnostics_dir or (seg / "geo_split_diagnostics")
    eval_dir = args.eval_dir or (seg / "eval_geo512_all")
    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    package_name = f"geo_split_transfer_{stamp}"
    package_root = args.output_dir / package_name
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)

    manifest: list[dict] = []

    if diagnostics.is_dir():
        _copy_tree_filtered(diagnostics, package_root / "diagnostics", manifest)
    else:
        print(f"[warn] diagnostics dir missing: {diagnostics}")

    if eval_dir.is_dir():
        _copy_tree_filtered(eval_dir, package_root / "eval_geo512_all", manifest)
    else:
        print(f"[warn] eval dir missing: {eval_dir}")

    runs = discover_geo512_runs(seg)
    if args.include_parent_2000:
        runs.extend(discover_parent_geo_runs(seg))
    dest_runs = package_root / "training_runs"
    for run in runs:
        _package_run(
            run,
            seg=seg,
            dest_runs=dest_runs,
            manifest=manifest,
            include_parent=args.include_parent_2000,
        )

    meta = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "package_name": package_name,
        "segmentation_dir": str(seg),
        "diagnostics_dir": str(diagnostics),
        "eval_dir": str(eval_dir),
        "n_files": len(manifest),
        "total_bytes": sum(m["bytes"] for m in manifest),
        "n_runs_packaged": len({Path(m["dst"]).parts[-2] for m in manifest if "training_runs" in Path(m["dst"]).parts}),
        "note": "Checkpoints and image tiles intentionally excluded.",
    }
    (package_root / "TRANSFER_MANIFEST.json").write_text(
        json.dumps({"meta": meta, "files": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    (package_root / "README.txt").write_text(
        "\n".join(
            [
                "geo_split transfer package",
                "==========================",
                f"Created: {meta['created_utc']}",
                f"Files:   {meta['n_files']}",
                f"Bytes:   {meta['total_bytes']}",
                "",
                "Contents:",
                "  diagnostics/          — Deliverable 1 JSONs + CSV",
                "  eval_geo512_all/      — Deliverable 2 per-run eval",
                "  training_runs/        — metrics_*.json + provenance (no .pth)",
                "",
                "On the laptop:",
                "  python BoulderCalculator/scripts/analyze_geo_split_transfer.py \\",
                f"    --package {package_name}.zip",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Packaged {meta['n_files']} files ({meta['total_bytes']/1e6:.2f} MB) → {package_root}")

    if args.no_zip:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / f"{package_name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in package_root.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(package_root.parent)))
    print(f"Wrote {zip_path} ({zip_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
