"""Run matching (and optional tile inference) for multiple models into separate outdirs.

Config YAML example (``match_models.yaml``)::

    project_root: ../..
    search_radius: 15.0
    min_score: 0.55
    candidate_radius: 25.0
    score_thresh: 0.4
    device: cpu
    models:
      - name: rgb_dsm_4000
        model: ../../segmentation/training_run_rgb_dsm_4000/model_final.pth
        outdir: ../../segmentation/match_runs/rgb_dsm_4000
      - name: geo_baseline
        model: ../../segmentation/training_run_geo_baseline/model_final.pth
        outdir: ../../segmentation/match_runs/geo_baseline
        # optional: skip inference and rematch existing polygons
        # before_polygons: ...
        # after_polygons: ...

Usage:
  python -m matching.run_match_models --config match_models.yaml
  python -m matching.run_match_models --config match_models.yaml --rematch-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_yaml(path: Path) -> dict:
    text = path.read_text()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        # Minimal fallback: JSON subset
        return json.loads(text)


def _resolve(base: Path, p: str | Path | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def run_one_inference_match(
    python: Path,
    match_dir: Path,
    model: Path,
    outdir: Path,
    project_root: Path,
    search_radius: float,
    min_score: float,
    candidate_radius: float | None,
    score_thresh: float,
    device: str,
    extra: list[str] | None = None,
) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python),
        "-m",
        "matching.run_inference_match",
        "--model",
        str(model),
        "--outdir",
        str(outdir),
        "--project-root",
        str(project_root),
        "--search-radius",
        str(search_radius),
        "--min-score",
        str(min_score),
        "--score-thresh",
        str(score_thresh),
        "--device",
        device,
        "--no-screenshots",
    ]
    if candidate_radius is not None:
        cmd.extend(["--candidate-radius", str(candidate_radius)])
    if extra:
        cmd.extend(extra)
    print("\n→", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(match_dir))


def run_one_rematch(
    python: Path,
    match_dir: Path,
    before: Path,
    after: Path,
    outdir: Path,
    search_radius: float,
    min_score: float,
    candidate_radius: float,
    before_dsm: Path | None = None,
    after_dsm: Path | None = None,
) -> int:
    cmd = [
        str(python),
        "-m",
        "matching.build_gt_dataset",
        "--before-polygons",
        str(before),
        "--after-polygons",
        str(after),
        "--outdir",
        str(outdir),
        "--search-radius",
        str(search_radius),
        "--min-score",
        str(min_score),
        "--candidate-radius",
        str(candidate_radius),
    ]
    if before_dsm:
        cmd.extend(["--before-dsm", str(before_dsm), "--compute-volume"])
    if after_dsm:
        cmd.extend(["--after-dsm", str(after_dsm)])
    print("\n→", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(match_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--rematch-only",
        action="store_true",
        help="Skip Detectron2; rematch from each model's existing prediction polygons or shared polygons",
    )
    parser.add_argument("--python", type=Path, default=None)
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    cfg = _load_yaml(cfg_path)
    cfg_dir = cfg_path.parent

    match_dir = Path(__file__).resolve().parents[1]
    project_root = _resolve(cfg_dir, cfg.get("project_root")) or match_dir.parents[1]
    python = args.python or (project_root / ".venv_boulder" / "bin" / "python")
    if not Path(python).exists():
        python = Path(sys.executable)

    search_radius = float(cfg.get("search_radius", 15.0))
    min_score = float(cfg.get("min_score", 0.55))
    candidate_radius = float(cfg.get("candidate_radius", 25.0))
    score_thresh = float(cfg.get("score_thresh", 0.4))
    device = str(cfg.get("device", "cpu"))

    models = cfg.get("models") or []
    if not models:
        raise SystemExit("Config has no models: list")

    report = {"started_at": _utc_now(), "config": str(cfg_path), "runs": []}

    for entry in models:
        name = entry["name"]
        outdir = _resolve(cfg_dir, entry["outdir"])
        assert outdir is not None
        print(f"\n======== {name} → {outdir} ========")

        before_poly = _resolve(cfg_dir, entry.get("before_polygons"))
        after_poly = _resolve(cfg_dir, entry.get("after_polygons"))
        # Shared rematch polygons from top-level config
        if before_poly is None:
            before_poly = _resolve(cfg_dir, cfg.get("before_polygons"))
        if after_poly is None:
            after_poly = _resolve(cfg_dir, cfg.get("after_polygons"))

        rc = 0
        if args.rematch_only or (before_poly and after_poly and entry.get("rematch_only")):
            if not before_poly or not after_poly:
                # Fall back to existing predictions under outdir
                before_poly = outdir / "predictions" / "before_inferred_boulders.geojson"
                after_poly = outdir / "predictions" / "after_inferred_boulders.geojson"
            if not before_poly.exists() or not after_poly.exists():
                print(f"Skip {name}: missing polygons for rematch")
                rc = 2
            else:
                rc = run_one_rematch(
                    python,
                    match_dir,
                    before_poly,
                    after_poly,
                    outdir,
                    search_radius,
                    min_score,
                    candidate_radius,
                    before_dsm=_resolve(cfg_dir, entry.get("before_dsm") or cfg.get("before_dsm")),
                    after_dsm=_resolve(cfg_dir, entry.get("after_dsm") or cfg.get("after_dsm")),
                )
        else:
            model = _resolve(cfg_dir, entry["model"])
            if model is None or not model.exists():
                print(f"Skip {name}: model not found ({model})")
                rc = 2
            else:
                rc = run_one_inference_match(
                    python,
                    match_dir,
                    model,
                    outdir,
                    project_root,
                    search_radius,
                    min_score,
                    candidate_radius,
                    score_thresh,
                    device,
                )

        report["runs"].append({"name": name, "outdir": str(outdir), "returncode": rc})

    report["finished_at"] = _utc_now()
    report_path = cfg_dir / "match_models_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")
    failed = [r for r in report["runs"] if r["returncode"] != 0]
    if failed:
        raise SystemExit(f"{len(failed)} run(s) failed")


if __name__ == "__main__":
    main()
