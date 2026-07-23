#!/usr/bin/env python3
"""Provenance records for COCO datasets and training runs.

Each dataset / tiling / training output dir gets a JSON sidecar that records
the tool, CLI flags, parents, split summaries, and (for training) metrics.

Files written
-------------
- ``dataset_provenance.json`` — COCO dirs from ``gpkg_to_coco``,
  ``augment_coco_dataset``, ``build_coco_rgb_dsm``, ``materialize_geo_split_coco``
- ``tiling_provenance.json`` — ``build_rgb_dsm_tiles`` output dirs
- ``training_run_provenance.json`` — ``train_boulder_local`` output dirs

Example::

    from run_provenance import write_dataset_provenance, load_provenance

    write_dataset_provenance(
        output_dir,
        tool=\"gpkg_to_coco.py\",
        flags={\"min_area_m2\": 1.0, \"boulder_only\": True, ...},
        splits_summary=summary,
    )
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

DATASET_PROVENANCE = "dataset_provenance.json"
TILING_PROVENANCE = "tiling_provenance.json"
TRAINING_PROVENANCE = "training_run_provenance.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def argv_command() -> list[str]:
    return [str(a) for a in sys.argv]


def load_provenance(path: Path | str) -> dict[str, Any] | None:
    path = Path(path)
    if path.is_dir():
        for name in (DATASET_PROVENANCE, TRAINING_PROVENANCE, TILING_PROVENANCE):
            cand = path / name
            if cand.is_file():
                return json.loads(cand.read_text())
        return None
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def parent_ref(path: Path | str | None) -> dict[str, Any] | None:
    """Pointer to a parent dataset/tiling dir and its provenance file if present."""
    if path is None:
        return None
    path = Path(path)
    ref: dict[str, Any] = {"path": str(path.resolve() if path.exists() else path)}
    prov = load_provenance(path)
    if prov is not None:
        # Prefer the actual sidecar name that loaded.
        for name in (DATASET_PROVENANCE, TRAINING_PROVENANCE, TILING_PROVENANCE):
            if (path / name).is_file():
                ref["provenance_file"] = name
                break
        ref["parent_tool"] = prov.get("tool")
        ref["parent_flags"] = prov.get("flags")
        ref["parent_kind"] = prov.get("kind")
    return ref


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    return path


def base_record(
    *,
    kind: str,
    tool: str,
    output_dir: Path,
    flags: dict[str, Any] | None = None,
    parents: list[dict[str, Any] | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_parents = [p for p in (parents or []) if p]
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "created_utc": utc_now(),
        "tool": tool,
        "command": argv_command(),
        "cwd": str(Path.cwd()),
        "output_dir": str(output_dir),
        "flags": _jsonable(flags or {}),
        "parents": clean_parents,
    }
    if extra:
        record.update(_jsonable(extra))
    return record


def write_dataset_provenance(
    output_dir: Path | str,
    *,
    tool: str,
    flags: dict[str, Any],
    splits_summary: Any = None,
    parents: list[Path | str | None] | None = None,
    notes: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``dataset_provenance.json`` into a COCO dataset directory."""
    output_dir = Path(output_dir)
    parent_refs = [parent_ref(p) for p in (parents or [])]
    payload = base_record(
        kind="coco_dataset",
        tool=tool,
        output_dir=output_dir,
        flags=flags,
        parents=parent_refs,
        extra=extra,
    )
    if splits_summary is not None:
        payload["splits"] = _jsonable(splits_summary)
    if notes:
        payload["notes"] = notes
    path = write_json(output_dir / DATASET_PROVENANCE, payload)
    print(f"Wrote {path}")
    return path


def write_tiling_provenance(
    output_dir: Path | str,
    *,
    tool: str,
    flags: dict[str, Any],
    tiles_summary: Any = None,
    parents: list[Path | str | None] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    payload = base_record(
        kind="tiling_rgb_dsm",
        tool=tool,
        output_dir=output_dir,
        flags=flags,
        parents=[parent_ref(p) for p in (parents or [])],
        extra=extra,
    )
    if tiles_summary is not None:
        # Keep the sidecar small — store counts, not every tile row.
        if isinstance(tiles_summary, list):
            payload["tile_count"] = len(tiles_summary)
            payload["tiles_sample"] = _jsonable(tiles_summary[:5])
        else:
            payload["tiles"] = _jsonable(tiles_summary)
    path = write_json(output_dir / TILING_PROVENANCE, payload)
    print(f"Wrote {path}")
    return path


def write_training_provenance(
    output_dir: Path | str,
    *,
    tool: str,
    flags: dict[str, Any],
    dataset_dir: Path | str | None = None,
    metrics_valid: dict[str, Any] | None = None,
    dataset_summary: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``training_run_provenance.json`` (flags + optional final metrics)."""
    output_dir = Path(output_dir)
    parents = [parent_ref(dataset_dir)] if dataset_dir is not None else []
    payload = base_record(
        kind="training_run",
        tool=tool,
        output_dir=output_dir,
        flags=flags,
        parents=parents,
        extra=extra,
    )
    if dataset_dir is not None:
        payload["dataset_dir"] = str(dataset_dir)
    if dataset_summary is not None:
        payload["dataset_summary"] = _jsonable(dataset_summary)
    if metrics_valid is not None:
        payload["metrics_valid"] = _jsonable(metrics_valid)
    path = write_json(output_dir / TRAINING_PROVENANCE, payload)
    print(f"Wrote {path}")
    return path


def update_training_metrics(
    output_dir: Path | str,
    metrics_valid: dict[str, Any],
) -> Path | None:
    """Attach final ``metrics_valid`` to an existing training provenance file."""
    output_dir = Path(output_dir)
    path = output_dir / TRAINING_PROVENANCE
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    data["metrics_valid"] = _jsonable(metrics_valid)
    data["metrics_valid_updated_utc"] = utc_now()
    write_json(path, data)
    print(f"Updated metrics in {path}")
    return path


def format_provenance(data: dict[str, Any]) -> str:
    lines = [
        f"kind:     {data.get('kind')}",
        f"tool:     {data.get('tool')}",
        f"created:  {data.get('created_utc')}",
        f"output:   {data.get('output_dir')}",
    ]
    flags = data.get("flags") or {}
    if flags:
        lines.append("flags:")
        for k in sorted(flags):
            lines.append(f"  {k}: {flags[k]}")
    parents = data.get("parents") or []
    if parents:
        lines.append("parents:")
        for p in parents:
            lines.append(f"  - {p.get('path')}")
            if p.get("parent_flags"):
                lines.append(f"    parent_flags: {json.dumps(p['parent_flags'])}")
    if data.get("metrics_valid"):
        lines.append("metrics_valid:")
        lines.append(json.dumps(data["metrics_valid"], indent=2))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show dataset / training provenance sidecars."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="A dataset/training dir, or a *_provenance.json file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of a short summary.",
    )
    args = parser.parse_args()
    data = load_provenance(args.path)
    if data is None:
        raise SystemExit(f"No provenance found at {args.path}")
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(format_provenance(data))


if __name__ == "__main__":
    main()
