#!/usr/bin/env python3
"""Shared reuse helpers for pipeline scripts (skip existing outputs unless --force).

Dataset / tile builds are expensive on Windows guests. Scripts default to keeping
outputs that already look complete; pass ``--force`` to rebuild.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

COCO_ANN_FILES: tuple[str, ...] = (
    "train_annotations.json",
    "validation_annotations.json",
    "testing_annotations.json",
)


def add_force_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even when they already exist on disk.",
    )


def coco_dataset_ready(path: Path | str) -> bool:
    """True when all three COCO annotation JSONs are present."""
    path = Path(path)
    return all((path / name).is_file() for name in COCO_ANN_FILES)


def _norm(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _norm(v) for k, v in value.items()}
    return value


def flags_match(existing: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    """Compare a subset of provenance flags (missing keys in existing → mismatch)."""
    existing = existing or {}
    for key, want in expected.items():
        if key not in existing:
            return False
        if _norm(existing[key]) != _norm(want):
            return False
    return True


def load_flags(path: Path | str) -> dict[str, Any] | None:
    try:
        from run_provenance import load_provenance
    except ImportError:
        return None
    prov = load_provenance(path)
    if not prov:
        return None
    flags = prov.get("flags")
    return flags if isinstance(flags, dict) else None


def should_skip_coco_dataset(
    output_dir: Path | str,
    *,
    force: bool,
    label: str,
    expected_flags: dict[str, Any] | None = None,
) -> bool:
    """Return True if the caller should skip rebuilding this COCO directory."""
    output_dir = Path(output_dir)
    if force:
        return False
    if not coco_dataset_ready(output_dir):
        return False
    if expected_flags:
        existing = load_flags(output_dir)
        if existing is not None and not flags_match(existing, expected_flags):
            print(
                f"[rebuild] {label}: existing {output_dir} provenance flags differ "
                f"(pass --force to rebuild without checking).",
                flush=True,
            )
            return False
    print(
        f"[skip] {label}: {output_dir} already complete "
        f"(pass --force to rebuild).",
        flush=True,
    )
    return True


def should_skip_file(path: Path | str, *, force: bool) -> bool:
    if force:
        return False
    return Path(path).is_file()
