"""Load project path configuration (paths.local.yaml / paths.example.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def code_root() -> Path:
    """``BoulderCalc/`` (parent of ``Matching/``)."""
    return Path(__file__).resolve().parents[2]


def default_project_root() -> Path:
    """Parent of the code directory (``reu/`` layout)."""
    return code_root().parent


def _resolve_config_path(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None

    code = code_root()
    for name in ("paths.local.yaml", "paths.example.yaml"):
        candidate = code / name
        if candidate.exists():
            return candidate
    return None


def load_paths(config_path: Path | None = None) -> dict[str, Any]:
    """
    Return a dict of resolved paths.

    Relative annotation / segmentation paths are resolved against ``project_root``.
    Missing optional keys fall back to conventional relative defaults.
    """
    root = default_project_root()
    cfg: dict[str, Any] = {}

    path = _resolve_config_path(config_path)
    if path is not None:
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load paths.local.yaml. "
                "Install with: pip install pyyaml"
            )
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Path config must be a mapping: {path}")
        cfg = loaded
        if cfg.get("project_root"):
            root = Path(cfg["project_root"]).expanduser().resolve()

    code_dir_name = cfg.get("code_dir", code_root().name)
    seg_name = cfg.get("segmentation_dir", "segmentation")

    def _abs(key: str, default: Path | None = None) -> Path | None:
        raw = cfg.get(key)
        if raw is None or raw == "":
            return default
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = root / p
        return p.resolve()

    ann24_default = root / seg_name / "annotations" / "july14_24.gpkg"
    ann25_default = root / seg_name / "annotations" / "july14_25.gpkg"
    ortho24_default = root / "2024" / "Sites1and2_2024_Orthomosaic.tif"
    dsm24_default = root / "2024" / "Sites1and2_2024_DSM_30mm.tif"
    ortho25_default = root / "2025" / "25IniSouthOrt.tif"
    dsm25_default = root / "2025" / "25IniSouthDSM.tif"

    return {
        "config_path": path,
        "project_root": root,
        "code_dir": root / code_dir_name,
        "code_dir_name": code_dir_name,
        "segmentation_dir": root / seg_name,
        "ortho_24": _abs("ortho_24", ortho24_default),
        "dsm_24": _abs("dsm_24", dsm24_default),
        "ortho_25": _abs("ortho_25", ortho25_default),
        "dsm_25": _abs("dsm_25", dsm25_default),
        "annotations_24": _abs("annotations_24", ann24_default),
        "annotations_25": _abs("annotations_25", ann25_default),
        "matching_dir": root / code_dir_name / "Matching",
    }
