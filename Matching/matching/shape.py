"""Polygon contour signatures and shape similarity for boulder matching."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


DEFAULT_N_POINTS = 64
# Chamfer distances on unit-scale contours; beyond this → score ~0.
CHAMFER_SCALE = 0.25
HAUSDORFF_SCALE = 0.40


def _exterior_coords(geom: BaseGeometry) -> np.ndarray:
    if geom is None or geom.is_empty:
        return np.zeros((0, 2), dtype=float)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if not isinstance(geom, Polygon):
        try:
            geom = geom.convex_hull
        except Exception:
            return np.zeros((0, 2), dtype=float)
    coords = np.asarray(geom.exterior.coords, dtype=float)
    if len(coords) < 2:
        return coords
    # Drop duplicate closing vertex for length calc; resampling re-closes.
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def resample_exterior(geom: BaseGeometry, n: int = DEFAULT_N_POINTS) -> np.ndarray:
    """Evenly resample the exterior ring to ``n`` points (closed ring not repeated)."""
    coords = _exterior_coords(geom)
    if len(coords) == 0:
        return np.zeros((n, 2), dtype=float)
    if len(coords) == 1:
        return np.repeat(coords, n, axis=0)

    seg = np.diff(coords, axis=0, append=coords[:1])
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 1e-12:
        return np.repeat(coords[:1], n, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    samples = np.linspace(0.0, total, n, endpoint=False)
    out = np.zeros((n, 2), dtype=float)
    for i, s in enumerate(samples):
        j = int(np.searchsorted(cum, s, side="right") - 1)
        j = min(max(j, 0), len(seg_len) - 1)
        t = (s - cum[j]) / seg_len[j] if seg_len[j] > 0 else 0.0
        out[i] = coords[j] + t * seg[j]
    return out


def _pca_align(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Rotate points so the major principal axis aligns with +X.
    Returns aligned points and orientation angle in degrees (atan2 of major axis).
    """
    if len(pts) < 2:
        return pts.copy(), 0.0

    cov = np.cov(pts.T)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return pts.copy(), 0.0

    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.atan2(major[1], major[0])
    c, s = math.cos(-angle), math.sin(-angle)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    aligned = pts @ rot.T

    # Ambiguity: flip 180° so mean x of the right half is non-negative when possible.
    if float(np.mean(aligned[:, 0])) < 0:
        aligned = -aligned
        angle = angle + math.pi
    return aligned, math.degrees(angle) % 180.0


def contour_signature(
    geom: BaseGeometry,
    n: int = DEFAULT_N_POINTS,
) -> tuple[np.ndarray, float]:
    """
    Translation-/scale-/rotation-normalized contour signature.

    Returns
    -------
    pts : (n, 2) array in unit-ish coordinates (divided by sqrt(area))
    orientation_deg : PCA major-axis angle in [0, 180)
    """
    pts = resample_exterior(geom, n=n)
    if geom is None or geom.is_empty:
        return pts, float("nan")

    centroid = np.array([geom.centroid.x, geom.centroid.y], dtype=float)
    pts = pts - centroid
    area = float(geom.area)
    scale = math.sqrt(max(area, 1e-12))
    pts = pts / scale
    aligned, orientation = _pca_align(pts)
    return aligned, orientation


def _nn_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each point in a, distance to nearest point in b."""
    # (na, 1, 2) - (1, nb, 2) → (na, nb)
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbor distance."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return float(0.5 * (_nn_dists(a, b).mean() + _nn_dists(b, a).mean()))


def hausdorff_approx(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric max nearest-neighbor distance (discrete Hausdorff)."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return float(max(_nn_dists(a, b).max(), _nn_dists(b, a).max()))


def shape_similarity(
    geom_a: BaseGeometry,
    geom_b: BaseGeometry,
    n: int = DEFAULT_N_POINTS,
    sig_a: tuple[np.ndarray, float] | None = None,
    sig_b: tuple[np.ndarray, float] | None = None,
) -> float:
    """
    Shape similarity in [0, 1] from PCA-aligned contour Chamfer (+ Hausdorff).

    Tries the 180° PCA flip of ``sig_b`` and keeps the better score.
    """
    if geom_a is None or geom_b is None or geom_a.is_empty or geom_b.is_empty:
        return 0.0

    pts_a, _ = sig_a if sig_a is not None else contour_signature(geom_a, n=n)
    pts_b, _ = sig_b if sig_b is not None else contour_signature(geom_b, n=n)

    best = 0.0
    for candidate in (pts_b, -pts_b):
        chamfer = chamfer_distance(pts_a, candidate)
        haus = hausdorff_approx(pts_a, candidate)
        if not math.isfinite(chamfer):
            continue
        chamfer_score = math.exp(-chamfer / CHAMFER_SCALE)
        haus_score = math.exp(-haus / HAUSDORFF_SCALE)
        score = 0.75 * chamfer_score + 0.25 * haus_score
        if score > best:
            best = score
    return best


def attach_shape_signatures(gdf, n: int = DEFAULT_N_POINTS):
    """Add ``shape_sig`` (ndarray) and PCA ``orientation`` columns in-place copy."""
    gdf = gdf.copy()
    sigs = []
    orients = []
    for geom in gdf.geometry:
        pts, ang = contour_signature(geom, n=n)
        sigs.append(pts)
        orients.append(ang)
    gdf["shape_sig"] = sigs
    gdf["orientation"] = orients
    return gdf
