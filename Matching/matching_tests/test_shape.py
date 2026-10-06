# test_shape.py

import math

import numpy as np
import pytest
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Polygon

from matching.shape import (
    chamfer_distance,
    contour_signature,
    resample_exterior,
    shape_similarity,
)


def _triangle():
    return Polygon([(0, 0), (3, 0), (1.2, 2.5)])


def _elongated():
    return Polygon([(0, 0), (4, 0), (4, 1), (0, 1)])


def test_resample_exterior_count():
    pts = resample_exterior(_triangle(), n=32)
    assert pts.shape == (32, 2)


def test_contour_signature_translation_invariance():
    geom = _triangle()
    moved = translate(geom, xoff=12.5, yoff=-7.3)
    a, _ = contour_signature(geom)
    b, _ = contour_signature(moved)
    assert np.allclose(a, b, atol=1e-6)


def test_contour_signature_scale_invariance():
    geom = _triangle()
    bigger = scale(geom, xfact=2.5, yfact=2.5, origin="centroid")
    a, _ = contour_signature(geom)
    b, _ = contour_signature(bigger)
    # After scale normalization + PCA, contours should nearly match.
    assert chamfer_distance(a, b) < 0.05 or chamfer_distance(a, -b) < 0.05


def test_shape_similarity_high_for_rotated_copy():
    geom = _elongated()
    rotated = rotate(geom, angle=37.0, origin="centroid")
    score = shape_similarity(geom, rotated)
    assert score > 0.85


def test_shape_similarity_low_for_dissimilar():
    score = shape_similarity(_triangle(), _elongated())
    assert score < 0.75


def test_shape_similarity_prefers_same_over_different():
    base = _triangle()
    twin = translate(rotate(base, 15, origin="centroid"), xoff=40, yoff=10)
    other = translate(_elongated(), xoff=40, yoff=10)
    assert shape_similarity(base, twin) > shape_similarity(base, other)
