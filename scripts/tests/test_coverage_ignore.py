#!/usr/bin/env python3
"""Synthetic tests for ocean-safe coverage ignore masks."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]


def _load_coverage_ignore():
    """Load coverage_ignore with lightweight stubs when heavy deps are missing."""
    try:
        sys.path.insert(0, str(SCRIPTS))
        import coverage_ignore as mod  # noqa: WPS433

        return mod
    except ImportError:
        pass

    # Stub rasterio / scipy / shapely enough to import array helpers only.
    if "rasterio" not in sys.modules:
        rio = types.ModuleType("rasterio")
        rio.band = None
        sys.modules["rasterio"] = rio
        sys.modules["rasterio.features"] = types.ModuleType("rasterio.features")
        sys.modules["rasterio.warp"] = types.SimpleNamespace(
            Resampling=types.SimpleNamespace(bilinear=1),
            reproject=None,
        )
    if "scipy" not in sys.modules:
        scipy = types.ModuleType("scipy")
        nd = types.ModuleType("scipy.ndimage")

        def _label(mask):
            from collections import deque

            mask = np.asarray(mask, dtype=bool)
            out = np.zeros(mask.shape, dtype=np.int32)
            n = 0
            h, w = mask.shape
            for y in range(h):
                for x in range(w):
                    if not mask[y, x] or out[y, x]:
                        continue
                    n += 1
                    q = deque([(y, x)])
                    out[y, x] = n
                    while q:
                        cy, cx = q.popleft()
                        for ny, nx in (
                            (cy - 1, cx),
                            (cy + 1, cx),
                            (cy, cx - 1),
                            (cy, cx + 1),
                        ):
                            if (
                                0 <= ny < h
                                and 0 <= nx < w
                                and mask[ny, nx]
                                and out[ny, nx] == 0
                            ):
                                out[ny, nx] = n
                                q.append((ny, nx))
            return out, n

        def _dilate(mask, iterations=1):
            out = np.asarray(mask, dtype=bool).copy()
            for _ in range(iterations):
                pad = np.pad(out, 1, mode="constant", constant_values=False)
                grown = (
                    pad[1:-1, 1:-1]
                    | pad[:-2, 1:-1]
                    | pad[2:, 1:-1]
                    | pad[1:-1, :-2]
                    | pad[1:-1, 2:]
                )
                out = grown
            return out

        nd.label = _label
        nd.binary_dilation = _dilate
        scipy.ndimage = nd
        sys.modules["scipy"] = scipy
        sys.modules["scipy.ndimage"] = nd
    if "shapely" not in sys.modules:
        shapely = types.ModuleType("shapely")
        geom = types.ModuleType("shapely.geometry")
        ops = types.ModuleType("shapely.ops")

        class _Poly:
            pass

        geom.Polygon = _Poly
        geom.shape = lambda g: g
        ops.unary_union = lambda x: x
        sys.modules["shapely"] = shapely
        sys.modules["shapely.geometry"] = geom
        sys.modules["shapely.ops"] = ops

    spec = importlib.util.spec_from_file_location(
        "coverage_ignore", SCRIPTS / "coverage_ignore.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestCoverageMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_coverage_ignore()

    def test_border_black_masked(self):
        # Left strip black (border-connected); rest mid-gray.
        rgb = np.full((3, 32, 32), 80, dtype=np.uint8)
        rgb[:, :, :4] = 0
        mask = self.mod.build_coverage_mask_from_arrays(
            rgb, pixel_size=0.03, dsm_valid=None, rgb_max=8, blur_m=0.0
        )
        self.assertTrue(mask[:, :4].all())
        self.assertFalse(mask[:, 8:].any())

    def test_interior_dark_ocean_not_masked(self):
        # Interior dark block not touching border → must stay unmasked.
        rgb = np.full((3, 40, 40), 90, dtype=np.uint8)
        rgb[:, 15:25, 15:25] = 2
        mask = self.mod.build_coverage_mask_from_arrays(
            rgb, pixel_size=0.03, dsm_valid=None, rgb_max=8, blur_m=0.0
        )
        self.assertFalse(mask[15:25, 15:25].any())
        self.assertFalse(mask.any())

    def test_dsm_gap_over_textured_rgb_not_masked(self):
        rgb = np.full((3, 24, 24), 60, dtype=np.uint8)
        dsm_valid = np.ones((24, 24), dtype=bool)
        dsm_valid[5:15, 5:15] = False  # gap over textured RGB
        mask = self.mod.build_coverage_mask_from_arrays(
            rgb, pixel_size=0.03, dsm_valid=dsm_valid, rgb_max=8, blur_m=0.0
        )
        self.assertFalse(mask.any())

    def test_dsm_gap_in_black_void_masked(self):
        rgb = np.full((3, 24, 24), 70, dtype=np.uint8)
        rgb[:, :, :6] = 0  # border black void
        dsm_valid = np.ones((24, 24), dtype=bool)
        dsm_valid[:, :3] = False
        mask = self.mod.build_coverage_mask_from_arrays(
            rgb, pixel_size=0.03, dsm_valid=dsm_valid, rgb_max=8, blur_m=0.0
        )
        self.assertTrue(mask[:, :3].all())
        self.assertTrue(mask[:, 3:6].all())  # RGB void even where DSM valid
        self.assertFalse(mask[:, 10:].any())

    def test_blur_dilate_extends_void(self):
        rgb = np.full((3, 20, 20), 100, dtype=np.uint8)
        rgb[:, :, 0] = 0
        # 0.09 m / 0.03 m/px = 3 px dilate
        mask = self.mod.build_coverage_mask_from_arrays(
            rgb, pixel_size=0.03, dsm_valid=None, rgb_max=8, blur_m=0.09
        )
        self.assertTrue(mask[:, 0].all())
        self.assertTrue(mask[:, 1:3].any())
        self.assertFalse(mask[:, 8:].any())


if __name__ == "__main__":
    unittest.main()
