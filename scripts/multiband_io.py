"""Shared helpers for RGB+DSM (4-band) Detectron2 training and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn


# Detectron2 COCO BGR means; DSM band uses mid-gray after uint8 stretch.
FOUR_BAND_PIXEL_MEAN = [103.530, 116.280, 123.675, 127.500]
FOUR_BAND_PIXEL_STD = [1.0, 1.0, 1.0, 1.0]


def load_rgbd_uint8(path: Path | str) -> np.ndarray:
    """Load a 4-band GeoTIFF as HxWx4 uint8 in RGB+DSM order."""
    with rasterio.open(path) as ds:
        arr = ds.read()
    if arr.shape[0] < 4:
        raise ValueError(f"{path} has {arr.shape[0]} bands; need 4 (RGB+DSM)")
    img = np.transpose(arr[:4], (1, 2, 0))
    return np.clip(img, 0, 255).astype(np.uint8)


def rgbd_to_bgrd(rgbd: np.ndarray) -> np.ndarray:
    """Convert HxWx4 RGB+DSM to BGR+DSM (Detectron2 INPUT.FORMAT=BGR)."""
    if rgbd.ndim != 3 or rgbd.shape[2] != 4:
        raise ValueError(f"Expected HxWx4, got {rgbd.shape}")
    bgrd = rgbd.copy()
    bgrd[:, :, :3] = rgbd[:, :, :3][:, :, ::-1]
    return bgrd


def load_bgrd_uint8(path: Path | str) -> np.ndarray:
    """Load 4-band GeoTIFF as HxWx4 uint8 in BGR+DSM order for the model."""
    return rgbd_to_bgrd(load_rgbd_uint8(path))


def rgb_preview_from_rgbd(rgbd: np.ndarray) -> np.ndarray:
    """First three bands as RGB for visualization overlays."""
    return np.ascontiguousarray(rgbd[:, :, :3])


def expand_stem_conv1_from_rgb(model: nn.Module, rgb_weight: torch.Tensor) -> None:
    """Copy COCO 3-channel conv1 weights into a 4-channel stem; init DSM from RGB mean."""
    stem_conv = model.backbone.bottom_up.stem.conv1
    weight = stem_conv.weight
    if weight.shape[1] != 4:
        raise ValueError(f"Expected 4-channel stem, got in_channels={weight.shape[1]}")
    if rgb_weight.shape[1] != 3:
        raise ValueError(f"Expected 3-channel checkpoint stem, got {rgb_weight.shape}")
    with torch.no_grad():
        weight[:, :3].copy_(rgb_weight.to(device=weight.device, dtype=weight.dtype))
        weight[:, 3:4].copy_(rgb_weight.mean(dim=1, keepdim=True).to(device=weight.device, dtype=weight.dtype))


def load_rgb_stem_weight(checkpoint_path: str | Path) -> torch.Tensor:
    """Load backbone.bottom_up.stem.conv1.weight from a Detectron2 checkpoint."""
    path = Path(checkpoint_path)
    key = "backbone.bottom_up.stem.conv1.weight"
    if path.suffix == ".pkl":
        import pickle

        with open(path, "rb") as f:
            ckpt = pickle.load(f, encoding="latin1")
        state = ckpt.get("model", ckpt)
        if key not in state:
            raise KeyError(f"{key} not found in checkpoint keys (sample: {list(state)[:8]})")
        weight = state[key]
        if not torch.is_tensor(weight):
            weight = torch.as_tensor(weight)
        return weight

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    if key not in state:
        raise KeyError(f"{key} not found in checkpoint keys (sample: {list(state)[:8]})")
    return state[key]


def patch_four_band_stem_from_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    """After loading a 3-channel zoo checkpoint into a 4-channel model, fix stem.conv1."""
    rgb_w = load_rgb_stem_weight(checkpoint_path)
    expand_stem_conv1_from_rgb(model, rgb_w)
