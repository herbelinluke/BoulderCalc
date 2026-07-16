"""Training-time augmentations for nadir boulder imagery (RGB and RGB+DSM).

Geometric augs treat orientation as arbitrary (full-circle rotation + both flips)
and apply bilinear resize/rotate to the image while Detectron2 keeps mask / seg
channels on nearest-neighbor (polygons are remapped by coordinates).

Photometric augs target Atlantic coastal domain shift and only touch the first
three channels so a DSM band is never color-jittered, noised, or shadowed.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from detectron2.data import transforms as T
from detectron2.data.transforms import ColorTransform, NoOpTransform
from PIL import Image


def _rgb_channels(img: np.ndarray) -> np.ndarray:
    """First three channels as a contiguous HxWx3 view (BGR or RGB)."""
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Expected HxWxC with C>=3, got {img.shape}")
    return np.ascontiguousarray(img[:, :, :3])


def _apply_rgb_only(img: np.ndarray, op: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Run ``op`` on bands 0–2; leave any extra bands (e.g. DSM) unchanged."""
    out = np.array(img, copy=True)
    out[:, :, :3] = op(_rgb_channels(out))
    return out


def _clip_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


class RGBOnlyColorTransform(ColorTransform):
    """Photometric ColorTransform that preserves band 4+ (DSM)."""

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return _apply_rgb_only(img, self.op)


class RandomBrightnessContrast(T.Augmentation):
    """Moderate brightness/contrast jitter on RGB only."""

    def __init__(self, brightness: float = 0.25, contrast: float = 0.25):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        b = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        c = 1.0 + np.random.uniform(-self.contrast, self.contrast)

        def op(rgb: np.ndarray) -> np.ndarray:
            x = rgb.astype(np.float32)
            mean = x.mean(axis=(0, 1), keepdims=True)
            x = (x - mean) * c + mean
            x = x * b
            return _clip_u8(x)

        return RGBOnlyColorTransform(op)


class RandomHueSaturation(T.Augmentation):
    """Mild hue/saturation jitter via OpenCV HSV (BGR input, RGB-only writeback)."""

    def __init__(self, hue_delta: float = 0.04, saturation: float = 0.2):
        """
        Args:
            hue_delta: Max |ΔH| as a fraction of the full hue circle (OpenCV H in [0, 180]).
            saturation: Relative saturation scale half-width (1±saturation).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        dh = np.random.uniform(-self.hue_delta, self.hue_delta) * 180.0
        sat_scale = 1.0 + np.random.uniform(-self.saturation, self.saturation)

        def op(bgr: np.ndarray) -> np.ndarray:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + dh) % 180.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
            return cv2.cvtColor(_clip_u8(hsv), cv2.COLOR_HSV2BGR)

        return RGBOnlyColorTransform(op)


class RandomGaussianNoise(T.Augmentation):
    """Sensor / session noise floor differences."""

    def __init__(self, std_max: float = 10.0):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        std = float(np.random.uniform(0.0, self.std_max))
        if std < 1e-3:
            return NoOpTransform()

        def op(rgb: np.ndarray) -> np.ndarray:
            noise = np.random.normal(0.0, std, size=rgb.shape).astype(np.float32)
            return _clip_u8(rgb.astype(np.float32) + noise)

        return RGBOnlyColorTransform(op)


class RandomMotionOrDefocusBlur(T.Augmentation):
    """Slight motion blur (drone) or defocus blur (altitude-dependent focus)."""

    def __init__(self, max_kernel: int = 7):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        k = int(np.random.choice(list(range(3, self.max_kernel + 1, 2))))
        use_motion = bool(np.random.rand() < 0.5)

        if use_motion:
            angle = float(np.random.uniform(0.0, 180.0))
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0
            rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
            kernel = cv2.warpAffine(kernel, rot, (k, k))
            kernel = kernel / max(kernel.sum(), 1e-6)

            def op(rgb: np.ndarray) -> np.ndarray:
                return cv2.filter2D(rgb, -1, kernel)

        else:

            def op(rgb: np.ndarray) -> np.ndarray:
                return cv2.GaussianBlur(rgb, (k, k), 0)

        return RGBOnlyColorTransform(op)


class RandomSyntheticShadow(T.Augmentation):
    """Directional soft shadow overlay for low-sun coastal domain shift."""

    def __init__(self, strength_min: float = 0.25, strength_max: float = 0.55):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        strength = float(np.random.uniform(self.strength_min, self.strength_max))
        # Ellipse center / axes in image coords; elongated to read as cast shadow.
        cx = float(np.random.uniform(0.15 * w, 0.85 * w))
        cy = float(np.random.uniform(0.15 * h, 0.85 * h))
        ax = float(np.random.uniform(0.25 * w, 0.7 * w))
        ay = float(np.random.uniform(0.12 * h, 0.45 * h))
        angle = float(np.random.uniform(0.0, 180.0))

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cos_a = np.cos(np.deg2rad(angle))
        sin_a = np.sin(np.deg2rad(angle))
        x = xx - cx
        y = yy - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        # Soft falloff: 1 inside core → 0 outside.
        r2 = (xr / max(ax, 1.0)) ** 2 + (yr / max(ay, 1.0)) ** 2
        mask = np.clip(1.0 - r2, 0.0, 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(ax, ay) * 0.08)
        shade = (1.0 - strength * mask).astype(np.float32)

        def op(rgb: np.ndarray) -> np.ndarray:
            return _clip_u8(rgb.astype(np.float32) * shade[:, :, None])

        return RGBOnlyColorTransform(op)


def build_boulder_train_augs(
    image_size: int,
    *,
    scale_min: float = 0.5,
    scale_max: float = 1.5,
    brightness: float = 0.25,
    contrast: float = 0.25,
    hue_delta: float = 0.04,
    saturation: float = 0.2,
    noise_std: float = 10.0,
    blur_prob: float = 0.35,
    shadow_prob: float = 0.4,
    noise_prob: float = 0.4,
    color_prob: float = 0.8,
) -> list:
    """Full train augmentation stack for crowd-ignore and RGB+DSM mappers.

    Scale jitter is for network generalization only — metric scale still comes
    from the SfM/DEM pipeline at inference/volume time, not from these pixels.
    """
    if image_size < 32:
        raise ValueError(f"image_size too small: {image_size}")

    augs: list = [
        # Full-circle rotation (nadir: no gravity "up"). Bilinear image; seg=nearest.
        T.RandomRotation([-180, 180], expand=True, sample_style="range", interp=cv2.INTER_LINEAR),
        T.RandomFlip(horizontal=True, vertical=False),
        T.RandomFlip(horizontal=False, vertical=True),
        # Altitude / GSD-like scale jitter, then square crop/pad to model size.
        T.ResizeScale(
            min_scale=scale_min,
            max_scale=scale_max,
            target_height=image_size,
            target_width=image_size,
            interp=Image.BILINEAR,
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size), pad=True, pad_value=0.0),
        # Photometric (lossy) — RGB only when a DSM band is present.
        T.RandomApply(RandomBrightnessContrast(brightness, contrast), prob=color_prob),
        T.RandomApply(RandomHueSaturation(hue_delta, saturation), prob=color_prob),
        T.RandomApply(RandomGaussianNoise(noise_std), prob=noise_prob),
        T.RandomApply(RandomMotionOrDefocusBlur(max_kernel=7), prob=blur_prob),
        T.RandomApply(RandomSyntheticShadow(), prob=shadow_prob),
    ]
    return augs


def build_boulder_test_augs(image_size: int) -> list:
    """Deterministic resize for val/test (no photometric / flip / rotate)."""
    return [T.ResizeShortestEdge(image_size, image_size, sample_style="choice", interp=Image.BILINEAR)]
